#pragma once
#include "common.hpp"
#include "logging.hpp"
#include "platform/platform_io.hpp"
#include "thread_pool.h"
#include <future>

// ════════════════════════════════════════════════════════════════════════════════
//  O_DIRECT Expert Loader with LRU Cache
// ════════════════════════════════════════════════════════════════════════════════

struct ExpertCacheEntry {
    int layer_id = -1;
    int expert_id = -1;
    int64_t last_used = 0;      // for LRU eviction
    void* gpu_data = nullptr;   // pointer into expert cache pool
    int slot_index = -1;
};

class ExpertLoader {
public:
    int expert_block_size_ = 0;
    int n_layers_ = 0;
    int n_experts_ = 0;
    moecher::platform::DirectFileHandle expert_file_;

    // L1 LRU cache (VRAM)
    int cache_capacity_ = 0;
    void* cache_pool_gpu_ = nullptr;
    std::vector<ExpertCacheEntry> cache_slots_;
    std::unordered_map<int64_t, int> key_to_slot_;  // (layer*n_experts+expert) -> L1 slot index
    std::vector<void*> flat_vram_ptrs_;             // Lock-free flat array for 0-latency expert lookups

    // L2 LRU cache (DRAM)
    int dram_cache_capacity_ = 0;
    void* dram_cache_pool_ = nullptr;
    bool dram_is_pinned_ = false;  // True if DRAM cache is pinned (cudaMallocHost or cudaHostRegister succeeded)
    std::vector<ExpertCacheEntry> dram_cache_slots_;
    std::unordered_map<int64_t, int> dram_key_to_slot_; // (layer*n_experts+expert) -> L2 slot index
    std::vector<void*> flat_dram_ptrs_;                 // Lock-free flat array for 0-latency DRAM lookups

    // Pinned staging buffers for DMA when DRAM cache is unpinned
    // cudaMemcpyAsync from pageable memory is SYNCHRONOUS — it blocks the stream.
    // These pinned buffers enable: memcpy(DRAM→pinned) + cudaMemcpyAsync(pinned→VRAM) = true async DMA.
    static constexpr int NUM_DMA_STAGING = 8;  // One per expert stream
    void* dma_staging_[NUM_DMA_STAGING] = {nullptr};

    // Ring buffer for staging (used when bypassing DRAM cache)
    static constexpr int NUM_STAGING_BUFFERS = 32;
    struct StagingBuffer {
        void* ptr = nullptr;
        cudaEvent_t event = nullptr;
    };
    std::vector<StagingBuffer> staging_ring_;
    int staging_idx_ = 0;

    int64_t access_counter_ = 0;
    std::mutex cache_mutex_;

    // L2 victim cache: writeback stream for async VRAM→DRAM demotion
    cudaStream_t dram_writeback_stream_ = nullptr;
    cudaEvent_t writeback_event_ = nullptr;  // Ensures writeback completes before slot reuse

    // ── Mixed Quantization Support (Hot: NVFP4, Cold: IQ2_XXS) ──
    bool is_mixed_ = false;
    moecher::platform::DirectFileHandle expert_file_cold_;
    moecher::platform::DirectFileHandle expert_file_hot_;
    int cold_block_size_ = 7077888;
    int hot_block_size_ = 10223616;
    void* hot_pool_gpu_ = nullptr;
    void* cold_pool_gpu_ = nullptr;
    std::vector<uint8_t> expert_type_map_host_;
    GPUTensor expert_type_map_gpu_;

    const uint8_t* expert_type_map_gpu() const {
        return expert_type_map_gpu_.u8();
    }

    int n_pinned_layers() const {
        return (n_experts_ > 0) ? std::min(n_layers_, cache_capacity_ / n_experts_) : 0;
    }

    bool is_layer_pinned(int layer_id) const {
        return layer_id < n_pinned_layers();
    }

    int expert_block_size() const { return expert_block_size_; }

    void* get_dram_expert_ptr(int layer_id, int expert_id) {
        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;
        if (key >= 0 && key < (int64_t)flat_dram_ptrs_.size() && flat_dram_ptrs_[key]) {
            return flat_dram_ptrs_[key];
        }
        return get_dram_or_disk_ptr(layer_id, expert_id);
    }


    inline void* get_dram_or_disk_ptr(int layer_id, int expert_id) {
        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;
        if (key >= 0 && key < (int64_t)flat_dram_ptrs_.size()) {
            void* ptr = flat_dram_ptrs_[key];
            if (ptr) return ptr;
        }
        return fetch_cold_expert_from_disk(layer_id, expert_id, key);
    }

    void* fetch_cold_expert_from_disk(int layer_id, int expert_id, int64_t key) {
        std::unique_lock<std::mutex> lock(cache_mutex_);
        if (dram_cache_capacity_ > 0) {
            auto it = dram_key_to_slot_.find(key);
            if (it != dram_key_to_slot_.end()) {
                return dram_cache_slots_[it->second].gpu_data;
            }
            access_counter_++;
            int evict_dram = 0;
            int64_t oldest_dram = INT64_MAX;
            for (int i = 0; i < dram_cache_capacity_; i++) {
                if (dram_cache_slots_[i].layer_id < 0) {
                    evict_dram = i;
                    break;
                }
                if (dram_cache_slots_[i].last_used < oldest_dram) {
                    oldest_dram = dram_cache_slots_[i].last_used;
                    evict_dram = i;
                }
            }
            auto& dslot = dram_cache_slots_[evict_dram];
            if (dslot.layer_id >= 0) {
                int64_t old_dram_key = (int64_t)dslot.layer_id * n_experts_ + dslot.expert_id;
                dram_key_to_slot_.erase(old_dram_key);
                if (old_dram_key >= 0 && old_dram_key < (int64_t)flat_dram_ptrs_.size()) {
                    flat_dram_ptrs_[old_dram_key] = nullptr;
                }
            }
            dslot.layer_id = layer_id;
            dslot.expert_id = expert_id;
            dslot.last_used = access_counter_;
            dram_key_to_slot_[key] = evict_dram;

            int64_t file_offset = (int64_t)key * expert_block_size_;
            expert_file_.pread_exact(dslot.gpu_data, expert_block_size_, file_offset);
            if (key >= 0 && key < (int64_t)flat_dram_ptrs_.size()) {
                flat_dram_ptrs_[key] = dslot.gpu_data;
            }
            return dslot.gpu_data;
        } else {
            int stage_idx = staging_idx_;
            staging_idx_ = (staging_idx_ + 1) % NUM_STAGING_BUFFERS;
            auto& stage = staging_ring_[stage_idx];
            int64_t file_offset = (int64_t)key * expert_block_size_;
            expert_file_.pread_exact(stage.ptr, expert_block_size_, file_offset);
            return stage.ptr;
        }
    }
    // ── L2 Victim Cache: demote evicted L1 expert to L2 DRAM ──────────────
    // Called under cache_mutex_. Queues async VRAM→DRAM copy on writeback stream.
    // Throttled to MAX_DEMOTIONS_PER_TOKEN to avoid saturating PCIe with D2H traffic
    // which competes with NVMe SSD reads on the same bus.
    static constexpr int MAX_DEMOTIONS_PER_TOKEN = 16;
    int demotions_this_token_ = 0;
    int64_t demotion_token_id_ = -1;  // access_counter_ of current token
    bool disable_demotion_ = false;

    void set_demotion_enabled(bool enabled) { disable_demotion_ = !enabled; }

    void demote_to_l2(int layer_id, int expert_id, int64_t key, void* vram_src) {
        if (disable_demotion_ || dram_cache_capacity_ <= 0 || !dram_writeback_stream_) return;

        // Reset demotion counter on new token
        if (access_counter_ != demotion_token_id_) {
            demotion_token_id_ = access_counter_;
            demotions_this_token_ = 0;
        }

        // Throttle: skip demotion if we've hit the per-token cap
        if (demotions_this_token_ >= MAX_DEMOTIONS_PER_TOKEN) return;

        // Skip if already in L2 (e.g. preloaded and never evicted)
        if (dram_key_to_slot_.count(key)) return;

        // Find LRU eviction candidate in L2
        int evict_dram = -1;
        int64_t oldest_d = INT64_MAX;
        for (int i = 0; i < dram_cache_capacity_; i++) {
            if (dram_cache_slots_[i].layer_id < 0) { evict_dram = i; break; }
            if (dram_cache_slots_[i].last_used < oldest_d) {
                oldest_d = dram_cache_slots_[i].last_used;
                evict_dram = i;
            }
        }
        if (evict_dram < 0) return;

        auto& ds = dram_cache_slots_[evict_dram];
        if (ds.layer_id >= 0) {
            int64_t old_key = (int64_t)ds.layer_id * n_experts_ + ds.expert_id;
            dram_key_to_slot_.erase(old_key);
        }

        ds.layer_id = layer_id;
        ds.expert_id = expert_id;
        ds.last_used = access_counter_;
        dram_key_to_slot_[key] = evict_dram;

        // Async copy VRAM → pinned DRAM (non-blocking, dedicated stream)
        cudaMemcpyAsync(ds.gpu_data, vram_src, expert_block_size_,
                        cudaMemcpyDeviceToHost, dram_writeback_stream_);
        // Record event so L2 hit path can wait for writeback to complete
        cudaEventRecord(writeback_event_, dram_writeback_stream_);
        demotions_this_token_++;
    }

    // Call before submitting new H2D DMA to an evicted L1 slot.
    // Makes the expert stream wait for any pending writeback to complete.
    void wait_for_writeback(cudaStream_t expert_stream) {
        if (dram_writeback_stream_ && writeback_event_) {
            cudaStreamWaitEvent(expert_stream, writeback_event_, 0);
        }
    }

    bool init(const std::string& expert_bin_path, int block_size,
              int n_layers, int n_experts, size_t cache_budget_bytes, size_t dram_budget_bytes,
              bool use_buffered_io = false) {
        expert_block_size_ = block_size;
        n_layers_ = n_layers;
        n_experts_ = n_experts;

        bool use_direct = !use_buffered_io;
        if (!expert_file_.open_read(expert_bin_path, use_direct)) {
            LOG_ERROR("Cannot open expert bin: %s", expert_bin_path.c_str());
            return false;
        }
        if (use_buffered_io) {
            LOG_INFO("Expert I/O: BUFFERED mode (OS file cache enabled — recommended for offloading)");
        } else {
            LOG_INFO("Expert I/O: DIRECT mode (FILE_FLAG_NO_BUFFERING / O_DIRECT)");
        }

        cache_capacity_ = (int)(cache_budget_bytes / block_size);
        if (cache_capacity_ < 64) cache_capacity_ = 64;
        LOG_INFO("Expert L1 cache: %d slots (%.1f GB)", cache_capacity_,
                 (double)cache_capacity_ * block_size / (1024.0 * 1024.0 * 1024.0));

        CUDA_CHECK(cudaMalloc(&cache_pool_gpu_, (size_t)cache_capacity_ * block_size));

        cache_slots_.resize(cache_capacity_);
        for (int i = 0; i < cache_capacity_; i++) {
            cache_slots_[i].slot_index = i;
            cache_slots_[i].gpu_data = (char*)cache_pool_gpu_ + (size_t)i * block_size;
        }

        dram_cache_capacity_ = (int)(dram_budget_bytes / block_size);
        if (dram_cache_capacity_ > 0) {
            LOG_INFO("Expert L2 cache: %d slots (%.1f GB)", dram_cache_capacity_,
                     (double)dram_cache_capacity_ * block_size / (1024.0 * 1024.0 * 1024.0));
            size_t total_dram_bytes = (size_t)dram_cache_capacity_ * block_size;
            cudaError_t err = cudaMallocHost(&dram_cache_pool_, total_dram_bytes);
            if (err != cudaSuccess) {
                cudaGetLastError(); // Clear error state
#if defined(_WIN32) || defined(_WIN64)
                dram_cache_pool_ = VirtualAlloc(NULL, total_dram_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
                dram_cache_pool_ = aligned_alloc(4096, total_dram_bytes);
#endif
                if (dram_cache_pool_) {
                    cudaError_t reg_err = cudaHostRegister(dram_cache_pool_, total_dram_bytes, cudaHostRegisterDefault);
                    if (reg_err != cudaSuccess) {
                        cudaGetLastError();
                        dram_is_pinned_ = false;
                        LOG_WARN("cudaHostRegister could not pin %.1f GB DRAM cache. Using pinned staging buffers for async DMA.",
                                 (double)total_dram_bytes / (1024.0 * 1024.0 * 1024.0));
                    } else {
                        dram_is_pinned_ = true;
                        LOG_INFO("Pinned %.1f GB DRAM cache into physical RAM with cudaHostRegister",
                                 (double)total_dram_bytes / (1024.0 * 1024.0 * 1024.0));
                    }
                } else {
                    LOG_ERROR("System RAM allocation also failed. Disabling L2 DRAM cache.");
                    dram_cache_capacity_ = 0;
                }
            } else {
                dram_is_pinned_ = true;
                LOG_INFO("Allocated %.1f GB pinned DRAM cache with cudaMallocHost",
                         (double)total_dram_bytes / (1024.0 * 1024.0 * 1024.0));
            }
            if (dram_cache_capacity_ > 0 && dram_cache_pool_) {
                dram_cache_slots_.resize(dram_cache_capacity_);
                for (int i = 0; i < dram_cache_capacity_; i++) {
                    dram_cache_slots_[i].slot_index = i;
                    dram_cache_slots_[i].gpu_data = (char*)dram_cache_pool_ + (size_t)i * block_size;
                }
            }
        }

        // Initialize staging ring buffer
        staging_ring_.resize(NUM_STAGING_BUFFERS);
        for (auto& s : staging_ring_) {
            CUDA_CHECK(cudaMallocHost(&s.ptr, block_size));
            CUDA_CHECK(cudaEventCreateWithFlags(&s.event, cudaEventDisableTiming));
        }

        flat_vram_ptrs_.assign((size_t)n_layers * n_experts, nullptr);
        flat_dram_ptrs_.assign((size_t)n_layers * n_experts, nullptr);
        flat_vram_ptrs_gpu_.alloc((size_t)n_layers * n_experts * sizeof(void*));
        CUDA_CHECK(cudaMemset(flat_vram_ptrs_gpu_.data, 0, (size_t)n_layers * n_experts * sizeof(void*)));

        // Create writeback stream for L2 victim cache demotion
        if (dram_cache_capacity_ > 0) {
            CUDA_CHECK(cudaStreamCreate(&dram_writeback_stream_));
            CUDA_CHECK(cudaEventCreateWithFlags(&writeback_event_, cudaEventDisableTiming));
            LOG_INFO("L2 victim cache enabled: evicted L1 experts will be demoted to L2 DRAM");
        }

        // Allocate pinned DMA staging buffers (used when DRAM cache is unpinned)
        if (!dram_is_pinned_ && dram_cache_capacity_ > 0) {
            LOG_INFO("Allocating %d pinned DMA staging buffers (%.1f MB each) for async DRAM→VRAM transfers",
                     NUM_DMA_STAGING, (double)block_size / (1024.0 * 1024.0));
            for (int i = 0; i < NUM_DMA_STAGING; i++) {
                CUDA_CHECK(cudaMallocHost(&dma_staging_[i], block_size));
            }
        }
        return true;
    }

    bool init_mixed(const std::string& cold_bin_path, int cold_block_size,
                    const std::string& hot_bin_path, int hot_block_size,
                    const std::string& map_file_path,
                    int n_layers, int n_experts,
                    bool use_buffered_io = false) {
        is_mixed_ = true;
        cold_block_size_ = cold_block_size;
        hot_block_size_ = hot_block_size;
        expert_block_size_ = cold_block_size; // For fallback compatibility
        n_layers_ = n_layers;
        n_experts_ = n_experts;

        bool use_direct = !use_buffered_io;
        if (!expert_file_cold_.open_read(cold_bin_path, use_direct)) {
            LOG_ERROR("Cannot open cold expert bin: %s", cold_bin_path.c_str());
            return false;
        }
        if (!expert_file_hot_.open_read(hot_bin_path, use_direct)) {
            LOG_ERROR("Cannot open hot expert bin: %s", hot_bin_path.c_str());
            return false;
        }

        // Read mixed expert map
        int total_experts = n_layers * n_experts;
        expert_type_map_host_.resize(total_experts, 0);

        std::ifstream mf(map_file_path, std::ios::binary);
        if (!mf.is_open()) {
            LOG_ERROR("Cannot open mixed expert map file: %s", map_file_path.c_str());
            return false;
        }
        mf.read(reinterpret_cast<char*>(expert_type_map_host_.data()), total_experts);
        if (!mf) {
            LOG_ERROR("Failed to read %d bytes from expert map: %s", total_experts, map_file_path.c_str());
            return false;
        }
        mf.close();

        int total_hot = 0;
        int total_cold = 0;
        for (int i = 0; i < total_experts; i++) {
            if (expert_type_map_host_[i] == 1) total_hot++;
            else total_cold++;
        }

        size_t hot_bytes = (size_t)total_hot * hot_block_size;
        size_t cold_bytes = (size_t)total_cold * cold_block_size;
        size_t total_bytes = hot_bytes + cold_bytes;

        LOG_INFO("══════════════════════════════════════════════════════════════════════");
        LOG_INFO("ExpertLoader: Initializing Mixed Quantization MoE (100%% VRAM Resident)");
        LOG_INFO("  Total Experts : %d (%d layers x %d)", total_experts, n_layers, n_experts);
        LOG_INFO("  Hot Experts   : %d (%.2f GB) in 2:4 Sparse NVFP4", total_hot, (double)hot_bytes / (1024.0 * 1024.0 * 1024.0));
        LOG_INFO("  Cold Experts  : %d (%.2f GB) in IQ2_XXS + Q2_K", total_cold, (double)cold_bytes / (1024.0 * 1024.0 * 1024.0));
        LOG_INFO("  Total MoE VRAM: %.2f GB", (double)total_bytes / (1024.0 * 1024.0 * 1024.0));
        LOG_INFO("══════════════════════════════════════════════════════════════════════");

        // Copy map to GPU
        expert_type_map_gpu_.alloc(total_experts);
        CUDA_CHECK(cudaMemcpy(expert_type_map_gpu_.data, expert_type_map_host_.data(),
                              total_experts, cudaMemcpyHostToDevice));

        // Allocate VRAM pools
        CUDA_CHECK(cudaMalloc(&hot_pool_gpu_, hot_bytes));
        CUDA_CHECK(cudaMalloc(&cold_pool_gpu_, cold_bytes));

        // All resident capacity: 100% of all experts in VRAM
        cache_capacity_ = total_experts;

        flat_vram_ptrs_.assign(total_experts, nullptr);
        flat_vram_ptrs_gpu_.alloc((size_t)total_experts * sizeof(void*));

        int cur_hot = 0;
        int cur_cold = 0;
        for (int i = 0; i < total_experts; i++) {
            if (expert_type_map_host_[i] == 1) {
                flat_vram_ptrs_[i] = (char*)hot_pool_gpu_ + (size_t)cur_hot * hot_block_size;
                cur_hot++;
            } else {
                flat_vram_ptrs_[i] = (char*)cold_pool_gpu_ + (size_t)cur_cold * cold_block_size;
                cur_cold++;
            }
        }

        return true;
    }

    bool all_resident(int active_layers = 0) const {
        int n_active = (active_layers > 0) ? active_layers : n_layers_;
        return cache_capacity_ >= n_active * n_experts_;
    }

    std::string get_expert_location(int layer_id, int expert_id) const {
        if (all_resident(n_layers_)) {
            return "L1 (VRAM)";
        }
        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;
        if (key >= 0 && key < (int64_t)flat_vram_ptrs_.size() && flat_vram_ptrs_[key] != nullptr) {
            return "L1 (VRAM)";
        }
        if (key >= 0 && key < (int64_t)flat_dram_ptrs_.size() && flat_dram_ptrs_[key] != nullptr) {
            return "L2 (DRAM)";
        }
        return "SSD (Disk)";
    }

    std::string get_expert_tier(int layer_id, int expert_id) const {
        if (all_resident(n_layers_)) {
            return "l1";
        }
        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;
        if (key >= 0 && key < (int64_t)flat_vram_ptrs_.size() && flat_vram_ptrs_[key] != nullptr) {
            return "l1";
        }
        if (key >= 0 && key < (int64_t)flat_dram_ptrs_.size() && flat_dram_ptrs_[key] != nullptr) {
            return "l2";
        }
        return "ssd";
    }

    inline void* try_get_expert_cached(int layer_id, int expert_id) {
        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;
        if (key >= 0 && key < (int64_t)flat_vram_ptrs_.size()) {
            return flat_vram_ptrs_[key];
        }
        return nullptr;
    }

    void* touch_expert_cached(int layer_id, int expert_id) {
        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;
        std::unique_lock<std::mutex> lock(cache_mutex_);
        auto it = key_to_slot_.find(key);
        if (it != key_to_slot_.end()) {
            access_counter_++;
            cache_slots_[it->second].last_used = access_counter_;
            return cache_slots_[it->second].gpu_data;
        }
        return nullptr;
    }

    void* get_expert(int layer_id, int expert_id, cudaStream_t stream) {
        return get_expert_async(layer_id, expert_id, stream);
    }

    void* get_expert_async(int layer_id, int expert_id, cudaStream_t stream) {
        std::unique_lock<std::mutex> lock(cache_mutex_);
        access_counter_++;

        int64_t key = (int64_t)layer_id * n_experts_ + expert_id;

        // 1. Check L1 Cache (VRAM)
        auto it = key_to_slot_.find(key);
        if (it != key_to_slot_.end()) {
            auto& slot = cache_slots_[it->second];
            slot.last_used = access_counter_;
            return slot.gpu_data;
        }

        // L1 Miss: find eviction candidate in L1 (never evict from pinned resident layers)
        int evict_slot = -1;
        int64_t oldest_time = INT64_MAX;
        for (int i = 0; i < cache_capacity_; i++) {
            if (cache_slots_[i].layer_id >= 0 && is_layer_pinned(cache_slots_[i].layer_id)) {
                continue;
            }
            if (cache_slots_[i].layer_id < 0) {
                evict_slot = i;
                break;
            }
            if (cache_slots_[i].last_used < oldest_time) {
                oldest_time = cache_slots_[i].last_used;
                evict_slot = i;
            }
        }

        if (evict_slot < 0) {
            return nullptr;
        }

        auto& slot = cache_slots_[evict_slot];
        if (slot.layer_id >= 0) {
            int64_t old_key = (int64_t)slot.layer_id * n_experts_ + slot.expert_id;
            key_to_slot_.erase(old_key);
            flat_vram_ptrs_[old_key] = nullptr;
            // Demote evicted expert to L2 DRAM (victim cache)
            demote_to_l2(slot.layer_id, slot.expert_id, old_key, slot.gpu_data);
        }

        slot.layer_id = layer_id;
        slot.expert_id = expert_id;
        slot.last_used = access_counter_;
        key_to_slot_[key] = evict_slot;
        flat_vram_ptrs_[key] = slot.gpu_data;

        void* host_src_ptr = nullptr;
        int stage_idx = -1;
        bool needs_disk_read = false;

        // 2. Check L2 Cache (DRAM) if available
        if (dram_cache_capacity_ > 0) {
            auto dram_it = dram_key_to_slot_.find(key);
            if (dram_it != dram_key_to_slot_.end()) {
                // L2 Hit!
                auto& dram_slot = dram_cache_slots_[dram_it->second];
                dram_slot.last_used = access_counter_;
                host_src_ptr = dram_slot.gpu_data;
                if (g_log_experts) {
                    LOG_INFO("[ExpertCache] L2 Hit: L%d E%d -> L1 slot %d", layer_id, expert_id, evict_slot);
                }
            } else {
                // L2 miss — use staging ring buffer for SSD read
                // Do NOT evict L2 entries: L2 is a stable fast cache, not SSD staging
                stage_idx = staging_idx_;
                staging_idx_ = (staging_idx_ + 1) % NUM_STAGING_BUFFERS;
                needs_disk_read = true;
                if (g_log_experts) {
                    LOG_INFO("[ExpertCache] L2 Miss (SSD read via staging): L%d E%d -> L1 slot %d", 
                             layer_id, expert_id, evict_slot);
                }
            }
        } else {
            // No L2 cache, fallback to staging buffer
            stage_idx = staging_idx_;
            staging_idx_ = (staging_idx_ + 1) % NUM_STAGING_BUFFERS;
            needs_disk_read = true;
            
            if (g_log_experts) {
                LOG_INFO("[ExpertCache] L1 Miss (SSD read, No L2): L%d E%d -> L1 slot %d", 
                         layer_id, expert_id, evict_slot);
            }
        }

        lock.unlock();

        if (stage_idx >= 0) {
            // SSD path: 2.5ms SSD read provides enough time for writeback to complete
            auto& stage = staging_ring_[stage_idx];
            CUDA_CHECK(cudaEventSynchronize(stage.event));
            
            int64_t file_offset = (int64_t)key * expert_block_size_;
            int64_t bytes_read = expert_file_.pread_exact(stage.ptr, expert_block_size_, file_offset);
            if (bytes_read != expert_block_size_) {
                LOG_ERROR("Expert read failed: layer=%d expert=%d offset=%ld got=%ld",
                          layer_id, expert_id, file_offset, (long)bytes_read);
                return nullptr;
            }
            host_src_ptr = stage.ptr;
            CUDA_CHECK(cudaMemcpyAsync(slot.gpu_data, host_src_ptr, expert_block_size_,
                                        cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(stage.event, stream));
        } else {
            // L2 hit path: DMA is immediate, must wait for writeback to complete first
            wait_for_writeback(stream);
            if (needs_disk_read) {
                int64_t file_offset = (int64_t)key * expert_block_size_;
                int64_t bytes_read = expert_file_.pread_exact(host_src_ptr, expert_block_size_, file_offset);
                if (bytes_read != expert_block_size_) {
                    LOG_ERROR("Expert read failed: layer=%d expert=%d offset=%ld got=%ld",
                              layer_id, expert_id, file_offset, (long)bytes_read);
                    return nullptr;
                }
            }
            // Memory in host_src_ptr (L2) is already populated, async copy to L1
            CUDA_CHECK(cudaMemcpyAsync(slot.gpu_data, host_src_ptr, expert_block_size_,
                                        cudaMemcpyHostToDevice, stream));
        }

        return slot.gpu_data;
    }

    // ── Batch expert loading: resolve all experts in ONE lock, parallel I/O outside ──
    // Returns the number of experts that needed DMA (non-cached).
    // gpu_ptrs_out[k] is populated with the VRAM pointer for expert k.
    // needs_dma_out[k] is true if expert k required a DMA transfer (not L1-cached).
    int batch_get_experts(int layer_id, const int32_t* expert_ids, int count,
                          void* gpu_ptrs_out[], bool needs_dma_out[],
                          cudaStream_t* streams, ThreadPool* pool) {
        struct BatchSlot {
            int64_t key;
            int l1_slot;           // L1 eviction slot index
            void* gpu_dst;         // VRAM destination
            void* host_src;        // DRAM source (L2 or staging)
            int stage_idx;         // staging ring index (-1 if L2)
            bool needs_disk;       // true if SSD read needed
            bool l1_hit;           // true if already in L1
        };

        BatchSlot batch[32];
        int n_uncached = 0;

        // ── Phase 1: Single lock — resolve all experts, allocate L1/L2 slots ──
        {
            std::unique_lock<std::mutex> lock(cache_mutex_);
            access_counter_++;

            for (int k = 0; k < count; k++) {
                int eid = expert_ids[k];
                if (eid < 0 || eid >= n_experts_) {
                    gpu_ptrs_out[k] = nullptr;
                    needs_dma_out[k] = false;
                    batch[k].l1_hit = true;
                    continue;
                }

                int64_t key = (int64_t)layer_id * n_experts_ + eid;
                batch[k].key = key;
                batch[k].stage_idx = -1;
                batch[k].needs_disk = false;
                batch[k].l1_hit = false;

                // Check L1 (VRAM)
                auto it = key_to_slot_.find(key);
                if (it != key_to_slot_.end()) {
                    auto& slot = cache_slots_[it->second];
                    slot.last_used = access_counter_;
                    gpu_ptrs_out[k] = slot.gpu_data;
                    needs_dma_out[k] = false;
                    batch[k].l1_hit = true;
                    continue;
                }

                // L1 miss — find eviction candidate (never evict from pinned resident layers)
                int evict_slot = -1;
                int64_t oldest_time = INT64_MAX;
                for (int i = 0; i < cache_capacity_; i++) {
                    if (cache_slots_[i].layer_id >= 0 && is_layer_pinned(cache_slots_[i].layer_id)) {
                        continue;
                    }
                    if (cache_slots_[i].layer_id < 0) {
                        evict_slot = i;
                        break;
                    }
                    if (cache_slots_[i].last_used < oldest_time) {
                        oldest_time = cache_slots_[i].last_used;
                        evict_slot = i;
                    }
                }

                auto& slot = cache_slots_[evict_slot];
                if (slot.layer_id >= 0) {
                    int64_t old_key = (int64_t)slot.layer_id * n_experts_ + slot.expert_id;
                    key_to_slot_.erase(old_key);
                    flat_vram_ptrs_[old_key] = nullptr;
                    // Demote evicted expert to L2 DRAM (victim cache)
                    demote_to_l2(slot.layer_id, slot.expert_id, old_key, slot.gpu_data);
                }

                slot.layer_id = layer_id;
                slot.expert_id = eid;
                slot.last_used = access_counter_;
                key_to_slot_[key] = evict_slot;
                flat_vram_ptrs_[key] = slot.gpu_data;

                batch[k].l1_slot = evict_slot;
                batch[k].gpu_dst = slot.gpu_data;
                gpu_ptrs_out[k] = slot.gpu_data;
                needs_dma_out[k] = true;
                n_uncached++;

                // Check L2 (DRAM)
                if (dram_cache_capacity_ > 0) {
                    auto dram_it = dram_key_to_slot_.find(key);
                    if (dram_it != dram_key_to_slot_.end()) {
                        // L2 hit
                        auto& dram_slot = dram_cache_slots_[dram_it->second];
                        dram_slot.last_used = access_counter_;
                        batch[k].host_src = dram_slot.gpu_data;
                    } else {
                        // L2 miss — use staging ring buffer for SSD read
                        // Do NOT evict L2 entries: L2 is a stable fast cache, not SSD staging
                        batch[k].stage_idx = staging_idx_;
                        staging_idx_ = (staging_idx_ + 1) % NUM_STAGING_BUFFERS;
                        batch[k].needs_disk = true;
                    }
                } else {
                    // No L2 — use staging ring
                    batch[k].stage_idx = staging_idx_;
                    staging_idx_ = (staging_idx_ + 1) % NUM_STAGING_BUFFERS;
                    batch[k].needs_disk = true;
                }
            }
        }
        // ── Mutex released ──

        if (n_uncached == 0) return 0;

        // ── Phase 2: Parallel I/O + DMA outside the lock ──
        // Submit all SSD reads and DRAM→VRAM DMAs concurrently
        std::future<bool> io_futures[32];

        for (int k = 0; k < count; k++) {
            if (batch[k].l1_hit) continue;

            io_futures[k] = pool->enqueue([this, &batch, k, &streams]() -> bool {
                auto& b = batch[k];
                void* host_ptr = b.host_src;

                if (b.stage_idx >= 0) {
                    // SSD path: 2.5ms SSD read provides enough time for writeback to complete
                    auto& stage = staging_ring_[b.stage_idx];
                    CUDA_CHECK(cudaEventSynchronize(stage.event));
                    int64_t bytes = expert_file_.pread_exact(stage.ptr, expert_block_size_, b.key * expert_block_size_);
                    if (bytes != expert_block_size_) return false;
                    host_ptr = stage.ptr;
                    CUDA_CHECK(cudaMemcpyAsync(b.gpu_dst, host_ptr, expert_block_size_,
                                                cudaMemcpyHostToDevice, streams[k]));
                    CUDA_CHECK(cudaEventRecord(stage.event, streams[k]));
                } else {
                    // L2 hit path: DMA is immediate, must wait for writeback to complete first
                    wait_for_writeback(streams[k]);
                    if (b.needs_disk) {
                        // L2 miss — read from SSD into L2 DRAM slot
                        int64_t bytes = expert_file_.pread_exact(host_ptr, expert_block_size_, b.key * expert_block_size_);
                        if (bytes != expert_block_size_) return false;
                    }
                    // L2 hit or just-read — DMA from DRAM to VRAM
                    CUDA_CHECK(cudaMemcpyAsync(b.gpu_dst, host_ptr, expert_block_size_,
                                                cudaMemcpyHostToDevice, streams[k]));
                }
                return true;
            });
        }

        // Wait for all I/O to complete
        for (int k = 0; k < count; k++) {
            if (batch[k].l1_hit) continue;
            if (!io_futures[k].get()) {
                LOG_ERROR("Failed to fetch expert L%d E%d", layer_id, expert_ids[k]);
                gpu_ptrs_out[k] = nullptr;
            }
        }

        return n_uncached;
    }

    bool preload_all(int n_threads = 16, const std::vector<uint32_t>& freq_counts = {}) {
        int total = n_layers_ * n_experts_;
        int to_load = std::min(total, cache_capacity_);
        flat_dram_ptrs_.assign(total, nullptr);
        LOG_INFO("Preloading %d/%d experts into VRAM L1 cache...", to_load, total);
        
        auto start = std::chrono::steady_clock::now();
        std::atomic<int> loaded{0};

        std::vector<std::thread> workers;
        int chunk_size = (to_load + n_threads - 1) / n_threads;

        for (int t = 0; t < n_threads; t++) {
            int start_idx = t * chunk_size;
            int end_idx = std::min(start_idx + chunk_size, to_load);
            if (start_idx >= end_idx) continue;

            workers.emplace_back([this, start_idx, end_idx, &loaded, to_load]() {
                void* stage_ptr = nullptr;
                cudaStream_t stream = nullptr;
                CUDA_CHECK(cudaMallocHost(&stage_ptr, expert_block_size_));
                CUDA_CHECK(cudaStreamCreate(&stream));

                for (int i = start_idx; i < end_idx; i++) {
                    int l = i / n_experts_;
                    int e = i % n_experts_;
                    int64_t file_offset = (int64_t)i * expert_block_size_;

                    int64_t bytes_read = expert_file_.pread_exact(stage_ptr, expert_block_size_, file_offset);
                    if (bytes_read == expert_block_size_) {
                        void* dst = cache_slots_[i].gpu_data;
                        cache_slots_[i].layer_id = l;
                        cache_slots_[i].expert_id = e;
                        cache_slots_[i].last_used = 1;
                        flat_vram_ptrs_[i] = dst;
                        CUDA_CHECK(cudaMemcpyAsync(dst, stage_ptr, expert_block_size_, cudaMemcpyHostToDevice, stream));
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                    }
                    int done = ++loaded;
                    if (done % 1000 == 0 || done == to_load) {
                        LOG_INFO("  Preloaded %d/%d experts into VRAM...", done, to_load);
                    }
                }
                CUDA_CHECK(cudaStreamDestroy(stream));
                CUDA_CHECK(cudaFreeHost(stage_ptr));
            });
        }

        for (auto& w : workers) {
            w.join();
        }

        for (int i = 0; i < to_load; i++) {
            if (cache_slots_[i].layer_id >= 0) {
                int64_t key = (int64_t)cache_slots_[i].layer_id * n_experts_ + cache_slots_[i].expert_id;
                key_to_slot_[key] = i;
            }
        }
        access_counter_ = 1;

        if (dram_cache_capacity_ > 0) {
            int dram_to_load = std::min(total - to_load, dram_cache_capacity_);
            int start_layer = to_load / n_experts_;
            int end_layer = (to_load + dram_to_load - 1) / n_experts_;

            if (!freq_counts.empty() && (int)freq_counts.size() == total) {
                // Frequency-guided smart preload: pick the hottest experts across all non-L1 slots
                struct Candidate {
                    int layer_id;
                    int expert_id;
                    uint32_t count;
                };
                std::vector<Candidate> candidates;
                candidates.reserve(total - to_load);
                for (int l = 0; l < n_layers_; l++) {
                    for (int e = 0; e < n_experts_; e++) {
                        int64_t key = (int64_t)l * n_experts_ + e;
                        if (key_to_slot_.find(key) == key_to_slot_.end()) {
                            candidates.push_back({l, e, freq_counts[(size_t)l * n_experts_ + e]});
                        }
                    }
                }
                std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                    if (a.count != b.count) return a.count > b.count;
                    if (a.layer_id != b.layer_id) return a.layer_id < b.layer_id;
                    return a.expert_id < b.expert_id;
                });

                LOG_INFO("Preloading %d/%d hot experts into DRAM L2 cache across all %d layers (frequency-guided)...",
                         dram_to_load, dram_cache_capacity_, n_layers_);
                std::vector<std::thread> dram_workers;
                int dram_chunk = (dram_to_load + n_threads - 1) / n_threads;
                for (int t = 0; t < n_threads; t++) {
                    int start_i = t * dram_chunk;
                    int end_i = std::min(start_i + dram_chunk, dram_to_load);
                    if (start_i >= end_i) continue;

                    dram_workers.emplace_back([this, start_i, end_i, &candidates]() {
                        for (int i = start_i; i < end_i; i++) {
                            int l = candidates[i].layer_id;
                            int e = candidates[i].expert_id;
                            int expert_global_idx = l * n_experts_ + e;
                            int64_t file_offset = (int64_t)expert_global_idx * expert_block_size_;
                            void* dst = dram_cache_slots_[i].gpu_data;
                            int64_t bytes_read = expert_file_.pread_exact(dst, expert_block_size_, file_offset);
                            if (bytes_read == expert_block_size_) {
                                dram_cache_slots_[i].layer_id = l;
                                dram_cache_slots_[i].expert_id = e;
                                dram_cache_slots_[i].last_used = 1;
                            }
                        }
                    });
                }
                for (auto& w : dram_workers) {
                    w.join();
                }

                for (int i = 0; i < dram_to_load; i++) {
                    if (dram_cache_slots_[i].layer_id >= 0) {
                        int64_t key = (int64_t)dram_cache_slots_[i].layer_id * n_experts_ + dram_cache_slots_[i].expert_id;
                        dram_key_to_slot_[key] = i;
                        if (key >= 0 && key < (int64_t)flat_dram_ptrs_.size()) {
                            flat_dram_ptrs_[key] = dram_cache_slots_[i].gpu_data;
                        }
                    }
                }
            } else {
                LOG_INFO("Preloading %d/%d experts into DRAM L2 cache (layers %d..%d)...",
                         dram_to_load, dram_cache_capacity_, start_layer, end_layer);
                std::vector<std::thread> dram_workers;
                int dram_chunk = (dram_to_load + n_threads - 1) / n_threads;
                for (int t = 0; t < n_threads; t++) {
                    int start_i = t * dram_chunk;
                    int end_i = std::min(start_i + dram_chunk, dram_to_load);
                    if (start_i >= end_i) continue;

                    dram_workers.emplace_back([this, start_i, end_i, to_load]() {
                        for (int i = start_i; i < end_i; i++) {
                            int expert_global_idx = to_load + i;
                            int l = expert_global_idx / n_experts_;
                            int e = expert_global_idx % n_experts_;
                            int64_t file_offset = (int64_t)expert_global_idx * expert_block_size_;
                            void* dst = dram_cache_slots_[i].gpu_data;
                            int64_t bytes_read = expert_file_.pread_exact(dst, expert_block_size_, file_offset);
                            if (bytes_read == expert_block_size_) {
                                dram_cache_slots_[i].layer_id = l;
                                dram_cache_slots_[i].expert_id = e;
                                dram_cache_slots_[i].last_used = 1;
                            }
                        }
                    });
                }
                for (auto& w : dram_workers) {
                    w.join();
                }

                for (int i = 0; i < dram_to_load; i++) {
                    if (dram_cache_slots_[i].layer_id >= 0) {
                        int64_t key = (int64_t)dram_cache_slots_[i].layer_id * n_experts_ + dram_cache_slots_[i].expert_id;
                        dram_key_to_slot_[key] = i;
                        if (key >= 0 && key < (int64_t)flat_dram_ptrs_.size()) {
                            flat_dram_ptrs_[key] = dram_cache_slots_[i].gpu_data;
                        }
                    }
                }
            }
        }

        // Copy flat pointer table to GPU for GPU-native kernel execution
        CUDA_CHECK(cudaMemcpy(flat_vram_ptrs_gpu_.data, flat_vram_ptrs_.data(),
                               flat_vram_ptrs_.size() * sizeof(void*), cudaMemcpyHostToDevice));

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        LOG_INFO("Preload completed in %.2f seconds", elapsed / 1000.0);
        return true;
    }

    bool preload_all_mixed(int n_threads = 16) {
        int total = n_layers_ * n_experts_;
        LOG_INFO("Preloading %d mixed experts into VRAM (Hot: NVFP4, Cold: IQ2_XXS)...", total);

        auto start = std::chrono::steady_clock::now();
        std::atomic<int> loaded{0};

        std::vector<std::thread> workers;
        int chunk_size = (total + n_threads - 1) / n_threads;

        for (int t = 0; t < n_threads; t++) {
            int start_idx = t * chunk_size;
            int end_idx = std::min(start_idx + chunk_size, total);
            if (start_idx >= end_idx) continue;

            workers.emplace_back([this, start_idx, end_idx, &loaded, total]() {
                size_t max_block = std::max(hot_block_size_, cold_block_size_);
                void* stage_ptr = nullptr;
                cudaStream_t stream = nullptr;
                CUDA_CHECK(cudaMallocHost(&stage_ptr, max_block));
                CUDA_CHECK(cudaStreamCreate(&stream));

                for (int i = start_idx; i < end_idx; i++) {
                    uint8_t is_hot = expert_type_map_host_[i];
                    void* dst = flat_vram_ptrs_[i];

                    if (is_hot) {
                        int64_t file_offset = (int64_t)i * hot_block_size_;
                        int64_t bytes = expert_file_hot_.pread_exact(stage_ptr, hot_block_size_, file_offset);
                        if (bytes == hot_block_size_) {
                            CUDA_CHECK(cudaMemcpyAsync(dst, stage_ptr, hot_block_size_, cudaMemcpyHostToDevice, stream));
                            CUDA_CHECK(cudaStreamSynchronize(stream));
                        } else {
                            LOG_ERROR("Failed reading hot expert %d at offset %lld (got %lld)", i, (long long)file_offset, (long long)bytes);
                        }
                    } else {
                        int64_t file_offset = (int64_t)i * cold_block_size_;
                        int64_t bytes = expert_file_cold_.pread_exact(stage_ptr, cold_block_size_, file_offset);
                        if (bytes == cold_block_size_) {
                            CUDA_CHECK(cudaMemcpyAsync(dst, stage_ptr, cold_block_size_, cudaMemcpyHostToDevice, stream));
                            CUDA_CHECK(cudaStreamSynchronize(stream));
                        } else {
                            LOG_ERROR("Failed reading cold expert %d at offset %lld (got %lld)", i, (long long)file_offset, (long long)bytes);
                        }
                    }

                    int done = ++loaded;
                    if (done % 1000 == 0 || done == total) {
                        LOG_INFO("  Preloaded %d/%d mixed experts into VRAM...", done, total);
                    }
                }
                CUDA_CHECK(cudaStreamDestroy(stream));
                CUDA_CHECK(cudaFreeHost(stage_ptr));
            });
        }

        for (auto& w : workers) {
            w.join();
        }

        // Copy flat pointer table to GPU for GPU-native kernel execution
        CUDA_CHECK(cudaMemcpy(flat_vram_ptrs_gpu_.data, flat_vram_ptrs_.data(),
                              flat_vram_ptrs_.size() * sizeof(void*), cudaMemcpyHostToDevice));

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        LOG_INFO("Preload completed: 100%% VRAM Resident (11008 experts) in %.2f seconds", elapsed / 1000.0);
        return true;
    }

    void* host_l1_backup_ = nullptr;
    size_t host_l1_backup_bytes_ = 0;

    bool evict_l1_to_host() {
        if (!cache_pool_gpu_ || cache_capacity_ <= 0) return true;
        std::unique_lock<std::mutex> lock(cache_mutex_);
        host_l1_backup_bytes_ = (size_t)cache_capacity_ * expert_block_size_;
        LOG_INFO("[ExpertCache] Evicting L1 expert cache (%.2f GB) to DRAM for Vision Delegate...",
                 (double)host_l1_backup_bytes_ / (1024.0 * 1024.0 * 1024.0));
#if defined(_WIN32) || defined(_WIN64)
        host_l1_backup_ = VirtualAlloc(NULL, host_l1_backup_bytes_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        host_l1_backup_ = malloc(host_l1_backup_bytes_);
#endif
        if (!host_l1_backup_) {
            LOG_WARN("[ExpertCache] Failed to allocate host DRAM for L1 eviction, will reload from SSD upon restore");
        } else {
            cudaError_t err = cudaMemcpy(host_l1_backup_, cache_pool_gpu_, host_l1_backup_bytes_, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                LOG_WARN("[ExpertCache] cudaMemcpy D2H failed: %s, will reload from SSD upon restore", cudaGetErrorString(err));
#if defined(_WIN32) || defined(_WIN64)
                VirtualFree(host_l1_backup_, 0, MEM_RELEASE);
#else
                free(host_l1_backup_);
#endif
                host_l1_backup_ = nullptr;
            }
        }

        cudaFree(cache_pool_gpu_);
        cache_pool_gpu_ = nullptr;
        if (flat_vram_ptrs_gpu_.data) {
            CUDA_CHECK(cudaMemset(flat_vram_ptrs_gpu_.data, 0, flat_vram_ptrs_.size() * sizeof(void*)));
        }
        for (auto& ptr : flat_vram_ptrs_) {
            ptr = nullptr;
        }
        LOG_INFO("[ExpertCache] L1 expert cache unloaded. VRAM freed!");
        return true;
    }

    bool restore_l1_from_host(const std::vector<uint32_t>& freq_counts = {}) {
        if (cache_pool_gpu_) return true;
        std::unique_lock<std::mutex> lock(cache_mutex_);
        size_t bytes = (size_t)cache_capacity_ * expert_block_size_;
        LOG_INFO("[ExpertCache] Restoring L1 expert cache (%.2f GB) to VRAM...",
                 (double)bytes / (1024.0 * 1024.0 * 1024.0));
        cudaError_t err = cudaMalloc(&cache_pool_gpu_, bytes);
        if (err != cudaSuccess) {
            LOG_ERROR("[ExpertCache] Failed to reallocate VRAM pool: %s", cudaGetErrorString(err));
            return false;
        }

        if (host_l1_backup_ != nullptr) {
            err = cudaMemcpy(cache_pool_gpu_, host_l1_backup_, bytes, cudaMemcpyHostToDevice);
#if defined(_WIN32) || defined(_WIN64)
            VirtualFree(host_l1_backup_, 0, MEM_RELEASE);
#else
            free(host_l1_backup_);
#endif
            host_l1_backup_ = nullptr;
            host_l1_backup_bytes_ = 0;

            if (err == cudaSuccess) {
                for (int i = 0; i < cache_capacity_; i++) {
                    cache_slots_[i].gpu_data = (char*)cache_pool_gpu_ + (size_t)i * expert_block_size_;
                    if (cache_slots_[i].layer_id >= 0) {
                        int64_t key = (int64_t)cache_slots_[i].layer_id * n_experts_ + cache_slots_[i].expert_id;
                        if (key >= 0 && key < (int64_t)flat_vram_ptrs_.size()) {
                            flat_vram_ptrs_[key] = cache_slots_[i].gpu_data;
                        }
                    }
                }
                if (flat_vram_ptrs_gpu_.data) {
                    CUDA_CHECK(cudaMemcpy(flat_vram_ptrs_gpu_.data, flat_vram_ptrs_.data(),
                                          flat_vram_ptrs_.size() * sizeof(void*), cudaMemcpyHostToDevice));
                }
                LOG_INFO("[ExpertCache] L1 cache restored from host DRAM in fast DMA transfer!");
                return true;
            }
        }

        // Fallback: reload from SSD if host backup was not present or failed
        LOG_INFO("[ExpertCache] Restoring L1 cache from SSD storage...");
        for (int i = 0; i < cache_capacity_; i++) {
            cache_slots_[i].gpu_data = (char*)cache_pool_gpu_ + (size_t)i * expert_block_size_;
        }
        lock.unlock();
        return preload_all(16, freq_counts);
    }

    const void* const* flat_vram_ptrs_gpu() const {
        return (const void* const*)flat_vram_ptrs_gpu_.data;
    }

    void cleanup() {
        expert_file_.close();
        if (is_mixed_) {
            expert_file_cold_.close();
            expert_file_hot_.close();
            if (hot_pool_gpu_) { cudaFree(hot_pool_gpu_); hot_pool_gpu_ = nullptr; }
            if (cold_pool_gpu_) { cudaFree(cold_pool_gpu_); cold_pool_gpu_ = nullptr; }
            expert_type_map_gpu_.free();
        }
        if (host_l1_backup_) {
#if defined(_WIN32) || defined(_WIN64)
            VirtualFree(host_l1_backup_, 0, MEM_RELEASE);
#else
            free(host_l1_backup_);
#endif
            host_l1_backup_ = nullptr;
        }
        if (cache_pool_gpu_) cudaFree(cache_pool_gpu_);
        if (dram_cache_pool_) {
            if (cudaFreeHost(dram_cache_pool_) != cudaSuccess) {
                cudaGetLastError();
#if defined(_WIN32) || defined(_WIN64)
                VirtualFree(dram_cache_pool_, 0, MEM_RELEASE);
#else
                free(dram_cache_pool_);
#endif
            }
            dram_cache_pool_ = nullptr;
        }
        flat_vram_ptrs_gpu_.free();
        for (auto& stage : staging_ring_) {
            if (stage.event) cudaEventDestroy(stage.event);
            if (stage.ptr) cudaFreeHost(stage.ptr);
        }
    }
private:
    GPUTensor flat_vram_ptrs_gpu_;
};


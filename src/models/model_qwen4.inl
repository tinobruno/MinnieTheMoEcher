// ============================================================================
//  Qwen3.8-Flash-Next (qwen4exp) Native MinnieTheMoEcher Engine Implementation
//  - 48 Layers: 36 Gated DeltaNet (GDN) + 12 Query-Sparse Attention (QSA)
//  - 4-Stream Hyper-Connection Gated Residuals (d_model=2560, rank=320)
//  - Layer 1 PLE + 131B Multi-Order N-Gram Table (Shard 2 MMap)
//  - 256 Routed Experts (Top-10 Softmax) + 1 Sigmoid-Gated Shared Expert
//  - Native Binary Pack (resident.bin + experts.bin + expert_freq.bin)
//  - Per-Expert 3-Tier Cache:
//      * Tier 1 (L1 VRAM): Static Hot Experts + Per-Layer Dynamic LRU Cache
//      * Tier 2 (L2 Pinned RAM): Contiguous [gate|up|down] Direct Async DMA
//      * Tier 2.5 (L2.5 Unpinned RAM): Contiguous [gate|up|down] Host Cache
//      * Tier 3 (L3 NVMe SSD): Direct Streaming from experts.bin
//  - L1-Hit / L1-Miss Split Execution: Overlaps L1-Hit GPU Compute on
//    main_stream_ with L2 -> L1 LRU Async DMA on copy_stream!
// ============================================================================

struct Qwen4GgufTensorInfo {
    std::string name;
    uint32_t n_dims = 0;
    uint64_t dims[4] = {1, 1, 1, 1};
    int32_t ggml_type = 0;
    uint64_t rel_offset = 0;
    const uint8_t* data_ptr = nullptr;
    size_t nbytes = 0;
};

struct Qwen4LayerWeights {
    bool is_qsa = false; // true if l % 4 == 3, false if GDN

    // 4-stream Gated Residual weights (always in VRAM)
    void* d_hc_attn_norm   = nullptr; // F32 [10240]
    void* d_hc_attn_down   = nullptr; // BF16 [320, 10240]
    void* d_hc_attn_up     = nullptr; // BF16 [10240, 320]
    void* d_hc_attn_inject = nullptr; // BF16 [4, 10240]

    void* d_hc_ffn_norm    = nullptr; // F32 [10240]
    void* d_hc_ffn_down    = nullptr; // BF16 [320, 10240]
    void* d_hc_ffn_up      = nullptr; // BF16 [10240, 320]
    void* d_hc_ffn_inject  = nullptr; // BF16 [4, 10240]

    // GDN weights (l % 4 != 3)
    void* d_gdn_qkv        = nullptr; int32_t gdn_qkv_type  = 0;
    void* d_gdn_gate       = nullptr; int32_t gdn_gate_type = 0;
    void* d_gdn_alpha      = nullptr; // BF16 [48, 2560]
    void* d_gdn_beta       = nullptr; // BF16 [48, 2560]
    void* d_gdn_conv1d     = nullptr; // F32 [10240, 4]
    void* d_gdn_ssm_a      = nullptr; // F32 [48]
    void* d_gdn_dt_bias    = nullptr; // F32 [48]
    void* d_gdn_ssm_norm   = nullptr; // F32 [128]
    void* d_gdn_out        = nullptr; int32_t gdn_out_type  = 0;

    // GDN recurrent state
    float* d_gdn_conv_state = nullptr; // F32 [3 * 10240]
    float* d_gdn_ssm_state  = nullptr; // F32 [48 * 128 * 128]

    // QSA weights (l % 4 == 3)
    void* d_qsa_q          = nullptr; int32_t qsa_q_type   = 0;
    void* d_qsa_k          = nullptr; int32_t qsa_k_type   = 0;
    void* d_qsa_v          = nullptr; int32_t qsa_v_type   = 0;
    void* d_qsa_q_norm     = nullptr; // F32 [256]
    void* d_qsa_k_norm     = nullptr; // F32 [256]
    void* d_qsa_out        = nullptr; int32_t qsa_out_type = 0;
    void* d_qsa_idx_k      = nullptr; // BF16 [128, 2560]
    void* d_qsa_idx_q      = nullptr; // BF16 [512, 2560]
    void* d_qsa_idx_k_norm = nullptr; // F32 [128]
    void* d_qsa_idx_q_norm = nullptr; // F32 [128]

    // QSA KV & Indexer caches
    uint16_t* d_qsa_k_cache = nullptr; // BF16 [max_seq_len, 512]
    uint16_t* d_qsa_v_cache = nullptr; // BF16 [max_seq_len, 512]
    float* d_qsa_idx_pooled = nullptr; // F32 [max_seq_len / 4, 128]
    float* d_qsa_idx_tail   = nullptr; // F32 [4, 128]
    float* d_qsa_idx_dead   = nullptr; // F32 [128]

    // MoE Router + Shared Expert (always in VRAM)
    void* d_moe_gate_inp       = nullptr; // BF16 [256, 2560]
    void* d_moe_gate_inp_shexp = nullptr; // BF16 [2560]
    void* d_moe_gate_shexp     = nullptr; int32_t moe_gate_shexp_type = 0;
    void* d_moe_up_shexp       = nullptr; int32_t moe_up_shexp_type   = 0;
    void* d_moe_down_shexp     = nullptr; int32_t moe_down_shexp_type = 0;

    // MoE Routed Expert Quantization & Contiguous Blob Layout ([gate_e | up_e | down_e])
    int32_t  moe_gate_exp_type = 0; size_t gate_exp_bytes  = 0; size_t gate_exp_row_bytes = 0;
    int32_t  moe_up_exp_type   = 0; size_t up_exp_bytes    = 0; size_t up_exp_row_bytes   = 0;
    int32_t  moe_down_exp_type = 0; size_t down_exp_bytes  = 0; size_t down_exp_row_bytes = 0;
    size_t   expert_blob_bytes = 0; // gate_exp_bytes + up_exp_bytes + down_exp_bytes
    uint64_t layer_file_offset = 0; // byte offset in experts.bin

    // Per-Expert 3-Tier Cache Table (256 experts per layer):
    //   d_vram_exp_ptr[e] != nullptr => Expert e is currently in Tier 1 L1 VRAM
    //     (either permanently in Static Hot L1 VRAM or dynamically in L1 LRU VRAM).
    //     Blob layout at d_vram_exp_ptr[e]:
    //       gate = d_vram_exp_ptr[e]
    //       up   = d_vram_exp_ptr[e] + gate_exp_bytes
    //       down = d_vram_exp_ptr[e] + gate_exp_bytes + up_exp_bytes
    static constexpr int MAX_LRU_SLOTS = 96;
    uint8_t* d_vram_exp_ptr[256]     = {nullptr};
    bool     is_static_vram[256]     = {false};
    int8_t   exp_lru_slot[256]       = {0}; // -1 if not in Dynamic LRU L1, else 0..n_lru_slots-1

    // Tier 1b: Per-layer Dynamic LRU L1 VRAM Pool (caches misses across tokens!)
    int      n_lru_slots             = 0;
    uint8_t* d_lru_base              = nullptr;
    int32_t  lru_slot_eid[MAX_LRU_SLOTS]       = {0};
    uint64_t lru_slot_last_used[MAX_LRU_SLOTS] = {0};
    uint64_t lru_slot_epoch[MAX_LRU_SLOTS]     = {0};

    // Tier 2: Per-expert Pinned Host RAM pointers (contiguous [gate|up|down], single-call async DMA)
    uint8_t* h_pinned_exp_ptr[256]   = {nullptr};

    // Tier 2.5: Per-expert Unpinned Host RAM pointers (contiguous [gate|up|down], zero SSD faults)
    uint8_t* h_unpinned_exp_ptr[256] = {nullptr};
};

struct Qwen4RuntimeState {
    bool loaded = false;
    moecher::platform::MemoryMappedFile shard1_mmap;
    moecher::platform::MemoryMappedFile shard2_mmap;
    FILE* experts_fp = nullptr;

    // Global weights in VRAM (token_embd kept in Host RAM with 1-row VRAM staging to save ~360 MB VRAM)
    uint8_t* h_token_embd = nullptr;
    size_t   token_embd_row_bytes = 0;
    void* d_token_embd  = nullptr; int32_t token_embd_type = 0;
    void* d_out_hc_norm = nullptr; // F32 [10240]
    void* d_out_hc_down = nullptr; // BF16 [320, 10240]
    void* d_out_hc_up   = nullptr; // BF16 [10240, 320]
    void* d_output      = nullptr; int32_t output_type = 0;

    // Layer 1 PLE weights & state in VRAM + Shard 2 N-gram MMap pointer
    bool has_ple = false;
    void* d_ple_norm_q  = nullptr; // F32 [10240]
    void* d_ple_norm_k  = nullptr; // F32 [10240]
    void* d_ple_norm_c  = nullptr; // F32 [10240]
    void* d_ple_key     = nullptr; int32_t ple_key_type = 0;
    void* d_ple_val     = nullptr; int32_t ple_val_type = 0;
    void* d_ple_conv1d  = nullptr; // F16 [10240, 4]
    float* d_ple_conv_state = nullptr; // F32 [9 * 10240]
    const uint8_t* mmap_ple_embed = nullptr; // Shard 2 [160, 320001536] IQ4_NL (90 B/row)
    uint8_t* h_ple_rows_iq4nl = nullptr; // Pinned [16 * 90 = 1440 B]
    void*    d_ple_rows_iq4nl = nullptr; // Device [1440 B]
    float*   d_ple_emb_flat   = nullptr; // Device [16 * 160 = 2560 floats]
    std::vector<int32_t> token_history;

    // Working activation buffers in VRAM
    float*   d_R            = nullptr; // [4 * 2560 = 10240]
    float*   d_x            = nullptr; // [2560]
    uint8_t* d_moe_pkt      = nullptr; // Contiguous [3072 B]: d_x_q8_1 (2880 B) + d_top_idx (40 B) + d_top_w (40 B)
    uint8_t* h_moe_pkt      = nullptr; // Pinned [3072 B]: h_x_q8_1 (2880 B) + h_top_idx (40 B) + h_top_w (40 B)
    void*    d_x_q8_1       = nullptr; // [2560 / 32 * 36 = 2880 B] (alias of d_moe_pkt + 0)
    void*    h_x_q8_1       = nullptr; // Pinned [2880 B] (alias of h_moe_pkt + 0)
    float*   d_sub_out      = nullptr; // [2560]
    float*   d_scratch      = nullptr; // [65536 floats = 256 KB]
    void*    d_q8_scratch   = nullptr; // [32768 B]
    int32_t* d_sel_indices  = nullptr; // [2056 int32]
    int32_t* d_top_idx      = nullptr; // [10 int32] (alias of d_moe_pkt + 2880)
    float*   d_top_w        = nullptr; // [10 float] (alias of d_moe_pkt + 2920)
    int32_t* h_top_idx      = nullptr; // Pinned [10 int32] (alias of h_moe_pkt + 2880)
    float*   h_top_w        = nullptr; // Pinned [10 float] (alias of h_moe_pkt + 2920)
    float*   h_cpu_moe_out  = nullptr; // Pinned Mapped [2560 floats] for Hybrid CPU Expert Compute
    float*   d_cpu_moe_out  = nullptr; // Device Mapped pointer to h_cpu_moe_out

    // Pinned host bounce buffer for Tier 2.5 (Unpinned RAM) & Tier 3 (SSD) misses (up to 10 experts)
    size_t   max_expert_blob_bytes = 0;
    uint8_t* h_bounce_exp          = nullptr; // Pinned [10 * max_expert_blob_bytes]
    Qwen4ActiveExpertPtrs* h_active_ptrs = nullptr; // Pinned
    Qwen4ActiveExpertPtrs* d_active_ptrs = nullptr;

    // LRU clock & token epoch for L1 Dynamic LRU VRAM slots
    uint64_t lru_clock       = 1;
    uint64_t cur_token_epoch = 1;

    // Tier hit telemetry counters
    uint64_t stat_l1_static_hits    = 0;
    uint64_t stat_l1_lru_hits       = 0;
    uint64_t stat_l2_pinned_hits    = 0;
    uint64_t stat_l25_unpinned_hits = 0;
    uint64_t stat_l3_ssd_reads      = 0;
    uint64_t stat_cpu_computed      = 0;
    uint64_t stat_tokens_recorded   = 0;

    // Optional empirical router calibration (enabled via MOECHER_CALIBRATE_EXPERTS=1)
    bool calibrate_experts = false;
    std::string calib_freq_bin_path;
    std::string calib_counts_json_path;
    std::vector<double> calib_scores; // [48 * 256]
    float calib_probs_host[256] = {0.0f};

    // Dedicated async DMA stream & events for overlapping L1-Hit GPU compute with L2 -> L1 DMA
    cudaStream_t copy_stream       = nullptr;
    cudaEvent_t  router_event      = nullptr;
    cudaEvent_t  top_idx_event     = nullptr;
    cudaEvent_t  stage_ready_event = nullptr;

    // Unpinned copy tasks (delegated to shared CpuMoEExpertPool when --no-cpu-experts is set)
    moecher::cpu_moe::CpuMemcpyTaskDesc copy_tasks[10];

    void init_async_and_pool() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&router_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&top_idx_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&stage_ready_event, cudaEventDisableTiming));
        (void)moecher::cpu_moe::CpuMoEExpertPool::instance();
    }

    void parallel_unpinned_gather(int count) {
        if (count <= 0) return;
        moecher::cpu_moe::CpuMoEExpertPool::instance().parallel_memcpy(copy_tasks, count);
    }

    std::vector<Qwen4LayerWeights> layers;
    std::vector<void*> device_allocs;
    std::vector<void*> pinned_allocs;
    std::vector<void*> unpinned_allocs;

    void* alloc_vram(size_t nbytes) {
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, nbytes));
        device_allocs.push_back(p);
        return p;
    }

    void* upload_vram(const void* src, size_t nbytes) {
        void* p = alloc_vram(nbytes);
        CUDA_CHECK(cudaMemcpy(p, src, nbytes, cudaMemcpyHostToDevice));
        return p;
    }

    void print_cache_stats() const {
        const uint64_t total = stat_l1_static_hits + stat_l1_lru_hits + stat_l2_pinned_hits + stat_l25_unpinned_hits + stat_l3_ssd_reads;
        if (total == 0) return;
        const double inv = 100.0 / (double)total;
        const uint64_t l1_total = stat_l1_static_hits + stat_l1_lru_hits;
        LOG_INFO("[Qwen4Exp] Expert Cache Telemetry (%llu tokens, %llu routed calls): "
                 "L1 VRAM Hit %.1f%% (Static %.1f%% + Dynamic LRU %.1f%%) | "
                 "L2 Pinned %.1f%% | L2.5 Unpinned %.1f%% | L3 SSD %.1f%% | CPU Computed %.1f%% (%llu experts)",
                 (unsigned long long)stat_tokens_recorded,
                 (unsigned long long)total,
                 (double)l1_total * inv,
                 (double)stat_l1_static_hits * inv,
                 (double)stat_l1_lru_hits * inv,
                 (double)stat_l2_pinned_hits * inv,
                 (double)stat_l25_unpinned_hits * inv,
                 (double)stat_l3_ssd_reads * inv,
                 (double)stat_cpu_computed * inv,
                 (unsigned long long)stat_cpu_computed);
    }

    ~Qwen4RuntimeState() {
        if (copy_stream) cudaStreamSynchronize(copy_stream);
        print_cache_stats();
        if (calibrate_experts && stat_tokens_recorded > 0 && calib_scores.size() == 48 * 256 && !calib_freq_bin_path.empty()) {
            std::vector<uint32_t> counts(48 * 256, 1);
            std::vector<int> order(48 * 256);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return calib_scores[a] > calib_scores[b];
            });
            for (int rank = 0; rank < 48 * 256; ++rank) {
                counts[order[rank]] = (uint32_t)(48 * 256 - rank);
            }
            if (FILE* fp = std::fopen(calib_freq_bin_path.c_str(), "wb")) {
                int32_t nl = 48, ne = 256;
                int64_t ntok = (int64_t)stat_tokens_recorded;
                std::fwrite(&nl, 4, 1, fp);
                std::fwrite(&ne, 4, 1, fp);
                std::fwrite(&ntok, 8, 1, fp);
                std::fwrite(counts.data(), sizeof(uint32_t), counts.size(), fp);
                std::fclose(fp);
                LOG_INFO("[Qwen4Exp] Saved calibrated empirical expert_freq.bin (%llu tokens) to %s",
                         (unsigned long long)stat_tokens_recorded, calib_freq_bin_path.c_str());
            }
            if (!calib_counts_json_path.empty()) {
                json cj;
                for (int l = 0; l < 48; ++l) {
                    std::vector<uint32_t> lc(counts.begin() + l * 256, counts.begin() + (l + 1) * 256);
                    cj["layer_" + std::to_string(l)] = lc;
                }
                std::ofstream out_j(calib_counts_json_path);
                if (out_j.is_open()) out_j << cj.dump();
            }
        }
        if (stage_ready_event) cudaEventDestroy(stage_ready_event);
        if (top_idx_event)     cudaEventDestroy(top_idx_event);
        if (router_event)      cudaEventDestroy(router_event);
        if (copy_stream)       cudaStreamDestroy(copy_stream);
        for (void* p : device_allocs)   if (p) cudaFree(p);
        for (void* p : pinned_allocs)   if (p) cudaFreeHost(p);
        for (void* p : unpinned_allocs) if (p) std::free(p);
        device_allocs.clear();
        pinned_allocs.clear();
        unpinned_allocs.clear();
        if (experts_fp) {
            std::fclose(experts_fp);
            experts_fp = nullptr;
        }
        shard1_mmap.close();
        shard2_mmap.close();
    }
};

std::unique_ptr<Qwen4RuntimeState> qwen4_state_;

// Helper to query true physical free VRAM on Windows via NVML (avoiding WDDM shared memory inflation)
static size_t query_true_free_vram_bytes() {
    size_t cuda_free = 0, cuda_total = 0;
    CUDA_CHECK(cudaMemGetInfo(&cuda_free, &cuda_total));
#if defined(_WIN32)
    HMODULE nvml = LoadLibraryA("nvml.dll");
    if (nvml) {
        using nvmlInit_t = int(*)();
        using nvmlDeviceGetHandleByIndex_t = int(*)(unsigned int, void**);
        struct nvmlMemory_t { unsigned long long total, free, used; };
        using nvmlDeviceGetMemoryInfo_t = int(*)(void*, nvmlMemory_t*);

        auto fn_init = (nvmlInit_t)GetProcAddress(nvml, "nvmlInit_v2");
        auto fn_dev  = (nvmlDeviceGetHandleByIndex_t)GetProcAddress(nvml, "nvmlDeviceGetHandleByIndex_v2");
        auto fn_mem  = (nvmlDeviceGetMemoryInfo_t)GetProcAddress(nvml, "nvmlDeviceGetMemoryInfo");
        if (fn_init && fn_dev && fn_mem && fn_init() == 0) {
            void* dev = nullptr;
            if (fn_dev(0, &dev) == 0) {
                nvmlMemory_t mem{};
                if (fn_mem(dev, &mem) == 0 && mem.free > 0) {
                    return std::min(cuda_free, (size_t)mem.free);
                }
            }
        }
    }
#endif
    return cuda_free;
}

bool load_qwen4_exp(
    const json& manifest,
    const std::filesystem::path& base_dir,
    float max_vram_gb,
    float dram_cache_gb)
{
    qwen4_state_ = std::make_unique<Qwen4RuntimeState>();
    auto& st = *qwen4_state_;

    if (!manifest.contains("dense_bin") || !manifest.contains("expert_bin") ||
        !manifest.contains("dense_tensors") || !manifest.contains("expert_layout")) {
        LOG_ERROR("[Qwen4Exp] Manifest is missing native MinnieTheMoEcher binary pack fields (dense_bin / expert_bin / dense_tensors / expert_layout). Run scripts/convert_qwen38_flash_next.py first.");
        return false;
    }

    const std::filesystem::path dense_bin_path  = base_dir / manifest.value("dense_bin", "resident.bin");
    const std::filesystem::path expert_bin_path = base_dir / manifest.value("expert_bin", "experts.bin");
    const std::filesystem::path freq_bin_path   = base_dir / manifest.value("expert_freq_bin", "expert_freq.bin");

    // 1. Open resident.bin via MMap, upload all dense/resident weights, then close resident.bin MMap!
    moecher::platform::MemoryMappedFile resident_mmap;
    LOG_INFO("[Qwen4Exp] Mapping native resident pack: %s", dense_bin_path.string().c_str());
    if (!resident_mmap.open_read(dense_bin_path.string())) {
        LOG_ERROR("[Qwen4Exp] Failed to open resident.bin: %s", dense_bin_path.string().c_str());
        return false;
    }
    const uint8_t* res_base = (const uint8_t*)resident_mmap.data();
    const auto& dense_map = manifest["dense_tensors"];

    auto require_dense = [&](const std::string& name, int32_t* out_type, size_t* out_nbytes) -> const uint8_t* {
        if (!dense_map.contains(name)) {
            LOG_ERROR("[Qwen4Exp] Missing dense tensor in resident.bin manifest: %s", name.c_str());
            std::exit(1);
        }
        const auto& ti = dense_map[name];
        const uint64_t off = ti["offset"].get<uint64_t>();
        const size_t nb    = ti["nbytes"].get<size_t>();
        const int32_t gt   = ti["ggml_type"].get<int32_t>();
        if (out_type)   *out_type   = gt;
        if (out_nbytes) *out_nbytes = nb;
        return res_base + off;
    };

    auto upload_dense = [&](const std::string& name, int32_t* out_type = nullptr) -> void* {
        size_t nb = 0;
        const uint8_t* ptr = require_dense(name, out_type, &nb);
        return st.upload_vram(ptr, nb);
    };

    // Keep token_embd in Host RAM (only 1 row needed per token -> saves ~360 MB VRAM for more L1 experts!)
    {
        size_t emb_nb = 0;
        const uint8_t* emb_ptr = require_dense("token_embd.weight", &st.token_embd_type, &emb_nb);
        st.token_embd_row_bytes = qwen4_ggml_row_bytes(st.token_embd_type, 2560);
        st.h_token_embd = (uint8_t*)std::malloc(emb_nb);
        st.unpinned_allocs.push_back(st.h_token_embd);
        std::memcpy(st.h_token_embd, emb_ptr, emb_nb);
        st.d_token_embd = st.alloc_vram(st.token_embd_row_bytes);
    }
    st.d_out_hc_norm = upload_dense("output_hc_norm.weight");
    st.d_out_hc_down = upload_dense("output_hc_down.weight");
    st.d_out_hc_up   = upload_dense("output_hc_up.weight");
    st.d_output      = upload_dense("output.weight", &st.output_type);
    st.init_async_and_pool();

    // 2. Upload Layer 1 PLE weights & bind Shard 2 131B N-gram table (~85 MB VRAM)
    if (dense_map.contains("blk.1.ple_key.weight") && manifest.contains("ple_ngram_table") && !manifest["ple_ngram_table"].is_null()) {
        const auto& ple_info = manifest["ple_ngram_table"];
        std::filesystem::path ple_file = base_dir / ple_info.value("file", "");
        if (std::filesystem::exists(ple_file) && st.shard2_mmap.open_read(ple_file.string())) {
            st.has_ple = true;
            st.d_ple_norm_q = upload_dense("blk.1.ple_norm_query.weight");
            st.d_ple_norm_k = upload_dense("blk.1.ple_norm_key.weight");
            st.d_ple_norm_c = upload_dense("blk.1.ple_norm_conv.weight");
            st.d_ple_key    = upload_dense("blk.1.ple_key.weight", &st.ple_key_type);
            st.d_ple_val    = upload_dense("blk.1.ple_value.weight", &st.ple_val_type);
            st.d_ple_conv1d = upload_dense("blk.1.ple_conv1d.weight");
            st.d_ple_conv_state = (float*)st.alloc_vram(9 * 10240 * sizeof(float));
            CUDA_CHECK(cudaMemset(st.d_ple_conv_state, 0, 9 * 10240 * sizeof(float)));

            const uint64_t ple_off = ple_info["offset"].get<uint64_t>();
            const uint64_t ple_nb  = ple_info["nbytes"].get<uint64_t>();
            st.mmap_ple_embed = (const uint8_t*)st.shard2_mmap.data() + ple_off;
            CUDA_CHECK(cudaMallocHost((void**)&st.h_ple_rows_iq4nl, 16 * 90));
            st.pinned_allocs.push_back(st.h_ple_rows_iq4nl);
            st.d_ple_rows_iq4nl = st.alloc_vram(16 * 90);
            st.d_ple_emb_flat   = (float*)st.alloc_vram(2560 * sizeof(float));
            LOG_INFO("[Qwen4Exp] Layer 1 PLE + 131B N-Gram Table active (ple_key=%s, ple_val=%s, ple_embed=%.2f GiB MMap)",
                     qwen4_ggml_type_name(st.ple_key_type),
                     qwen4_ggml_type_name(st.ple_val_type),
                     (double)ple_nb / (1024.0 * 1024.0 * 1024.0));
        }
    }

    // 3. Upload all 48 layers' resident weights (GR, GDN/QSA, Router, Shared Expert) + allocate states
    const int n_layers = cfg_.num_hidden_layers; // 48
    st.layers.resize(n_layers);
    const auto& exp_layers_json = manifest["expert_layout"]["layers"];

    for (int l = 0; l < n_layers; ++l) {
        auto& lw = st.layers[l];
        lw.is_qsa = (l % 4 == 3);
        const std::string pfx = "blk." + std::to_string(l) + ".";

        lw.d_hc_attn_norm   = upload_dense(pfx + "hc_attn_norm.weight");
        lw.d_hc_attn_down   = upload_dense(pfx + "hc_attn_down.weight");
        lw.d_hc_attn_up     = upload_dense(pfx + "hc_attn_up.weight");
        lw.d_hc_attn_inject = upload_dense(pfx + "hc_attn_inject.weight");

        lw.d_hc_ffn_norm    = upload_dense(pfx + "hc_ffn_norm.weight");
        lw.d_hc_ffn_down    = upload_dense(pfx + "hc_ffn_down.weight");
        lw.d_hc_ffn_up      = upload_dense(pfx + "hc_ffn_up.weight");
        lw.d_hc_ffn_inject  = upload_dense(pfx + "hc_ffn_inject.weight");

        if (!lw.is_qsa) {
            lw.d_gdn_qkv      = upload_dense(pfx + "attn_qkv.weight", &lw.gdn_qkv_type);
            lw.d_gdn_gate     = upload_dense(pfx + "attn_gate.weight", &lw.gdn_gate_type);
            lw.d_gdn_alpha    = upload_dense(pfx + "ssm_alpha.weight");
            lw.d_gdn_beta     = upload_dense(pfx + "ssm_beta.weight");
            lw.d_gdn_conv1d   = upload_dense(pfx + "ssm_conv1d.weight");
            lw.d_gdn_ssm_a    = upload_dense(pfx + "ssm_a");
            lw.d_gdn_dt_bias  = upload_dense(pfx + "ssm_dt.bias");
            lw.d_gdn_ssm_norm = upload_dense(pfx + "ssm_norm.weight");
            lw.d_gdn_out      = upload_dense(pfx + "ssm_out.weight", &lw.gdn_out_type);

            lw.d_gdn_conv_state = (float*)st.alloc_vram(3 * 10240 * sizeof(float));
            lw.d_gdn_ssm_state  = (float*)st.alloc_vram(48 * 128 * 128 * sizeof(float));
            CUDA_CHECK(cudaMemset(lw.d_gdn_conv_state, 0, 3 * 10240 * sizeof(float)));
            CUDA_CHECK(cudaMemset(lw.d_gdn_ssm_state,  0, 48 * 128 * 128 * sizeof(float)));
        } else {
            lw.d_qsa_q          = upload_dense(pfx + "attn_q.weight", &lw.qsa_q_type);
            lw.d_qsa_k          = upload_dense(pfx + "attn_k.weight", &lw.qsa_k_type);
            lw.d_qsa_v          = upload_dense(pfx + "attn_v.weight", &lw.qsa_v_type);
            lw.d_qsa_q_norm     = upload_dense(pfx + "attn_q_norm.weight");
            lw.d_qsa_k_norm     = upload_dense(pfx + "attn_k_norm.weight");
            lw.d_qsa_out        = upload_dense(pfx + "attn_output.weight", &lw.qsa_out_type);
            lw.d_qsa_idx_k      = upload_dense(pfx + "indexer.k_proj.weight");
            lw.d_qsa_idx_q      = upload_dense(pfx + "indexer.q_proj.weight");
            lw.d_qsa_idx_k_norm = upload_dense(pfx + "indexer.k_norm.weight");
            lw.d_qsa_idx_q_norm = upload_dense(pfx + "indexer.q_norm.weight");

            const size_t max_seq = (size_t)cfg_.max_seq_len;
            const size_t max_blocks = (max_seq + 3) / 4;
            lw.d_qsa_k_cache    = (uint16_t*)st.alloc_vram(max_seq * 512 * sizeof(uint16_t));
            lw.d_qsa_v_cache    = (uint16_t*)st.alloc_vram(max_seq * 512 * sizeof(uint16_t));
            lw.d_qsa_idx_pooled = (float*)st.alloc_vram(max_blocks * 128 * sizeof(float));
            lw.d_qsa_idx_tail   = (float*)st.alloc_vram(4 * 128 * sizeof(float));
            lw.d_qsa_idx_dead   = (float*)st.alloc_vram(128 * sizeof(float));
            CUDA_CHECK(cudaMemset(lw.d_qsa_k_cache,    0, max_seq * 512 * sizeof(uint16_t)));
            CUDA_CHECK(cudaMemset(lw.d_qsa_v_cache,    0, max_seq * 512 * sizeof(uint16_t)));
            CUDA_CHECK(cudaMemset(lw.d_qsa_idx_pooled, 0, max_blocks * 128 * sizeof(float)));
            CUDA_CHECK(cudaMemset(lw.d_qsa_idx_tail,   0, 4 * 128 * sizeof(float)));
            CUDA_CHECK(cudaMemset(lw.d_qsa_idx_dead,   0, 128 * sizeof(float)));
        }

        // Router + Shared Expert
        lw.d_moe_gate_inp       = upload_dense(pfx + "ffn_gate_inp.weight");
        lw.d_moe_gate_inp_shexp = upload_dense(pfx + "ffn_gate_inp_shexp.weight");
        lw.d_moe_gate_shexp     = upload_dense(pfx + "ffn_gate_shexp.weight", &lw.moe_gate_shexp_type);
        lw.d_moe_up_shexp       = upload_dense(pfx + "ffn_up_shexp.weight",   &lw.moe_up_shexp_type);
        lw.d_moe_down_shexp     = upload_dense(pfx + "ffn_down_shexp.weight", &lw.moe_down_shexp_type);

        // Routed expert layout from manifest["expert_layout"]["layers"][l]
        const auto& el = exp_layers_json[l];
        lw.layer_file_offset = el["offset"].get<uint64_t>();
        lw.expert_blob_bytes = el["blob_bytes"].get<size_t>();
        lw.gate_exp_bytes    = el["gate_bytes"].get<size_t>();
        lw.up_exp_bytes      = el["up_bytes"].get<size_t>();
        lw.down_exp_bytes    = el["down_bytes"].get<size_t>();
        lw.moe_gate_exp_type  = el["gate_type"].get<int32_t>();
        lw.moe_up_exp_type    = el["up_type"].get<int32_t>();
        lw.moe_down_exp_type  = el["down_type"].get<int32_t>();
        lw.gate_exp_row_bytes = qwen4_ggml_row_bytes(lw.moe_gate_exp_type, 2560);
        lw.up_exp_row_bytes   = qwen4_ggml_row_bytes(lw.moe_up_exp_type, 2560);
        lw.down_exp_row_bytes = qwen4_ggml_row_bytes(lw.moe_down_exp_type, 640);

        std::fill(std::begin(lw.exp_lru_slot), std::end(lw.exp_lru_slot), (int8_t)-1);
        std::fill(std::begin(lw.lru_slot_eid), std::end(lw.lru_slot_eid), -1);

        st.max_expert_blob_bytes = std::max(st.max_expert_blob_bytes, lw.expert_blob_bytes);
    }

    // Close resident.bin MMap now that all resident weights are in VRAM!
    resident_mmap.close();
#if defined(_WIN32)
    SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
#endif

    // 4. Allocate working activation & bounce buffers
    st.d_R           = (float*)st.alloc_vram(10240 * sizeof(float));
    st.d_x           = (float*)st.alloc_vram(2560 * sizeof(float));
    st.d_moe_pkt     = (uint8_t*)st.alloc_vram(3072);
    CUDA_CHECK(cudaMallocHost((void**)&st.h_moe_pkt, 3072));
    st.pinned_allocs.push_back(st.h_moe_pkt);
    st.d_x_q8_1      = (void*)(st.d_moe_pkt + 0);
    st.h_x_q8_1      = (void*)(st.h_moe_pkt + 0);
    st.d_top_idx     = (int32_t*)(st.d_moe_pkt + 2880);
    st.h_top_idx     = (int32_t*)(st.h_moe_pkt + 2880);
    st.d_top_w       = (float*)(st.d_moe_pkt + 2920);
    st.h_top_w       = (float*)(st.h_moe_pkt + 2920);
    CUDA_CHECK(cudaHostAlloc((void**)&st.h_cpu_moe_out, 2560 * sizeof(float), cudaHostAllocMapped));
    st.pinned_allocs.push_back(st.h_cpu_moe_out);
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&st.d_cpu_moe_out, st.h_cpu_moe_out, 0));
    st.d_sub_out     = (float*)st.alloc_vram(2560 * sizeof(float));
    const size_t scratch_floats = std::max<size_t>(65536, 32768 + ((size_t)cfg_.max_seq_len + 3) / 4 + 256);
    st.d_scratch     = (float*)st.alloc_vram(scratch_floats * sizeof(float));
    st.d_q8_scratch  = st.alloc_vram(10 * qwen4_q8_1_bytes(10240));
    st.d_sel_indices = (int32_t*)st.alloc_vram(2056 * sizeof(int32_t));

    CUDA_CHECK(cudaMallocHost((void**)&st.h_bounce_exp, 10 * st.max_expert_blob_bytes));
    st.pinned_allocs.push_back(st.h_bounce_exp);
    CUDA_CHECK(cudaMallocHost((void**)&st.h_active_ptrs, sizeof(Qwen4ActiveExpertPtrs)));
    st.pinned_allocs.push_back(st.h_active_ptrs);
    st.d_active_ptrs = (Qwen4ActiveExpertPtrs*)st.alloc_vram(sizeof(Qwen4ActiveExpertPtrs));

    // Standard engine control & sampling buffers
    buf_logits_.alloc((size_t)cfg_.vocab_size * sizeof(float));
    buf_argmax_out_.alloc(sizeof(int32_t));
    buf_input_ids_.alloc(sizeof(int32_t));
    buf_input_token_.alloc(sizeof(int32_t));
    buf_input_pos_.alloc(sizeof(int32_t));
    buf_mrope_pos_.alloc(2 * sizeof(int32_t));
    int dummy_mrope[2] = {-1, 0};
    CUDA_CHECK(cudaMemcpy(buf_mrope_pos_.i32(), dummy_mrope, 2 * sizeof(int32_t), cudaMemcpyHostToDevice));
    buf_track_flag_.alloc(sizeof(int32_t));
    CUDA_CHECK(cudaMemset(buf_track_flag_.data, 0, sizeof(int32_t)));
    d_expert_counts_.alloc((size_t)cfg_.num_hidden_layers * std::max(cfg_.n_routed_experts, 256) * sizeof(uint32_t));
    CUDA_CHECK(cudaMemset(d_expert_counts_.data, 0, d_expert_counts_.size_bytes));
    size_t step_topk_sz = (size_t)cfg_.num_hidden_layers * std::max(cfg_.num_experts_per_tok, 10) * sizeof(int32_t);
    d_step_topk_.alloc(step_topk_sz);
    CUDA_CHECK(cudaMemset(d_step_topk_.data, 0, d_step_topk_.size_bytes));
    if (step_topk_host_) cudaFreeHost(step_topk_host_);
    CUDA_CHECK(cudaMallocHost(&step_topk_host_, 2 * step_topk_sz));
    std::memset(step_topk_host_, 0, 2 * step_topk_sz);
    router_indices_host_.resize(cfg_.n_routed_experts);
    logits_host_.resize(cfg_.vocab_size);
    probs_host_.resize(cfg_.vocab_size);
    if (logits_bf16_host_) cudaFreeHost(logits_bf16_host_);
    CUDA_CHECK(cudaMallocHost(&logits_bf16_host_, cfg_.vocab_size * sizeof(__nv_bfloat16)));

    // 5. Load Empirical Expert Routing Frequencies from expert_freq.bin
    const int n_experts = 256;
    const int total_experts = n_layers * n_experts; // 12,288
    std::vector<uint32_t> exp_freq(total_experts, 1);
    if (std::filesystem::exists(freq_bin_path)) {
        FILE* fp_f = std::fopen(freq_bin_path.string().c_str(), "rb");
        if (fp_f) {
            int32_t fl = 0, fe = 0;
            int64_t ftok = 0;
            if (std::fread(&fl, 4, 1, fp_f) == 1 &&
                std::fread(&fe, 4, 1, fp_f) == 1 &&
                std::fread(&ftok, 8, 1, fp_f) == 1 &&
                fl == n_layers && fe == n_experts) {
                std::fread(exp_freq.data(), sizeof(uint32_t), total_experts, fp_f);
                LOG_INFO("[Qwen4Exp] Loaded empirical per-expert routing profile (%d x %d = %d experts) from %s",
                         fl, fe, total_experts, freq_bin_path.filename().string().c_str());
            }
            std::fclose(fp_f);
        }
    }

    const char* env_calib = std::getenv("MOECHER_CALIBRATE_EXPERTS");
    if (env_calib && std::string(env_calib) == "1") {
        st.calibrate_experts      = true;
        st.calib_freq_bin_path    = freq_bin_path.string();
        st.calib_counts_json_path = (base_dir / "imatrix.dat.counts.json").string();
        st.calib_scores.assign(total_experts, 0.0);
        LOG_INFO("[Qwen4Exp] Empirical Router Calibration ACTIVE — will write calibrated expert_freq.bin on exit.");
    }

    // 6. Plan Per-Expert 3-Tier Cache Placement across all 12,288 experts:
    //    - Tier 1b: Dynamic LRU L1 VRAM (16 LRU slots per layer = 768 slots across 48 layers, pre-warmed!)
    //    - Tier 1a: Static Hot L1 VRAM (fills remaining VRAM budget: per-layer floor + global frequency top-K)
    //    - Tier 2 : L2 Pinned Host RAM (up to 9.00 GiB cudaMallocHost, contiguous [gate|up|down] blobs)
    //    - Tier 2.5: L2.5 Unpinned Host RAM (remaining ~3,646 coldest experts in physical RAM, 0 pagefile/SSD reads)
    //    - Tier 3 : L3 NVMe SSD fallback (experts.bin)
    size_t free_vram = query_true_free_vram_bytes();
    const size_t vram_safety_reserve = 640ULL * 1024 * 1024; // 640 MiB WDDM display/page-table reserve
    size_t vram_budget = (free_vram > vram_safety_reserve) ? (free_vram - vram_safety_reserve) : 0;
    if (max_vram_gb > 0.0f) {
        size_t user_max = (size_t)(max_vram_gb * 1024.0 * 1024.0 * 1024.0);
        size_t used_vram = (gpu_caps_.total_vram_bytes > free_vram) ? (gpu_caps_.total_vram_bytes - free_vram) : 0;
        size_t user_rem = (user_max > used_vram) ? (user_max - used_vram) : 0;
        vram_budget = std::min(vram_budget, user_rem);
    }

    const size_t max_pinned_cap   = 10240ULL * 1024 * 1024; // 10.00 GiB Pinned RAM (safe below 12 GiB WDDM limit)
    const size_t max_unpinned_cap = 10240ULL * 1024 * 1024; // 10.00 GiB Unpinned RAM
    size_t pinned_budget   = (dram_cache_gb < 0.0f) ? 0 : max_pinned_cap;
    size_t unpinned_budget = (dram_cache_gb < 0.0f) ? 0 : max_unpinned_cap;
    if (dram_cache_gb > 0.0f) {
        size_t total_ram_budget = (size_t)(dram_cache_gb * 1024.0 * 1024.0 * 1024.0);
        pinned_budget   = std::min(total_ram_budget, max_pinned_cap);
        unpinned_budget = (total_ram_budget > pinned_budget) ? (total_ram_budget - pinned_budget) : 0;
    }

    // Reserve 18 Dynamic LRU L1 VRAM slots per layer (864 slots = 1.65 GiB across 48 layers)
    constexpr int TARGET_LRU_SLOTS = 18;
    std::vector<int> layer_lru_cnt(n_layers, 0);
    for (int l = 0; l < n_layers; ++l) {
        const size_t need = (size_t)TARGET_LRU_SLOTS * st.layers[l].expert_blob_bytes;
        if (vram_budget >= need) {
            layer_lru_cnt[l] = TARGET_LRU_SLOTS;
            vram_budget -= need;
        }
    }

    // Select Static Hot L1 VRAM experts:
    //   Pass 1: Guarantee a minimum floor of 36 hottest experts per layer across all 48 layers
    //   Pass 2: Fill remaining VRAM budget globally in strict empirical frequency order
    enum ExpTier : uint8_t { TIER_NONE = 0, TIER_L1_STATIC = 1, TIER_L2_PINNED = 2, TIER_L25_UNPINNED = 3, TIER_L3_SSD = 4 };
    std::vector<uint8_t> exp_tier(total_experts, TIER_NONE);
    std::vector<int> layer_static_cnt(n_layers, 0);

    constexpr int STATIC_FLOOR_PER_LAYER = 36;
    for (int l = 0; l < n_layers; ++l) {
        std::vector<int> e_order(n_experts);
        std::iota(e_order.begin(), e_order.end(), 0);
        std::sort(e_order.begin(), e_order.end(), [&](int a, int b) {
            return exp_freq[l * n_experts + a] > exp_freq[l * n_experts + b];
        });
        const size_t eb = st.layers[l].expert_blob_bytes;
        for (int i = 0; i < STATIC_FLOOR_PER_LAYER && vram_budget >= eb; ++i) {
            const int e = e_order[i];
            exp_tier[l * n_experts + e] = TIER_L1_STATIC;
            layer_static_cnt[l]++;
            vram_budget -= eb;
        }
    }

    // Global frequency ranking across all remaining experts
    std::vector<int> global_order(total_experts);
    std::iota(global_order.begin(), global_order.end(), 0);
    std::sort(global_order.begin(), global_order.end(), [&](int a, int b) {
        return exp_freq[a] > exp_freq[b];
    });

    for (int flat_idx : global_order) {
        if (exp_tier[flat_idx] != TIER_NONE) continue;
        const int l = flat_idx / n_experts;
        const size_t eb = st.layers[l].expert_blob_bytes;
        // Leave at least layer_lru_cnt[l] non-static experts in each layer unless all 256 fit
        if (vram_budget >= eb && layer_static_cnt[l] < (n_experts - layer_lru_cnt[l])) {
            exp_tier[flat_idx] = TIER_L1_STATIC;
            layer_static_cnt[l]++;
            vram_budget -= eb;
        }
    }

    // Allocate 1 single contiguous L1 VRAM block per layer (Static Hot + Dynamic LRU combined -> 0 VRAM fragmentation!)
    size_t total_static_vram_bytes = 0, total_lru_vram_bytes = 0;
    int total_static_experts = 0, total_lru_slots = 0;
    for (int l = 0; l < n_layers; ++l) {
        auto& lw = st.layers[l];
        const int s_cnt = layer_static_cnt[l];
        const int l_cnt = layer_lru_cnt[l];
        if (s_cnt + l_cnt <= 0) continue;
        const size_t bytes = (size_t)(s_cnt + l_cnt) * lw.expert_blob_bytes;
        void* d_block = nullptr;
        if (cudaMalloc(&d_block, bytes) == cudaSuccess) {
            st.device_allocs.push_back(d_block);
            int slot = 0;
            for (int e = 0; e < n_experts; ++e) {
                if (exp_tier[l * n_experts + e] == TIER_L1_STATIC) {
                    lw.d_vram_exp_ptr[e] = (uint8_t*)d_block + (size_t)slot * lw.expert_blob_bytes;
                    lw.is_static_vram[e] = true;
                    slot++;
                }
            }
            if (l_cnt > 0) {
                lw.d_lru_base  = (uint8_t*)d_block + (size_t)s_cnt * lw.expert_blob_bytes;
                lw.n_lru_slots = l_cnt;
                total_lru_vram_bytes += (size_t)l_cnt * lw.expert_blob_bytes;
                total_lru_slots += l_cnt;
            }
            total_static_vram_bytes += (size_t)s_cnt * lw.expert_blob_bytes;
            total_static_experts += s_cnt;
        } else {
            cudaGetLastError();
            for (int e = 0; e < n_experts; ++e) {
                if (exp_tier[l * n_experts + e] == TIER_L1_STATIC) {
                    exp_tier[l * n_experts + e] = TIER_NONE;
                }
            }
            layer_static_cnt[l] = 0;
            layer_lru_cnt[l]    = 0;
        }
    }

    // Assign remaining non-static experts globally by frequency to Tier 2 (L2 Pinned RAM) -> Tier 2.5 (L2.5 Unpinned RAM) -> Tier 3 (SSD)
    std::vector<int> layer_pinned_cnt(n_layers, 0);
    std::vector<int> layer_unpinned_cnt(n_layers, 0);
    for (int flat_idx : global_order) {
        if (exp_tier[flat_idx] != TIER_NONE) continue;
        const int l = flat_idx / n_experts;
        const size_t eb = st.layers[l].expert_blob_bytes;
        if (pinned_budget >= eb) {
            exp_tier[flat_idx] = TIER_L2_PINNED;
            layer_pinned_cnt[l]++;
            pinned_budget -= eb;
        } else if (unpinned_budget >= eb) {
            exp_tier[flat_idx] = TIER_L25_UNPINNED;
            layer_unpinned_cnt[l]++;
            unpinned_budget -= eb;
        } else {
            exp_tier[flat_idx] = TIER_L3_SSD;
        }
    }

    // Allocate per-layer contiguous L2 Pinned RAM and L2.5 Unpinned RAM blocks
    size_t total_pinned_bytes = 0, total_unpinned_bytes = 0;
    int total_pinned_experts = 0, total_unpinned_experts = 0, total_ssd_experts = 0;

    for (int l = 0; l < n_layers; ++l) {
        auto& lw = st.layers[l];
        const size_t eb = lw.expert_blob_bytes;

        if (layer_pinned_cnt[l] > 0) {
            const size_t p_bytes = (size_t)layer_pinned_cnt[l] * eb;
            void* h_pin = nullptr;
            if (cudaMallocHost(&h_pin, p_bytes) == cudaSuccess) {
                st.pinned_allocs.push_back(h_pin);
                int slot = 0;
                for (int e = 0; e < n_experts; ++e) {
                    if (exp_tier[l * n_experts + e] == TIER_L2_PINNED) {
                        lw.h_pinned_exp_ptr[e] = (uint8_t*)h_pin + (size_t)slot * eb;
                        slot++;
                    }
                }
                total_pinned_bytes += p_bytes;
                total_pinned_experts += layer_pinned_cnt[l];
            } else {
                cudaGetLastError();
                // Demote to Tier 2.5 Unpinned RAM if WDDM pinned limit is reached
                for (int e = 0; e < n_experts; ++e) {
                    if (exp_tier[l * n_experts + e] == TIER_L2_PINNED) {
                        exp_tier[l * n_experts + e] = TIER_L25_UNPINNED;
                        layer_unpinned_cnt[l]++;
                    }
                }
                layer_pinned_cnt[l] = 0;
            }
        }

        if (layer_unpinned_cnt[l] > 0) {
            const size_t u_bytes = (size_t)layer_unpinned_cnt[l] * eb;
            void* h_unpin = std::malloc(u_bytes);
            if (h_unpin) {
                st.unpinned_allocs.push_back(h_unpin);
                int slot = 0;
                for (int e = 0; e < n_experts; ++e) {
                    if (exp_tier[l * n_experts + e] == TIER_L25_UNPINNED) {
                        lw.h_unpinned_exp_ptr[e] = (uint8_t*)h_unpin + (size_t)slot * eb;
                        slot++;
                    }
                }
                total_unpinned_bytes += u_bytes;
                total_unpinned_experts += layer_unpinned_cnt[l];
            } else {
                for (int e = 0; e < n_experts; ++e) {
                    if (exp_tier[l * n_experts + e] == TIER_L25_UNPINNED) {
                        exp_tier[l * n_experts + e] = TIER_L3_SSD;
                        total_ssd_experts++;
                    }
                }
                layer_unpinned_cnt[l] = 0;
            }
        }

        for (int e = 0; e < n_experts; ++e) {
            if (exp_tier[l * n_experts + e] == TIER_L3_SSD) {
                total_ssd_experts++;
            }
        }
    }

    // 7. Stream experts.bin sequentially from byte 0 to end into L1 Static VRAM, L2 Pinned RAM, and L2.5 Unpinned RAM,
    //    and pre-warm all 576 Dynamic LRU L1 VRAM slots with the hottest L2 experts per layer!
    LOG_INFO("[Qwen4Exp] Streaming experts.bin (%.2f GiB) into Per-Expert 3-Tier Cache...",
             (double)manifest["expert_layout"]["total_bytes"].get<uint64_t>() / (1024.0 * 1024.0 * 1024.0));

    st.experts_fp = std::fopen(expert_bin_path.string().c_str(), "rb");
    if (!st.experts_fp) {
        LOG_ERROR("[Qwen4Exp] Failed to open experts.bin: %s", expert_bin_path.string().c_str());
        return false;
    }
    // Use a 16 MiB stdio buffer for fast sequential NVMe reads
    std::setvbuf(st.experts_fp, nullptr, _IOFBF, 16 * 1024 * 1024);

    int total_prewarmed_lru = 0;
    for (int l = 0; l < n_layers; ++l) {
        auto& lw = st.layers[l];
        const size_t eb = lw.expert_blob_bytes;
        _fseeki64(st.experts_fp, (int64_t)lw.layer_file_offset, SEEK_SET);

        for (int e = 0; e < n_experts; ++e) {
            const uint8_t tier = exp_tier[l * n_experts + e];
            if (tier == TIER_L1_STATIC) {
                if (std::fread(st.h_bounce_exp, 1, eb, st.experts_fp) != eb) {
                    LOG_ERROR("[Qwen4Exp] Short read on experts.bin layer %d expert %d", l, e);
                    return false;
                }
                CUDA_CHECK(cudaMemcpy(lw.d_vram_exp_ptr[e], st.h_bounce_exp, eb, cudaMemcpyHostToDevice));
            } else if (tier == TIER_L2_PINNED) {
                if (std::fread(lw.h_pinned_exp_ptr[e], 1, eb, st.experts_fp) != eb) {
                    LOG_ERROR("[Qwen4Exp] Short read on experts.bin layer %d expert %d", l, e);
                    return false;
                }
            } else if (tier == TIER_L25_UNPINNED) {
                if (std::fread(lw.h_unpinned_exp_ptr[e], 1, eb, st.experts_fp) != eb) {
                    LOG_ERROR("[Qwen4Exp] Short read on experts.bin layer %d expert %d", l, e);
                    return false;
                }
            } else {
                _fseeki64(st.experts_fp, (int64_t)eb, SEEK_CUR);
            }
        }

        // Pre-warm layer l's Dynamic LRU L1 VRAM slots with the hottest non-static experts in layer l!
        if (lw.n_lru_slots > 0 && lw.d_lru_base != nullptr) {
            std::vector<int> non_static;
            non_static.reserve(n_experts);
            for (int e = 0; e < n_experts; ++e) {
                if (!lw.is_static_vram[e] && (lw.h_pinned_exp_ptr[e] || lw.h_unpinned_exp_ptr[e])) {
                    non_static.push_back(e);
                }
            }
            std::sort(non_static.begin(), non_static.end(), [&](int a, int b) {
                return exp_freq[l * n_experts + a] > exp_freq[l * n_experts + b];
            });
            const int n_warm = std::min((int)non_static.size(), lw.n_lru_slots);
            // Seed hotter experts with higher initial lru_clock so colder pre-warmed slots are evicted first
            for (int s = 0; s < n_warm; ++s) {
                const int e = non_static[n_warm - 1 - s]; // s=0 gets coldest of top-12, s=n_warm-1 gets hottest of top-12
                const uint8_t* src_host = lw.h_pinned_exp_ptr[e] ? lw.h_pinned_exp_ptr[e] : lw.h_unpinned_exp_ptr[e];
                uint8_t* dst_vram = lw.d_lru_base + (size_t)s * eb;
                CUDA_CHECK(cudaMemcpy(dst_vram, src_host, eb, cudaMemcpyHostToDevice));
                lw.d_vram_exp_ptr[e]     = dst_vram;
                lw.exp_lru_slot[e]       = (int8_t)s;
                lw.lru_slot_eid[s]       = e;
                lw.lru_slot_last_used[s] = ++st.lru_clock;
                lw.lru_slot_epoch[s]     = 0;
                total_prewarmed_lru++;
            }
        }
    }

    if (total_ssd_experts == 0 && st.experts_fp) {
        std::fclose(st.experts_fp);
        st.experts_fp = nullptr;
        LOG_INFO("[Qwen4Exp] All 12,288 experts resident in L1 VRAM + L2/L2.5 RAM — closed experts.bin (0 SSD reads)!");
    }

    LOG_INFO("[Qwen4Exp] Per-Expert Tier Summary: "
             "L1 VRAM = %d experts (%.2f GiB: %d Static Hot [%.2f GiB] + %d Dynamic LRU Slots [%.2f GiB]) | "
             "L2 Pinned RAM = %d experts (%.2f GiB) | "
             "L2.5 Unpinned RAM = %d experts (%.2f GiB) | "
             "L3 SSD = %d experts",
             total_static_experts + total_prewarmed_lru,
             (double)(total_static_vram_bytes + total_lru_vram_bytes) / (1024.0 * 1024.0 * 1024.0),
             total_static_experts,
             (double)total_static_vram_bytes / (1024.0 * 1024.0 * 1024.0),
             total_lru_slots,
             (double)total_lru_vram_bytes / (1024.0 * 1024.0 * 1024.0),
             total_pinned_experts,
             (double)total_pinned_bytes / (1024.0 * 1024.0 * 1024.0),
             total_unpinned_experts,
             (double)total_unpinned_bytes / (1024.0 * 1024.0 * 1024.0),
             total_ssd_experts);
    LOG_INFO("[Qwen4Exp] Hybrid CPU Expert Compute: %s (%d AVX2 worker threads)",
             enable_cpu_experts_ ? "ENABLED" : "DISABLED",
             moecher::cpu_moe::CpuMoEExpertPool::instance().num_threads());

    st.loaded = true;
    return true;
}

void reset_state_qwen4_exp() {
    if (!qwen4_state_ || !qwen4_state_->loaded) return;
    auto& st = *qwen4_state_;
    if (st.copy_stream) {
        CUDA_CHECK(cudaStreamSynchronize(st.copy_stream));
    }
    st.token_history.clear();
    if (st.has_ple) {
        CUDA_CHECK(cudaMemsetAsync(st.d_ple_conv_state, 0, 9 * 10240 * sizeof(float), main_stream_));
    }
    for (auto& lw : st.layers) {
        if (!lw.is_qsa) {
            CUDA_CHECK(cudaMemsetAsync(lw.d_gdn_conv_state, 0, 3 * 10240 * sizeof(float), main_stream_));
            CUDA_CHECK(cudaMemsetAsync(lw.d_gdn_ssm_state,  0, 48 * 128 * 128 * sizeof(float), main_stream_));
        } else {
            CUDA_CHECK(cudaMemsetAsync(lw.d_qsa_idx_tail, 0, 4 * 128 * sizeof(float), main_stream_));
            CUDA_CHECK(cudaMemsetAsync(lw.d_qsa_idx_dead, 0, 128 * sizeof(float), main_stream_));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(main_stream_));
}

void forward_token_qwen4_exp(int token_id, int position, bool do_compute_logits = true) {
    auto& st = *qwen4_state_;
    st.stat_tokens_recorded++;

    // Track token history for Layer 1 PLE multi-order N-gram hash
    if ((int)st.token_history.size() <= position) {
        st.token_history.resize(position + 1, 0);
    }
    st.token_history[position] = token_id;

    // 1. Token embedding single-row H2D + dequantize + broadcast into 4-stream hyper-connection state R[4, 2560]
    CUDA_CHECK(cudaMemcpyAsync(
        st.d_token_embd,
        st.h_token_embd + (size_t)token_id * st.token_embd_row_bytes,
        st.token_embd_row_bytes,
        cudaMemcpyHostToDevice,
        main_stream_));
    qwen4_embed_to_hc_cuda(
        st.token_embd_type, st.d_token_embd, 0, st.d_R, 2560, 4, main_stream_);

    // 2. Process all 48 layers
    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
        auto& lw = st.layers[l];
        const uint64_t step_epoch = ++st.cur_token_epoch;

        // Layer 1 PLE + 131B N-gram cross-attention block (runs BEFORE blk.1.hc_attn_*)
        if (l == 1 && st.has_ple) {
            static constexpr uint64_t PLE_MULT[3] = {
                23703573157769ULL, 20109073645365ULL, 8052911324071ULL
            };
            static constexpr uint64_t PLE_OFF[16] = {
                0ULL, 20000003ULL, 40000026ULL, 60000059ULL,
                80000106ULL, 100000165ULL, 120000228ULL, 140000297ULL,
                160000374ULL, 180000455ULL, 200000548ULL, 220000655ULL,
                240000802ULL, 260000955ULL, 280001114ULL, 300001275ULL
            };
            static constexpr uint64_t PLE_VOCAB[16] = {
                20000003ULL, 20000023ULL, 20000033ULL, 20000047ULL,
                20000059ULL, 20000063ULL, 20000069ULL, 20000077ULL,
                20000081ULL, 20000093ULL, 20000107ULL, 20000147ULL,
                20000153ULL, 20000159ULL, 20000161ULL, 20000171ULL
            };
            static constexpr int64_t PLE_EOS = 248044;
            int64_t ctx[3];
            ctx[0] = (int64_t)token_id;
            bool cut = false;
            for (int s = 1; s < 3; ++s) {
                const int q = position - s;
                const int64_t tk = (q >= 0) ? (int64_t)st.token_history[q] : -1;
                cut = cut || tk < 0 || tk == PLE_EOS;
                ctx[s] = cut ? PLE_EOS : tk;
            }
            uint64_t ple_rows[16];
            for (int n = 2; n <= 3; ++n) {
                uint64_t mixed = (uint64_t)ctx[0] * PLE_MULT[0];
                for (int j = 1; j < n; ++j) mixed ^= (uint64_t)ctx[j] * PLE_MULT[j];
                for (int gi = 0; gi < 8; ++gi) {
                    const int h = (n - 2) * 8 + gi;
                    ple_rows[h] = mixed % PLE_VOCAB[h] + PLE_OFF[h];
                }
            }
#if defined(_WIN32)
            WIN32_MEMORY_RANGE_ENTRY ple_ranges[16];
            for (int h = 0; h < 16; ++h) {
                ple_ranges[h].VirtualAddress = (PVOID)(st.mmap_ple_embed + ple_rows[h] * 90);
                ple_ranges[h].NumberOfBytes  = 90;
            }
            PrefetchVirtualMemory(GetCurrentProcess(), 16, ple_ranges, 0);
#endif
            for (int h = 0; h < 16; ++h) {
                std::memcpy(st.h_ple_rows_iq4nl + h * 90, st.mmap_ple_embed + ple_rows[h] * 90, 90);
            }
            CUDA_CHECK(cudaMemcpyAsync(
                st.d_ple_rows_iq4nl, st.h_ple_rows_iq4nl, 16 * 90, cudaMemcpyHostToDevice, main_stream_));
            qwen4_ple_iq4nl_dequant_cuda(st.d_ple_rows_iq4nl, st.d_ple_emb_flat, main_stream_);
            qwen4_ple_forward_cuda(
                st.d_R,
                (const float*)st.d_ple_norm_q,
                (const float*)st.d_ple_norm_k,
                (const float*)st.d_ple_norm_c,
                st.d_ple_key, st.ple_key_type,
                st.d_ple_val, st.ple_val_type,
                (const uint16_t*)st.d_ple_conv1d,
                st.d_ple_emb_flat,
                st.d_ple_conv_state,
                st.d_scratch,
                st.d_q8_scratch,
                cfg_.rms_norm_eps,
                main_stream_);
        }

        // --- Attention Sub-layer (GR Read -> GDN or QSA -> GR Write) ---
        qwen4_gr_read_cuda(
            st.d_R,
            (const float*)lw.d_hc_attn_norm,
            (const uint16_t*)lw.d_hc_attn_down,
            (const uint16_t*)lw.d_hc_attn_up,
            st.d_x,
            st.d_scratch,
            cfg_.rms_norm_eps,
            main_stream_);

        qwen4_quantize_q8_1_cuda(st.d_x, st.d_x_q8_1, 2560, main_stream_);

        if (!lw.is_qsa) {
            qwen4_gdn_decode_cuda(
                st.d_x, st.d_x_q8_1,
                lw.d_gdn_qkv, lw.gdn_qkv_type,
                lw.d_gdn_gate, lw.gdn_gate_type,
                (const uint16_t*)lw.d_gdn_alpha,
                (const uint16_t*)lw.d_gdn_beta,
                (const float*)lw.d_gdn_conv1d,
                (const float*)lw.d_gdn_ssm_a,
                (const float*)lw.d_gdn_dt_bias,
                (const float*)lw.d_gdn_ssm_norm,
                lw.d_gdn_out, lw.gdn_out_type,
                lw.d_gdn_conv_state,
                lw.d_gdn_ssm_state,
                st.d_sub_out,
                st.d_scratch,
                st.d_q8_scratch,
                cfg_.rms_norm_eps,
                main_stream_);
        } else {
            qwen4_qsa_decode_cuda(
                st.d_x, st.d_x_q8_1,
                lw.d_qsa_q, lw.qsa_q_type,
                lw.d_qsa_k, lw.qsa_k_type,
                lw.d_qsa_v, lw.qsa_v_type,
                (const float*)lw.d_qsa_q_norm,
                (const float*)lw.d_qsa_k_norm,
                lw.d_qsa_out, lw.qsa_out_type,
                (const uint16_t*)lw.d_qsa_idx_k,
                (const uint16_t*)lw.d_qsa_idx_q,
                (const float*)lw.d_qsa_idx_k_norm,
                (const float*)lw.d_qsa_idx_q_norm,
                lw.d_qsa_k_cache,
                lw.d_qsa_v_cache,
                lw.d_qsa_idx_pooled,
                lw.d_qsa_idx_tail,
                lw.d_qsa_idx_dead,
                position,
                cfg_.max_seq_len,
                st.d_sub_out,
                st.d_scratch,
                st.d_sel_indices,
                st.d_q8_scratch,
                cfg_.rms_norm_eps,
                main_stream_);
        }

        qwen4_gr_write_cuda(
            st.d_R,
            (const float*)lw.d_hc_attn_norm,
            (const uint16_t*)lw.d_hc_attn_inject,
            st.d_sub_out,
            st.d_scratch,
            cfg_.rms_norm_eps,
            main_stream_);

        // --- MoE Sub-layer (GR Read -> Router + Shared Expert + Per-Expert L1/L2/L2.5/L3 Routed Experts -> GR Write) ---
        qwen4_gr_read_cuda(
            st.d_R,
            (const float*)lw.d_hc_ffn_norm,
            (const uint16_t*)lw.d_hc_ffn_down,
            (const uint16_t*)lw.d_hc_ffn_up,
            st.d_x,
            st.d_scratch,
            cfg_.rms_norm_eps,
            main_stream_);

        qwen4_quantize_q8_1_cuda(st.d_x, st.d_x_q8_1, 2560, main_stream_);

        // Launch Top-10 of 256 Router
        qwen4_router_top10_cuda(
            (const uint16_t*)lw.d_moe_gate_inp,
            st.d_x,
            st.d_scratch,
            st.d_top_idx,
            st.d_top_w,
            main_stream_);

        if (st.calibrate_experts) {
            CUDA_CHECK(cudaMemcpyAsync(
                st.calib_probs_host, st.d_scratch, 256 * sizeof(float), cudaMemcpyDeviceToHost, main_stream_));
        }

        // Copy Top-10 expert IDs (or single contiguous 2,960 B MoE packet when CPU experts enabled) on copy_stream while Shared Expert runs on main_stream_
        CUDA_CHECK(cudaEventRecord(st.router_event, main_stream_));
        CUDA_CHECK(cudaStreamWaitEvent(st.copy_stream, st.router_event, 0));
        if (enable_cpu_experts_) {
            CUDA_CHECK(cudaMemcpyAsync(
                st.h_moe_pkt, st.d_moe_pkt, 2960, cudaMemcpyDeviceToHost, st.copy_stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(
                st.h_top_idx, st.d_top_idx, 10 * sizeof(int32_t), cudaMemcpyDeviceToHost, st.copy_stream));
        }
        CUDA_CHECK(cudaEventRecord(st.top_idx_event, st.copy_stream));

        // Launch Shared Expert on main_stream_
        qwen4_shared_expert_cuda(
            st.d_x, st.d_x_q8_1,
            lw.d_moe_gate_shexp, lw.moe_gate_shexp_type,
            lw.d_moe_up_shexp,   lw.moe_up_shexp_type,
            lw.d_moe_down_shexp, lw.moe_down_shexp_type,
            (const uint16_t*)lw.d_moe_gate_inp_shexp,
            st.d_sub_out,
            st.d_scratch,
            st.d_q8_scratch,
            main_stream_);

        CUDA_CHECK(cudaEventSynchronize(st.top_idx_event));
        if (st.calibrate_experts) {
            for (int e = 0; e < 256; ++e) {
                st.calib_scores[l * 256 + e] += 100.0 * (double)st.calib_probs_host[e];
            }
            for (int k = 0; k < 10; ++k) {
                const int eid = st.h_top_idx[k];
                if (eid >= 0 && eid < 256) {
                    st.calib_scores[l * 256 + eid] += (double)(20 - k);
                }
            }
        }

        // Partition the 10 active experts into L1 VRAM Hits [0..n_hits-1] and L1 VRAM Misses
        const size_t g_off = 0;
        const size_t u_off = lw.gate_exp_bytes;
        const size_t d_off = lw.gate_exp_bytes + lw.up_exp_bytes;
        const size_t eb    = lw.expert_blob_bytes;

        int n_hits = 0;
        int miss_eids[10];
        int miss_orig_k[10];
        int n_misses = 0;

        for (int k = 0; k < 10; ++k) {
            const int eid = st.h_top_idx[k];
            uint8_t* vptr = lw.d_vram_exp_ptr[eid];
            if (vptr != nullptr) {
                // L1 VRAM Hit (either Static Hot L1 or Dynamic LRU L1)
                if (lw.is_static_vram[eid]) {
                    st.stat_l1_static_hits++;
                } else {
                    st.stat_l1_lru_hits++;
                    const int s = (int)lw.exp_lru_slot[eid];
                    if (s >= 0) {
                        lw.lru_slot_last_used[s] = ++st.lru_clock;
                        lw.lru_slot_epoch[s]     = step_epoch;
                    }
                }
                const int p = n_hits++;
                st.h_active_ptrs->gate[p] = vptr + g_off;
                st.h_active_ptrs->up[p]   = vptr + u_off;
                st.h_active_ptrs->down[p] = vptr + d_off;
                st.h_active_ptrs->slot[p] = k;
            } else {
                miss_eids[n_misses]   = eid;
                miss_orig_k[n_misses] = k;
                n_misses++;
            }
        }

        // Helper to allocate a Dynamic LRU L1 VRAM victim slot for a missed expert
        auto alloc_lru_slot = [&](int eid) -> uint8_t* {
            int best_s = 0;
            uint64_t best_time = UINT64_MAX;
            for (int s = 0; s < lw.n_lru_slots; ++s) {
                if (lw.lru_slot_epoch[s] == step_epoch) continue;
                if (lw.lru_slot_eid[s] < 0) {
                    best_s = s;
                    break;
                }
                if (lw.lru_slot_last_used[s] < best_time) {
                    best_time = lw.lru_slot_last_used[s];
                    best_s = s;
                }
            }
            const int old_eid = lw.lru_slot_eid[best_s];
            if (old_eid >= 0) {
                lw.d_vram_exp_ptr[old_eid] = nullptr;
                lw.exp_lru_slot[old_eid]   = -1;
            }
            uint8_t* slot_vram = lw.d_lru_base + (size_t)best_s * eb;
            lw.d_vram_exp_ptr[eid]         = slot_vram;
            lw.exp_lru_slot[eid]           = (int8_t)best_s;
            lw.lru_slot_eid[best_s]       = eid;
            lw.lru_slot_last_used[best_s] = ++st.lru_clock;
            lw.lru_slot_epoch[best_s]     = step_epoch;
            return slot_vram;
        };

        // Decide which misses (if any) are computed on the CPU vs DMA'd to the GPU
        moecher::cpu_moe::CpuExpertTaskDesc cpu_tasks[10];
        int n_cpu_misses = 0;
        bool is_cpu_miss[10] = {false};

        if (enable_cpu_experts_ && n_misses > 0) {
            // Hybrid CPU/GPU 3-Way Overlap Policy:
            // - Warm Tier 2 (L2 Pinned RAM) misses are ALWAYS promoted to L1 Dynamic LRU VRAM on st.copy_stream
            //   so frequent experts hit L1 VRAM on subsequent tokens.
            // - Cold Tier 2.5 (L2.5 Unpinned RAM) tail misses (orig_k >= 7 when n_misses >= 2, or when lw.n_lru_slots == 0)
            //   are offloaded to the CPU AVX2 pool so they execute inside the PCIe DMA shadow without requiring a host bounce
            //   copy and without evicting warm experts from L1 Dynamic LRU VRAM!
            if (n_misses >= 2 || lw.n_lru_slots == 0) {
                for (int m = n_misses - 1; m >= 0; --m) {
                    const int eid    = miss_eids[m];
                    const int orig_k = miss_orig_k[m];
                    if (lw.h_unpinned_exp_ptr[eid] != nullptr && (orig_k >= 7 || lw.n_lru_slots == 0)) {
                        is_cpu_miss[m] = true;
                        st.stat_l25_unpinned_hits++;
                        st.stat_cpu_computed++;
                        const uint8_t* base = lw.h_unpinned_exp_ptr[eid];
                        auto& ct = cpu_tasks[n_cpu_misses++];
                        ct.gate_ptr = base + g_off;
                        ct.up_ptr   = base + u_off;
                        ct.down_ptr = base + d_off;
                        ct.weight   = st.h_top_w[orig_k];
                        break;
                    }
                }
            }
        }

        // Assign Dynamic LRU L1 VRAM victim slots for remaining GPU misses
        int gpu_miss_eids[10];
        uint8_t* gpu_miss_dst_vram[10];
        int n_gpu_misses = 0;

        for (int m = 0; m < n_misses; ++m) {
            if (is_cpu_miss[m]) continue;
            const int eid    = miss_eids[m];
            const int orig_k = miss_orig_k[m];
            uint8_t* slot_vram = alloc_lru_slot(eid);

            const int gm = n_gpu_misses++;
            gpu_miss_eids[gm]     = eid;
            gpu_miss_dst_vram[gm] = slot_vram;

            const int p = n_hits + gm;
            st.h_active_ptrs->gate[p] = slot_vram + g_off;
            st.h_active_ptrs->up[p]   = slot_vram + u_off;
            st.h_active_ptrs->down[p] = slot_vram + d_off;
            st.h_active_ptrs->slot[p] = orig_k;
        }

        // 1) Upload active pointer table (280 B) and IMMEDIATELY launch L1-Hit experts [0, n_hits) on main_stream_!
        CUDA_CHECK(cudaMemcpyAsync(
            st.d_active_ptrs, st.h_active_ptrs, sizeof(Qwen4ActiveExpertPtrs), cudaMemcpyHostToDevice, main_stream_));

        if (n_hits > 0) {
            qwen4_routed_experts_range_cuda(
                st.d_x_q8_1,
                nullptr, 0, lw.moe_gate_exp_type,
                nullptr, 0, lw.moe_up_exp_type,
                nullptr, 0, lw.moe_down_exp_type,
                st.d_active_ptrs,
                nullptr,
                st.d_top_w,
                0,
                n_hits,
                /*do_accum=*/(n_gpu_misses == 0 && n_cpu_misses == 0),
                st.d_sub_out,
                st.d_scratch,
                st.d_q8_scratch,
                main_stream_,
                nullptr);
        }

        // 2) Queue any remaining GPU misses on copy_stream and launch their GPU kernels on main_stream_
        //    BEFORE starting CPU expert compute so PCIe DMA + GPU miss kernels run concurrently with CPU AVX2!
        if (n_gpu_misses > 0) {
            int unpinned_task_cnt = 0;
            int unpinned_miss_idx[10];
            for (int gm = 0; gm < n_gpu_misses; ++gm) {
                const int eid = gpu_miss_eids[gm];
                if (lw.h_pinned_exp_ptr[eid] != nullptr) {
                    st.stat_l2_pinned_hits++;
                    CUDA_CHECK(cudaMemcpyAsync(
                        gpu_miss_dst_vram[gm], lw.h_pinned_exp_ptr[eid], eb, cudaMemcpyHostToDevice, st.copy_stream));
                } else if (lw.h_unpinned_exp_ptr[eid] != nullptr) {
                    st.stat_l25_unpinned_hits++;
                    const int tidx = unpinned_task_cnt++;
                    unpinned_miss_idx[tidx] = gm;
                    st.copy_tasks[tidx].src    = lw.h_unpinned_exp_ptr[eid];
                    st.copy_tasks[tidx].dst    = st.h_bounce_exp + (size_t)gm * st.max_expert_blob_bytes;
                    st.copy_tasks[tidx].nbytes = eb;
                } else if (st.experts_fp != nullptr) {
                    st.stat_l3_ssd_reads++;
                    uint8_t* bounce = st.h_bounce_exp + (size_t)gm * st.max_expert_blob_bytes;
                    _fseeki64(st.experts_fp, (int64_t)(lw.layer_file_offset + (uint64_t)eid * eb), SEEK_SET);
                    std::fread(bounce, 1, eb, st.experts_fp);
                    CUDA_CHECK(cudaMemcpyAsync(
                        gpu_miss_dst_vram[gm], bounce, eb, cudaMemcpyHostToDevice, st.copy_stream));
                }
            }

            if (unpinned_task_cnt > 0) {
                (void)cudaStreamQuery(st.copy_stream);
                (void)cudaStreamQuery(main_stream_);
                st.parallel_unpinned_gather(unpinned_task_cnt);
                for (int t = 0; t < unpinned_task_cnt; ++t) {
                    const int gm = unpinned_miss_idx[t];
                    CUDA_CHECK(cudaMemcpyAsync(
                        gpu_miss_dst_vram[gm],
                        st.h_bounce_exp + (size_t)gm * st.max_expert_blob_bytes,
                        eb,
                        cudaMemcpyHostToDevice,
                        st.copy_stream));
                }
            }

            CUDA_CHECK(cudaEventRecord(st.stage_ready_event, st.copy_stream));
            CUDA_CHECK(cudaStreamWaitEvent(main_stream_, st.stage_ready_event, 0));

            qwen4_routed_experts_range_cuda(
                st.d_x_q8_1,
                nullptr, 0, lw.moe_gate_exp_type,
                nullptr, 0, lw.moe_up_exp_type,
                nullptr, 0, lw.moe_down_exp_type,
                st.d_active_ptrs,
                nullptr,
                st.d_top_w,
                n_hits,
                n_gpu_misses,
                /*do_accum=*/(n_cpu_misses == 0),
                st.d_sub_out,
                st.d_scratch,
                st.d_q8_scratch,
                main_stream_,
                nullptr);
        }

        // 3) Evaluate CPU expert(s) on the AVX2 worker pool while GPU L1-Hits + PCIe DMA + GPU Misses run on the GPU!
        if (n_cpu_misses > 0) {
            // Flush WDDM command batch on both streams before CPU threads begin heavy AVX2 compute
            (void)cudaStreamQuery(st.copy_stream);
            (void)cudaStreamQuery(main_stream_);

            moecher::cpu_moe::CpuMoEExpertPool::instance().compute_experts_sync(
                n_cpu_misses,
                cpu_tasks,
                lw.moe_gate_exp_type, lw.gate_exp_row_bytes,
                lw.moe_up_exp_type,   lw.up_exp_row_bytes,
                lw.moe_down_exp_type, lw.down_exp_row_bytes,
                st.h_x_q8_1,
                st.h_cpu_moe_out,
                2560,
                640);

            // Accumulate all GPU routed experts [0, n_hits + n_gpu_misses) and CPU output (d_cpu_moe_out) into d_sub_out
            qwen4_routed_experts_range_cuda(
                st.d_x_q8_1,
                nullptr, 0, lw.moe_gate_exp_type,
                nullptr, 0, lw.moe_up_exp_type,
                nullptr, 0, lw.moe_down_exp_type,
                st.d_active_ptrs,
                nullptr,
                st.d_top_w,
                n_hits + n_gpu_misses,
                0,
                /*do_accum=*/true,
                st.d_sub_out,
                st.d_scratch,
                st.d_q8_scratch,
                main_stream_,
                st.d_cpu_moe_out);
        }

        qwen4_gr_write_cuda(
            st.d_R,
            (const float*)lw.d_hc_ffn_norm,
            (const uint16_t*)lw.d_hc_ffn_inject,
            st.d_sub_out,
            st.d_scratch,
            cfg_.rms_norm_eps,
            main_stream_);
    }

    // 3. Final Output Gated Residual Read + Output Head Logits (only on final prefill token or decode step)
    if (do_compute_logits) {
        qwen4_gr_read_cuda(
            st.d_R,
            (const float*)st.d_out_hc_norm,
            (const uint16_t*)st.d_out_hc_down,
            (const uint16_t*)st.d_out_hc_up,
            st.d_x,
            st.d_scratch,
            cfg_.rms_norm_eps,
            main_stream_);

        qwen4_quantize_q8_1_cuda(st.d_x, st.d_x_q8_1, 2560, main_stream_);
        qwen4_project_cuda(
            st.output_type, st.d_output, st.d_x, st.d_x_q8_1, buf_logits_.f32(), cfg_.vocab_size, 2560, main_stream_);
    }
}

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
    int       device_id                  = 0;
    uint8_t*  d_all_vram_base            = nullptr; // Non-null when all 256 experts reside contiguously in L1 Static VRAM
    bool      all_decode_gpu_accessible  = false;   // True when all 256 experts have valid device-accessible pointers in d_decode_exp_table (L1 VRAM + Mapped L2 Pinned RAM)
    uint8_t*  d_vram_exp_ptr[256]        = {nullptr};
    uint8_t*  d_mapped_exp_ptr[256]      = {nullptr}; // Zero-copy device pointer for L2 Pinned Host RAM experts
    uint8_t** d_vram_exp_table           = nullptr; // Device [256 * sizeof(uint8_t*)] mirror of d_vram_exp_ptr (L1 VRAM only)
    uint8_t** h_vram_exp_table           = nullptr; // Pinned Host [256 * sizeof(uint8_t*)] mirror of d_vram_exp_ptr
    uint8_t** d_decode_exp_table         = nullptr; // Device [256 * sizeof(uint8_t*)] mirror of (d_vram_exp_ptr ? d_vram_exp_ptr : d_mapped_exp_ptr)
    uint8_t** h_decode_exp_table         = nullptr; // Pinned Host [256 * sizeof(uint8_t*)] mirror for d_decode_exp_table
    bool      is_static_vram[256]        = {false};
    int8_t    exp_lru_slot[256]          = {0}; // -1 if not in Dynamic LRU L1, else 0..n_lru_slots-1
    uint16_t  decode_miss_cnt[256]       = {0}; // Count of decode misses while in L2 Mapped Pinned RAM

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

    // Pre-sorted non-static expert IDs by descending empirical router frequency (for Attention-Window Prefetching)
    uint8_t  non_static_order[256]   = {0};
    int      n_non_static            = 0;
};

struct Qwen4RuntimeState {
    static constexpr int QWEN4_MAX_SUB_M     = 576;
    static constexpr int QWEN4_MAX_PREFILL_M = 2064;
    bool loaded = false;
    int  num_gpus = 1;
    int  pp_split_layer = 48;
    moecher::platform::MemoryMappedFile shard1_mmap;
    moecher::platform::MemoryMappedFile shard2_mmap;
    FILE* experts_fp = nullptr;

    // Global weights in VRAM (token_embd kept in Host RAM with VRAM staging to save ~360 MB VRAM)
    uint8_t* h_token_embd = nullptr;
    uint8_t* h_token_embd_staging = nullptr; // Pinned Mapped [2 * QWEN4_MAX_PREFILL_M * token_embd_row_bytes]
    uint8_t* d_host_token_embd_staging = nullptr;
    size_t   token_embd_row_bytes = 0;
    void* d_token_embd  = nullptr; int32_t token_embd_type = 0;
    void* d_out_hc_norm = nullptr; // F32 [10240]
    void* d_out_hc_down = nullptr; // BF16 [320, 10240]
    void* d_out_hc_up   = nullptr; // BF16 [10240, 320]
    void* d_output      = nullptr; int32_t output_type = 0;
    void* d_output_dev[2] = {nullptr, nullptr};
    int   vocab_split   = 0;

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
    uint8_t* h_ple_rows_iq4nl = nullptr; // Pinned Mapped [64 * 64 * 16 * 90 B ring buffer]
    uint8_t* d_host_ple_rows_iq4nl = nullptr;
    void*    d_ple_rows_iq4nl = nullptr; // Device [1440 B]
    float*   d_ple_emb_flat   = nullptr; // Device [16 * 160 = 2560 floats]
    std::vector<int32_t> token_history;

    // Working activation buffers in VRAM (per-device for 2-GPU Pipeline Parallelism)
    float*   d_R_dev[2]            = {nullptr, nullptr}; // [4 * 2560 = 10240]
    float*   d_x_dev[2]            = {nullptr, nullptr}; // [2560]
    uint8_t* d_moe_pkt_dev[2]      = {nullptr, nullptr}; // Contiguous [3072 B]: d_x_q8_1 (2880 B) + d_top_idx (40 B) + d_top_w (40 B)
    void*    d_x_q8_1_dev[2]       = {nullptr, nullptr}; // [2560 / 32 * 36 = 2880 B]
    float*   d_sub_out_dev[2]      = {nullptr, nullptr}; // [2560]
    float*   d_scratch_dev[2]      = {nullptr, nullptr}; // [65536 floats = 256 KB]
    void*    d_q8_scratch_dev[2]   = {nullptr, nullptr}; // [32768 B]
    int32_t* d_sel_indices_dev[2]  = {nullptr, nullptr}; // [2056 int32]
    int32_t* d_top_idx_dev[2]      = {nullptr, nullptr}; // [10 int32]
    float*   d_top_w_dev[2]        = {nullptr, nullptr}; // [10 float]
    float*   d_cpu_moe_out_dev[2]  = {nullptr, nullptr}; // Device Mapped pointer to h_cpu_moe_out
    Qwen4ActiveExpertPtrs* d_active_ptrs_dev[2] = {nullptr, nullptr};
    float*   d_logits_dev1         = nullptr;            // [vocab_size] on GPU 1 when num_gpus == 2

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
    int32_t* h_top_idx      = nullptr; // Pinned Mapped [10 int32] (alias of h_moe_pkt + 2880)
    float*   h_top_w        = nullptr; // Pinned Mapped [10 float] (alias of h_moe_pkt + 2920)
    int32_t* d_host_top_idx_dev[2] = {nullptr, nullptr};
    float*   d_host_top_w_dev[2]   = {nullptr, nullptr};
    float*   h_cpu_moe_out  = nullptr; // Pinned Mapped [2560 floats] for Hybrid CPU Expert Compute
    float*   d_cpu_moe_out  = nullptr; // Device Mapped pointer to h_cpu_moe_out
    int32_t* h_argmax_out   = nullptr; // Pinned Mapped [1 int32] for zero-copy L2-resident argmax
    int32_t* d_argmax_out_dev[2] = {nullptr, nullptr};
    int      greedy_mask_id0 = -1;
    int      greedy_mask_id1 = -1;
    bool     fast_greedy     = false;
    bool     argmax_ready    = false;

    // Pinned host bounce buffer for Tier 2.5 (Unpinned RAM) & Tier 3 (SSD) misses (up to 48 experts)
    size_t   max_expert_blob_bytes = 0;
    uint8_t* h_bounce_exp          = nullptr; // Pinned [48 * max_expert_blob_bytes]
    Qwen4ActiveExpertPtrs* h_active_ptrs = nullptr; // Pinned
    Qwen4ActiveExpertPtrs* d_active_ptrs = nullptr;
    uint8_t** h_vram_exp_tables_all   = nullptr; // Pinned [48 * 256 * sizeof(uint8_t*)]
    uint8_t** h_decode_exp_tables_all = nullptr; // Pinned [48 * 256 * sizeof(uint8_t*)]
    bool      has_unpinned_experts       = false;
    bool      has_pending_decode_top_idx = false;

    struct Qwen4PrefillTaskPack {
        const void* gate_ptrs[256];
        const void* up_ptrs[256];
        const void* down_ptrs[256];
        Qwen4MoETileTask tasks[400];
    };
    struct Qwen4PrefillMissPair {
        int32_t tok;
        float   wt;
    };
    Qwen4PrefillTaskPack* h_prefill_pack = nullptr; // Pinned Mapped
    Qwen4PrefillTaskPack* d_prefill_pack_dev[2] = {nullptr, nullptr};
    Qwen4PrefillMissPair flat_miss_pairs[10 * QWEN4_MAX_PREFILL_M];

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
    bool profile_timing    = false;
    double prof_dec[10]    = {0.0};
    uint64_t prof_dec_cnt  = 0;
    std::string calib_freq_bin_path;
    std::string calib_counts_json_path;
    std::vector<double> calib_scores; // [48 * 256]
    float calib_probs_host[256] = {0.0f};

    // Dedicated async DMA stream & events for overlapping L1-Hit GPU compute with L2 -> L1 DMA
    cudaStream_t copy_stream_dev[2]          = {nullptr, nullptr};
    cudaEvent_t  router_event_dev[2]         = {nullptr, nullptr};
    cudaEvent_t  top_idx_event_dev[2]        = {nullptr, nullptr};
    cudaEvent_t  stage_ready_event_dev[2]    = {nullptr, nullptr};
    cudaEvent_t  logits_ready_event_dev[2]   = {nullptr, nullptr};
    cudaEvent_t  wave_ready_event_dev[2][8]  = {{nullptr}};
    cudaEvent_t  wave_done_event_dev[2][4]   = {{nullptr}};

    cudaStream_t copy_stream       = nullptr;
    cudaEvent_t  router_event      = nullptr;
    cudaEvent_t  top_idx_event     = nullptr;
    cudaEvent_t  stage_ready_event = nullptr;

    // Unpinned copy tasks (delegated to shared CpuMoEExpertPool)
    moecher::cpu_moe::CpuMemcpyTaskDesc copy_tasks[256];

    void init_async_and_pool() {
        for (int dev = 0; dev < num_gpus; ++dev) {
            if (num_gpus > 1) CUDA_CHECK(cudaSetDevice(dev));
            CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_dev[dev], cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&router_event_dev[dev], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&top_idx_event_dev[dev], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&stage_ready_event_dev[dev], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&logits_ready_event_dev[dev], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(stage_ready_event_dev[dev], copy_stream_dev[dev]));
            for (int w = 0; w < 8; ++w) {
                CUDA_CHECK(cudaEventCreateWithFlags(&wave_ready_event_dev[dev][w], cudaEventDisableTiming));
            }
            for (int b = 0; b < 4; ++b) {
                CUDA_CHECK(cudaEventCreateWithFlags(&wave_done_event_dev[dev][b], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(wave_done_event_dev[dev][b], copy_stream_dev[dev]));
            }
        }
        if (num_gpus > 1) CUDA_CHECK(cudaSetDevice(0));
        copy_stream       = copy_stream_dev[0];
        router_event      = router_event_dev[0];
        top_idx_event     = top_idx_event_dev[0];
        stage_ready_event = stage_ready_event_dev[0];
        (void)moecher::cpu_moe::CpuMoEExpertPool::instance();
    }

    int consume_argmax() {
        CUDA_CHECK(cudaEventSynchronize(stage_ready_event_dev[0]));
        if (num_gpus > 1 && d_output_dev[1] != nullptr) {
            CUDA_CHECK(cudaEventSynchronize(stage_ready_event_dev[1]));
            const int32_t i0 = h_argmax_out[0];
            const float   v0 = ((const float*)h_argmax_out)[1];
            const int32_t i1 = h_argmax_out[2];
            const float   v1 = ((const float*)h_argmax_out)[3];
            argmax_ready = false;
            return (v1 > v0) ? i1 : i0;
        }
        argmax_ready = false;
        return h_argmax_out[0];
    }

    void parallel_unpinned_gather(int count) {
        if (count <= 0) return;
        moecher::cpu_moe::CpuMoEExpertPool::instance().parallel_memcpy(copy_tasks, count);
    }

    std::vector<Qwen4LayerWeights> layers;
    std::vector<void*> device_allocs_dev[2];
    std::vector<void*> pinned_allocs;
    std::vector<void*> unpinned_allocs;

    void* alloc_vram(size_t nbytes, int dev = 0) {
        if (num_gpus > 1) CUDA_CHECK(cudaSetDevice(dev));
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, nbytes));
        device_allocs_dev[dev].push_back(p);
        return p;
    }

    void* upload_vram(const void* src, size_t nbytes, int dev = 0) {
        void* p = alloc_vram(nbytes, dev);
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
        for (int dev = 0; dev < num_gpus; ++dev) {
            if (num_gpus > 1) cudaSetDevice(dev);
            if (copy_stream_dev[dev]) cudaStreamSynchronize(copy_stream_dev[dev]);
        }
        if (num_gpus > 1) cudaSetDevice(0);
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
        for (int dev = 0; dev < num_gpus; ++dev) {
            if (num_gpus > 1) cudaSetDevice(dev);
            for (int w = 0; w < 8; ++w) {
                if (wave_ready_event_dev[dev][w]) cudaEventDestroy(wave_ready_event_dev[dev][w]);
            }
            for (int b = 0; b < 4; ++b) {
                if (wave_done_event_dev[dev][b])  cudaEventDestroy(wave_done_event_dev[dev][b]);
            }
            if (stage_ready_event_dev[dev])  cudaEventDestroy(stage_ready_event_dev[dev]);
            if (logits_ready_event_dev[dev]) cudaEventDestroy(logits_ready_event_dev[dev]);
            if (top_idx_event_dev[dev])      cudaEventDestroy(top_idx_event_dev[dev]);
            if (router_event_dev[dev])      cudaEventDestroy(router_event_dev[dev]);
            if (copy_stream_dev[dev])       cudaStreamDestroy(copy_stream_dev[dev]);
            for (void* p : device_allocs_dev[dev]) if (p) cudaFree(p);
            device_allocs_dev[dev].clear();
        }
        if (num_gpus > 1) cudaSetDevice(0);
        for (void* p : pinned_allocs)   if (p) cudaFreeHost(p);
        for (void* p : unpinned_allocs) if (p) std::free(p);
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
static size_t query_true_free_vram_bytes(int dev_idx = 0) {
    CUDA_CHECK(cudaSetDevice(dev_idx));
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
            if (fn_dev((unsigned int)dev_idx, &dev) == 0) {
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
    st.num_gpus = num_gpus_;
    st.pp_split_layer = (st.num_gpus > 1) ? (cfg_.num_hidden_layers / 2) : cfg_.num_hidden_layers;
    const int out_dev = (st.num_gpus > 1) ? 1 : 0;

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

    auto upload_dense = [&](const std::string& name, int32_t* out_type = nullptr, int dev = 0) -> void* {
        size_t nb = 0;
        const uint8_t* ptr = require_dense(name, out_type, &nb);
        return st.upload_vram(ptr, nb, dev);
    };

    // Keep token_embd in Host RAM (only 1 row needed per token -> saves ~360 MB VRAM for more L1 experts!)
    {
        size_t emb_nb = 0;
        const uint8_t* emb_ptr = require_dense("token_embd.weight", &st.token_embd_type, &emb_nb);
        st.token_embd_row_bytes = qwen4_ggml_row_bytes(st.token_embd_type, 2560);
        st.h_token_embd = (uint8_t*)std::malloc(emb_nb);
        st.unpinned_allocs.push_back(st.h_token_embd);
        std::memcpy(st.h_token_embd, emb_ptr, emb_nb);
        CUDA_CHECK(cudaMallocHost((void**)&st.h_token_embd_staging, 2 * Qwen4RuntimeState::QWEN4_MAX_PREFILL_M * st.token_embd_row_bytes));
        st.pinned_allocs.push_back(st.h_token_embd_staging);
        st.d_token_embd = st.alloc_vram(Qwen4RuntimeState::QWEN4_MAX_PREFILL_M * st.token_embd_row_bytes, 0);
    }
    st.d_out_hc_norm = upload_dense("output_hc_norm.weight", nullptr, out_dev);
    st.d_out_hc_down = upload_dense("output_hc_down.weight", nullptr, out_dev);
    st.d_out_hc_up   = upload_dense("output_hc_up.weight",   nullptr, out_dev);
    {
        size_t out_nb = 0;
        const uint8_t* out_ptr = require_dense("output.weight", &st.output_type, &out_nb);
        const size_t out_row_bytes = qwen4_ggml_row_bytes(st.output_type, 2560);
        if (st.num_gpus > 1 && out_nb == (size_t)cfg_.vocab_size * out_row_bytes) {
            st.vocab_split = (cfg_.vocab_size / 2) & ~31;
            const size_t nb0 = (size_t)st.vocab_split * out_row_bytes;
            const size_t nb1 = out_nb - nb0;
            st.d_output_dev[0] = st.upload_vram(out_ptr, nb0, 0);
            st.d_output_dev[1] = st.upload_vram(out_ptr + nb0, nb1, 1);
            st.d_output = st.d_output_dev[1];
        } else {
            st.vocab_split = cfg_.vocab_size;
            st.d_output_dev[0] = st.upload_vram(out_ptr, out_nb, out_dev);
            st.d_output = st.d_output_dev[0];
        }
    }
    st.init_async_and_pool();

    // 2. Upload Layer 1 PLE weights & bind Shard 2 131B N-gram table (~85 MB VRAM on GPU 0)
    if (dense_map.contains("blk.1.ple_key.weight") && manifest.contains("ple_ngram_table") && !manifest["ple_ngram_table"].is_null()) {
        const auto& ple_info = manifest["ple_ngram_table"];
        std::filesystem::path ple_file = base_dir / ple_info.value("file", "");
        if (std::filesystem::exists(ple_file) && st.shard2_mmap.open_read(ple_file.string())) {
            st.has_ple = true;
            st.d_ple_norm_q = upload_dense("blk.1.ple_norm_query.weight", nullptr, 0);
            st.d_ple_norm_k = upload_dense("blk.1.ple_norm_key.weight",   nullptr, 0);
            st.d_ple_norm_c = upload_dense("blk.1.ple_norm_conv.weight",  nullptr, 0);
            st.d_ple_key    = upload_dense("blk.1.ple_key.weight",   &st.ple_key_type, 0);
            st.d_ple_val    = upload_dense("blk.1.ple_value.weight", &st.ple_val_type, 0);
            st.d_ple_conv1d = upload_dense("blk.1.ple_conv1d.weight", nullptr, 0);
            st.d_ple_conv_state = (float*)st.alloc_vram(9 * 10240 * sizeof(float), 0);
            CUDA_CHECK(cudaMemset(st.d_ple_conv_state, 0, 9 * 10240 * sizeof(float)));

            const uint64_t ple_off = ple_info["offset"].get<uint64_t>();
            const uint64_t ple_nb  = ple_info["nbytes"].get<uint64_t>();
            st.mmap_ple_embed = (const uint8_t*)st.shard2_mmap.data() + ple_off;
            CUDA_CHECK(cudaMallocHost((void**)&st.h_ple_rows_iq4nl, 64 * 64 * 16 * 90));
            st.pinned_allocs.push_back(st.h_ple_rows_iq4nl);
            st.d_ple_rows_iq4nl = st.alloc_vram(64 * 16 * 90, 0);
            st.d_ple_emb_flat   = (float*)st.alloc_vram(64 * 2560 * sizeof(float), 0);
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
        const int l_dev = (st.num_gpus > 1 && l >= st.pp_split_layer) ? 1 : 0;
        lw.device_id = l_dev;
        if (st.num_gpus > 1) CUDA_CHECK(cudaSetDevice(l_dev));
        lw.is_qsa = (l % 4 == 3);
        const std::string pfx = "blk." + std::to_string(l) + ".";

        lw.d_hc_attn_norm   = upload_dense(pfx + "hc_attn_norm.weight",   nullptr, l_dev);
        lw.d_hc_attn_down   = upload_dense(pfx + "hc_attn_down.weight",   nullptr, l_dev);
        lw.d_hc_attn_up     = upload_dense(pfx + "hc_attn_up.weight",     nullptr, l_dev);
        lw.d_hc_attn_inject = upload_dense(pfx + "hc_attn_inject.weight", nullptr, l_dev);

        lw.d_hc_ffn_norm    = upload_dense(pfx + "hc_ffn_norm.weight",    nullptr, l_dev);
        lw.d_hc_ffn_down    = upload_dense(pfx + "hc_ffn_down.weight",    nullptr, l_dev);
        lw.d_hc_ffn_up      = upload_dense(pfx + "hc_ffn_up.weight",      nullptr, l_dev);
        lw.d_hc_ffn_inject  = upload_dense(pfx + "hc_ffn_inject.weight",  nullptr, l_dev);

        if (!lw.is_qsa) {
            lw.d_gdn_qkv      = upload_dense(pfx + "attn_qkv.weight",  &lw.gdn_qkv_type,  l_dev);
            lw.d_gdn_gate     = upload_dense(pfx + "attn_gate.weight", &lw.gdn_gate_type, l_dev);
            lw.d_gdn_alpha    = upload_dense(pfx + "ssm_alpha.weight",  nullptr, l_dev);
            lw.d_gdn_beta     = upload_dense(pfx + "ssm_beta.weight",   nullptr, l_dev);
            lw.d_gdn_conv1d   = upload_dense(pfx + "ssm_conv1d.weight", nullptr, l_dev);
            lw.d_gdn_ssm_a    = upload_dense(pfx + "ssm_a",             nullptr, l_dev);
            lw.d_gdn_dt_bias  = upload_dense(pfx + "ssm_dt.bias",       nullptr, l_dev);
            lw.d_gdn_ssm_norm = upload_dense(pfx + "ssm_norm.weight",   nullptr, l_dev);
            lw.d_gdn_out      = upload_dense(pfx + "ssm_out.weight",   &lw.gdn_out_type,  l_dev);

            lw.d_gdn_conv_state = (float*)st.alloc_vram(3 * 10240 * sizeof(float), l_dev);
            lw.d_gdn_ssm_state  = (float*)st.alloc_vram(48 * 128 * 128 * sizeof(float), l_dev);
            CUDA_CHECK(cudaMemset(lw.d_gdn_conv_state, 0, 3 * 10240 * sizeof(float)));
            CUDA_CHECK(cudaMemset(lw.d_gdn_ssm_state,  0, 48 * 128 * 128 * sizeof(float)));
        } else {
            lw.d_qsa_q          = upload_dense(pfx + "attn_q.weight",          &lw.qsa_q_type,   l_dev);
            lw.d_qsa_k          = upload_dense(pfx + "attn_k.weight",          &lw.qsa_k_type,   l_dev);
            lw.d_qsa_v          = upload_dense(pfx + "attn_v.weight",          &lw.qsa_v_type,   l_dev);
            lw.d_qsa_q_norm     = upload_dense(pfx + "attn_q_norm.weight",      nullptr,         l_dev);
            lw.d_qsa_k_norm     = upload_dense(pfx + "attn_k_norm.weight",      nullptr,         l_dev);
            lw.d_qsa_out        = upload_dense(pfx + "attn_output.weight",     &lw.qsa_out_type, l_dev);
            lw.d_qsa_idx_k      = upload_dense(pfx + "indexer.k_proj.weight",   nullptr,         l_dev);
            lw.d_qsa_idx_q      = upload_dense(pfx + "indexer.q_proj.weight",   nullptr,         l_dev);
            lw.d_qsa_idx_k_norm = upload_dense(pfx + "indexer.k_norm.weight",   nullptr,         l_dev);
            lw.d_qsa_idx_q_norm = upload_dense(pfx + "indexer.q_norm.weight",   nullptr,         l_dev);

            const size_t max_seq = (size_t)cfg_.max_seq_len + 64;
            const size_t max_blocks = (max_seq + 3) / 4 + 8;
            lw.d_qsa_k_cache    = (uint16_t*)st.alloc_vram(max_seq * 512 * sizeof(uint16_t), l_dev);
            lw.d_qsa_v_cache    = (uint16_t*)st.alloc_vram(max_seq * 512 * sizeof(uint16_t), l_dev);
            lw.d_qsa_idx_pooled = (float*)st.alloc_vram(max_blocks * 128 * sizeof(float), l_dev);
            lw.d_qsa_idx_tail   = (float*)st.alloc_vram(4 * 128 * sizeof(float), l_dev);
            lw.d_qsa_idx_dead   = (float*)st.alloc_vram(128 * sizeof(float), l_dev);
            CUDA_CHECK(cudaMemset(lw.d_qsa_k_cache,    0, max_seq * 512 * sizeof(uint16_t)));
            CUDA_CHECK(cudaMemset(lw.d_qsa_v_cache,    0, max_seq * 512 * sizeof(uint16_t)));
            CUDA_CHECK(cudaMemset(lw.d_qsa_idx_pooled, 0, max_blocks * 128 * sizeof(float)));
            CUDA_CHECK(cudaMemset(lw.d_qsa_idx_tail,   0, 4 * 128 * sizeof(float)));
            CUDA_CHECK(cudaMemset(lw.d_qsa_idx_dead,   0, 128 * sizeof(float)));
        }

        // Router + Shared Expert
        lw.d_moe_gate_inp       = upload_dense(pfx + "ffn_gate_inp.weight",       nullptr, l_dev);
        lw.d_moe_gate_inp_shexp = upload_dense(pfx + "ffn_gate_inp_shexp.weight", nullptr, l_dev);
        lw.d_moe_gate_shexp     = upload_dense(pfx + "ffn_gate_shexp.weight", &lw.moe_gate_shexp_type, l_dev);
        lw.d_moe_up_shexp       = upload_dense(pfx + "ffn_up_shexp.weight",   &lw.moe_up_shexp_type,   l_dev);
        lw.d_moe_down_shexp     = upload_dense(pfx + "ffn_down_shexp.weight", &lw.moe_down_shexp_type, l_dev);

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
    if (st.num_gpus > 1) CUDA_CHECK(cudaSetDevice(0));

    // Close resident.bin MMap now that all resident weights are in VRAM!
    resident_mmap.close();
#if defined(_WIN32)
    SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
#endif

    // 4. Allocate working activation & bounce buffers on all active GPUs
    constexpr int SUB_M = Qwen4RuntimeState::QWEN4_MAX_SUB_M;
    const int dev_max_m = (st.num_gpus > 1) ? SUB_M : Qwen4RuntimeState::QWEN4_MAX_PREFILL_M;
    CUDA_CHECK(cudaHostAlloc((void**)&st.h_moe_pkt, 256 * 1024, cudaHostAllocMapped | cudaHostAllocPortable));
    st.pinned_allocs.push_back(st.h_moe_pkt);
    st.h_x_q8_1  = (void*)(st.h_moe_pkt + 0);
    st.h_top_idx = (int32_t*)(st.h_moe_pkt + 2880);
    st.h_top_w   = (float*)(st.h_moe_pkt + 2920);
    CUDA_CHECK(cudaHostAlloc((void**)&st.h_cpu_moe_out, 2564 * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable));
    st.pinned_allocs.push_back(st.h_cpu_moe_out);
    st.h_argmax_out = (int32_t*)(st.h_cpu_moe_out + 2560);
    CUDA_CHECK(cudaMallocHost((void**)&st.h_bounce_exp, 72 * st.max_expert_blob_bytes));
    st.pinned_allocs.push_back(st.h_bounce_exp);
    CUDA_CHECK(cudaHostAlloc((void**)&st.h_active_ptrs, 48 * sizeof(Qwen4ActiveExpertPtrs), cudaHostAllocMapped | cudaHostAllocPortable));
    st.pinned_allocs.push_back(st.h_active_ptrs);
    CUDA_CHECK(cudaHostAlloc((void**)&st.h_prefill_pack, 8 * sizeof(Qwen4RuntimeState::Qwen4PrefillTaskPack), cudaHostAllocMapped | cudaHostAllocPortable));
    st.pinned_allocs.push_back(st.h_prefill_pack);
    CUDA_CHECK(cudaMallocHost((void**)&st.h_vram_exp_tables_all, (size_t)n_layers * 256 * sizeof(uint8_t*)));
    st.pinned_allocs.push_back(st.h_vram_exp_tables_all);
    std::memset(st.h_vram_exp_tables_all, 0, (size_t)n_layers * 256 * sizeof(uint8_t*));
    CUDA_CHECK(cudaMallocHost((void**)&st.h_decode_exp_tables_all, (size_t)n_layers * 256 * sizeof(uint8_t*)));
    st.pinned_allocs.push_back(st.h_decode_exp_tables_all);
    std::memset(st.h_decode_exp_tables_all, 0, (size_t)n_layers * 256 * sizeof(uint8_t*));
    for (int l = 0; l < n_layers; ++l) {
        st.layers[l].h_vram_exp_table   = st.h_vram_exp_tables_all + (size_t)l * 256;
        st.layers[l].d_vram_exp_table   = (uint8_t**)st.alloc_vram(256 * sizeof(uint8_t*), st.layers[l].device_id);
        st.layers[l].h_decode_exp_table = st.h_decode_exp_tables_all + (size_t)l * 256;
        st.layers[l].d_decode_exp_table = (uint8_t**)st.alloc_vram(256 * sizeof(uint8_t*), st.layers[l].device_id);
    }

    constexpr size_t bf16_ws_offset_floats = 20447232ULL;
    const size_t min_scratch_floats = (st.num_gpus == 1) ? (48ULL * 1024 * 1024) : (39ULL * 1024 * 1024);
    const size_t scratch_floats = std::max<size_t>(min_scratch_floats, 32768 + ((size_t)cfg_.max_seq_len + 3) / 4 + 256);
    for (int dev = 0; dev < st.num_gpus; ++dev) {
        if (st.num_gpus > 1) set_active_pp_device(dev);
        st.d_R_dev[dev]           = (float*)st.alloc_vram(((size_t)dev_max_m * 10240 + (size_t)dev_max_m * 4 + 256) * sizeof(float), dev);
        st.d_x_dev[dev]           = (float*)st.alloc_vram((size_t)dev_max_m * 2560 * sizeof(float), dev);
        st.d_moe_pkt_dev[dev]     = (uint8_t*)st.alloc_vram((st.num_gpus > 1 ? 2ULL : 8ULL) * 1024 * 1024, dev);
        st.d_x_q8_1_dev[dev]      = (void*)(st.d_moe_pkt_dev[dev] + 0);
        st.d_top_idx_dev[dev]     = (int32_t*)(st.d_moe_pkt_dev[dev] + 2880);
        st.d_top_w_dev[dev]       = (float*)(st.d_moe_pkt_dev[dev] + 2920);
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&st.d_cpu_moe_out_dev[dev], st.h_cpu_moe_out, 0));
        st.d_argmax_out_dev[dev]  = (int32_t*)(st.d_cpu_moe_out_dev[dev] + 2560 + dev * 2);
        uint8_t* d_host_pkt_dev   = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_host_pkt_dev, st.h_moe_pkt, 0));
        st.d_host_top_idx_dev[dev] = (int32_t*)(d_host_pkt_dev + 2880);
        st.d_host_top_w_dev[dev]   = (float*)(d_host_pkt_dev + 2920);
        st.d_sub_out_dev[dev]     = (float*)st.alloc_vram((size_t)dev_max_m * 2560 * sizeof(float), dev);
        st.d_scratch_dev[dev]     = (float*)st.alloc_vram(scratch_floats * sizeof(float), dev);
        st.d_q8_scratch_dev[dev]  = st.alloc_vram((size_t)SUB_M * qwen4_q8_1_bytes(10240), dev);
        st.d_sel_indices_dev[dev] = (int32_t*)st.alloc_vram(2056 * sizeof(int32_t), dev);
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&st.d_active_ptrs_dev[dev], st.h_active_ptrs, 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&st.d_prefill_pack_dev[dev], st.h_prefill_pack, 0));
        qwen4_bind_device_workspace_cuda(
            dev,
            (void*)cublas_handle_,
            st.d_scratch_dev[dev] + bf16_ws_offset_floats,
            (scratch_floats - bf16_ws_offset_floats) * sizeof(float));
    }
    if (st.num_gpus > 1) {
        st.d_logits_dev1 = (float*)st.alloc_vram((size_t)cfg_.vocab_size * sizeof(float), 1);
        set_active_pp_device(0);
    }
    st.d_R           = st.d_R_dev[0];
    st.d_x           = st.d_x_dev[0];
    st.d_moe_pkt     = st.d_moe_pkt_dev[0];
    st.d_x_q8_1      = st.d_x_q8_1_dev[0];
    st.d_top_idx     = st.d_top_idx_dev[0];
    st.d_top_w       = st.d_top_w_dev[0];
    st.d_cpu_moe_out = st.d_cpu_moe_out_dev[0];
    st.d_sub_out     = st.d_sub_out_dev[0];
    st.d_scratch     = st.d_scratch_dev[0];
    st.d_q8_scratch  = st.d_q8_scratch_dev[0];
    st.d_sel_indices = st.d_sel_indices_dev[0];
    st.d_active_ptrs = st.d_active_ptrs_dev[0];

    // Standard engine control & sampling buffers (on GPU 0)
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
                (void)std::fread(exp_freq.data(), sizeof(uint32_t), total_experts, fp_f);
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

    // 6. Plan Per-Expert 3-Tier Cache Placement across all 12,288 experts (per-GPU when num_gpus == 2):
    //    - When 2x 24 GB GPUs are active, 100% of all 6,144 experts per stage (11.71 GiB/GPU) fit in L1 Static VRAM!
    //    - Otherwise:
    //      * Tier 1b: Dynamic LRU L1 VRAM (18 LRU slots per layer, pre-warmed!)
    //      * Tier 1a: Static Hot L1 VRAM (per-layer floor + global frequency top-K)
    //      * Tier 2 : L2 Pinned Host RAM
    //      * Tier 2.5: L2.5 Unpinned Host RAM
    //      * Tier 3 : L3 NVMe SSD fallback (experts.bin)
    const size_t vram_safety_reserve = 256ULL * 1024 * 1024; // 256 MiB safety reserve for Windows WDDM + cuBLAS workspace
    size_t vram_budget_dev[2] = {0, 0};
    for (int dev = 0; dev < st.num_gpus; ++dev) {
        size_t free_vram = query_true_free_vram_bytes(dev);
        size_t budget = (free_vram > vram_safety_reserve) ? (free_vram - vram_safety_reserve) : 0;
        if (max_vram_gb > 0.0f) {
            size_t user_max = (size_t)(max_vram_gb * 1024.0 * 1024.0 * 1024.0);
            size_t used_vram = (gpu_caps_.total_vram_bytes > free_vram) ? (gpu_caps_.total_vram_bytes - free_vram) : 0;
            size_t user_rem = (user_max > used_vram) ? (user_max - used_vram) : 0;
            budget = std::min(budget, user_rem);
        }
        vram_budget_dev[dev] = budget;
        LOG_INFO("[Qwen4Exp] GPU %d free VRAM for routed experts: %.2f GiB (after %zu MiB safety reserve)",
                 dev, (double)budget / (1024.0 * 1024.0 * 1024.0), vram_safety_reserve / (1024 * 1024));
    }
    if (st.num_gpus > 1) CUDA_CHECK(cudaSetDevice(0));

    const size_t max_pinned_cap   = 16384ULL * 1024 * 1024; // 16.00 GiB Pinned RAM (with automatic fallback to Tier 2.5 Unpinned RAM)
    const size_t max_unpinned_cap = 10240ULL * 1024 * 1024; // 10.00 GiB Unpinned RAM
    size_t pinned_budget   = (dram_cache_gb < 0.0f) ? 0 : max_pinned_cap;
    size_t unpinned_budget = (dram_cache_gb < 0.0f) ? 0 : max_unpinned_cap;
    if (dram_cache_gb > 0.0f) {
        size_t total_ram_budget = (size_t)(dram_cache_gb * 1024.0 * 1024.0 * 1024.0);
        pinned_budget   = std::min(total_ram_budget, max_pinned_cap);
        unpinned_budget = (total_ram_budget > pinned_budget) ? (total_ram_budget - pinned_budget) : 0;
    }

    enum ExpTier : uint8_t { TIER_NONE = 0, TIER_L1_STATIC = 1, TIER_L2_PINNED = 2, TIER_L25_UNPINNED = 3, TIER_L3_SSD = 4 };
    std::vector<uint8_t> exp_tier(total_experts, TIER_NONE);
    std::vector<int> layer_static_cnt(n_layers, 0);
    std::vector<int> layer_lru_cnt(n_layers, 0);

    constexpr int TARGET_LRU_SLOTS = 48;
    constexpr int STATIC_FLOOR_PER_LAYER = 56;

    // Global calibrated ranking across all 12,288 experts (expert_freq.bin stores 12288 - global_rank)
    std::vector<int> global_order(total_experts);
    std::iota(global_order.begin(), global_order.end(), 0);
    std::sort(global_order.begin(), global_order.end(), [&](int a, int b) {
        return exp_freq[a] > exp_freq[b];
    });

    for (int dev = 0; dev < st.num_gpus; ++dev) {
        const int l_begin = (dev == 0) ? 0 : st.pp_split_layer;
        const int l_end   = (dev == 0) ? st.pp_split_layer : n_layers;

        size_t stage_total_exp_bytes = 0;
        for (int l = l_begin; l < l_end; ++l) {
            stage_total_exp_bytes += (size_t)n_experts * st.layers[l].expert_blob_bytes;
        }

        if (vram_budget_dev[dev] >= stage_total_exp_bytes) {
            // 100% of all routed experts in this pipeline stage fit permanently in L1 Static VRAM!
            for (int l = l_begin; l < l_end; ++l) {
                layer_lru_cnt[l]    = 0;
                layer_static_cnt[l] = n_experts;
                for (int e = 0; e < n_experts; ++e) {
                    exp_tier[l * n_experts + e] = TIER_L1_STATIC;
                }
                vram_budget_dev[dev] -= (size_t)n_experts * st.layers[l].expert_blob_bytes;
            }
            LOG_INFO("[Qwen4Exp] GPU %d: 100%% of stage experts (Layers %d..%d, %d experts, %.2f GiB) fit in L1 Static VRAM (zero-sync GPU fast path enabled)!",
                     dev, l_begin, l_end - 1, (l_end - l_begin) * n_experts,
                     (double)stage_total_exp_bytes / (1024.0 * 1024.0 * 1024.0));
        } else {
            // Allocate 36 Dynamic LRU L1 VRAM slots per layer first (so prompt/topic experts are cached in 936 GB/s VRAM!)
            for (int l = l_begin; l < l_end; ++l) {
                const size_t need = (size_t)TARGET_LRU_SLOTS * st.layers[l].expert_blob_bytes;
                if (vram_budget_dev[dev] >= need) {
                    layer_lru_cnt[l] = TARGET_LRU_SLOTS;
                    vram_budget_dev[dev] -= need;
                }
            }
            // Pass 1: Guarantee minimum floor of 36 hottest experts per layer
            for (int l = l_begin; l < l_end; ++l) {
                std::vector<int> e_order(n_experts);
                std::iota(e_order.begin(), e_order.end(), 0);
                std::sort(e_order.begin(), e_order.end(), [&](int a, int b) {
                    return exp_freq[l * n_experts + a] > exp_freq[l * n_experts + b];
                });
                const size_t eb = st.layers[l].expert_blob_bytes;
                for (int i = 0; i < STATIC_FLOOR_PER_LAYER && vram_budget_dev[dev] >= eb; ++i) {
                    const int e = e_order[i];
                    exp_tier[l * n_experts + e] = TIER_L1_STATIC;
                    layer_static_cnt[l]++;
                    vram_budget_dev[dev] -= eb;
                }
            }
            // Pass 2: Fill remaining VRAM budget on this GPU in global frequency order
            for (int flat_idx : global_order) {
                if (exp_tier[flat_idx] != TIER_NONE) continue;
                const int l = flat_idx / n_experts;
                if (l < l_begin || l >= l_end) continue;
                const size_t eb = st.layers[l].expert_blob_bytes;
                if (vram_budget_dev[dev] >= eb && layer_static_cnt[l] < (n_experts - layer_lru_cnt[l])) {
                    exp_tier[flat_idx] = TIER_L1_STATIC;
                    layer_static_cnt[l]++;
                    vram_budget_dev[dev] -= eb;
                }
            }
        }
    }

    // Allocate 1 single contiguous L1 VRAM block per layer on its target GPU (Static Hot + Dynamic LRU combined -> 0 VRAM fragmentation!)
    size_t total_static_vram_bytes = 0, total_lru_vram_bytes = 0;
    int total_static_experts = 0, total_lru_slots = 0;
    for (int l = 0; l < n_layers; ++l) {
        auto& lw = st.layers[l];
        if (st.num_gpus > 1) CUDA_CHECK(cudaSetDevice(lw.device_id));
        const int s_cnt = layer_static_cnt[l];
        const int l_cnt = layer_lru_cnt[l];
        if (s_cnt + l_cnt <= 0) continue;
        const size_t bytes = (size_t)(s_cnt + l_cnt) * lw.expert_blob_bytes;
        void* d_block = nullptr;
        if (cudaMalloc(&d_block, bytes) == cudaSuccess) {
            st.device_allocs_dev[lw.device_id].push_back(d_block);
            int slot = 0;
            for (int e = 0; e < n_experts; ++e) {
                if (exp_tier[l * n_experts + e] == TIER_L1_STATIC) {
                    lw.d_vram_exp_ptr[e] = (uint8_t*)d_block + (size_t)slot * lw.expert_blob_bytes;
                    lw.is_static_vram[e] = true;
                    slot++;
                }
            }
            if (s_cnt == n_experts) {
                lw.d_all_vram_base = (uint8_t*)d_block;
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
    if (st.num_gpus > 1) CUDA_CHECK(cudaSetDevice(0));

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
            if (cudaHostAlloc(&h_pin, p_bytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess) {
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
    //    and pre-warm all Dynamic LRU L1 VRAM slots with the hottest L2 experts per layer!
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
    int total_zero_sync_layers = 0;
    for (int l = 0; l < n_layers; ++l) {
        auto& lw = st.layers[l];
        if (st.num_gpus > 1) CUDA_CHECK(cudaSetDevice(lw.device_id));
        const size_t eb = lw.expert_blob_bytes;
#if defined(_WIN32)
        _fseeki64(st.experts_fp, (int64_t)lw.layer_file_offset, SEEK_SET);
#else
        fseeko(st.experts_fp, (off_t)lw.layer_file_offset, SEEK_SET);
#endif

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
#if defined(_WIN32)
                _fseeki64(st.experts_fp, (int64_t)eb, SEEK_CUR);
#else
                fseeko(st.experts_fp, (off_t)eb, SEEK_CUR);
#endif
            }
        }

        // Pre-warm layer l's Dynamic LRU L1 VRAM slots with the hottest non-static experts in layer l!
        {
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
            lw.n_non_static = (int)non_static.size();
            for (int i = 0; i < lw.n_non_static; ++i) {
                lw.non_static_order[i] = (uint8_t)non_static[i];
            }
            if (lw.n_lru_slots > 0 && lw.d_lru_base != nullptr) {
                const int n_warm = std::min((int)non_static.size(), lw.n_lru_slots);
                // Seed hotter experts with higher initial lru_clock so colder pre-warmed slots are evicted first
                for (int s = 0; s < n_warm; ++s) {
                    const int e = non_static[n_warm - 1 - s]; // s=0 gets coldest of top-36, s=n_warm-1 gets hottest of top-36
                    const uint8_t* src_host = lw.h_pinned_exp_ptr[e] ? lw.h_pinned_exp_ptr[e] : lw.h_unpinned_exp_ptr[e];
                    uint8_t* dst_vram = lw.d_lru_base + (size_t)s * eb;
                    CUDA_CHECK(cudaMemcpy(dst_vram, src_host, eb, cudaMemcpyHostToDevice));
                    lw.d_vram_exp_ptr[e]     = dst_vram;
                    lw.exp_lru_slot[e]       = (int8_t)s;
                    lw.lru_slot_eid[s]       = e;
                    lw.lru_slot_last_used[s] = (uint64_t)(s + 1) * 16;
                    lw.lru_slot_epoch[s]     = 0;
                    total_prewarmed_lru++;
                }
            }
        }
        // Build both d_vram_exp_table (L1 VRAM only) and d_decode_exp_table (L1 VRAM + Mapped L2 Pinned RAM for zero-sync decode!)
        bool all_decode_mapped = true;
        for (int e = 0; e < n_experts; ++e) {
            if (lw.h_pinned_exp_ptr[e] != nullptr) {
                void* d_mapped = nullptr;
                if (cudaHostGetDevicePointer(&d_mapped, lw.h_pinned_exp_ptr[e], 0) == cudaSuccess && d_mapped != nullptr) {
                    lw.d_mapped_exp_ptr[e] = (uint8_t*)d_mapped;
                } else {
                    cudaGetLastError();
                }
            }
            lw.h_vram_exp_table[e]   = lw.d_vram_exp_ptr[e];
            lw.h_decode_exp_table[e] = lw.d_vram_exp_ptr[e] ? lw.d_vram_exp_ptr[e] : lw.d_mapped_exp_ptr[e];
            if (lw.h_decode_exp_table[e] == nullptr) {
                all_decode_mapped = false;
            }
        }
        lw.all_decode_gpu_accessible = all_decode_mapped;
        if (all_decode_mapped) total_zero_sync_layers++;
        CUDA_CHECK(cudaMemcpy(lw.d_vram_exp_table, lw.h_vram_exp_table, n_experts * sizeof(uint8_t*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lw.d_decode_exp_table, lw.h_decode_exp_table, n_experts * sizeof(uint8_t*), cudaMemcpyHostToDevice));
    }
    if (st.num_gpus > 1) CUDA_CHECK(cudaSetDevice(0));
    st.lru_clock = (uint64_t)TARGET_LRU_SLOTS * 16;
    if (total_zero_sync_layers > 0) {
        LOG_INFO("[Qwen4Exp] Zero-Sync GPU Fast Path enabled on %d / %d layers (L1 Static VRAM + Mapped L2 Pinned RAM)!",
                 total_zero_sync_layers, n_layers);
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

    st.has_unpinned_experts = (total_unpinned_experts > 0);
    st.profile_timing = (std::getenv("MOECHER_PROFILE") != nullptr);
    enable_batched_prefill_ = (batched_prefill_mode_ != "off");
    st.loaded = true;
    return true;
}

void reset_state_qwen4_exp() {
    if (!qwen4_state_ || !qwen4_state_->loaded) return;
    auto& st = *qwen4_state_;
    for (int dev = 0; dev < st.num_gpus; ++dev) {
        if (st.num_gpus > 1) set_active_pp_device(dev);
        if (st.copy_stream_dev[dev]) {
            CUDA_CHECK(cudaStreamSynchronize(st.copy_stream_dev[dev]));
        }
    }
    if (st.num_gpus > 1) set_active_pp_device(0);
    st.token_history.clear();
    st.has_pending_decode_top_idx = false;
    if (st.has_ple) {
        CUDA_CHECK(cudaMemsetAsync(st.d_ple_conv_state, 0, 9 * 10240 * sizeof(float), main_stream_));
    }
    for (auto& lw : st.layers) {
        if (st.num_gpus > 1 && active_pp_dev_ != lw.device_id) {
            set_active_pp_device(lw.device_id);
        }
        if (!lw.is_qsa) {
            CUDA_CHECK(cudaMemsetAsync(lw.d_gdn_conv_state, 0, 3 * 10240 * sizeof(float), main_stream_));
            CUDA_CHECK(cudaMemsetAsync(lw.d_gdn_ssm_state,  0, 48 * 128 * 128 * sizeof(float), main_stream_));
        } else {
            CUDA_CHECK(cudaMemsetAsync(lw.d_qsa_idx_tail, 0, 4 * 128 * sizeof(float), main_stream_));
            CUDA_CHECK(cudaMemsetAsync(lw.d_qsa_idx_dead, 0, 128 * sizeof(float), main_stream_));
        }
    }
    for (int dev = 0; dev < st.num_gpus; ++dev) {
        if (st.num_gpus > 1) set_active_pp_device(dev);
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
    }
    if (st.num_gpus > 1) set_active_pp_device(0);
}

void snapshot_kv_qwen4_exp(KVSnapshot& snap, const std::vector<int>& tokens) {
    if (!qwen4_state_ || !qwen4_state_->loaded || tokens.empty()) return;
    auto& st = *qwen4_state_;
    snap.tokens = tokens;
    snap.layers.resize(cfg_.num_hidden_layers);
    const bool use_host_snap = (st.num_gpus == 1);

    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
        auto& lw = st.layers[l];
        if (st.num_gpus > 1) set_active_pp_device(lw.device_id);
        auto& l_snap = snap.layers[l];
        if (!lw.is_qsa) {
            const size_t conv_bytes = 3 * 10240 * sizeof(float);
            const size_t ssm_bytes  = 48 * 128 * 128 * sizeof(float);
            if (use_host_snap) {
                l_snap.host_conv_state.resize(conv_bytes);
                l_snap.host_ssm_state.resize(ssm_bytes);
                CUDA_CHECK(cudaMemcpyAsync(l_snap.host_conv_state.data(), lw.d_gdn_conv_state, conv_bytes, cudaMemcpyDeviceToHost, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(l_snap.host_ssm_state.data(),  lw.d_gdn_ssm_state,  ssm_bytes,  cudaMemcpyDeviceToHost, main_stream_));
            } else {
                if (!l_snap.snap_conv_state.data) l_snap.snap_conv_state.alloc(conv_bytes);
                if (!l_snap.snap_ssm_state.data)  l_snap.snap_ssm_state.alloc(ssm_bytes);
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_conv_state.data, lw.d_gdn_conv_state, conv_bytes, cudaMemcpyDeviceToDevice, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_ssm_state.data,  lw.d_gdn_ssm_state,  ssm_bytes,  cudaMemcpyDeviceToDevice, main_stream_));
            }
        } else {
            const size_t max_seq      = (size_t)cfg_.max_seq_len;
            const size_t max_blocks   = (max_seq + 3) / 4;
            const size_t clamped_toks = std::clamp<size_t>(tokens.size(), 1, max_seq);
            const size_t clamped_blks = std::min<size_t>((clamped_toks + 3) / 4 + 2, max_blocks);
            const size_t kv_bytes     = clamped_toks * 512 * sizeof(uint16_t);
            const size_t pool_bytes   = clamped_blks * 128 * sizeof(float);
            const size_t tail_bytes   = 4 * 128 * sizeof(float);
            const size_t dead_bytes   = 128 * sizeof(float);
            if (use_host_snap) {
                l_snap.host_k_cache.resize(kv_bytes);
                l_snap.host_v_cache.resize(kv_bytes);
                l_snap.host_ssm_state.resize(pool_bytes);
                l_snap.host_conv_state.resize(tail_bytes);
                l_snap.host_comp_score_state.resize(dead_bytes);
                CUDA_CHECK(cudaMemcpyAsync(l_snap.host_k_cache.data(),          lw.d_qsa_k_cache,    kv_bytes,   cudaMemcpyDeviceToHost, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(l_snap.host_v_cache.data(),          lw.d_qsa_v_cache,    kv_bytes,   cudaMemcpyDeviceToHost, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(l_snap.host_ssm_state.data(),        lw.d_qsa_idx_pooled, pool_bytes, cudaMemcpyDeviceToHost, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(l_snap.host_conv_state.data(),       lw.d_qsa_idx_tail,   tail_bytes, cudaMemcpyDeviceToHost, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(l_snap.host_comp_score_state.data(), lw.d_qsa_idx_dead,   dead_bytes, cudaMemcpyDeviceToHost, main_stream_));
            } else {
                if (!l_snap.snap_k_cache_gqa.data || l_snap.snap_k_cache_gqa.size_bytes < kv_bytes) l_snap.snap_k_cache_gqa.alloc(kv_bytes);
                if (!l_snap.snap_v_cache_gqa.data || l_snap.snap_v_cache_gqa.size_bytes < kv_bytes) l_snap.snap_v_cache_gqa.alloc(kv_bytes);
                if (!l_snap.snap_ssm_state.data   || l_snap.snap_ssm_state.size_bytes   < pool_bytes) l_snap.snap_ssm_state.alloc(pool_bytes);
                if (!l_snap.snap_conv_state.data) l_snap.snap_conv_state.alloc(tail_bytes);
                if (!l_snap.snap_comp_score_state.data) l_snap.snap_comp_score_state.alloc(dead_bytes);
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_k_cache_gqa.data,      lw.d_qsa_k_cache,    kv_bytes,   cudaMemcpyDeviceToDevice, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_v_cache_gqa.data,      lw.d_qsa_v_cache,    kv_bytes,   cudaMemcpyDeviceToDevice, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_ssm_state.data,        lw.d_qsa_idx_pooled, pool_bytes, cudaMemcpyDeviceToDevice, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_conv_state.data,       lw.d_qsa_idx_tail,   tail_bytes, cudaMemcpyDeviceToDevice, main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(l_snap.snap_comp_score_state.data, lw.d_qsa_idx_dead,   dead_bytes, cudaMemcpyDeviceToDevice, main_stream_));
            }
        }
    }

    const int out_dev = (st.num_gpus > 1) ? 1 : 0;
    if (st.num_gpus > 1) set_active_pp_device(out_dev);
    const size_t r_bytes = 10240 * sizeof(float);
    if (use_host_snap) {
        snap.host_hc_state.resize(r_bytes);
        CUDA_CHECK(cudaMemcpyAsync(snap.host_hc_state.data(), st.d_R_dev[out_dev], r_bytes, cudaMemcpyDeviceToHost, main_stream_));
    } else {
        if (!snap.snap_hc_state.data) snap.snap_hc_state.alloc(r_bytes);
        CUDA_CHECK(cudaMemcpyAsync(snap.snap_hc_state.data, st.d_R_dev[out_dev], r_bytes, cudaMemcpyDeviceToDevice, main_stream_));
    }
    if (st.num_gpus > 1) {
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        set_active_pp_device(0);
    }
    if (st.has_ple && st.d_ple_conv_state) {
        const size_t ple_bytes = 9 * 10240 * sizeof(float);
        if (use_host_snap) {
            snap.host_hc_after_attn.resize(ple_bytes);
            CUDA_CHECK(cudaMemcpyAsync(snap.host_hc_after_attn.data(), st.d_ple_conv_state, ple_bytes, cudaMemcpyDeviceToHost, main_stream_));
        } else {
            if (!snap.snap_hc_after_attn.data) snap.snap_hc_after_attn.alloc(ple_bytes);
            CUDA_CHECK(cudaMemcpyAsync(snap.snap_hc_after_attn.data, st.d_ple_conv_state, ple_bytes, cudaMemcpyDeviceToDevice, main_stream_));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(main_stream_));
    snap.valid = true;

    size_t total_snap_bytes = snap.snap_hc_state.size_bytes + snap.snap_hc_after_attn.size_bytes +
                              snap.host_hc_state.size() + snap.host_hc_after_attn.size();
    for (const auto& ls : snap.layers) {
        total_snap_bytes += ls.snap_k_cache_gqa.size_bytes + ls.snap_v_cache_gqa.size_bytes +
                            ls.snap_ssm_state.size_bytes + ls.snap_conv_state.size_bytes + ls.snap_comp_score_state.size_bytes +
                            ls.host_k_cache.size() + ls.host_v_cache.size() +
                            ls.host_ssm_state.size() + ls.host_conv_state.size() + ls.host_comp_score_state.size();
    }
    if (&snap == &system_kv_snapshot_) {
        system_snapshot_tokens_.store(tokens.size(), std::memory_order_relaxed);
        system_snapshot_bytes_.store(total_snap_bytes, std::memory_order_relaxed);
    } else if (&snap == &turn_kv_snapshot_) {
        turn_snapshot_tokens_.store(tokens.size(), std::memory_order_relaxed);
        turn_snapshot_bytes_.store(total_snap_bytes, std::memory_order_relaxed);
    }
}

void restore_kv_qwen4_exp(const KVSnapshot& snap) {
    if (!snap.valid || !qwen4_state_ || !qwen4_state_->loaded) return;
    auto& st = *qwen4_state_;
    const bool use_host_snap = (st.num_gpus == 1);

    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
        auto& lw = st.layers[l];
        if (st.num_gpus > 1) set_active_pp_device(lw.device_id);
        const auto& l_snap = snap.layers[l];
        if (!lw.is_qsa) {
            if (use_host_snap) {
                if (!l_snap.host_conv_state.empty()) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_gdn_conv_state, l_snap.host_conv_state.data(), l_snap.host_conv_state.size(), cudaMemcpyHostToDevice, main_stream_));
                }
                if (!l_snap.host_ssm_state.empty()) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_gdn_ssm_state, l_snap.host_ssm_state.data(), l_snap.host_ssm_state.size(), cudaMemcpyHostToDevice, main_stream_));
                }
            } else {
                if (l_snap.snap_conv_state.data) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_gdn_conv_state, l_snap.snap_conv_state.data, l_snap.snap_conv_state.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
                }
                if (l_snap.snap_ssm_state.data) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_gdn_ssm_state, l_snap.snap_ssm_state.data, l_snap.snap_ssm_state.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
                }
            }
        } else {
            const size_t max_seq      = (size_t)cfg_.max_seq_len;
            const size_t max_blocks   = (max_seq + 3) / 4;
            const size_t clamped_toks = std::clamp<size_t>(snap.tokens.size(), 1, max_seq);
            const size_t clamped_blks = std::min<size_t>((clamped_toks + 3) / 4 + 2, max_blocks);
            if (use_host_snap) {
                const size_t kv_bytes   = std::min(l_snap.host_k_cache.size(), clamped_toks * 512 * sizeof(uint16_t));
                const size_t pool_bytes = std::min(l_snap.host_ssm_state.size(), clamped_blks * 128 * sizeof(float));
                if (!l_snap.host_k_cache.empty() && kv_bytes > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_qsa_k_cache, l_snap.host_k_cache.data(), kv_bytes, cudaMemcpyHostToDevice, main_stream_));
                }
                if (!l_snap.host_v_cache.empty() && kv_bytes > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_qsa_v_cache, l_snap.host_v_cache.data(), kv_bytes, cudaMemcpyHostToDevice, main_stream_));
                }
                if (!l_snap.host_ssm_state.empty() && pool_bytes > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_qsa_idx_pooled, l_snap.host_ssm_state.data(), pool_bytes, cudaMemcpyHostToDevice, main_stream_));
                }
                if (!l_snap.host_conv_state.empty()) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_qsa_idx_tail, l_snap.host_conv_state.data(), l_snap.host_conv_state.size(), cudaMemcpyHostToDevice, main_stream_));
                }
                if (!l_snap.host_comp_score_state.empty()) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_qsa_idx_dead, l_snap.host_comp_score_state.data(), l_snap.host_comp_score_state.size(), cudaMemcpyHostToDevice, main_stream_));
                }
            } else {
                const size_t kv_bytes   = std::min(l_snap.snap_k_cache_gqa.size_bytes, clamped_toks * 512 * sizeof(uint16_t));
                const size_t pool_bytes = std::min(l_snap.snap_ssm_state.size_bytes, clamped_blks * 128 * sizeof(float));
                if (l_snap.snap_k_cache_gqa.data && kv_bytes > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_qsa_k_cache, l_snap.snap_k_cache_gqa.data, kv_bytes, cudaMemcpyDeviceToDevice, main_stream_));
                }
                if (l_snap.snap_v_cache_gqa.data && kv_bytes > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_qsa_v_cache, l_snap.snap_v_cache_gqa.data, kv_bytes, cudaMemcpyDeviceToDevice, main_stream_));
                }
                if (l_snap.snap_ssm_state.data && pool_bytes > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_qsa_idx_pooled, l_snap.snap_ssm_state.data, pool_bytes, cudaMemcpyDeviceToDevice, main_stream_));
                }
                if (l_snap.snap_conv_state.data) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_qsa_idx_tail, l_snap.snap_conv_state.data, l_snap.snap_conv_state.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
                }
                if (l_snap.snap_comp_score_state.data) {
                    CUDA_CHECK(cudaMemcpyAsync(lw.d_qsa_idx_dead, l_snap.snap_comp_score_state.data, l_snap.snap_comp_score_state.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
                }
            }
        }
    }

    const int out_dev = (st.num_gpus > 1) ? 1 : 0;
    if (st.num_gpus > 1) set_active_pp_device(out_dev);
    if (use_host_snap) {
        if (!snap.host_hc_state.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(st.d_R_dev[out_dev], snap.host_hc_state.data(), snap.host_hc_state.size(), cudaMemcpyHostToDevice, main_stream_));
        }
    } else if (snap.snap_hc_state.data) {
        CUDA_CHECK(cudaMemcpyAsync(st.d_R_dev[out_dev], snap.snap_hc_state.data, snap.snap_hc_state.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
    }
    if (st.num_gpus > 1) {
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        set_active_pp_device(0);
    }
    if (st.has_ple && st.d_ple_conv_state) {
        if (use_host_snap) {
            if (!snap.host_hc_after_attn.empty()) {
                CUDA_CHECK(cudaMemcpyAsync(st.d_ple_conv_state, snap.host_hc_after_attn.data(), snap.host_hc_after_attn.size(), cudaMemcpyHostToDevice, main_stream_));
            }
        } else if (snap.snap_hc_after_attn.data) {
            CUDA_CHECK(cudaMemcpyAsync(st.d_ple_conv_state, snap.snap_hc_after_attn.data, snap.snap_hc_after_attn.size_bytes, cudaMemcpyDeviceToDevice, main_stream_));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(main_stream_));
    st.token_history = snap.tokens;
    cached_tokens_ = snap.tokens;
    live_context_tokens_.store(snap.tokens.size(), std::memory_order_relaxed);
    argmax_cache_valid_ = false;
}

void forward_token_qwen4_exp(int token_id, int position, bool do_compute_logits = true) {
    auto& st = *qwen4_state_;
    st.stat_tokens_recorded++;

    if (st.num_gpus > 1 && active_pp_dev_ != 0) {
        set_active_pp_device(0);
    }

    // Track token history for Layer 1 PLE multi-order N-gram hash
    if ((int)st.token_history.size() <= position) {
        st.token_history.resize(position + 1, 0);
    }
    st.token_history[position] = token_id;

    // 1. Token embedding single-row H2D + dequantize + broadcast into 4-stream hyper-connection state R[4, 2560] on GPU 0
    if (active_visual_embeddings_ && position >= visual_embed_start_pos_ && position < visual_embed_start_pos_ + visual_embed_num_tokens_) {
        const int v_idx = position - visual_embed_start_pos_;
        qwen4_embed_to_hc_cuda(
            30 /* BF16 */, active_visual_embeddings_, v_idx, st.d_R_dev[0], 2560, 4, main_stream_);
    } else {
        uint8_t* h_emb_slot = st.h_token_embd_staging + (size_t)(st.stat_tokens_recorded & 63) * st.token_embd_row_bytes;
        std::memcpy(h_emb_slot, st.h_token_embd + (size_t)token_id * st.token_embd_row_bytes, st.token_embd_row_bytes);
        CUDA_CHECK(cudaMemcpyAsync(
            st.d_token_embd,
            h_emb_slot,
            st.token_embd_row_bytes,
            cudaMemcpyHostToDevice,
            main_stream_));
        qwen4_embed_to_hc_cuda(
            st.token_embd_type, st.d_token_embd, 0, st.d_R_dev[0], 2560, 4, main_stream_);
    }

    auto prof_sync = [&]() -> double {
        if (!st.profile_timing) return 0.0;
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double t_mark = prof_sync();
    auto prof_acc = [&](int slot) {
        if (!st.profile_timing) return;
        double now = prof_sync();
        st.prof_dec[slot] += (now - t_mark);
        t_mark = now;
    };

    // 2. Process all 48 layers across GPU 0 (Layers 0..pp_split_layer-1) and GPU 1 (Layers pp_split_layer..47)
    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
        if (st.num_gpus > 1 && l == st.pp_split_layer) {
            // Wait until GPU 1 has finished consuming d_R_dev[1] from the previous token, then transfer d_R over NVLink P2P
            CUDA_CHECK(cudaStreamWaitEvent(main_stream_, st.stage_ready_event_dev[1], 0));
            pp_copy_d2d(st.d_R_dev[1], 1, st.d_R_dev[0], 0, 10240 * sizeof(float));
            set_active_pp_device(1);
            prof_acc(8);
        }

        auto& lw = st.layers[l];
        const int dev = lw.device_id;
        float*   d_R           = st.d_R_dev[dev];
        float*   d_x           = st.d_x_dev[dev];
        uint8_t* d_moe_pkt     = st.d_moe_pkt_dev[dev];
        void*    d_x_q8_1      = st.d_x_q8_1_dev[dev];
        int32_t* d_top_idx     = st.d_top_idx_dev[dev];
        float*   d_top_w       = st.d_top_w_dev[dev];
        float*   d_sub_out     = st.d_sub_out_dev[dev];
        float*   d_scratch     = st.d_scratch_dev[dev];
        void*    d_q8_scratch  = st.d_q8_scratch_dev[dev];
        int32_t* d_sel_indices = st.d_sel_indices_dev[dev];
        Qwen4ActiveExpertPtrs* d_active_ptrs = st.d_active_ptrs_dev[dev];
        float*   d_cpu_moe_out = st.d_cpu_moe_out_dev[dev];
        cudaStream_t copy_stream       = st.copy_stream_dev[dev];
        cudaEvent_t  router_event      = st.router_event_dev[dev];
        cudaEvent_t  top_idx_event     = st.top_idx_event_dev[dev];
        cudaEvent_t  stage_ready_event = st.stage_ready_event_dev[dev];

        const uint64_t step_epoch = ++st.cur_token_epoch;

        // Layer 1 PLE + 131B N-gram cross-attention block (runs BEFORE blk.1.hc_attn_* on GPU 0)
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
            uint8_t* h_ple_slot = st.h_ple_rows_iq4nl + (size_t)(st.stat_tokens_recorded & 63) * (16 * 90);
            for (int h = 0; h < 16; ++h) {
                std::memcpy(h_ple_slot + h * 90, st.mmap_ple_embed + ple_rows[h] * 90, 90);
            }
            CUDA_CHECK(cudaMemcpyAsync(
                st.d_ple_rows_iq4nl, h_ple_slot, 16 * 90, cudaMemcpyHostToDevice, main_stream_));
            qwen4_ple_iq4nl_dequant_cuda(st.d_ple_rows_iq4nl, st.d_ple_emb_flat, main_stream_, d_q8_scratch);
            qwen4_ple_forward_cuda(
                d_R,
                (const float*)st.d_ple_norm_q,
                (const float*)st.d_ple_norm_k,
                (const float*)st.d_ple_norm_c,
                st.d_ple_key, st.ple_key_type,
                st.d_ple_val, st.ple_val_type,
                (const uint16_t*)st.d_ple_conv1d,
                st.d_ple_emb_flat,
                st.d_ple_conv_state,
                d_scratch,
                d_q8_scratch,
                cfg_.rms_norm_eps,
                main_stream_);
            prof_acc(0);
        }

        // --- Attention Sub-layer (GR Read -> GDN or QSA -> GR Write) ---
        const bool attn_pre_normed = (l > 0) && !(l == 1 && st.has_ple) && !(st.num_gpus > 1 && l == st.pp_split_layer);
        const float* next_attn_norm_w = nullptr;
        if (l + 1 < cfg_.num_hidden_layers) {
            if (!(l == 0 && st.has_ple) && !(st.num_gpus > 1 && l + 1 == st.pp_split_layer)) {
                next_attn_norm_w = (const float*)st.layers[l + 1].d_hc_attn_norm;
            }
        } else if (do_compute_logits) {
            next_attn_norm_w = (const float*)st.d_out_hc_norm;
        }

        qwen4_gr_read_cuda(
            d_R,
            (const float*)lw.d_hc_attn_norm,
            (const uint16_t*)lw.d_hc_attn_down,
            (const uint16_t*)lw.d_hc_attn_up,
            d_x,
            d_scratch,
            cfg_.rms_norm_eps,
            main_stream_,
            (const uint16_t*)lw.d_hc_attn_inject,
            d_x_q8_1,
            nullptr,
            attn_pre_normed);
        prof_acc(1);

        if (!lw.is_qsa) {
            qwen4_gdn_decode_cuda(
                d_x, d_x_q8_1,
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
                d_sub_out,
                d_scratch,
                d_q8_scratch,
                cfg_.rms_norm_eps,
                main_stream_);
            prof_acc(2);
        } else {
            qwen4_qsa_decode_cuda(
                d_x, d_x_q8_1,
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
                d_sub_out,
                d_scratch,
                d_sel_indices,
                d_q8_scratch,
                cfg_.rms_norm_eps,
                main_stream_);
            prof_acc(3);
        }

        // --- MoE Sub-layer (Fused Attention GR Write + MoE GR Read -> Router + Shared Expert + Routed Experts -> GR Write) ---
        qwen4_gr_read_cuda(
            d_R,
            (const float*)lw.d_hc_ffn_norm,
            (const uint16_t*)lw.d_hc_ffn_down,
            (const uint16_t*)lw.d_hc_ffn_up,
            d_x,
            d_scratch,
            cfg_.rms_norm_eps,
            main_stream_,
            (const uint16_t*)lw.d_hc_ffn_inject,
            d_x_q8_1,
            d_sub_out);
        prof_acc(4);

        // Launch Top-10 of 256 Router + Shared Expert Scalar Sigmoid Gate (stored in d_scratch[256])
        const bool need_host_top = (lw.d_all_vram_base == nullptr || st.calibrate_experts);
        qwen4_router_top10_cuda(
            (const uint16_t*)lw.d_moe_gate_inp,
            d_x,
            d_scratch,
            d_top_idx,
            d_top_w,
            main_stream_,
            (const uint16_t*)lw.d_moe_gate_inp_shexp,
            need_host_top ? st.d_host_top_idx_dev[dev] : nullptr,
            need_host_top ? st.d_host_top_w_dev[dev] : nullptr);
        prof_acc(5);

        float* d_shexp_gate = d_scratch + 256;
        float* d_down_11    = d_scratch + 512; // [11 * 2560]

        // Fast Path: All 256 experts of this layer reside contiguously in L1 Static VRAM -> Zero CPU-GPU Sync!
        if (lw.d_all_vram_base != nullptr && !st.calibrate_experts) {
            qwen4_moe_decode_shexp_and_l1_cuda(
                d_x_q8_1,
                lw.d_moe_gate_shexp, lw.moe_gate_shexp_type,
                lw.d_moe_up_shexp,   lw.moe_up_shexp_type,
                lw.d_moe_down_shexp, lw.moe_down_shexp_type,
                lw.d_all_vram_base,
                (const uint8_t* const*)lw.d_vram_exp_table,
                lw.gate_exp_bytes,
                lw.up_exp_bytes,
                lw.expert_blob_bytes,
                lw.moe_gate_exp_type,
                lw.moe_up_exp_type,
                lw.moe_down_exp_type,
                d_top_idx,
                d_down_11,
                d_q8_scratch,
                main_stream_);

            st.stat_l1_static_hits += 10;

            qwen4_moe_accum_gr_write_cuda(
                d_R,
                d_down_11,
                d_top_w,
                d_shexp_gate,
                nullptr,
                main_stream_,
                next_attn_norm_w,
                d_scratch,
                cfg_.rms_norm_eps);
            prof_acc(6);
            continue;
        }

        if (st.calibrate_experts) {
            CUDA_CHECK(cudaMemcpyAsync(
                st.calib_probs_host, d_scratch, 256 * sizeof(float), cudaMemcpyDeviceToHost, main_stream_));
        }

        // 1) Record router completion on main_stream_ (or queue h_x_q8_1 D2H if unpinned/no-LRU CPU offload is needed)
        //    AND IMMEDIATELY launch Shared Expert + all L1-Hit Routed Experts BEFORE cudaEventSynchronize(top_idx_event)!
        if (enable_cpu_experts_ && (st.has_unpinned_experts || lw.n_lru_slots == 0)) {
            CUDA_CHECK(cudaEventRecord(router_event, main_stream_));
            CUDA_CHECK(cudaStreamWaitEvent(copy_stream, router_event, 0));
            CUDA_CHECK(cudaMemcpyAsync(
                st.h_moe_pkt, d_moe_pkt, 2880, cudaMemcpyDeviceToHost, copy_stream));
            CUDA_CHECK(cudaEventRecord(top_idx_event, copy_stream));
        } else {
            CUDA_CHECK(cudaEventRecord(top_idx_event, main_stream_));
        }

        qwen4_moe_decode_shexp_and_l1_cuda(
            d_x_q8_1,
            lw.d_moe_gate_shexp, lw.moe_gate_shexp_type,
            lw.d_moe_up_shexp,   lw.moe_up_shexp_type,
            lw.d_moe_down_shexp, lw.moe_down_shexp_type,
            nullptr,
            (const uint8_t* const*)lw.d_vram_exp_table,
            lw.gate_exp_bytes,
            lw.up_exp_bytes,
            lw.expert_blob_bytes,
            lw.moe_gate_exp_type,
            lw.moe_up_exp_type,
            lw.moe_down_exp_type,
            d_top_idx,
            d_down_11,
            d_q8_scratch,
            main_stream_);
        prof_acc(6);

        CUDA_CHECK(cudaEventSynchronize(top_idx_event));
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

        // Partition the 10 active experts into L1 VRAM Hits (already executing on main_stream_!) and L1 VRAM Misses
        const size_t g_off = 0;
        const size_t u_off = lw.gate_exp_bytes;
        const size_t d_off = lw.gate_exp_bytes + lw.up_exp_bytes;
        const size_t eb    = lw.expert_blob_bytes;

        int miss_eids[10];
        int miss_orig_k[10];
        int n_misses = 0;

        for (int k = 0; k < 10; ++k) {
            const int eid = st.h_top_idx[k];
            uint8_t* vptr = lw.d_vram_exp_ptr[eid];
            if (vptr != nullptr) {
                if (lw.is_static_vram[eid]) {
                    st.stat_l1_static_hits++;
                } else {
                    st.stat_l1_lru_hits++;
                    const int s = (int)lw.exp_lru_slot[eid];
                    if (s >= 0) {
                        lw.lru_slot_last_used[s] = (++st.lru_clock) + 24;
                        lw.lru_slot_epoch[s]     = step_epoch;
                    }
                }
            } else {
                miss_eids[n_misses]   = eid;
                miss_orig_k[n_misses] = k;
                n_misses++;
            }
        }

        if (n_misses > 0) {
            // Helper to allocate a Dynamic LRU L1 VRAM victim slot for a missed expert
            auto alloc_lru_slot = [&](int eid, int32_t* out_old_eid) -> uint8_t* {
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
                *out_old_eid = old_eid;
                if (old_eid >= 0) {
                    lw.d_vram_exp_ptr[old_eid]     = nullptr;
                    lw.h_vram_exp_table[old_eid]   = nullptr;
                    lw.h_decode_exp_table[old_eid] = lw.d_mapped_exp_ptr[old_eid];
                    lw.exp_lru_slot[old_eid]       = -1;
                }
                uint8_t* slot_vram = lw.d_lru_base + (size_t)best_s * eb;
                lw.d_vram_exp_ptr[eid]         = slot_vram;
                lw.h_vram_exp_table[eid]       = slot_vram;
                lw.h_decode_exp_table[eid]     = slot_vram;
                lw.exp_lru_slot[eid]           = (int8_t)best_s;
                lw.lru_slot_eid[best_s]       = eid;
                lw.lru_slot_last_used[best_s] = ++st.lru_clock;
                lw.lru_slot_epoch[best_s]     = step_epoch;
                lw.decode_miss_cnt[eid]       = 0;
                return slot_vram;
            };

            uint16_t miss_mcnt[10];
            for (int m = 0; m < n_misses; ++m) {
                miss_mcnt[m] = ++lw.decode_miss_cnt[miss_eids[m]];
            }

            moecher::cpu_moe::CpuExpertTaskDesc cpu_tasks[10];
            int n_cpu_misses = 0;
            bool is_cpu_miss[10] = {false};

            if (enable_cpu_experts_) {
                for (int m = n_misses - 1; m >= 0; --m) {
                    const int eid    = miss_eids[m];
                    const int orig_k = miss_orig_k[m];
                    const uint8_t* host_ptr = lw.h_unpinned_exp_ptr[eid]
                        ? lw.h_unpinned_exp_ptr[eid]
                        : lw.h_pinned_exp_ptr[eid];
                    if (host_ptr == nullptr) continue;

                    const bool is_unpinned = (lw.h_unpinned_exp_ptr[eid] != nullptr);
                    const bool offload_to_cpu =
                        (lw.n_lru_slots == 0) ||
                        (is_unpinned && n_misses >= 2 && n_cpu_misses == 0 && orig_k >= 4 && miss_mcnt[m] == 1);

                    if (offload_to_cpu) {
                        is_cpu_miss[m] = true;
                        if (lw.h_unpinned_exp_ptr[eid] != nullptr) {
                            st.stat_l25_unpinned_hits++;
                        } else {
                            st.stat_l2_pinned_hits++;
                        }
                        st.stat_cpu_computed++;
                        auto& ct = cpu_tasks[n_cpu_misses++];
                        ct.gate_ptr = host_ptr + g_off;
                        ct.up_ptr   = host_ptr + u_off;
                        ct.down_ptr = host_ptr + d_off;
                        ct.weight   = st.h_top_w[orig_k];
                        if (n_cpu_misses >= 2 || n_misses <= 2) break;
                    }
                }
            }

            Qwen4ActiveExpertPtrs* h_layer_ptrs = st.h_active_ptrs + l;
            Qwen4ActiveExpertPtrs* d_layer_ptrs = d_active_ptrs + l;
            int gpu_miss_eids[10];
            uint8_t* gpu_miss_dst_vram[10];
            int n_gpu_misses = 0;

            for (int m = 0; m < n_misses; ++m) {
                if (is_cpu_miss[m]) continue;
                const int eid    = miss_eids[m];
                const int orig_k = miss_orig_k[m];
                int32_t old_eid  = -1;
                uint8_t* slot_vram = alloc_lru_slot(eid, &old_eid);

                const int gm = n_gpu_misses++;
                gpu_miss_eids[gm]     = eid;
                gpu_miss_dst_vram[gm] = slot_vram;

                h_layer_ptrs->gate[gm]    = slot_vram + g_off;
                h_layer_ptrs->up[gm]      = slot_vram + u_off;
                h_layer_ptrs->down[gm]    = slot_vram + d_off;
                h_layer_ptrs->slot[gm]    = orig_k;
                h_layer_ptrs->old_eid[gm] = old_eid;
                h_layer_ptrs->new_eid[gm] = eid;
            }

            const bool need_ondemand_x_q8 = (n_cpu_misses > 0) && !(st.has_unpinned_experts || lw.n_lru_slots == 0);
            if (need_ondemand_x_q8) {
                CUDA_CHECK(cudaMemcpyAsync(st.h_moe_pkt, d_moe_pkt, 2880, cudaMemcpyDeviceToHost, copy_stream));
                CUDA_CHECK(cudaEventRecord(router_event, copy_stream));
            }

            if (n_gpu_misses > 0) {
                int unpinned_task_cnt = 0;
                int unpinned_miss_idx[10];
                for (int gm = 0; gm < n_gpu_misses; ++gm) {
                    const int eid = gpu_miss_eids[gm];
                    if (lw.h_pinned_exp_ptr[eid] != nullptr) {
                        st.stat_l2_pinned_hits++;
                        CUDA_CHECK(cudaMemcpyAsync(
                            gpu_miss_dst_vram[gm], lw.h_pinned_exp_ptr[eid], eb, cudaMemcpyHostToDevice, copy_stream));
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
#if defined(_WIN32)
                        _fseeki64(st.experts_fp, (int64_t)(lw.layer_file_offset + (uint64_t)eid * eb), SEEK_SET);
#else
                        fseeko(st.experts_fp, (off_t)(lw.layer_file_offset + (uint64_t)eid * eb), SEEK_SET);
#endif
                        (void)std::fread(bounce, 1, eb, st.experts_fp);
                        CUDA_CHECK(cudaMemcpyAsync(
                            gpu_miss_dst_vram[gm], bounce, eb, cudaMemcpyHostToDevice, copy_stream));
                    }
                }

                if (unpinned_task_cnt > 0) {
                    (void)cudaStreamQuery(copy_stream);
                    (void)cudaStreamQuery(main_stream_);
                    st.parallel_unpinned_gather(unpinned_task_cnt);
                    for (int t = 0; t < unpinned_task_cnt; ++t) {
                        const int gm = unpinned_miss_idx[t];
                        CUDA_CHECK(cudaMemcpyAsync(
                            gpu_miss_dst_vram[gm],
                            st.h_bounce_exp + (size_t)gm * st.max_expert_blob_bytes,
                            eb,
                            cudaMemcpyHostToDevice,
                            copy_stream));
                    }
                }

                CUDA_CHECK(cudaEventRecord(stage_ready_event, copy_stream));
                CUDA_CHECK(cudaStreamWaitEvent(main_stream_, stage_ready_event, 0));

                qwen4_moe_decode_misses_cuda(
                    d_x_q8_1,
                    d_layer_ptrs,
                    n_gpu_misses,
                    lw.moe_gate_exp_type,
                    lw.moe_up_exp_type,
                    lw.moe_down_exp_type,
                    d_down_11,
                    d_q8_scratch,
                    main_stream_,
                    lw.d_vram_exp_table);
            }

            if (n_cpu_misses > 0) {
                if (need_ondemand_x_q8) {
                    CUDA_CHECK(cudaEventSynchronize(router_event));
                }
                (void)cudaStreamQuery(copy_stream);
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
            }

            qwen4_moe_accum_gr_write_cuda(
                d_R,
                d_down_11,
                d_top_w,
                d_shexp_gate,
                (n_cpu_misses > 0) ? d_cpu_moe_out : nullptr,
                main_stream_,
                next_attn_norm_w,
                d_scratch,
                cfg_.rms_norm_eps);
        } else {
            qwen4_moe_accum_gr_write_cuda(
                d_R,
                d_down_11,
                d_top_w,
                d_shexp_gate,
                nullptr,
                main_stream_,
                next_attn_norm_w,
                d_scratch,
                cfg_.rms_norm_eps);
        }
        prof_acc(7);
    }

    // 3. Final Output Gated Residual Read + Output Head Logits (only on final prefill token or decode step)
    const int out_dev = (st.num_gpus > 1) ? 1 : 0;
    if (do_compute_logits) {
        void* peer_q8_out = (st.num_gpus > 1 && p2p_enabled_ && st.d_output_dev[1] != nullptr) ? st.d_x_q8_1_dev[0] : nullptr;
        qwen4_gr_read_cuda(
            st.d_R_dev[out_dev],
            (const float*)st.d_out_hc_norm,
            (const uint16_t*)st.d_out_hc_down,
            (const uint16_t*)st.d_out_hc_up,
            st.d_x_dev[out_dev],
            st.d_scratch_dev[out_dev],
            cfg_.rms_norm_eps,
            main_stream_,
            nullptr,
            st.d_x_q8_1_dev[out_dev],
            nullptr,
            true,
            peer_q8_out);

        if (st.num_gpus > 1 && st.d_output_dev[1] != nullptr) {
            const int n_rows0 = st.vocab_split;
            const int n_rows1 = cfg_.vocab_size - n_rows0;
            CUDA_CHECK(cudaEventRecord(st.logits_ready_event_dev[1], main_stream_));
            qwen4_project_cuda(
                st.output_type, st.d_output_dev[1], st.d_x_dev[1], st.d_x_q8_1_dev[1],
                st.d_logits_dev1 + n_rows0, n_rows1, 2560, main_stream_);
            if (st.fast_greedy) {
                qwen4_argmax_f32_cuda(
                    st.d_logits_dev1 + n_rows0, n_rows1,
                    st.greedy_mask_id0, st.greedy_mask_id1,
                    st.d_argmax_out_dev[1], main_stream_, n_rows0);
            }
            CUDA_CHECK(cudaEventRecord(st.stage_ready_event_dev[1], main_stream_));

            set_active_pp_device(0);
            CUDA_CHECK(cudaStreamWaitEvent(main_stream_, st.logits_ready_event_dev[1], 0));
            if (st.output_type == 30 || st.output_type == 1 || st.output_type == 0) {
                CUDA_CHECK(cudaMemcpyPeerAsync(st.d_x_dev[0], 0, st.d_x_dev[1], 1, 2560 * sizeof(float), main_stream_));
            } else if (!p2p_enabled_) {
                CUDA_CHECK(cudaMemcpyAsync(st.d_x_q8_1_dev[0], st.d_x_q8_1_dev[1], 2880, cudaMemcpyDefault, main_stream_));
            }
            float* d_logits0 = buf_logits_.f32();
            qwen4_project_cuda(
                st.output_type, st.d_output_dev[0], st.d_x_dev[0], st.d_x_q8_1_dev[0],
                d_logits0, n_rows0, 2560, main_stream_);
            if (st.fast_greedy) {
                qwen4_argmax_f32_cuda(
                    d_logits0, n_rows0,
                    st.greedy_mask_id0, st.greedy_mask_id1,
                    st.d_argmax_out_dev[0], main_stream_, 0);
                CUDA_CHECK(cudaEventRecord(st.stage_ready_event_dev[0], main_stream_));
                st.argmax_ready = true;
            } else {
                CUDA_CHECK(cudaStreamWaitEvent(main_stream_, st.stage_ready_event_dev[1], 0));
                CUDA_CHECK(cudaMemcpyPeerAsync(
                    d_logits0 + n_rows0, 0, st.d_logits_dev1 + n_rows0, 1,
                    (size_t)n_rows1 * sizeof(float), main_stream_));
                st.argmax_ready = false;
            }
        } else {
            float* d_logits_out = (st.num_gpus > 1) ? st.d_logits_dev1 : buf_logits_.f32();
            qwen4_project_cuda(
                st.output_type, st.d_output, st.d_x_dev[out_dev], st.d_x_q8_1_dev[out_dev], d_logits_out, cfg_.vocab_size, 2560, main_stream_);

            if (st.fast_greedy) {
                qwen4_argmax_f32_cuda(
                    d_logits_out, cfg_.vocab_size, st.greedy_mask_id0, st.greedy_mask_id1, st.d_argmax_out_dev[out_dev], main_stream_);
                CUDA_CHECK(cudaEventRecord(st.stage_ready_event_dev[out_dev], main_stream_));
                if (st.num_gpus > 1) {
                    set_active_pp_device(0);
                }
                st.argmax_ready = true;
            } else if (st.num_gpus > 1) {
                CUDA_CHECK(cudaEventRecord(st.stage_ready_event_dev[1], main_stream_));
                pp_copy_d2d(pp_peer_.buf_logits_.data, 0, st.d_logits_dev1, 1, (size_t)cfg_.vocab_size * sizeof(float));
                set_active_pp_device(0);
                st.argmax_ready = false;
            } else {
                st.argmax_ready = false;
            }
        }
        prof_acc(9);
        if (st.profile_timing) {
            st.prof_dec_cnt++;
            if ((st.prof_dec_cnt & 31) == 0) {
                const double inv = 1.0 / (double)st.prof_dec_cnt;
                LOG_WARN("[DECODE PROFILE n=%llu] PLE=%.2fms | GR_Attn=%.2fms | GDN=%.2fms | QSA=%.2fms | GR_FFN=%.2fms | Router=%.2fms | MoE_L1=%.2fms | MoE_Miss=%.2fms | PP_Sync=%.2fms | Logits=%.2fms",
                         (unsigned long long)st.prof_dec_cnt,
                         st.prof_dec[0] * inv, st.prof_dec[1] * inv, st.prof_dec[2] * inv, st.prof_dec[3] * inv,
                         st.prof_dec[4] * inv, st.prof_dec[5] * inv, st.prof_dec[6] * inv, st.prof_dec[7] * inv,
                         st.prof_dec[8] * inv, st.prof_dec[9] * inv);
            }
        }
    } else if (st.num_gpus > 1) {
        CUDA_CHECK(cudaEventRecord(st.stage_ready_event_dev[1], main_stream_));
        set_active_pp_device(0);
        if ((st.stat_tokens_recorded & 7) == 0) {
            CUDA_CHECK(cudaEventSynchronize(st.stage_ready_event_dev[1]));
        }
    } else {
        if (st.layers[0].d_all_vram_base != nullptr && (st.stat_tokens_recorded & 7) == 0) {
            CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        }
    }
}

void prefill_prompt_batched_qwen4_exp(
    const std::vector<int>& prompt,
    int start_pos = 0,
    bool compute_last_logits = true)
{
    const int total_tokens = (int)prompt.size();
    if (start_pos >= total_tokens) return;
    if (total_tokens - start_pos == 1) {
        forward_token_qwen4_exp(prompt[start_pos], start_pos, compute_last_logits);
        return;
    }

    auto& st = *qwen4_state_;
    st.has_pending_decode_top_idx = false;
    constexpr int MAX_M = Qwen4RuntimeState::QWEN4_MAX_PREFILL_M;
    constexpr int SUB_M = Qwen4RuntimeState::QWEN4_MAX_SUB_M;
    const int dev_max_m = (st.num_gpus > 1) ? SUB_M : MAX_M;
    const int total_to_prefill = total_tokens - start_pos;
    int max_chunk_m = dev_max_m;
    int first_chunk_m = 0;
    if (st.num_gpus > 1) {
        if (total_to_prefill <= 96) {
            max_chunk_m = total_to_prefill;
        } else if (total_to_prefill <= 768) {
            max_chunk_m = (total_to_prefill + 1) / 2;
        } else {
            const int n_chunks = (total_to_prefill + 527) / 528;
            if (n_chunks >= 3) {
                max_chunk_m = dev_max_m; // 576 for middle chunks
                const int rem = total_to_prefill - (n_chunks - 2) * max_chunk_m;
                if (rem > 64) {
                    first_chunk_m = std::min(max_chunk_m, ((rem / 2) + 15) & ~15);
                } else {
                    max_chunk_m = (total_to_prefill + n_chunks - 1) / n_chunks;
                }
            } else {
                max_chunk_m = (total_to_prefill + n_chunks - 1) / n_chunks;
            }
        }
    } else {
        const int eff_max_m = dev_max_m;
        if (total_to_prefill > eff_max_m) {
            const int n_chunks = (total_to_prefill + eff_max_m - 1) / eff_max_m;
            max_chunk_m = (total_to_prefill + n_chunks - 1) / n_chunks;
        }
    }
    if (total_to_prefill > max_chunk_m) {
        max_chunk_m = std::min(dev_max_m, (max_chunk_m + 15) & ~15);
    }

    if ((int)st.token_history.size() < total_tokens) {
        st.token_history.resize(total_tokens, 0);
    }
    for (int p = start_pos; p < total_tokens; ++p) {
        st.token_history[p] = prompt[p];
    }

    int pos_start = start_pos;
    int chunk_idx = 0;
    double pre_prof[10] = {0.0};
    auto pre_sync = [&]() -> double {
        if (!st.profile_timing) return 0.0;
        CUDA_CHECK(cudaStreamSynchronize(main_stream_));
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double pre_mark = pre_sync();
    auto pre_acc = [&](int slot) {
        if (!st.profile_timing) return;
        double now = pre_sync();
        pre_prof[slot] += (now - pre_mark);
        pre_mark = now;
    };

    while (pos_start < total_tokens) {
        const int target_m = (chunk_idx == 0 && first_chunk_m > 0) ? first_chunk_m : max_chunk_m;
        const int M_chunk  = std::min(target_m, total_tokens - pos_start);
        int M = M_chunk;
        const bool is_last_chunk = (pos_start + M_chunk >= total_tokens);
        st.stat_tokens_recorded += (uint64_t)M;

        if (st.num_gpus > 1 && active_pp_dev_ != 0) {
            set_active_pp_device(0);
        }

        // 1. Batch-stage M token embeddings into double-buffered pinned host memory & upload to GPU 0
        uint8_t* h_emb_base = st.h_token_embd_staging + (size_t)(chunk_idx & 1) * MAX_M * st.token_embd_row_bytes;
        for (int m = 0; m < M; ++m) {
            const int tid = prompt[pos_start + m];
            std::memcpy(
                h_emb_base + (size_t)m * st.token_embd_row_bytes,
                st.h_token_embd + (size_t)tid * st.token_embd_row_bytes,
                st.token_embd_row_bytes);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            st.d_token_embd,
            h_emb_base,
            (size_t)M * st.token_embd_row_bytes,
            cudaMemcpyHostToDevice,
            main_stream_));
        qwen4_embed_to_hc_batchM_cuda(
            st.token_embd_type, st.d_token_embd, nullptr, M, st.d_R_dev[0], 2560, 4, main_stream_);
        if (active_visual_embeddings_ && visual_embed_start_pos_ >= 0 && visual_embed_num_tokens_ > 0) {
            const int v_beg = std::max(pos_start, visual_embed_start_pos_);
            const int v_end = std::min(pos_start + M, visual_embed_start_pos_ + visual_embed_num_tokens_);
            if (v_beg < v_end) {
                const int m_off = v_beg - pos_start;
                const int v_off = v_beg - visual_embed_start_pos_;
                const int n_vis = v_end - v_beg;
                qwen4_embed_to_hc_batchM_cuda(
                    30 /* BF16 */,
                    active_visual_embeddings_ + (size_t)v_off * 2560,
                    nullptr,
                    n_vis,
                    st.d_R_dev[0] + (size_t)m_off * 10240,
                    2560,
                    4,
                    main_stream_);
            }
        }
        pre_acc(8);

        // 2. Process all 48 layers for the M-token chunk
        for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
            if (st.num_gpus > 1 && l == st.pp_split_layer) {
                CUDA_CHECK(cudaStreamWaitEvent(main_stream_, st.stage_ready_event_dev[1], 0));
                pp_copy_d2d(st.d_R_dev[1], 1, st.d_R_dev[0], 0, (size_t)M * 10240 * sizeof(float));
                set_active_pp_device(1);
                pre_acc(8);
            }

            auto& lw = st.layers[l];
            const int dev = lw.device_id;
            float*   d_R           = st.d_R_dev[dev];
            float*   d_x           = st.d_x_dev[dev];
            int32_t* d_top_idx_b   = (int32_t*)(st.d_moe_pkt_dev[dev] + 0);
            float*   d_top_w_b     = (float*)(st.d_moe_pkt_dev[dev] + 98304);
            void*    d_x_q8_1_b    = (void*)(st.d_moe_pkt_dev[dev] + 196608);
            float*   d_sub_out     = st.d_sub_out_dev[dev];
            float*   d_scratch     = st.d_scratch_dev[dev];
            void*    d_q8_scratch  = st.d_q8_scratch_dev[dev];
            int32_t* d_sel_indices = st.d_sel_indices_dev[dev];
            cudaStream_t copy_stream       = st.copy_stream_dev[dev];
            cudaEvent_t  router_event      = st.router_event_dev[dev];
            cudaEvent_t  top_idx_event     = st.top_idx_event_dev[dev];
            cudaEvent_t  stage_ready_event = st.stage_ready_event_dev[dev];

            const uint64_t step_epoch = ++st.cur_token_epoch;

            // Layer 1 PLE + 131B N-gram cross-attention block (batched in sub-chunks of up to 64 tokens)
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

                for (int m_off = 0; m_off < M; m_off += 64) {
                    const int sub_m = std::min(64, M - m_off);
                    const int n_ple_rows = sub_m * 16;
                    uint8_t* h_ple_base = st.h_ple_rows_iq4nl + (size_t)((chunk_idx * 32 + (m_off / 64)) & 63) * (64 * 16 * 90);
                    uint64_t ple_rows_idx[64 * 16];
                    for (int lm = 0; lm < sub_m; ++lm) {
                        const int pos_m = pos_start + m_off + lm;
                        int64_t ctx[3];
                        ctx[0] = (int64_t)st.token_history[pos_m];
                        bool cut = false;
                        for (int s = 1; s < 3; ++s) {
                            const int q = pos_m - s;
                            const int64_t tk = (q >= 0) ? (int64_t)st.token_history[q] : -1;
                            cut = cut || tk < 0 || tk == PLE_EOS;
                            ctx[s] = cut ? PLE_EOS : tk;
                        }
                        uint64_t* r = ple_rows_idx + lm * 16;
                        for (int n = 2; n <= 3; ++n) {
                            uint64_t mixed = (uint64_t)ctx[0] * PLE_MULT[0];
                            for (int j = 1; j < n; ++j) mixed ^= (uint64_t)ctx[j] * PLE_MULT[j];
                            for (int gi = 0; gi < 8; ++gi) {
                                const int h = (n - 2) * 8 + gi;
                                const uint64_t row_idx = mixed % PLE_VOCAB[h] + PLE_OFF[h];
                                r[h] = row_idx;
                                _mm_prefetch((const char*)(st.mmap_ple_embed + row_idx * 90), _MM_HINT_T0);
                            }
                        }
                    }
                    for (int i = 0; i < n_ple_rows; ++i) {
                        if (i + 16 < n_ple_rows) {
                            _mm_prefetch((const char*)(st.mmap_ple_embed + ple_rows_idx[i + 16] * 90), _MM_HINT_T0);
                        }
                        std::memcpy(h_ple_base + (size_t)i * 90, st.mmap_ple_embed + ple_rows_idx[i] * 90, 90);
                    }
                    CUDA_CHECK(cudaMemcpyAsync(
                        st.d_ple_rows_iq4nl, h_ple_base, (size_t)n_ple_rows * 90, cudaMemcpyHostToDevice, main_stream_));
                    qwen4_ple_forward_batchM_cuda(
                        d_R + (size_t)m_off * 10240,
                        (const float*)st.d_ple_norm_q,
                        (const float*)st.d_ple_norm_k,
                        (const float*)st.d_ple_norm_c,
                        st.d_ple_key, st.ple_key_type,
                        st.d_ple_val, st.ple_val_type,
                        (const uint16_t*)st.d_ple_conv1d,
                        st.d_ple_rows_iq4nl,
                        st.d_ple_emb_flat,
                        st.d_ple_conv_state,
                        sub_m,
                        d_scratch,
                        d_q8_scratch,
                        cfg_.rms_norm_eps,
                        main_stream_);
                }
                pre_acc(0);
            }

            // --- Attention Sub-layer (Batched GR Read -> Sub-chunked GDN or QSA) ---
            const bool attn_fuses_prev_moe = (l > 0) && !(l == 1 && st.has_ple) && !(st.num_gpus > 1 && l == st.pp_split_layer);
            const bool defer_moe_gr_write  = (l + 1 < cfg_.num_hidden_layers) && !(l == 0 && st.has_ple) && !(st.num_gpus > 1 && l + 1 == st.pp_split_layer);

            qwen4_gr_read_batchM_cuda(
                d_R,
                (const float*)lw.d_hc_attn_norm,
                (const uint16_t*)lw.d_hc_attn_down,
                (const uint16_t*)lw.d_hc_attn_up,
                d_x,
                M,
                d_scratch,
                cfg_.rms_norm_eps,
                main_stream_,
                (const uint16_t*)lw.d_hc_attn_inject,
                ((M < 64) || (M > 1032 && (M % 1032) > 0 && (M % 1032) < 64)) ? d_x_q8_1_b : nullptr,
                attn_fuses_prev_moe ? d_sub_out : nullptr);
            pre_acc(1);

            if (!lw.is_qsa) {
                qwen4_gdn_prefill_batchM_cuda(
                    d_x, d_x_q8_1_b,
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
                    M,
                    d_sub_out,
                    d_scratch,
                    d_q8_scratch,
                    cfg_.rms_norm_eps,
                    main_stream_);
            } else {
                qwen4_qsa_prefill_batchM_cuda(
                    d_x, d_x_q8_1_b,
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
                    pos_start,
                    M,
                    cfg_.max_seq_len,
                    d_sub_out,
                    d_scratch,
                    d_sel_indices,
                    d_q8_scratch,
                    cfg_.rms_norm_eps,
                    main_stream_);
            }
            pre_acc(lw.is_qsa ? 3 : 2);

            // --- MoE Sub-layer (Fused Attention GR Write + Batched MoE GR Read -> Router + Shared Expert + Grouped Routed Experts -> Batched GR Write) ---
            // On the final layer of the final chunk when only the last token's logits are needed,
            // Layer 47's Attention/GDN state has already been updated for all M tokens;
            // compact token (M - 1) into slot 0 and evaluate Layer 47's stateless MoE FFN for only the final token!
            if (is_last_chunk && compute_last_logits && (l + 1 == cfg_.num_hidden_layers) && M > 1) {
                CUDA_CHECK(cudaMemcpyAsync(
                    d_R,
                    d_R + (size_t)(M - 1) * 10240,
                    10240 * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(
                    d_R + 10240,
                    d_R + (size_t)M * 10240 + (size_t)(M - 1) * 4,
                    4 * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    main_stream_));
                CUDA_CHECK(cudaMemcpyAsync(
                    d_sub_out,
                    d_sub_out + (size_t)(M - 1) * 2560,
                    2560 * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    main_stream_));
                M = 1;
            }
            qwen4_gr_read_batchM_cuda(
                d_R,
                (const float*)lw.d_hc_ffn_norm,
                (const uint16_t*)lw.d_hc_ffn_down,
                (const uint16_t*)lw.d_hc_ffn_up,
                d_x,
                M,
                d_scratch,
                cfg_.rms_norm_eps,
                main_stream_,
                (const uint16_t*)lw.d_hc_ffn_inject,
                d_x_q8_1_b,
                d_sub_out);
            pre_acc(4);

            const bool need_host_top10 = (lw.d_all_vram_base == nullptr) || st.calibrate_experts;
            int32_t* d_host_top_idx_b = need_host_top10 ? (int32_t*)((uint8_t*)st.d_host_top_idx_dev[dev] - 2880) : nullptr;
            float*   d_host_top_w_b   = need_host_top10 ? (float*)((uint8_t*)st.d_host_top_idx_dev[dev] - 2880 + 98304) : nullptr;
            qwen4_router_top10_batchM_cuda(
                (const uint16_t*)lw.d_moe_gate_inp,
                d_x,
                M,
                d_scratch,
                d_top_idx_b,
                d_top_w_b,
                main_stream_,
                d_host_top_idx_b,
                d_host_top_w_b);
            if (need_host_top10) {
                CUDA_CHECK(cudaEventRecord(top_idx_event, main_stream_));
            }

            const size_t g_off = 0;
            const size_t u_off = lw.gate_exp_bytes;
            const size_t d_off = lw.gate_exp_bytes + lw.up_exp_bytes;
            const size_t eb    = lw.expert_blob_bytes;

            // Fast Path: All 256 experts reside in L1 Static VRAM -> GPU-grouped 8-token tiles, zero CPU sync!
            if (lw.d_all_vram_base != nullptr && !st.calibrate_experts) {
                qwen4_shared_expert_batchM_cuda(
                    d_x, d_x_q8_1_b,
                    lw.d_moe_gate_shexp, lw.moe_gate_shexp_type,
                    lw.d_moe_up_shexp,   lw.moe_up_shexp_type,
                    lw.d_moe_down_shexp, lw.moe_down_shexp_type,
                    (const uint16_t*)lw.d_moe_gate_inp_shexp,
                    M,
                    d_sub_out,
                    d_scratch,
                    d_q8_scratch,
                    main_stream_);
                pre_acc(5);

                qwen4_routed_experts_resident_batchM_cuda(
                    d_x_q8_1_b,
                    lw.d_all_vram_base + g_off, eb, lw.moe_gate_exp_type,
                    lw.d_all_vram_base + u_off, eb, lw.moe_up_exp_type,
                    lw.d_all_vram_base + d_off, eb, lw.moe_down_exp_type,
                    d_top_idx_b,
                    d_top_w_b,
                    M,
                    d_sub_out,
                    d_scratch,
                    d_q8_scratch,
                    main_stream_);

                st.stat_l1_static_hits += (uint64_t)M * 10;
                if (!defer_moe_gr_write) {
                    qwen4_gr_write_batchM_cuda(d_R, d_sub_out, M, main_stream_);
                }
                pre_acc(6);
                continue;
            }

            // Tiered Cache Path (1-GPU):
            // 1) Router top-10 IDs + weights are written directly to mapped pinned host memory by router_top10_kernel;
            //    IMMEDIATELY launch Shared Expert + GPU-grouped L1-Hit Experts on main_stream_ BEFORE cudaEventSynchronize(top_idx_event),
            //    overlapping L1-Hit compute on main_stream_ with PCIe Miss DMAs on copy_stream!
            int32_t* h_top_idx_b = (int32_t*)(st.h_moe_pkt + 0);
            float*   h_top_w_b   = (float*)(st.h_moe_pkt + 98304);

            qwen4_shared_expert_batchM_cuda(
                d_x, d_x_q8_1_b,
                lw.d_moe_gate_shexp, lw.moe_gate_shexp_type,
                lw.d_moe_up_shexp,   lw.moe_up_shexp_type,
                lw.d_moe_down_shexp, lw.moe_down_shexp_type,
                (const uint16_t*)lw.d_moe_gate_inp_shexp,
                M,
                d_sub_out,
                d_scratch,
                d_q8_scratch,
                main_stream_);
            pre_acc(5);

            qwen4_routed_experts_l1_hits_batchM_cuda(
                d_x_q8_1_b,
                (const uint8_t* const*)lw.d_vram_exp_table,
                lw.gate_exp_bytes,
                lw.up_exp_bytes,
                lw.moe_gate_exp_type,
                lw.moe_up_exp_type,
                lw.moe_down_exp_type,
                d_top_idx_b,
                d_top_w_b,
                M,
                d_sub_out,
                d_scratch + 2359296,
                main_stream_);
            pre_acc(6);

            CUDA_CHECK(cudaEventSynchronize(top_idx_event));

            // Fast Pass 1 (2 KB stack-resident L1 cache): count expert frequencies & last token index so copy_stream H2D DMAs can launch immediately!
            int exp_cnt[256] = {0};
            int last_tok[256] = {0};
            for (int m = 0; m < M; ++m) {
                const int32_t* idx_m = h_top_idx_b + m * 10;
                for (int k = 0; k < 10; ++k) {
                    const int eid = idx_m[k];
                    if ((unsigned)eid < 256u) {
                        exp_cnt[eid]++;
                        last_tok[eid] = m;
                    }
                }
            }

            int miss_eids[256];
            int miss_base[256];
            int miss_write[256];
            int n_unique_misses = 0;

            for (int eid = 0; eid < 256; ++eid) {
                miss_base[eid]  = -1;
                miss_write[eid] = -1;
                const int cnt = exp_cnt[eid];
                if (cnt == 0) continue;
                uint8_t* vptr = lw.d_vram_exp_ptr[eid];
                if (vptr != nullptr) {
                    if (lw.is_static_vram[eid]) {
                        st.stat_l1_static_hits += (uint64_t)cnt;
                    } else {
                        st.stat_l1_lru_hits += (uint64_t)cnt;
                        const int s = (int)lw.exp_lru_slot[eid];
                        if (s >= 0) {
                            lw.lru_slot_last_used[s] = (++st.lru_clock) + (uint64_t)std::min(cnt, 8);
                            lw.lru_slot_epoch[s]     = step_epoch;
                        }
                    }
                } else {
                    miss_eids[n_unique_misses++] = eid;
                }
            }

            // Sort missed experts by prompt token frequency descending (tie-breaking by latest token index)
            if (n_unique_misses > 1) {
                std::sort(miss_eids, miss_eids + n_unique_misses, [&](int a, int b) {
                    if (is_last_chunk && M <= 64 && ((last_tok[a] >= M - 24) != (last_tok[b] >= M - 24))) return (last_tok[a] >= M - 24);
                    if (exp_cnt[a] != exp_cnt[b]) return exp_cnt[a] > exp_cnt[b];
                    return last_tok[a] > last_tok[b];
                });
            }

            int miss_pair_offset = 0;
            for (int i = 0; i < n_unique_misses; ++i) {
                const int eid = miss_eids[i];
                miss_base[eid]  = miss_pair_offset;
                miss_write[eid] = miss_pair_offset;
                miss_pair_offset += exp_cnt[eid];
            }
            bool miss_pairs_populated = false;

            // Scratch Layout on 1-GPU (48M floats = 192 MiB):
            //   [0 .. 2,359,296 floats):             Shared Expert workspace
            //   [2,359,296 .. 6,815,744 floats):     GPU L1-Hit tasks + q8_interm (up to 1,520 tasks of 16 tokens for M <= 2,064)
            //   [6,815,744 .. 9,961,472 floats):     Wave Buffers (2x 6.0 MiB for L2 Pinned up to 385 tasks/wave, or 3x 4.0 MiB for Bounce up to 340 tasks/wave)
            //   [9,961,472 .. 50,331,648 floats):    154.0 MiB Contiguous Staging Pool (d_stage_pool, up to 77 missed experts)
            const bool need_host_bounce_sync = st.has_unpinned_experts || (st.experts_fp != nullptr);
            Qwen4RuntimeState::Qwen4PrefillTaskPack* d_pack_buf[3] = {
                (Qwen4RuntimeState::Qwen4PrefillTaskPack*)(d_scratch + 6815744),
                (Qwen4RuntimeState::Qwen4PrefillTaskPack*)(d_scratch + (need_host_bounce_sync ? 7864320 : 8388608)),
                (Qwen4RuntimeState::Qwen4PrefillTaskPack*)(d_scratch + 8912896)
            };
            void* d_q8_interm_buf[3] = {
                (void*)(d_pack_buf[0] + 1),
                (void*)(d_pack_buf[1] + 1),
                (void*)(d_pack_buf[2] + 1)
            };
            uint8_t* d_stage_pool = (uint8_t*)(d_scratch + 9961472);
            const int max_stage_experts = std::min(80, (int)((161480704ULL) / std::max<size_t>(1, eb)));
            const int max_wave_exp = need_host_bounce_sync ? 24 : 64;
            const int max_wave_tasks = need_host_bounce_sync ? 340 : 350;
            const int n_wave_bufs = need_host_bounce_sync ? 3 : 2;

            // 2) Asynchronous Multi-Wave DMA Pipeline with Contiguous 154 MiB Staging Pool + Unused LRU Overflow:
            auto alloc_prefill_lru_slot = [&](int eid, bool retain_hot) -> uint8_t* {
                if (lw.n_lru_slots <= 0) return nullptr;
                int best_s = -1;
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
                if (best_s < 0) return nullptr;
                const int old_eid = lw.lru_slot_eid[best_s];
                if (old_eid >= 0) {
                    lw.d_vram_exp_ptr[old_eid]     = nullptr;
                    lw.h_vram_exp_table[old_eid]   = nullptr;
                    lw.h_decode_exp_table[old_eid] = lw.d_mapped_exp_ptr[old_eid];
                    lw.exp_lru_slot[old_eid]       = -1;
                }
                uint8_t* slot_vram = lw.d_lru_base + (size_t)best_s * eb;
                lw.d_vram_exp_ptr[eid]         = slot_vram;
                lw.h_vram_exp_table[eid]       = slot_vram;
                lw.h_decode_exp_table[eid]     = slot_vram;
                lw.exp_lru_slot[eid]           = (int8_t)best_s;
                lw.lru_slot_eid[best_s]       = eid;
                lw.lru_slot_last_used[best_s] = retain_hot
                    ? ((++st.lru_clock) + (uint64_t)std::min(exp_cnt[eid], 8))
                    : 1ULL;
                lw.lru_slot_epoch[best_s]     = step_epoch;
                lw.decode_miss_cnt[eid]       = 0;
                return slot_vram;
            };

            bool lru_table_dirty = false;
            int prefill_lru_used = 0;
            const int max_prefill_lru = (M <= 64) ? lw.n_lru_slots : std::min(8, lw.n_lru_slots);
            int miss_cursor = 0;
            int w_idx = 0;
            int stage_used = 0;
            while (miss_cursor < n_unique_misses) {
                const int buf_idx = w_idx % n_wave_bufs;
                const int bounce_base = (w_idx % 3) * 24;
                auto* h_pack      = st.h_prefill_pack + (w_idx & 7);
                auto* d_pack      = d_pack_buf[buf_idx];
                void* d_q8_interm = d_q8_interm_buf[buf_idx];

                if (need_host_bounce_sync && w_idx >= 3) {
                    CUDA_CHECK(cudaEventSynchronize(st.wave_ready_event_dev[dev][(w_idx - 3) & 7]));
                }
                if (w_idx >= n_wave_bufs) {
                    CUDA_CHECK(cudaStreamWaitEvent(copy_stream, st.wave_done_event_dev[dev][buf_idx], 0));
                }
                if (stage_used >= max_stage_experts && w_idx > 0) {
                    CUDA_CHECK(cudaStreamWaitEvent(copy_stream, st.wave_done_event_dev[dev][0], 0));
                    stage_used = 0;
                }

                int wave_exp_cnt = 0;
                int unpinned_task_cnt = 0;
                int unpinned_idx[256];
                uint8_t* wave_dst_vram[256];
                int wave_eids[256];
                int n_miss_tasks = 0;

                while (miss_cursor < n_unique_misses) {
                    if (wave_exp_cnt >= max_wave_exp) break;
                    const int eid = miss_eids[miss_cursor];
                    const int cnt = exp_cnt[eid];
                    const int tiles_needed = (cnt + QWEN4_MOE_TILE_M - 1) / QWEN4_MOE_TILE_M;
                    if (n_miss_tasks + tiles_needed > max_wave_tasks) break;

                    const bool want_lru = (M <= 64 || cnt >= 2 || (is_last_chunk && prefill_lru_used < max_prefill_lru));
                    uint8_t* dst_vram = (want_lru && prefill_lru_used < max_prefill_lru)
                        ? alloc_prefill_lru_slot(eid, true)
                        : nullptr;
                    if (dst_vram != nullptr) {
                        lru_table_dirty = true;
                        prefill_lru_used++;
                    } else if (stage_used < max_stage_experts) {
                        lw.decode_miss_cnt[eid] = 1;
                        dst_vram = d_stage_pool + (size_t)(stage_used++) * eb;
                    } else {
                        dst_vram = alloc_prefill_lru_slot(eid, false);
                        if (dst_vram != nullptr) {
                            lru_table_dirty = true;
                        } else {
                            break;
                        }
                    }

                    const int wi = wave_exp_cnt++;
                    wave_dst_vram[wi] = dst_vram;
                    wave_eids[wi]     = eid;
                    n_miss_tasks     += tiles_needed;
                    miss_cursor++;

                    if (lw.h_pinned_exp_ptr[eid] != nullptr) {
                        st.stat_l2_pinned_hits += (uint64_t)cnt;
                        CUDA_CHECK(cudaMemcpyAsync(
                            dst_vram, lw.h_pinned_exp_ptr[eid], eb, cudaMemcpyHostToDevice, copy_stream));
                    } else if (lw.h_unpinned_exp_ptr[eid] != nullptr) {
                        st.stat_l25_unpinned_hits += (uint64_t)cnt;
                        const int tidx = std::min(unpinned_task_cnt++, 23);
                        unpinned_idx[tidx] = wi;
                        st.copy_tasks[tidx].src    = lw.h_unpinned_exp_ptr[eid];
                        st.copy_tasks[tidx].dst    = st.h_bounce_exp + (size_t)(bounce_base + tidx) * st.max_expert_blob_bytes;
                        st.copy_tasks[tidx].nbytes = eb;
                    } else if (st.experts_fp != nullptr) {
                        st.stat_l3_ssd_reads += (uint64_t)cnt;
                        uint8_t* bounce = st.h_bounce_exp + (size_t)(bounce_base + (wi % 24)) * st.max_expert_blob_bytes;
#if defined(_WIN32)
                        _fseeki64(st.experts_fp, (int64_t)(lw.layer_file_offset + (uint64_t)eid * eb), SEEK_SET);
#else
                        fseeko(st.experts_fp, (off_t)(lw.layer_file_offset + (uint64_t)eid * eb), SEEK_SET);
#endif
                        (void)std::fread(bounce, 1, eb, st.experts_fp);
                        CUDA_CHECK(cudaMemcpyAsync(
                            dst_vram, bounce, eb, cudaMemcpyHostToDevice, copy_stream));
                    }

                    h_pack->gate_ptrs[eid] = dst_vram + g_off;
                    h_pack->up_ptrs[eid]   = dst_vram + u_off;
                    h_pack->down_ptrs[eid] = dst_vram + d_off;
                }

                // While PCIe DMA is actively streaming expert blobs on copy_stream, scatter missed (tok, wt) pairs into compact L1-resident flat_miss_pairs & pack tasks!
                if (!miss_pairs_populated) {
                    miss_pairs_populated = true;
                    for (int m = 0; m < M; ++m) {
                        const int32_t* idx_m = h_top_idx_b + m * 10;
                        const float*   w_m   = h_top_w_b   + m * 10;
                        for (int k = 0; k < 10; ++k) {
                            const int eid = idx_m[k];
                            if ((unsigned)eid < 256u) {
                                const int pos = miss_write[eid];
                                if (pos >= 0) {
                                    miss_write[eid] = pos + 1;
                                    st.flat_miss_pairs[pos] = {m, w_m[k]};
                                }
                            }
                        }
                    }
                }

                int task_cursor = 0;
                for (int wi = 0; wi < wave_exp_cnt; ++wi) {
                    const int eid = wave_eids[wi];
                    const int cnt = exp_cnt[eid];
                    const auto* pairs = st.flat_miss_pairs + miss_base[eid];
                    for (int base = 0; base < cnt; base += QWEN4_MOE_TILE_M) {
                        const int nt = (cnt - base < QWEN4_MOE_TILE_M) ? (cnt - base) : QWEN4_MOE_TILE_M;
                        Qwen4MoETileTask& t = h_pack->tasks[task_cursor++];
                        t.exp_id = eid;
                        t.n_toks = nt;
                        for (int i = 0; i < QWEN4_MOE_TILE_M; ++i) {
                            t.tok_idx[i] = (i < nt) ? pairs[base + i].tok : 0;
                            t.tok_w[i]   = (i < nt) ? pairs[base + i].wt  : 0.0f;
                        }
                    }
                }

                if (unpinned_task_cnt > 0) {
                    const int n_tasks_clamped = std::min(unpinned_task_cnt, 24);
                    (void)cudaStreamQuery(copy_stream);
                    (void)cudaStreamQuery(main_stream_);
                    st.parallel_unpinned_gather(n_tasks_clamped);
                    for (int t = 0; t < n_tasks_clamped; ++t) {
                        const int wi = unpinned_idx[t];
                        CUDA_CHECK(cudaMemcpyAsync(
                            wave_dst_vram[wi],
                            st.h_bounce_exp + (size_t)(bounce_base + t) * st.max_expert_blob_bytes,
                            eb,
                            cudaMemcpyHostToDevice,
                            copy_stream));
                    }
                }

                const size_t pack_bytes = offsetof(Qwen4RuntimeState::Qwen4PrefillTaskPack, tasks)
                                        + (size_t)n_miss_tasks * sizeof(Qwen4MoETileTask);
                CUDA_CHECK(cudaMemcpyAsync(
                    d_pack, h_pack, pack_bytes, cudaMemcpyHostToDevice, copy_stream));
                if (miss_cursor >= n_unique_misses && lru_table_dirty) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        lw.d_vram_exp_table, lw.h_vram_exp_table, 256 * sizeof(uint8_t*), cudaMemcpyHostToDevice, copy_stream));
                    CUDA_CHECK(cudaMemcpyAsync(
                        lw.d_decode_exp_table, lw.h_decode_exp_table, 256 * sizeof(uint8_t*), cudaMemcpyHostToDevice, copy_stream));
                    lru_table_dirty = false;
                }
                CUDA_CHECK(cudaEventRecord(st.wave_ready_event_dev[dev][w_idx & 7], copy_stream));
                CUDA_CHECK(cudaStreamWaitEvent(main_stream_, st.wave_ready_event_dev[dev][w_idx & 7], 0));

                qwen4_routed_experts_tasks_cuda(
                    d_x_q8_1_b,
                    nullptr, 0, lw.moe_gate_exp_type,
                    nullptr, 0, lw.moe_up_exp_type,
                    nullptr, 0, lw.moe_down_exp_type,
                    d_pack->gate_ptrs,
                    d_pack->up_ptrs,
                    d_pack->down_ptrs,
                    d_pack->tasks,
                    nullptr,
                    n_miss_tasks,
                    d_sub_out,
                    d_q8_interm,
                    main_stream_);
                CUDA_CHECK(cudaEventRecord(st.wave_done_event_dev[dev][buf_idx], main_stream_));
                w_idx++;
            }
            st.lru_clock += 16;

            if (lru_table_dirty) {
                CUDA_CHECK(cudaMemcpyAsync(
                    lw.d_vram_exp_table, lw.h_vram_exp_table, 256 * sizeof(uint8_t*), cudaMemcpyHostToDevice, copy_stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    lw.d_decode_exp_table, lw.h_decode_exp_table, 256 * sizeof(uint8_t*), cudaMemcpyHostToDevice, copy_stream));
            }

            if (!defer_moe_gr_write) {
                qwen4_gr_write_batchM_cuda(d_R, d_sub_out, M, main_stream_);
            }
            pre_acc(7);
        }

        // 3. End of chunk: on the final chunk, place the final token's R into d_R_dev[out_dev][0..10239] and optionally compute logits
        const int out_dev = (st.num_gpus > 1) ? 1 : 0;
        if (is_last_chunk) {
            if (M > 1) {
                CUDA_CHECK(cudaMemcpyAsync(
                    st.d_R_dev[out_dev],
                    st.d_R_dev[out_dev] + (size_t)(M - 1) * 10240,
                    10240 * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    main_stream_));
            }
            if (compute_last_logits) {
                void* peer_q8_out = (st.num_gpus > 1 && p2p_enabled_ && st.d_output_dev[1] != nullptr) ? st.d_x_q8_1_dev[0] : nullptr;
                qwen4_gr_read_cuda(
                    st.d_R_dev[out_dev],
                    (const float*)st.d_out_hc_norm,
                    (const uint16_t*)st.d_out_hc_down,
                    (const uint16_t*)st.d_out_hc_up,
                    st.d_x_dev[out_dev],
                    st.d_scratch_dev[out_dev],
                    cfg_.rms_norm_eps,
                    main_stream_,
                    nullptr,
                    st.d_x_q8_1_dev[out_dev],
                    nullptr,
                    false,
                    peer_q8_out);

                if (st.num_gpus > 1 && st.d_output_dev[1] != nullptr) {
                    const int n_rows0 = st.vocab_split;
                    const int n_rows1 = cfg_.vocab_size - n_rows0;
                    CUDA_CHECK(cudaEventRecord(st.logits_ready_event_dev[1], main_stream_));
                    qwen4_project_cuda(
                        st.output_type, st.d_output_dev[1], st.d_x_dev[1], st.d_x_q8_1_dev[1],
                        st.d_logits_dev1 + n_rows0, n_rows1, 2560, main_stream_);
                    if (st.fast_greedy) {
                        qwen4_argmax_f32_cuda(
                            st.d_logits_dev1 + n_rows0, n_rows1,
                            st.greedy_mask_id0, st.greedy_mask_id1,
                            st.d_argmax_out_dev[1], main_stream_, n_rows0);
                    }
                    CUDA_CHECK(cudaEventRecord(st.stage_ready_event_dev[1], main_stream_));

                    set_active_pp_device(0);
                    CUDA_CHECK(cudaStreamWaitEvent(main_stream_, st.logits_ready_event_dev[1], 0));
                    if (st.output_type == 30 || st.output_type == 1 || st.output_type == 0) {
                        CUDA_CHECK(cudaMemcpyPeerAsync(st.d_x_dev[0], 0, st.d_x_dev[1], 1, 2560 * sizeof(float), main_stream_));
                    } else if (!p2p_enabled_) {
                        CUDA_CHECK(cudaMemcpyAsync(st.d_x_q8_1_dev[0], st.d_x_q8_1_dev[1], 2880, cudaMemcpyDefault, main_stream_));
                    }
                    float* d_logits0 = buf_logits_.f32();
                    qwen4_project_cuda(
                        st.output_type, st.d_output_dev[0], st.d_x_dev[0], st.d_x_q8_1_dev[0],
                        d_logits0, n_rows0, 2560, main_stream_);
                    if (st.fast_greedy) {
                        qwen4_argmax_f32_cuda(
                            d_logits0, n_rows0,
                            st.greedy_mask_id0, st.greedy_mask_id1,
                            st.d_argmax_out_dev[0], main_stream_, 0);
                        CUDA_CHECK(cudaEventRecord(st.stage_ready_event_dev[0], main_stream_));
                        st.argmax_ready = true;
                    } else {
                        CUDA_CHECK(cudaStreamWaitEvent(main_stream_, st.stage_ready_event_dev[1], 0));
                        CUDA_CHECK(cudaMemcpyPeerAsync(
                            d_logits0 + n_rows0, 0, st.d_logits_dev1 + n_rows0, 1,
                            (size_t)n_rows1 * sizeof(float), main_stream_));
                        st.argmax_ready = false;
                    }
                } else {
                    float* d_logits_out = (st.num_gpus > 1) ? st.d_logits_dev1 : buf_logits_.f32();
                    qwen4_project_cuda(
                        st.output_type, st.d_output, st.d_x_dev[out_dev], st.d_x_q8_1_dev[out_dev], d_logits_out, cfg_.vocab_size, 2560, main_stream_);

                    if (st.fast_greedy) {
                        qwen4_argmax_f32_cuda(
                            d_logits_out, cfg_.vocab_size, st.greedy_mask_id0, st.greedy_mask_id1, st.d_argmax_out_dev[out_dev], main_stream_);
                        CUDA_CHECK(cudaEventRecord(st.stage_ready_event_dev[out_dev], main_stream_));
                        if (st.num_gpus > 1) {
                            set_active_pp_device(0);
                            CUDA_CHECK(cudaStreamWaitEvent(main_stream_, st.stage_ready_event_dev[1], 0));
                        }
                        st.argmax_ready = true;
                    } else if (st.num_gpus > 1) {
                        CUDA_CHECK(cudaEventRecord(st.stage_ready_event_dev[1], main_stream_));
                        pp_copy_d2d(pp_peer_.buf_logits_.data, 0, st.d_logits_dev1, 1, (size_t)cfg_.vocab_size * sizeof(float));
                        set_active_pp_device(0);
                        st.argmax_ready = false;
                    } else {
                        st.argmax_ready = false;
                    }
                }
            } else if (st.num_gpus > 1) {
                CUDA_CHECK(cudaEventRecord(st.stage_ready_event_dev[1], main_stream_));
                set_active_pp_device(0);
                CUDA_CHECK(cudaStreamWaitEvent(main_stream_, st.stage_ready_event_dev[1], 0));
            }
            pre_acc(9);
        } else if (st.num_gpus > 1) {
            // Intermediate chunk on 2-GPU: record stage_ready_event_dev[1] on GPU 1 and immediately switch to GPU 0
            // so GPU 0 executes Layers 0..23 of chunk (chunk_idx + 1) concurrently with GPU 1 executing Layers 24..47 of chunk_idx!
            CUDA_CHECK(cudaEventRecord(st.stage_ready_event_dev[1], main_stream_));
            set_active_pp_device(0);
            pre_acc(8);
        }

        pos_start += M_chunk;
        chunk_idx++;
    }
    if (st.profile_timing) {
        LOG_INFO("[PREFILL PROFILE toks=%d chunks=%d] PLE=%.1fms | GR_Attn=%.1fms | GDN=%.1fms | QSA=%.1fms | GR_FFN=%.1fms | ShExp=%.1fms | MoE_L1=%.1fms | MoE_Miss=%.1fms | PP=%.1fms | Logits=%.1fms",
                 total_to_prefill, chunk_idx,
                 pre_prof[0], pre_prof[1], pre_prof[2], pre_prof[3], pre_prof[4],
                 pre_prof[5], pre_prof[6], pre_prof[7], pre_prof[8], pre_prof[9]);
    }
}

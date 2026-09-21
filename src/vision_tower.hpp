#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <nlohmann/json.hpp>

#include "cuda/vision_kernels.cuh"
#include "image_loader.hpp"
#include "platform/platform_io.hpp"

namespace moecher::vision {

using json = nlohmann::json;

struct ViTBlockWeights {
    __nv_bfloat16* norm1_w = nullptr;
    __nv_bfloat16* norm1_b = nullptr;
    __nv_bfloat16* qkv_w   = nullptr;
    __nv_bfloat16* qkv_b   = nullptr;
    __nv_bfloat16* proj_w  = nullptr;
    __nv_bfloat16* proj_b  = nullptr;
    __nv_bfloat16* norm2_w = nullptr;
    __nv_bfloat16* norm2_b = nullptr;
    __nv_bfloat16* mlp1_w  = nullptr;
    __nv_bfloat16* mlp1_b  = nullptr;
    __nv_bfloat16* mlp2_w  = nullptr;
    __nv_bfloat16* mlp2_b  = nullptr;
};

struct MergerWeights {
    __nv_bfloat16* norm_w = nullptr;
    __nv_bfloat16* norm_b = nullptr;
    __nv_bfloat16* fc1_w  = nullptr;
    __nv_bfloat16* fc1_b  = nullptr;
    __nv_bfloat16* fc2_w  = nullptr;
    __nv_bfloat16* fc2_b  = nullptr;
};

class QwenVisionTower {
public:
    static constexpr int NUM_BLOCKS = 27;
    static constexpr int EMBED_DIM = 1152;
    static constexpr int NUM_HEADS = 16;
    static constexpr int HEAD_DIM = 72; // 1152 / 16
    static constexpr int MLP_INTERMEDIATE = 4304;
    static constexpr int PATCH_SIZE = 16;
    static constexpr int TEMPORAL_SIZE = 2;
    static constexpr int IMAGE_SIZE = 768;
    static constexpr int NUM_PATCHES = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE); // 2304
    static constexpr int MERGED_PATCHES = (IMAGE_SIZE / (PATCH_SIZE * 2)) * (IMAGE_SIZE / (PATCH_SIZE * 2)); // 576
    static constexpr int MERGER_IN_DIM = 4 * EMBED_DIM; // 4608
    static constexpr int OUTPUT_DIM = 5120; // Matches Qwen 27B language model hidden_size

    QwenVisionTower() = default;
    ~QwenVisionTower() {
        free_all();
    }

    bool is_loaded() const { return is_loaded_; }

    bool load_from_manifest(
        const std::string& dense_path,
        const json& tensor_map,
        cublasHandle_t cublas_handle,
        cudaStream_t stream = 0)
    {
        cublas_handle_ = cublas_handle;
        stream_ = stream;

        moecher::platform::MemoryMappedFile dense_mmap;
        if (!dense_mmap.open_read(dense_path)) {
            std::cerr << "[Vision] Error: cannot open dense bin: " << dense_path << std::endl;
            return false;
        }
        void* mapped = dense_mmap.data();

        auto load_tensor_ptr = [&](__nv_bfloat16*& dev_ptr, const std::string& name) -> bool {
            if (!tensor_map.contains(name)) {
                std::cerr << "[Vision] Error: tensor not in map: " << name << std::endl;
                return false;
            }
            auto& info = tensor_map[name];
            int64_t offset = info["offset"].get<int64_t>();
            int64_t nbytes = info["nbytes"].get<int64_t>();
            cudaError_t err = cudaMalloc(&dev_ptr, nbytes);
            if (err != cudaSuccess) {
                std::cerr << "[Vision] Error: cudaMalloc failed (" << cudaGetErrorString(err) << ") for " << name << std::endl;
                return false;
            }
            allocated_ptrs_.push_back(dev_ptr);
            cudaMemcpyAsync(dev_ptr, (char*)mapped + offset, nbytes, cudaMemcpyHostToDevice, stream_);
            return true;
        };

        // 1. Patch Embed & Pos Embed
        if (!load_tensor_ptr(patch_embed_w_, "model.visual.patch_embed.proj.weight") ||
            !load_tensor_ptr(patch_embed_b_, "model.visual.patch_embed.proj.bias") ||
            !load_tensor_ptr(pos_embed_w_,   "model.visual.pos_embed.weight")) {
            return false;
        }

        // 2. 32 ViT Blocks
        blocks_.resize(NUM_BLOCKS);
        for (int i = 0; i < NUM_BLOCKS; i++) {
            std::string prefix = "model.visual.blocks." + std::to_string(i) + ".";
            auto& b = blocks_[i];
            if (!load_tensor_ptr(b.norm1_w, prefix + "norm1.weight") ||
                !load_tensor_ptr(b.norm1_b, prefix + "norm1.bias") ||
                !load_tensor_ptr(b.qkv_w,   prefix + "attn.qkv.weight") ||
                !load_tensor_ptr(b.qkv_b,   prefix + "attn.qkv.bias") ||
                !load_tensor_ptr(b.proj_w,  prefix + "attn.proj.weight") ||
                !load_tensor_ptr(b.proj_b,  prefix + "attn.proj.bias") ||
                !load_tensor_ptr(b.norm2_w, prefix + "norm2.weight") ||
                !load_tensor_ptr(b.norm2_b, prefix + "norm2.bias") ||
                !load_tensor_ptr(b.mlp1_w,  prefix + "mlp.linear_fc1.weight") ||
                !load_tensor_ptr(b.mlp1_b,  prefix + "mlp.linear_fc1.bias") ||
                !load_tensor_ptr(b.mlp2_w,  prefix + "mlp.linear_fc2.weight") ||
                !load_tensor_ptr(b.mlp2_b,  prefix + "mlp.linear_fc2.bias")) {
                return false;
            }
        }

        // 3. Spatial Merger
        if (!load_tensor_ptr(merger_.norm_w, "model.visual.merger.norm.weight") ||
            !load_tensor_ptr(merger_.norm_b, "model.visual.merger.norm.bias") ||
            !load_tensor_ptr(merger_.fc1_w,  "model.visual.merger.linear_fc1.weight") ||
            !load_tensor_ptr(merger_.fc1_b,  "model.visual.merger.linear_fc1.bias") ||
            !load_tensor_ptr(merger_.fc2_w,  "model.visual.merger.linear_fc2.weight") ||
            !load_tensor_ptr(merger_.fc2_b,  "model.visual.merger.linear_fc2.bias")) {
            return false;
        }

        cudaStreamSynchronize(stream_);

        // 4. Allocate Reusable Working Scratch Buffers
        if (!alloc_scratch_buffers()) {
            return false;
        }

        is_loaded_ = true;
        return true;
    }

    // Runs forward inference on a normalized preprocessed image
    // Returns pointer to device memory of shape [576, 5120] in BF16
    const __nv_bfloat16* forward(const ProcessedImage& img, cudaStream_t stream = 0) {
        if (!is_loaded_) return nullptr;
        cudaStream_t active_stream = stream ? stream : stream_;
        cublasSetStream(cublas_handle_, active_stream);

        // 1. Copy host normalized image to GPU
        size_t img_bytes = img.data.size() * sizeof(float);
        cudaMemcpyAsync(d_img_norm_, img.data.data(), img_bytes, cudaMemcpyHostToDevice, active_stream);

        // 2. Fast Conv3D Patch Embedding via Im2Col + GEMM
        im2col_patch_embed_bf16_cuda(
            d_im2col_, d_img_norm_, IMAGE_SIZE, IMAGE_SIZE, PATCH_SIZE, TEMPORAL_SIZE, 3, active_stream);

        gemm_bf16(d_patches_, NUM_PATCHES, EMBED_DIM, 3 * TEMPORAL_SIZE * PATCH_SIZE * PATCH_SIZE,
                  d_im2col_, patch_embed_w_);

        add_bias_and_posemb_bf16_cuda(
            d_patches_, patch_embed_b_, pos_embed_w_, NUM_PATCHES, EMBED_DIM, active_stream);
        // 3. ViT Blocks
        for (int i = 0; i < NUM_BLOCKS; i++) {
            const auto& b = blocks_[i];

            // LayerNorm 1
            layernorm_affine_bf16_cuda(
                d_norm_out_, d_patches_, b.norm1_w, b.norm1_b, NUM_PATCHES, EMBED_DIM, 1e-6f, active_stream);

            // QKV projection: [NUM_PATCHES, 1152] x [3456, 1152]^T -> [NUM_PATCHES, 3456]
            gemm_bf16(d_qkv_, NUM_PATCHES, 3 * EMBED_DIM, EMBED_DIM, d_norm_out_, b.qkv_w);

            // Split Q, K, V and add bias with 2D RoPE: [16, NUM_PATCHES, 72]
            vit_split_qkv_bias_bf16_cuda(
                d_Q_, d_K_, d_V_, d_qkv_, b.qkv_b, d_cos_, d_sin_, NUM_PATCHES, NUM_HEADS, HEAD_DIM, active_stream);

            // Multi-Head Attention:
            // S = Q @ K^T / sqrt(72) -> [16, NUM_PATCHES, NUM_PATCHES]
            float alpha = 1.0f, beta = 0.0f;
            cublasGemmStridedBatchedEx(
                cublas_handle_,
                CUBLAS_OP_T, CUBLAS_OP_N,
                NUM_PATCHES, NUM_PATCHES, HEAD_DIM,
                &alpha,
                d_K_, CUDA_R_16BF, HEAD_DIM, (long long)NUM_PATCHES * HEAD_DIM,
                d_Q_, CUDA_R_16BF, HEAD_DIM, (long long)NUM_PATCHES * HEAD_DIM,
                &beta,
                d_scores_, CUDA_R_16BF, NUM_PATCHES, (long long)NUM_PATCHES * NUM_PATCHES,
                NUM_HEADS,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

            // Softmax with scale 1.0 / sqrt(72)
            float scale = 1.0f / std::sqrt((float)HEAD_DIM);
            vit_softmax_bf16_cuda(
                d_scores_, NUM_HEADS * NUM_PATCHES, NUM_PATCHES, scale, active_stream);

            // Context = Scores @ V -> [16, NUM_PATCHES, 72]
            cublasGemmStridedBatchedEx(
                cublas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                HEAD_DIM, NUM_PATCHES, NUM_PATCHES,
                &alpha,
                d_V_, CUDA_R_16BF, HEAD_DIM, (long long)NUM_PATCHES * HEAD_DIM,
                d_scores_, CUDA_R_16BF, NUM_PATCHES, (long long)NUM_PATCHES * NUM_PATCHES,
                &beta,
                d_attn_ctx_, CUDA_R_16BF, HEAD_DIM, (long long)NUM_PATCHES * HEAD_DIM,
                NUM_HEADS,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

            // Merge heads: [16, NUM_PATCHES, 72] -> [NUM_PATCHES, 1152]
            vit_merge_heads_bf16_cuda(
                d_attn_merged_, d_attn_ctx_, NUM_PATCHES, NUM_HEADS, HEAD_DIM, active_stream);

            // Attention out projection: [NUM_PATCHES, 1152] x [1152, 1152]^T -> [NUM_PATCHES, 1152]
            gemm_bf16(d_attn_proj_, NUM_PATCHES, EMBED_DIM, EMBED_DIM, d_attn_merged_, b.proj_w);

            // Residual add + bias: d_patches_ = d_patches_ + d_attn_proj_ + b.proj_b
            add_residual_bias_bf16_cuda(
                d_patches_, d_attn_proj_, b.proj_b, NUM_PATCHES, EMBED_DIM, active_stream);

            // LayerNorm 2
            layernorm_affine_bf16_cuda(
                d_norm_out_, d_patches_, b.norm2_w, b.norm2_b, NUM_PATCHES, EMBED_DIM, 1e-6f, active_stream);

            // MLP FC1: [NUM_PATCHES, 1152] x [4304, 1152]^T -> [NUM_PATCHES, 4304]
            gemm_bf16(d_mlp_fc1_, NUM_PATCHES, MLP_INTERMEDIATE, EMBED_DIM, d_norm_out_, b.mlp1_w);

            // GELU + bias
            gelu_bias_bf16_cuda(
                d_mlp_fc1_, d_mlp_fc1_, b.mlp1_b, NUM_PATCHES, MLP_INTERMEDIATE, active_stream);

            // MLP FC2: [NUM_PATCHES, 4304] x [1152, 4304]^T -> [NUM_PATCHES, 1152]
            gemm_bf16(d_mlp_fc2_, NUM_PATCHES, EMBED_DIM, MLP_INTERMEDIATE, d_mlp_fc1_, b.mlp2_w);

            // Residual add + bias: d_patches_ = d_patches_ + d_mlp_fc2_ + b.mlp2_b
            add_residual_bias_bf16_cuda(
                d_patches_, d_mlp_fc2_, b.mlp2_b, NUM_PATCHES, EMBED_DIM, active_stream);
        }

        // 4. Spatial Merger
        // LayerNorm
        layernorm_affine_bf16_cuda(
            d_norm_out_, d_patches_, merger_.norm_w, merger_.norm_b, NUM_PATCHES, EMBED_DIM, 1e-6f, active_stream);

        // Gather 2x2 patches: [48, 48, 1152] -> [576, 4608]
        spatial_merge_gather_bf16_cuda(
            d_merge_in_, d_norm_out_, 48, 48, EMBED_DIM, active_stream);

        // Merger FC1: [576, 4608] x [4608, 4608]^T -> [576, 4608]
        gemm_bf16(d_merge_fc1_, MERGED_PATCHES, MERGER_IN_DIM, MERGER_IN_DIM, d_merge_in_, merger_.fc1_w);

        // GELU + bias
        gelu_bias_bf16_cuda(
            d_merge_fc1_, d_merge_fc1_, merger_.fc1_b, MERGED_PATCHES, MERGER_IN_DIM, active_stream);

        // Merger FC2: [576, 4608] x [5120, 4608]^T -> [576, 5120]
        gemm_bf16(d_visual_out_, MERGED_PATCHES, OUTPUT_DIM, MERGER_IN_DIM, d_merge_fc1_, merger_.fc2_w);

        // Add FC2 bias: [576, 5120]
        add_bias_bf16_cuda(
            d_visual_out_, merger_.fc2_b, MERGED_PATCHES, OUTPUT_DIM, active_stream);

        return d_visual_out_;
    }

private:
    void gemm_bf16(
        __nv_bfloat16* C, int M, int N, int K,
        const __nv_bfloat16* A, const __nv_bfloat16* B)
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(
            cublas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, CUDA_R_16BF, K,
            A, CUDA_R_16BF, K,
            &beta,
            C, CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
    }

    bool alloc_scratch_buffers() {
        size_t img_bytes = (size_t)TEMPORAL_SIZE * 3 * IMAGE_SIZE * IMAGE_SIZE * sizeof(float);
        if (cudaMalloc(&d_img_norm_, img_bytes) != cudaSuccess) return false;
        allocated_ptrs_.push_back(d_img_norm_);

        auto alloc_bf16 = [&](__nv_bfloat16*& ptr, size_t elements) -> bool {
            if (cudaMalloc(&ptr, elements * sizeof(__nv_bfloat16)) != cudaSuccess) return false;
            allocated_ptrs_.push_back(ptr);
            return true;
        };

        if (!alloc_bf16(d_im2col_, (size_t)NUM_PATCHES * (3 * TEMPORAL_SIZE * PATCH_SIZE * PATCH_SIZE)) ||
            !alloc_bf16(d_patches_, (size_t)NUM_PATCHES * EMBED_DIM) ||
            !alloc_bf16(d_norm_out_, (size_t)NUM_PATCHES * EMBED_DIM) ||
            !alloc_bf16(d_qkv_, (size_t)NUM_PATCHES * (3 * EMBED_DIM)) ||
            !alloc_bf16(d_Q_, (size_t)NUM_HEADS * NUM_PATCHES * HEAD_DIM) ||
            !alloc_bf16(d_K_, (size_t)NUM_HEADS * NUM_PATCHES * HEAD_DIM) ||
            !alloc_bf16(d_V_, (size_t)NUM_HEADS * NUM_PATCHES * HEAD_DIM) ||
            !alloc_bf16(d_scores_, (size_t)NUM_HEADS * NUM_PATCHES * NUM_PATCHES) ||
            !alloc_bf16(d_attn_ctx_, (size_t)NUM_HEADS * NUM_PATCHES * HEAD_DIM) ||
            !alloc_bf16(d_attn_merged_, (size_t)NUM_PATCHES * EMBED_DIM) ||
            !alloc_bf16(d_attn_proj_, (size_t)NUM_PATCHES * EMBED_DIM) ||
            !alloc_bf16(d_mlp_fc1_, (size_t)NUM_PATCHES * MLP_INTERMEDIATE) ||
            !alloc_bf16(d_mlp_fc2_, (size_t)NUM_PATCHES * EMBED_DIM) ||
            !alloc_bf16(d_merge_in_, (size_t)MERGED_PATCHES * MERGER_IN_DIM) ||
            !alloc_bf16(d_merge_fc1_, (size_t)MERGED_PATCHES * MERGER_IN_DIM) ||
            !alloc_bf16(d_visual_out_, (size_t)MERGED_PATCHES * OUTPUT_DIM) ||
            !init_rotary_tables()) {
            return false;
        }

        return true;
    }

    bool init_rotary_tables() {
        size_t total_elements = (size_t)NUM_PATCHES * HEAD_DIM; // 2304 * 72 = 165888
        size_t bytes = total_elements * sizeof(float);
        if (cudaMalloc(&d_cos_, bytes) != cudaSuccess) return false;
        if (cudaMalloc(&d_sin_, bytes) != cudaSuccess) return false;
        allocated_ptrs_.push_back(d_cos_);
        allocated_ptrs_.push_back(d_sin_);

        std::vector<float> h_cos(total_elements);
        std::vector<float> h_sin(total_elements);

        // Precompute inv_freq for head_dim=72: dim = 36 -> 18 frequencies
        float inv_freq[18];
        for (int i = 0; i < 18; i++) {
            inv_freq[i] = 1.0f / std::pow(10000.0f, (float)i / 18.0f);
        }

        for (int p = 0; p < NUM_PATCHES; p++) {
            int block_idx = p / 4;
            int my = block_idx / 24;
            int mx = block_idx % 24;
            int sub = p % 4;
            int py = my * 2 + (sub / 2);
            int px = mx * 2 + (sub % 2);

            float emb[72];
            for (int i = 0; i < 18; i++) {
                float v_h = (float)py * inv_freq[i];
                float v_w = (float)px * inv_freq[i];
                emb[i]      = v_h;
                emb[18 + i] = v_w;
                emb[36 + i] = v_h;
                emb[54 + i] = v_w;
            }

            for (int d = 0; d < 72; d++) {
                h_cos[(size_t)p * 72 + d] = std::cos(emb[d]);
                h_sin[(size_t)p * 72 + d] = std::sin(emb[d]);
            }
        }

        cudaMemcpy(d_cos_, h_cos.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sin_, h_sin.data(), bytes, cudaMemcpyHostToDevice);
        return true;
    }

    void free_all() {
        for (void* p : allocated_ptrs_) {
            if (p) cudaFree(p);
        }
        allocated_ptrs_.clear();
        is_loaded_ = false;
    }

    bool is_loaded_ = false;
    cublasHandle_t cublas_handle_ = nullptr;
    cudaStream_t stream_ = 0;
    std::vector<void*> allocated_ptrs_;

    // Weights
    __nv_bfloat16* patch_embed_w_ = nullptr;
    __nv_bfloat16* patch_embed_b_ = nullptr;
    __nv_bfloat16* pos_embed_w_   = nullptr;
    std::vector<ViTBlockWeights> blocks_;
    MergerWeights merger_;

    // Working Buffers
    float* d_img_norm_ = nullptr;
    __nv_bfloat16* d_im2col_ = nullptr;
    __nv_bfloat16* d_patches_ = nullptr;
    __nv_bfloat16* d_norm_out_ = nullptr;
    __nv_bfloat16* d_qkv_ = nullptr;
    __nv_bfloat16* d_Q_ = nullptr;
    __nv_bfloat16* d_K_ = nullptr;
    __nv_bfloat16* d_V_ = nullptr;
    __nv_bfloat16* d_scores_ = nullptr;
    __nv_bfloat16* d_attn_ctx_ = nullptr;
    __nv_bfloat16* d_attn_merged_ = nullptr;
    __nv_bfloat16* d_attn_proj_ = nullptr;
    __nv_bfloat16* d_mlp_fc1_ = nullptr;
    __nv_bfloat16* d_mlp_fc2_ = nullptr;
    __nv_bfloat16* d_merge_in_ = nullptr;
    __nv_bfloat16* d_merge_fc1_ = nullptr;
    __nv_bfloat16* d_visual_out_ = nullptr;
    float* d_cos_ = nullptr;
    float* d_sin_ = nullptr;
};

} // namespace moecher::vision

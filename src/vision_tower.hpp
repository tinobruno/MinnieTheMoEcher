#pragma once
#if defined(__APPLE__)
#include "metal/metal_backend.h"
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#endif
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <fstream>
#include <cmath>
#include <nlohmann/json.hpp>

#include "cuda/vision_kernels.cuh"
#include "cuda/activations.cuh"
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
    // Qwen 3.8 GELU MLP
    __nv_bfloat16* mlp1_w  = nullptr;
    __nv_bfloat16* mlp1_b  = nullptr;
    __nv_bfloat16* mlp2_w  = nullptr;
    __nv_bfloat16* mlp2_b  = nullptr;
    // Qwen 2.5 SwiGLU MLP
    __nv_bfloat16* gate_w  = nullptr;
    __nv_bfloat16* gate_b  = nullptr;
    __nv_bfloat16* up_w    = nullptr;
    __nv_bfloat16* up_b    = nullptr;
    __nv_bfloat16* down_w  = nullptr;
    __nv_bfloat16* down_b  = nullptr;
};

struct MergerWeights {
    // Qwen 3.8 Merger
    __nv_bfloat16* norm_w = nullptr;
    __nv_bfloat16* norm_b = nullptr;
    __nv_bfloat16* fc1_w  = nullptr;
    __nv_bfloat16* fc1_b  = nullptr;
    __nv_bfloat16* fc2_w  = nullptr;
    __nv_bfloat16* fc2_b  = nullptr;
    // Qwen 2.5 Merger
    __nv_bfloat16* ln_q_w = nullptr;
    __nv_bfloat16* mlp0_w = nullptr;
    __nv_bfloat16* mlp0_b = nullptr;
    __nv_bfloat16* mlp2_w = nullptr;
    __nv_bfloat16* mlp2_b = nullptr;
};

class QwenVisionTower {
public:
    bool is_qwen25_ = false;
    int num_blocks_ = 27;
    int embed_dim_ = 1152;
    int num_heads_ = 16;
    int head_dim_ = 72;
    int mlp_intermediate_ = 4304;
    int patch_size_ = 16;
    int temporal_size_ = 2;
    int image_size_ = 768;
    int num_patches_ = 2304;
    int merged_patches_ = 576;
    int merger_in_dim_ = 4608;
    int output_dim_ = 5120;

    int output_dim() const { return output_dim_; }
    bool is_qwen25() const { return is_qwen25_; }

    QwenVisionTower() = default;
    ~QwenVisionTower() {
        free_all();
    }

    bool is_loaded() const { return is_loaded_; }
    const __nv_bfloat16* merge_fc1_output() const { return d_merge_fc1_; }

    bool alloc_vram_or_managed(void*& ptr, size_t bytes) {
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            cudaGetLastError();
            err = cudaMallocManaged(&ptr, bytes);
            if (err != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
        }
        allocated_ptrs_.push_back(ptr);
        return true;
    }

    bool load_from_mmproj_gguf(
        const std::string& gguf_path,
        cublasHandle_t cublas_handle,
        cudaStream_t stream = 0,
        int output_dim = 2560)
    {
        cublas_handle_ = cublas_handle;
        stream_ = stream;

        is_qwen25_ = false;
        num_blocks_ = 27;
        embed_dim_ = 1152;
        num_heads_ = 16;
        head_dim_ = 72;
        mlp_intermediate_ = 4304;
        patch_size_ = 16;
        temporal_size_ = 2;
        image_size_ = 768;
        num_patches_ = 2304;
        merged_patches_ = 576;
        merger_in_dim_ = 4608;
        output_dim_ = (output_dim > 0) ? output_dim : 2560;

        moecher::platform::MemoryMappedFile mm_mmap;
        if (!mm_mmap.open_read(gguf_path)) {
            std::cerr << "[Vision] Error: failed to open mmproj GGUF: " << gguf_path << std::endl;
            return false;
        }
        const uint8_t* base = static_cast<const uint8_t*>(mm_mmap.data());
        const size_t file_size = mm_mmap.size();
        if (file_size < 24 || std::memcmp(base, "GGUF", 4) != 0) {
            std::cerr << "[Vision] Error: invalid GGUF magic in " << gguf_path << std::endl;
            return false;
        }

        size_t pos = 4;
        auto rd_u32 = [&]() -> uint32_t { uint32_t v; std::memcpy(&v, base + pos, 4); pos += 4; return v; };
        auto rd_u64 = [&]() -> uint64_t { uint64_t v; std::memcpy(&v, base + pos, 8); pos += 8; return v; };
        auto rd_str = [&]() -> std::string {
            uint64_t len = rd_u64();
            std::string s(reinterpret_cast<const char*>(base + pos), (size_t)len);
            pos += (size_t)len;
            return s;
        };
        auto skip_val = [&](auto& self, uint32_t vtype) -> void {
            switch (vtype) {
                case 0: case 1: case 7: pos += 1; break;
                case 2: case 3:         pos += 2; break;
                case 4: case 5: case 6: pos += 4; break;
                case 10: case 11: case 12: pos += 8; break;
                case 8: { uint64_t len = rd_u64(); pos += (size_t)len; break; }
                case 9: {
                    uint32_t etype = rd_u32();
                    uint64_t cnt   = rd_u64();
                    for (uint64_t i = 0; i < cnt; ++i) self(self, etype);
                    break;
                }
                default: break;
            }
        };

        (void)rd_u32(); // version
        uint64_t n_tensors = rd_u64();
        uint64_t n_kv      = rd_u64();
        size_t alignment   = 32;

        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string k = rd_str();
            uint32_t vt = rd_u32();
            if (k == "general.alignment" && vt == 4) {
                alignment = rd_u32();
            } else {
                skip_val(skip_val, vt);
            }
        }

        struct GgufTensorMeta {
            uint64_t n_elems = 1;
            uint32_t ggml_type = 0;
            uint64_t rel_offset = 0;
            const uint8_t* ptr = nullptr;
        };
        std::unordered_map<std::string, GgufTensorMeta> tmap;
        tmap.reserve((size_t)n_tensors);

        for (uint64_t i = 0; i < n_tensors; ++i) {
            std::string name = rd_str();
            uint32_t n_dims  = rd_u32();
            uint64_t ne = 1;
            for (uint32_t d = 0; d < n_dims; ++d) {
                ne *= rd_u64();
            }
            uint32_t gtype = rd_u32();
            uint64_t roff  = rd_u64();
            tmap[name] = GgufTensorMeta{ne, gtype, roff, nullptr};
        }

        const size_t data_offset = (pos + alignment - 1) & ~(alignment - 1);
        for (auto& kv : tmap) {
            kv.second.ptr = base + data_offset + kv.second.rel_offset;
        }

        std::vector<__nv_bfloat16> host_conv_buf;
        auto load_gguf_bf16 = [&](__nv_bfloat16*& dev_ptr, const std::string& name) -> bool {
            auto it = tmap.find(name);
            if (it == tmap.end()) {
                std::cerr << "[Vision] Error: GGUF tensor missing: " << name << std::endl;
                return false;
            }
            const auto& tm = it->second;
            const size_t out_bytes = (size_t)tm.n_elems * sizeof(__nv_bfloat16);
            void* raw_dev = nullptr;
            if (!alloc_vram_or_managed(raw_dev, out_bytes)) {
                std::cerr << "[Vision] Error: VRAM allocation failed for " << name << std::endl;
                return false;
            }
            dev_ptr = static_cast<__nv_bfloat16*>(raw_dev);

            if (tm.ggml_type == 30) { // BF16
                cudaMemcpyAsync(dev_ptr, tm.ptr, out_bytes, cudaMemcpyHostToDevice, stream_);
            } else if (tm.ggml_type == 0) { // F32 -> BF16
                host_conv_buf.resize((size_t)tm.n_elems);
                const float* src_f32 = reinterpret_cast<const float*>(tm.ptr);
                for (size_t i = 0; i < (size_t)tm.n_elems; ++i) {
                    host_conv_buf[i] = __float2bfloat16(src_f32[i]);
                }
                cudaMemcpyAsync(dev_ptr, host_conv_buf.data(), out_bytes, cudaMemcpyHostToDevice, stream_);
                cudaStreamSynchronize(stream_);
            } else {
                std::cerr << "[Vision] Error: unsupported ggml_type=" << tm.ggml_type << " for " << name << std::endl;
                return false;
            }
            return true;
        };

        // 1. Interleave v.patch_embd.weight (t=0) and v.patch_embd.weight.1 (t=1) into [1152, 3, 2, 16, 16] BF16
        {
            auto it0 = tmap.find("v.patch_embd.weight");
            auto it1 = tmap.find("v.patch_embd.weight.1");
            if (it0 == tmap.end() || it1 == tmap.end()) {
                std::cerr << "[Vision] Error: missing v.patch_embd.weight / v.patch_embd.weight.1 in GGUF" << std::endl;
                return false;
            }
            const size_t total_elems = 1152ULL * 3 * 2 * 16 * 16; // 1,769,472
            host_conv_buf.resize(total_elems);
            const float* w0_f32 = reinterpret_cast<const float*>(it0->second.ptr);
            const float* w1_f32 = reinterpret_cast<const float*>(it1->second.ptr);
            const __nv_bfloat16* w0_bf16 = reinterpret_cast<const __nv_bfloat16*>(it0->second.ptr);
            const __nv_bfloat16* w1_bf16 = reinterpret_cast<const __nv_bfloat16*>(it1->second.ptr);
            const bool is_f32 = (it0->second.ggml_type == 0);

            for (size_t oc = 0; oc < 1152; ++oc) {
                for (size_t c = 0; c < 3; ++c) {
                    const size_t src_off = (oc * 3 + c) * 256;
                    const size_t dst_t0  = ((oc * 3 + c) * 2 + 0) * 256;
                    const size_t dst_t1  = ((oc * 3 + c) * 2 + 1) * 256;
                    for (size_t sp = 0; sp < 256; ++sp) {
                        host_conv_buf[dst_t0 + sp] = is_f32 ? __float2bfloat16(w0_f32[src_off + sp]) : w0_bf16[src_off + sp];
                        host_conv_buf[dst_t1 + sp] = is_f32 ? __float2bfloat16(w1_f32[src_off + sp]) : w1_bf16[src_off + sp];
                    }
                }
            }
            void* raw_pe = nullptr;
            if (!alloc_vram_or_managed(raw_pe, total_elems * sizeof(__nv_bfloat16))) return false;
            patch_embed_w_ = static_cast<__nv_bfloat16*>(raw_pe);
            cudaMemcpyAsync(patch_embed_w_, host_conv_buf.data(), total_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream_);
            cudaStreamSynchronize(stream_);
        }

        if (!load_gguf_bf16(patch_embed_b_, "v.patch_embd.bias") ||
            !load_gguf_bf16(pos_embed_w_,   "v.position_embd.weight")) {
            return false;
        }

        // 2. 27 ViT Blocks
        blocks_.resize(num_blocks_);
        for (int i = 0; i < num_blocks_; ++i) {
            std::string pfx = "v.blk." + std::to_string(i) + ".";
            auto& b = blocks_[i];
            if (!load_gguf_bf16(b.norm1_w, pfx + "ln1.weight") ||
                !load_gguf_bf16(b.norm1_b, pfx + "ln1.bias") ||
                !load_gguf_bf16(b.qkv_w,   pfx + "attn_qkv.weight") ||
                !load_gguf_bf16(b.qkv_b,   pfx + "attn_qkv.bias") ||
                !load_gguf_bf16(b.proj_w,  pfx + "attn_out.weight") ||
                !load_gguf_bf16(b.proj_b,  pfx + "attn_out.bias") ||
                !load_gguf_bf16(b.norm2_w, pfx + "ln2.weight") ||
                !load_gguf_bf16(b.norm2_b, pfx + "ln2.bias") ||
                !load_gguf_bf16(b.mlp1_w,  pfx + "ffn_up.weight") ||
                !load_gguf_bf16(b.mlp1_b,  pfx + "ffn_up.bias") ||
                !load_gguf_bf16(b.mlp2_w,  pfx + "ffn_down.weight") ||
                !load_gguf_bf16(b.mlp2_b,  pfx + "ffn_down.bias")) {
                return false;
            }
        }

        // 3. Spatial Merger (4608 -> 4608 -> 2560)
        if (!load_gguf_bf16(merger_.norm_w, "v.post_ln.weight") ||
            !load_gguf_bf16(merger_.norm_b, "v.post_ln.bias") ||
            !load_gguf_bf16(merger_.fc1_w,  "mm.0.weight") ||
            !load_gguf_bf16(merger_.fc1_b,  "mm.0.bias") ||
            !load_gguf_bf16(merger_.fc2_w,  "mm.2.weight") ||
            !load_gguf_bf16(merger_.fc2_b,  "mm.2.bias")) {
            return false;
        }

        cudaStreamSynchronize(stream_);
        if (!init_persistent_buffers()) {
            return false;
        }

        is_loaded_ = true;
        return true;
    }

    bool load_from_manifest(
        const std::string& dense_path,
        const json& tensor_map,
        cublasHandle_t cublas_handle,
        cudaStream_t stream = 0,
        int output_dim = 5120,
        const std::string& vision_bin = "",
        const std::string& bridge_bin = "",
        const std::string& embed_mean_bin = "")
    {
        cublas_handle_ = cublas_handle;
        stream_ = stream;

        // Auto-detect Qwen 2.5 ViT vs Qwen 3.8 ViT
        is_qwen25_ = tensor_map.contains("model.visual.merger.mlp.0.weight");
        if (is_qwen25_) {
            num_blocks_ = 32;
            embed_dim_ = 1280;
            num_heads_ = 16;
            head_dim_ = 80;
            mlp_intermediate_ = 3420;
            patch_size_ = 14;
            temporal_size_ = 2;
            image_size_ = 672;
            num_patches_ = (image_size_ / patch_size_) * (image_size_ / patch_size_); // 48 * 48 = 2304
            merged_patches_ = 24 * 24; // 576
            merger_in_dim_ = 4 * embed_dim_; // 5120
            output_dim_ = (output_dim > 0 && output_dim != 5120) ? output_dim : 2048;
            std::cout << "[Vision] Detected Qwen2.5-VL ViT architecture (32 blocks, embed=1280, head=80, out=" << output_dim_ << ")" << std::endl;
        } else {
            num_blocks_ = 27;
            embed_dim_ = 1152;
            num_heads_ = 16;
            head_dim_ = 72;
            mlp_intermediate_ = 4304;
            patch_size_ = 16;
            temporal_size_ = 2;
            image_size_ = 768;
            num_patches_ = 2304;
            merged_patches_ = 576;
            merger_in_dim_ = 4608;
            if (output_dim > 0) output_dim_ = output_dim;
            std::cout << "[Vision] Detected Qwen 3.8 ViT architecture (27 blocks, embed=1152, head=72, out=" << output_dim_ << ")" << std::endl;
        }

        moecher::platform::MemoryMappedFile dense_mmap;
        void* dense_mapped = nullptr;
        if (!dense_path.empty() && dense_mmap.open_read(dense_path)) {
            dense_mapped = dense_mmap.data();
        }

        moecher::platform::MemoryMappedFile vision_mmap;
        void* vision_mapped = nullptr;
        if (!vision_bin.empty() && vision_mmap.open_read(vision_bin)) {
            vision_mapped = vision_mmap.data();
        }

        moecher::platform::MemoryMappedFile bridge_mmap;
        void* bridge_mapped = nullptr;
        if (!bridge_bin.empty() && bridge_mmap.open_read(bridge_bin)) {
            bridge_mapped = bridge_mmap.data();
        }

        auto load_tensor_ptr = [&](__nv_bfloat16*& dev_ptr, const std::string& name) -> bool {
            if (!tensor_map.contains(name)) {
                std::cerr << "[Vision] Error: tensor not in map: " << name << std::endl;
                return false;
            }
            auto& info = tensor_map[name];
            int64_t offset = info["offset"].get<int64_t>();
            int64_t nbytes = info["nbytes"].get<int64_t>();

            std::string file_source = info.value("file", "");
            void* source_mapped = dense_mapped;
            if (file_source == "vision" && vision_mapped) {
                source_mapped = vision_mapped;
            } else if (file_source == "bridge" && bridge_mapped) {
                source_mapped = bridge_mapped;
            } else if (source_mapped == nullptr && vision_mapped) {
                source_mapped = vision_mapped;
            }

            if (!source_mapped) {
                std::cerr << "[Vision] Error: no valid memory mapping for tensor " << name << " (file_source=" << file_source << ")" << std::endl;
                return false;
            }

            cudaError_t err = cudaMalloc(&dev_ptr, nbytes);
            if (err != cudaSuccess) {
                std::cerr << "[Vision] Error: cudaMalloc failed (" << cudaGetErrorString(err) << ") for " << name << std::endl;
                return false;
            }
            allocated_ptrs_.push_back(dev_ptr);
            cudaMemcpyAsync(dev_ptr, (char*)source_mapped + offset, nbytes, cudaMemcpyHostToDevice, stream_);
            return true;
        };

        if (is_qwen25_) {
            // 1. Qwen 2.5 Patch Embed
            if (!load_tensor_ptr(patch_embed_w_, "model.visual.patch_embed.proj.weight")) {
                return false;
            }

            // 2. 32 ViT Blocks (SwiGLU)
            blocks_.resize(num_blocks_);
            for (int i = 0; i < num_blocks_; i++) {
                std::string prefix = "model.visual.blocks." + std::to_string(i) + ".";
                auto& b = blocks_[i];
                if (!load_tensor_ptr(b.norm1_w, prefix + "norm1.weight") ||
                    !load_tensor_ptr(b.qkv_w,   prefix + "attn.qkv.weight") ||
                    !load_tensor_ptr(b.qkv_b,   prefix + "attn.qkv.bias") ||
                    !load_tensor_ptr(b.proj_w,  prefix + "attn.proj.weight") ||
                    !load_tensor_ptr(b.proj_b,  prefix + "attn.proj.bias") ||
                    !load_tensor_ptr(b.norm2_w, prefix + "norm2.weight") ||
                    !load_tensor_ptr(b.gate_w,  prefix + "mlp.gate_proj.weight") ||
                    !load_tensor_ptr(b.gate_b,  prefix + "mlp.gate_proj.bias") ||
                    !load_tensor_ptr(b.up_w,    prefix + "mlp.up_proj.weight") ||
                    !load_tensor_ptr(b.up_b,    prefix + "mlp.up_proj.bias") ||
                    !load_tensor_ptr(b.down_w,  prefix + "mlp.down_proj.weight") ||
                    !load_tensor_ptr(b.down_b,  prefix + "mlp.down_proj.bias")) {
                    return false;
                }
            }

            // 3. Qwen 2.5 Spatial Merger
            if (!load_tensor_ptr(merger_.ln_q_w, "model.visual.merger.ln_q.weight") ||
                !load_tensor_ptr(merger_.mlp0_w, "model.visual.merger.mlp.0.weight") ||
                !load_tensor_ptr(merger_.mlp0_b, "model.visual.merger.mlp.0.bias") ||
                !load_tensor_ptr(merger_.mlp2_w, "model.visual.merger.mlp.2.weight") ||
                !load_tensor_ptr(merger_.mlp2_b, "model.visual.merger.mlp.2.bias")) {
                return false;
            }
        } else {
            // 1. Qwen 3.8 Patch Embed & Pos Embed
            if (!load_tensor_ptr(patch_embed_w_, "model.visual.patch_embed.proj.weight") ||
                !load_tensor_ptr(patch_embed_b_, "model.visual.patch_embed.proj.bias") ||
                !load_tensor_ptr(pos_embed_w_,   "model.visual.pos_embed.weight")) {
                return false;
            }

            // 2. 27 ViT Blocks (GELU)
            blocks_.resize(num_blocks_);
            for (int i = 0; i < num_blocks_; i++) {
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

            // 3. Qwen 3.8 Spatial Merger
            if (!load_tensor_ptr(merger_.norm_w, "model.visual.merger.norm.weight") ||
                !load_tensor_ptr(merger_.norm_b, "model.visual.merger.norm.bias") ||
                !load_tensor_ptr(merger_.fc1_w,  "model.visual.merger.linear_fc1.weight") ||
                !load_tensor_ptr(merger_.fc1_b,  "model.visual.merger.linear_fc1.bias") ||
                !load_tensor_ptr(merger_.fc2_w,  "model.visual.merger.linear_fc2.weight") ||
                !load_tensor_ptr(merger_.fc2_b,  "model.visual.merger.linear_fc2.bias")) {
                return false;
            }
        }

        cudaStreamSynchronize(stream_);

        // 4. Load optional DeepSeek text manifold centroid
        if (!embed_mean_bin.empty()) {
            std::ifstream fin(embed_mean_bin, std::ios::binary);
            if (fin.good()) {
                size_t bytes = (size_t)output_dim_ * sizeof(__nv_bfloat16);
                std::vector<char> buf(bytes);
                fin.read(buf.data(), bytes);
                if (fin.gcount() == (std::streamsize)bytes) {
                    if (cudaMalloc(&d_embed_mean_, bytes) == cudaSuccess) {
                        cudaMemcpy(d_embed_mean_, buf.data(), bytes, cudaMemcpyHostToDevice);
                        allocated_ptrs_.push_back(d_embed_mean_);
                        std::cout << "[Vision] Loaded DeepSeek embedding centroid from " << embed_mean_bin << " (" << bytes << " bytes)" << std::endl;
                    }
                }
            }
        }

        // 5. Allocate Persistent Working Buffers (RoPE tables, window indices, output buffer)
        if (!init_persistent_buffers()) {
            return false;
        }

        is_loaded_ = true;
        return true;
    }

    // Runs forward inference on a normalized preprocessed image
    // Returns pointer to device memory of shape [576, output_dim_] in BF16
    const __nv_bfloat16* forward(const ProcessedImage& img, cudaStream_t stream = 0) {
        if (!is_loaded_) return nullptr;
        cudaStream_t active_stream = stream ? stream : stream_;
        cublasSetStream(cublas_handle_, active_stream);

        // Allocate transient scratch buffers (~265 MB) on-demand for the ViT forward pass
        if (!alloc_scratch_buffers()) {
            std::cerr << "[Vision] Error: Failed to allocate on-demand ViT scratch buffers!" << std::endl;
            return nullptr;
        }

        // 1. Copy host normalized image to GPU
        size_t img_bytes = img.data.size() * sizeof(float);
        cudaMemcpyAsync(d_img_norm_, img.data.data(), img_bytes, cudaMemcpyHostToDevice, active_stream);

        if (is_qwen25_) {
            // 2. Fast Conv3D Patch Embedding via Im2Col + GEMM
            im2col_patch_embed_bf16_cuda(
                d_im2col_, d_img_norm_, image_size_, image_size_, patch_size_, temporal_size_, 3, active_stream);

            int filter_dim = 3 * temporal_size_ * patch_size_ * patch_size_; // 1176
            gemm_bf16(d_patches_, num_patches_, embed_dim_, filter_dim, d_im2col_, patch_embed_w_);

            // Permute patches into window order: [576, 4, 1280] -> permuted by window_index
            permute_patches_by_window_bf16_cuda(
                d_patches_win_, d_patches_, d_window_index_, merged_patches_, 4, embed_dim_, active_stream);
            cudaMemcpyAsync(d_patches_, d_patches_win_, (size_t)num_patches_ * embed_dim_ * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToDevice, active_stream);

            // 3. 32 ViT Blocks (Window Attention on 28 blocks, Full Attention on blocks [7, 15, 23, 31])
            float alpha = 1.0f, beta = 0.0f;
            float scale = 1.0f / std::sqrt((float)head_dim_);

            for (int i = 0; i < num_blocks_; i++) {
                const auto& b = blocks_[i];

                // RMSNorm 1
                rms_norm_cuda_batched(d_norm_out_, d_patches_, b.norm1_w, num_patches_, embed_dim_, 1e-6f, active_stream);

                // QKV projection: [num_patches_, 1280] x [3840, 1280]^T -> [num_patches_, 3840]
                gemm_bf16(d_qkv_, num_patches_, 3 * embed_dim_, embed_dim_, d_norm_out_, b.qkv_w);

                bool is_full_att = (i == 7 || i == 15 || i == 23 || i == 31);

                if (is_full_att) {
                    // Full Attention: split into [16, num_patches_, 80]
                    vit_split_qkv_bias_bf16_cuda(
                        d_Q_, d_K_, d_V_, d_qkv_, b.qkv_b, d_cos_, d_sin_, num_patches_, num_heads_, head_dim_, active_stream);

                    // S = Q @ K^T / sqrt(head_dim_) -> [16, num_patches_, num_patches_]
                    cublasGemmStridedBatchedEx(
                        cublas_handle_,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        num_patches_, num_patches_, head_dim_,
                        &alpha,
                        d_K_, CUDA_R_16BF, head_dim_, (long long)num_patches_ * head_dim_,
                        d_Q_, CUDA_R_16BF, head_dim_, (long long)num_patches_ * head_dim_,
                        &beta,
                        d_scores_, CUDA_R_16BF, num_patches_, (long long)num_patches_ * num_patches_,
                        num_heads_,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

                    // Softmax
                    vit_softmax_bf16_cuda(
                        d_scores_, num_heads_ * num_patches_, num_patches_, scale, active_stream);

                    // Context = Scores @ V -> [16, num_patches_, head_dim_]
                    cublasGemmStridedBatchedEx(
                        cublas_handle_,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        head_dim_, num_patches_, num_patches_,
                        &alpha,
                        d_V_, CUDA_R_16BF, head_dim_, (long long)num_patches_ * head_dim_,
                        d_scores_, CUDA_R_16BF, num_patches_, (long long)num_patches_ * num_patches_,
                        &beta,
                        d_attn_ctx_, CUDA_R_16BF, head_dim_, (long long)num_patches_ * head_dim_,
                        num_heads_,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

                    // Merge heads: [16, num_patches_, 80] -> [num_patches_, 1280]
                    vit_merge_heads_bf16_cuda(
                        d_attn_merged_, d_attn_ctx_, num_patches_, num_heads_, head_dim_, active_stream);
                } else {
                    // Window Attention: 36 windows of 64 tokens each
                    // Split Q, K, V into layout [36 windows, 16 heads, 64 tokens, 80 head_dim] (576 batches)
                    int window_len = 64;
                    int num_windows = 36;
                    int total_batches = num_windows * num_heads_; // 576

                    vit_split_qkv_bias_window_bf16_cuda(
                        d_Q_, d_K_, d_V_, d_qkv_, b.qkv_b, d_cos_, d_sin_, num_patches_, num_heads_, head_dim_, window_len, active_stream);

                    // S = Q @ K^T / sqrt(head_dim_) for each batch: [64, 64]
                    cublasGemmStridedBatchedEx(
                        cublas_handle_,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        window_len, window_len, head_dim_,
                        &alpha,
                        d_K_, CUDA_R_16BF, head_dim_, (long long)window_len * head_dim_,
                        d_Q_, CUDA_R_16BF, head_dim_, (long long)window_len * head_dim_,
                        &beta,
                        d_scores_, CUDA_R_16BF, window_len, (long long)window_len * window_len,
                        total_batches,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

                    // Softmax over 576 * 64 rows of 64
                    vit_softmax_bf16_cuda(
                        d_scores_, total_batches * window_len, window_len, scale, active_stream);

                    // Context = Scores @ V -> [80, 64] in column-major
                    cublasGemmStridedBatchedEx(
                        cublas_handle_,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        head_dim_, window_len, window_len,
                        &alpha,
                        d_V_, CUDA_R_16BF, head_dim_, (long long)window_len * head_dim_,
                        d_scores_, CUDA_R_16BF, window_len, (long long)window_len * window_len,
                        &beta,
                        d_attn_ctx_, CUDA_R_16BF, head_dim_, (long long)window_len * head_dim_,
                        total_batches,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

                    // Merge window heads: [36, 16, 64, 80] -> [2304, 1280]
                    vit_merge_heads_window_bf16_cuda(
                        d_attn_merged_, d_attn_ctx_, num_patches_, num_heads_, head_dim_, window_len, active_stream);
                }

                // Attention out projection: [num_patches_, 1280] x [1280, 1280]^T -> [num_patches_, 1280]
                gemm_bf16(d_attn_proj_, num_patches_, embed_dim_, embed_dim_, d_attn_merged_, b.proj_w);
                add_bias_bf16_cuda(d_attn_proj_, b.proj_b, num_patches_, embed_dim_, active_stream);

                // Residual add: d_patches_ = d_patches_ + d_attn_proj_
                vector_add_bf16_cuda(d_patches_, d_attn_proj_, (size_t)num_patches_ * embed_dim_, active_stream);

                // RMSNorm 2
                rms_norm_cuda_batched(d_norm_out_, d_patches_, b.norm2_w, num_patches_, embed_dim_, 1e-6f, active_stream);

                // SwiGLU MLP:
                // gate = norm_out @ gate_proj^T + gate_b
                gemm_bf16(d_gate_buf_, num_patches_, mlp_intermediate_, embed_dim_, d_norm_out_, b.gate_w);
                add_bias_bf16_cuda(d_gate_buf_, b.gate_b, num_patches_, mlp_intermediate_, active_stream);

                // up = norm_out @ up_proj^T + up_b
                gemm_bf16(d_up_buf_, num_patches_, mlp_intermediate_, embed_dim_, d_norm_out_, b.up_w);
                add_bias_bf16_cuda(d_up_buf_, b.up_b, num_patches_, mlp_intermediate_, active_stream);

                // silu(gate) * up
                silu_mul_cuda(d_gate_buf_, d_gate_buf_, d_up_buf_, num_patches_ * mlp_intermediate_, 10.0f, active_stream);

                // down = gate @ down_proj^T + down_b
                gemm_bf16(d_down_buf_, num_patches_, embed_dim_, mlp_intermediate_, d_gate_buf_, b.down_w);
                add_bias_bf16_cuda(d_down_buf_, b.down_b, num_patches_, embed_dim_, active_stream);

                // Residual add: d_patches_ = d_patches_ + d_down_buf_
                vector_add_bf16_cuda(d_patches_, d_down_buf_, (size_t)num_patches_ * embed_dim_, active_stream);
            }

            // 4. Qwen 2.5 Merger
            // RMSNorm ln_q
            rms_norm_cuda_batched(d_ln_q_, d_patches_, merger_.ln_q_w, num_patches_, embed_dim_, 1e-6f, active_stream);

            // In window order, consecutive 4 patches already form 2x2 merged units [576, 5120]
            // MLP.0: [576, 5120] x [5120, 5120]^T + bias -> GELU -> [576, 5120]
            gemm_bf16(d_mlp0_, merged_patches_, merger_in_dim_, merger_in_dim_, d_ln_q_, merger_.mlp0_w);
            gelu_bias_bf16_cuda(d_mlp0_, d_mlp0_, merger_.mlp0_b, merged_patches_, merger_in_dim_, active_stream);

            // MLP.2: [576, 5120] x [2048, 5120]^T + bias -> [576, 2048]
            gemm_bf16(d_mlp2_out_, merged_patches_, output_dim_, merger_in_dim_, d_mlp0_, merger_.mlp2_w);
            add_bias_bf16_cuda(d_mlp2_out_, merger_.mlp2_b, merged_patches_, output_dim_, active_stream);

            // 5. Unpermute merged tokens back to original raster order [24, 24]:
            // d_visual_out_[t] = d_mlp2_out_[reverse_indices[t]]
            unpermute_merged_tokens_bf16_cuda(
                d_visual_out_, d_mlp2_out_, d_reverse_indices_, merged_patches_, output_dim_, active_stream);

            std::cout << "[Vision] Qwen2.5 ViT forward complete: generated 576 visual tokens x " << output_dim_ << " dim (window attention + unpermuted)." << std::endl;
            return d_visual_out_;
        }


        // Qwen 3.8 Forward Path
        // 2. Fast Conv3D Patch Embedding via Im2Col + GEMM
        im2col_patch_embed_bf16_cuda(
            d_im2col_, d_img_norm_, image_size_, image_size_, patch_size_, temporal_size_, 3, active_stream);

        gemm_bf16(d_patches_, num_patches_, embed_dim_, 3 * temporal_size_ * patch_size_ * patch_size_,
                  d_im2col_, patch_embed_w_);

        add_bias_and_posemb_bf16_cuda(
            d_patches_, patch_embed_b_, pos_embed_w_, num_patches_, embed_dim_, active_stream);

        // 3. ViT Blocks
        for (int i = 0; i < num_blocks_; i++) {
            const auto& b = blocks_[i];

            // LayerNorm 1
            layernorm_affine_bf16_cuda(
                d_norm_out_, d_patches_, b.norm1_w, b.norm1_b, num_patches_, embed_dim_, 1e-6f, active_stream);

            // QKV projection
            gemm_bf16(d_qkv_, num_patches_, 3 * embed_dim_, embed_dim_, d_norm_out_, b.qkv_w);

            // Split Q, K, V and add bias with 2D RoPE
            vit_split_qkv_bias_bf16_cuda(
                d_Q_, d_K_, d_V_, d_qkv_, b.qkv_b, d_cos_, d_sin_, num_patches_, num_heads_, head_dim_, active_stream);

            // Multi-Head Attention
            float alpha = 1.0f, beta = 0.0f;
            cublasGemmStridedBatchedEx(
                cublas_handle_,
                CUBLAS_OP_T, CUBLAS_OP_N,
                num_patches_, num_patches_, head_dim_,
                &alpha,
                d_K_, CUDA_R_16BF, head_dim_, (long long)num_patches_ * head_dim_,
                d_Q_, CUDA_R_16BF, head_dim_, (long long)num_patches_ * head_dim_,
                &beta,
                d_scores_, CUDA_R_16BF, num_patches_, (long long)num_patches_ * num_patches_,
                num_heads_,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

            // Softmax
            float scale = 1.0f / std::sqrt((float)head_dim_);
            vit_softmax_bf16_cuda(
                d_scores_, num_heads_ * num_patches_, num_patches_, scale, active_stream);

            // Context
            cublasGemmStridedBatchedEx(
                cublas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                head_dim_, num_patches_, num_patches_,
                &alpha,
                d_V_, CUDA_R_16BF, head_dim_, (long long)num_patches_ * head_dim_,
                d_scores_, CUDA_R_16BF, num_patches_, (long long)num_patches_ * num_patches_,
                &beta,
                d_attn_ctx_, CUDA_R_16BF, head_dim_, (long long)num_patches_ * head_dim_,
                num_heads_,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

            // Merge heads
            vit_merge_heads_bf16_cuda(
                d_attn_merged_, d_attn_ctx_, num_patches_, num_heads_, head_dim_, active_stream);

            // Attention out projection
            gemm_bf16(d_attn_proj_, num_patches_, embed_dim_, embed_dim_, d_attn_merged_, b.proj_w);

            // Residual add + bias
            add_residual_bias_bf16_cuda(
                d_patches_, d_attn_proj_, b.proj_b, num_patches_, embed_dim_, active_stream);

            // LayerNorm 2
            layernorm_affine_bf16_cuda(
                d_norm_out_, d_patches_, b.norm2_w, b.norm2_b, num_patches_, embed_dim_, 1e-6f, active_stream);

            // MLP FC1
            gemm_bf16(d_mlp_fc1_, num_patches_, mlp_intermediate_, embed_dim_, d_norm_out_, b.mlp1_w);
            gelu_bias_bf16_cuda(d_mlp_fc1_, d_mlp_fc1_, b.mlp1_b, num_patches_, mlp_intermediate_, active_stream);

            // MLP FC2
            gemm_bf16(d_mlp_fc2_, num_patches_, embed_dim_, mlp_intermediate_, d_mlp_fc1_, b.mlp2_w);
            add_residual_bias_bf16_cuda(
                d_patches_, d_mlp_fc2_, b.mlp2_b, num_patches_, embed_dim_, active_stream);
        }

        // 4. Spatial Merger
        layernorm_affine_bf16_cuda(
            d_norm_out_, d_patches_, merger_.norm_w, merger_.norm_b, num_patches_, embed_dim_, 1e-6f, active_stream);

        spatial_merge_gather_bf16_cuda(
            d_merge_in_, d_norm_out_, 24, 24, embed_dim_, active_stream);

        gemm_bf16(d_merge_fc1_, merged_patches_, merger_in_dim_, merger_in_dim_, d_merge_in_, merger_.fc1_w);
        gelu_bias_bf16_cuda(
            d_merge_fc1_, d_merge_fc1_, merger_.fc1_b, merged_patches_, merger_in_dim_, active_stream);

        gemm_bf16(d_visual_out_, merged_patches_, output_dim_, merger_in_dim_, d_merge_fc1_, merger_.fc2_w);
        add_bias_bf16_cuda(
            d_visual_out_, merger_.fc2_b, merged_patches_, output_dim_, active_stream);

        // Synchronize active stream and immediately reclaim ~265 MB scratch buffers for LLM decode
        cudaStreamSynchronize(active_stream);
        free_scratch_buffers();

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

    bool init_persistent_buffers() {
        void* raw_vis = nullptr;
        if (!alloc_vram_or_managed(raw_vis, (size_t)merged_patches_ * output_dim_ * sizeof(__nv_bfloat16))) return false;
        d_visual_out_ = static_cast<__nv_bfloat16*>(raw_vis);

        if (is_qwen25_) {
            void* raw_win = nullptr;
            if (!alloc_vram_or_managed(raw_win, (size_t)merged_patches_ * sizeof(int))) return false;
            d_window_index_ = static_cast<int*>(raw_win);
            void* raw_rev = nullptr;
            if (!alloc_vram_or_managed(raw_rev, (size_t)merged_patches_ * sizeof(int))) return false;
            d_reverse_indices_ = static_cast<int*>(raw_rev);
        }

        if (!init_rotary_tables()) {
            return false;
        }

        size_t col_mean_bytes = (size_t)(output_dim_ + 256) * sizeof(float);
        void* raw_cm = nullptr;
        if (!alloc_vram_or_managed(raw_cm, col_mean_bytes)) return false;
        d_col_mean_scratch_ = static_cast<float*>(raw_cm);

        return true;
    }

    bool alloc_scratch_buffers() {
        if (!scratch_ptrs_.empty()) return true;

        auto alloc_raw = [&](void*& ptr, size_t bytes) -> bool {
            cudaError_t err = cudaMalloc(&ptr, bytes);
            if (err != cudaSuccess) {
                cudaGetLastError();
                if (cudaMallocManaged(&ptr, bytes) != cudaSuccess) {
                    cudaGetLastError();
                    return false;
                }
            }
            scratch_ptrs_.push_back(ptr);
            return true;
        };
        auto alloc_bf16 = [&](__nv_bfloat16*& ptr, size_t elements) -> bool {
            void* raw = nullptr;
            if (!alloc_raw(raw, elements * sizeof(__nv_bfloat16))) return false;
            ptr = static_cast<__nv_bfloat16*>(raw);
            return true;
        };

        size_t img_bytes = (size_t)temporal_size_ * 3 * image_size_ * image_size_ * sizeof(float);
        void* raw_img = nullptr;
        if (!alloc_raw(raw_img, img_bytes)) return false;
        d_img_norm_ = static_cast<float*>(raw_img);

        size_t patch_filter_elems = (size_t)3 * temporal_size_ * patch_size_ * patch_size_;
        if (!alloc_bf16(d_im2col_, (size_t)num_patches_ * patch_filter_elems) ||
            !alloc_bf16(d_patches_, (size_t)num_patches_ * embed_dim_) ||
            !alloc_bf16(d_norm_out_, (size_t)num_patches_ * embed_dim_) ||
            !alloc_bf16(d_qkv_, (size_t)num_patches_ * (3 * embed_dim_)) ||
            !alloc_bf16(d_Q_, (size_t)num_heads_ * num_patches_ * head_dim_) ||
            !alloc_bf16(d_K_, (size_t)num_heads_ * num_patches_ * head_dim_) ||
            !alloc_bf16(d_V_, (size_t)num_heads_ * num_patches_ * head_dim_) ||
            !alloc_bf16(d_scores_, (size_t)num_heads_ * num_patches_ * num_patches_) ||
            !alloc_bf16(d_attn_ctx_, (size_t)num_heads_ * num_patches_ * head_dim_) ||
            !alloc_bf16(d_attn_merged_, (size_t)num_patches_ * embed_dim_) ||
            !alloc_bf16(d_attn_proj_, (size_t)num_patches_ * embed_dim_) ||
            !alloc_bf16(d_merge_in_, (size_t)merged_patches_ * merger_in_dim_)) {
            free_scratch_buffers();
            return false;
        }

        if (is_qwen25_) {
            if (!alloc_bf16(d_gate_buf_, (size_t)num_patches_ * mlp_intermediate_) ||
                !alloc_bf16(d_up_buf_, (size_t)num_patches_ * mlp_intermediate_) ||
                !alloc_bf16(d_down_buf_, (size_t)num_patches_ * embed_dim_) ||
                !alloc_bf16(d_ln_q_, (size_t)num_patches_ * embed_dim_) ||
                !alloc_bf16(d_mlp0_, (size_t)merged_patches_ * merger_in_dim_) ||
                !alloc_bf16(d_patches_win_, (size_t)num_patches_ * embed_dim_) ||
                !alloc_bf16(d_mlp2_out_, (size_t)merged_patches_ * output_dim_)) {
                free_scratch_buffers();
                return false;
            }
        } else {
            if (!alloc_bf16(d_mlp_fc1_, (size_t)num_patches_ * mlp_intermediate_) ||
                !alloc_bf16(d_mlp_fc2_, (size_t)num_patches_ * embed_dim_) ||
                !alloc_bf16(d_merge_fc1_, (size_t)merged_patches_ * merger_in_dim_)) {
                free_scratch_buffers();
                return false;
            }
        }

        return true;
    }

    void free_scratch_buffers() {
        for (void* p : scratch_ptrs_) {
            if (p) cudaFree(p);
        }
        scratch_ptrs_.clear();
        d_img_norm_ = nullptr;
        d_im2col_ = nullptr;
        d_patches_ = nullptr;
        d_norm_out_ = nullptr;
        d_qkv_ = nullptr;
        d_Q_ = nullptr;
        d_K_ = nullptr;
        d_V_ = nullptr;
        d_scores_ = nullptr;
        d_attn_ctx_ = nullptr;
        d_attn_merged_ = nullptr;
        d_attn_proj_ = nullptr;
        d_merge_in_ = nullptr;
        d_gate_buf_ = nullptr;
        d_up_buf_ = nullptr;
        d_down_buf_ = nullptr;
        d_ln_q_ = nullptr;
        d_mlp0_ = nullptr;
        d_patches_win_ = nullptr;
        d_mlp2_out_ = nullptr;
        d_mlp_fc1_ = nullptr;
        d_mlp_fc2_ = nullptr;
        d_merge_fc1_ = nullptr;
    }

    bool init_rotary_tables() {
        size_t total_elements = (size_t)num_patches_ * head_dim_;
        size_t bytes = total_elements * sizeof(float);
        void* raw_cos = nullptr;
        void* raw_sin = nullptr;
        if (!alloc_vram_or_managed(raw_cos, bytes)) return false;
        if (!alloc_vram_or_managed(raw_sin, bytes)) return false;
        d_cos_ = static_cast<float*>(raw_cos);
        d_sin_ = static_cast<float*>(raw_sin);

        std::vector<float> h_cos(total_elements);
        std::vector<float> h_sin(total_elements);

        if (is_qwen25_) {
            // Compute window_index and reverse_indices for 24x24 merged grid (6x6 windows of 4x4 blocks)
            std::vector<int> h_window_index(merged_patches_);
            std::vector<int> h_reverse_indices(merged_patches_);
            int idx = 0;
            for (int wy = 0; wy < 6; wy++) {
                for (int wx = 0; wx < 6; wx++) {
                    for (int y = 0; y < 4; y++) {
                        for (int x = 0; x < 4; x++) {
                            int my = wy * 4 + y;
                            int mx = wx * 4 + x;
                            h_window_index[idx++] = my * 24 + mx;
                        }
                    }
                }
            }
            for (int i = 0; i < merged_patches_; i++) {
                h_reverse_indices[h_window_index[i]] = i;
            }

            cudaMemcpy(d_window_index_, h_window_index.data(), merged_patches_ * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_reverse_indices_, h_reverse_indices.data(), merged_patches_ * sizeof(int), cudaMemcpyHostToDevice);

            // Compute 2D RoPE table in window order matching PyTorch Qwen2.5-VL ViT
            int n_freq = head_dim_ / 4; // 80 / 4 = 20
            std::vector<float> inv_freq(n_freq);
            for (int i = 0; i < n_freq; i++) {
                inv_freq[i] = 1.0f / std::pow(10000.0f, (float)i / (float)n_freq);
            }

            int w_blocks = 24;
            for (int p = 0; p < num_patches_; p++) {
                int w_blk = p / 4;
                int sub = p % 4;
                int orig_blk = h_window_index[w_blk];
                int my = orig_blk / w_blocks;
                int mx = orig_blk % w_blocks;
                int py = my * 2 + (sub / 2);
                int px = mx * 2 + (sub % 2);

                std::vector<float> emb(head_dim_);
                for (int i = 0; i < n_freq; i++) {
                    float v_h = (float)py * inv_freq[i];
                    float v_w = (float)px * inv_freq[i];
                    emb[i]              = v_h;
                    emb[n_freq + i]     = v_w;
                    emb[2 * n_freq + i] = v_h;
                    emb[3 * n_freq + i] = v_w;
                }

                for (int d = 0; d < head_dim_; d++) {
                    h_cos[(size_t)p * head_dim_ + d] = std::cos(emb[d]);
                    h_sin[(size_t)p * head_dim_ + d] = std::sin(emb[d]);
                }
            }
        } else {
            int n_freq = head_dim_ / 4;
            std::vector<float> inv_freq(n_freq);
            for (int i = 0; i < n_freq; i++) {
                inv_freq[i] = 1.0f / std::pow(10000.0f, (float)i / (float)n_freq);
            }

            int w_blocks = 24;
            for (int p = 0; p < num_patches_; p++) {
                int block_idx = p / 4;
                int my = block_idx / w_blocks;
                int mx = block_idx % w_blocks;
                int sub = p % 4;
                int py = my * 2 + (sub / 2);
                int px = mx * 2 + (sub % 2);

                std::vector<float> emb(head_dim_);
                for (int i = 0; i < n_freq; i++) {
                    float v_h = (float)py * inv_freq[i];
                    float v_w = (float)px * inv_freq[i];
                    emb[i]              = v_h;
                    emb[n_freq + i]     = v_w;
                    emb[2 * n_freq + i] = v_h;
                    emb[3 * n_freq + i] = v_w;
                }

                for (int d = 0; d < head_dim_; d++) {
                    h_cos[(size_t)p * head_dim_ + d] = std::cos(emb[d]);
                    h_sin[(size_t)p * head_dim_ + d] = std::sin(emb[d]);
                }
            }
        }

        cudaMemcpy(d_cos_, h_cos.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sin_, h_sin.data(), bytes, cudaMemcpyHostToDevice);
        return true;
    }


    void free_all() {
        free_scratch_buffers();
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
    std::vector<void*> scratch_ptrs_;

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
    __nv_bfloat16* d_gate_buf_ = nullptr;
    __nv_bfloat16* d_up_buf_ = nullptr;
    __nv_bfloat16* d_down_buf_ = nullptr;
    __nv_bfloat16* d_ln_q_ = nullptr;
    __nv_bfloat16* d_mlp0_ = nullptr;
    __nv_bfloat16* d_merge_in_ = nullptr;
    __nv_bfloat16* d_merge_fc1_ = nullptr;
    __nv_bfloat16* d_visual_out_ = nullptr;
    __nv_bfloat16* d_patches_win_ = nullptr;
    __nv_bfloat16* d_mlp2_out_ = nullptr;
    int* d_window_index_ = nullptr;
    int* d_reverse_indices_ = nullptr;
    float* d_cos_ = nullptr;
    float* d_sin_ = nullptr;
    float* d_col_mean_scratch_ = nullptr;
    __nv_bfloat16* d_embed_mean_ = nullptr;
};

} // namespace moecher::vision

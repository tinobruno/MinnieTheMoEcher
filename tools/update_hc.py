import re

with open("src/server_single.cpp", "r") as f:
    content = f.read()

# Replace hc_pre
pre_old = """        CUDA_CHECK(cudaMemset(buf_hidden_.data, 0, dim * sizeof(__nv_bfloat16)));
        for (int i = 0; i < hc; i++) {
            weighted_add_cuda(buf_hidden_.bf16(),
                              buf_hc_state_.bf16() + (size_t)i * dim,
                              pre_host[i], dim, main_stream_);
        }"""
pre_new = """        hc_pre_cuda(buf_hidden_.bf16(), buf_hc_state_.bf16(), buf_hc_pre_.f32(), hc, dim, main_stream_);"""
content = content.replace(pre_old, pre_new)

# Replace hc_post
post_old = """        for (int i = 0; i < hc; i++) {
            // Start with post[i] * sublayer_out
            __nv_bfloat16* dst = buf_hc_state_.bf16() + (size_t)i * dim;
            // Scale buf_hidden_ by post[i] and write to dst
            CUDA_CHECK(cudaMemset(dst, 0, dim * sizeof(__nv_bfloat16)));
            weighted_add_cuda(dst, buf_hidden_.bf16(), post_host[i], dim, main_stream_);

            // Add sum_j(comb[j][i] * residual[j])
            for (int j = 0; j < hc; j++) {
                float c = comb_host[j * hc + i];
                if (fabsf(c) < 1e-10f) continue;
                weighted_add_cuda(dst, buf_hc_residual_.bf16() + (size_t)j * dim,
                                  c, dim, main_stream_);
            }
        }"""
post_new = """        hc_post_cuda(buf_hc_state_.bf16(), buf_hc_residual_.bf16(), buf_hidden_.bf16(), buf_hc_post_.f32(), buf_hc_comb_.f32(), hc, dim, main_stream_);"""
content = content.replace(post_old, post_new)

# Update head reduce
head_old = """        // Weighted sum
        CUDA_CHECK(cudaMemset(buf_hidden_.data, 0, dim * sizeof(__nv_bfloat16)));
        for (int h = 0; h < hc; h++) {
            weighted_add_cuda(buf_hidden_.bf16(),
                              buf_hc_state_.bf16() + (size_t)h * dim,
                              pre_host[h], dim, main_stream_);
        }"""

# In head reduce we can't just pass pre_host from memory without moving it, wait, we can allocate memory or use an existing buffer!
# Actually buf_hc_pre_ is not used during head reduce! Oh wait, it's not a buffer, we compute pre_host in a local array!
head_new = """        // Weighted sum
        CUDA_CHECK(cudaMemcpy(buf_hc_mixes_.f32(), pre_host, hc * sizeof(float), cudaMemcpyHostToDevice));
        hc_pre_cuda(buf_hidden_.bf16(), buf_hc_state_.bf16(), buf_hc_mixes_.f32(), hc, dim, main_stream_);"""
content = content.replace(head_old, head_new)

with open("src/server_single.cpp", "w") as f:
    f.write(content)

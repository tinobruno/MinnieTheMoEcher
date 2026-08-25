import sys
content = open('src/server_single.cpp').read()
old_code = """    void gemm_bf16(
        __nv_bfloat16* C, int M, int N, int K,
        const __nv_bfloat16* A,  // [M, K]
        const __nv_bfloat16* B,  // [N, K] — stored as weight[out, in]
        float alpha = 1.0f, float beta = 0.0f)
    {
        // In column-major: A_col = K x M, B_col = K x N"""
        
new_code = """    void gemm_bf16(
        __nv_bfloat16* C, int M, int N, int K,
        const __nv_bfloat16* A,  // [M, K]
        const __nv_bfloat16* B,  // [N, K] — stored as weight[out, in]
        float alpha = 1.0f, float beta = 0.0f)
    {
        if (M == 1 && K % 8 == 0 && (size_t)B % 16 == 0 && (size_t)A % 16 == 0) {
            gemv_bf16_out_bf16_cuda(C, B, A, N, K, main_stream_);
            return;
        }
        // In column-major: A_col = K x M, B_col = K x N"""

content = content.replace(old_code, new_code)
open('src/server_single.cpp', 'w').write(content)

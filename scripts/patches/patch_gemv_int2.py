import sys
content = open('src/cuda/activations.cu').read()
old_code = """void gemv_int2_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec,
                    const uint8_t* weight, const __nv_bfloat16* scale_min,
                    int rows, int cols_packed, int block_size,
                    cudaStream_t stream) {
    int threads = 256;
    // We can use a different grid layout if we want, but let's stick to the current one.
    gemv_int2_kernel<<<rows, threads, 0, stream>>>(
        out, vec, weight, scale_min, rows, cols_packed, block_size);
}"""
new_code = """void gemv_int2_cuda(__nv_bfloat16* out, const __nv_bfloat16* vec,
                    const uint8_t* weight, const __nv_bfloat16* scale_min,
                    int rows, int cols_packed, int block_size,
                    cudaStream_t stream) {
    int threads = 256;
    // Launch a kernel with more threads/blocks to optimize?
    // Actually, rows (dim=7168 or something). So we have 7168 blocks, 256 threads.
    // That's plenty of parallelism.
    gemv_int2_kernel<<<rows, threads, 0, stream>>>(
        out, vec, weight, scale_min, rows, cols_packed, block_size);
}"""
content = content.replace(old_code, new_code)
open('src/cuda/activations.cu', 'w').write(content)

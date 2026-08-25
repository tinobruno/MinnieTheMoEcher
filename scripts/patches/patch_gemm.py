import sys
content = open('src/server_single.cpp').read()
target = """        if (M == 1) {
            gemv_int2_cuda(C, A, weight, scale_min, N, K_packed, block_size, main_stream_);
        } else {
            gemm_int2_cuda(C, A, weight, scale_min, M, N, K_packed, block_size, main_stream_);
        }"""
repl = """        if (M == 1) {
            gemv_int2_cuda(C, A, weight, scale_min, N, K_packed, block_size, main_stream_);
        } else {
            int2_dequant_cuda(buf_dequant_.bf16(), weight, scale_min, N, K_packed, block_size, main_stream_);
            gemm_bf16(C, M, N, K_logical, A, buf_dequant_.bf16());
        }"""
if target not in content:
    print("Could not find target")
    sys.exit(1)
open('src/server_single.cpp', 'w').write(content.replace(target, repl))
print("Patched!")

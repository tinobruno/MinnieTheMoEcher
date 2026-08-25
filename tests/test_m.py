content = open('src/server_single.cpp').read()
old = """        if (M == 1) {
            gemv_int2_cuda(C, A, weight, scale_min, N, K_packed, block_size, main_stream_);
        } else {"""
new = """        if (M == 1) {
            gemv_int2_cuda(C, A, weight, scale_min, N, K_packed, block_size, main_stream_);
        } else {
            // Needs INT2 GEMM implementation!"""
print(content.find(old))

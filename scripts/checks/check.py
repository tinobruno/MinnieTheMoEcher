content = open('src/server_single.cpp').read()
print("gemm_bf16 in file:", "gemm_bf16" in content)
idx = content.find("void gemm_bf16")
if idx != -1:
    print(content[idx:idx+500])

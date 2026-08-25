content = open('src/cuda/activations.cu').read()
start = content.find('__global__ void dequantize_int2_kernel')
end = content.find('}', start + 1000)
print(content[start:end+1])

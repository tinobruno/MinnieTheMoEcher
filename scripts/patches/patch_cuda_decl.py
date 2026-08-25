import sys
content = open('src/cuda/activations.cu').read()
content = content.replace('extern "C" void rms_norm_f32_cuda', 'void rms_norm_f32_cuda')
open('src/cuda/activations.cu', 'w').write(content)

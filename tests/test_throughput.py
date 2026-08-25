import torch
import time
from scripts.quantize_experts import quantize_to_int2_asymmetric

x = torch.randn(4096, 4096, device='cuda')
torch.cuda.synchronize()
start = time.time()
quantize_to_int2_asymmetric(x, 256)
torch.cuda.synchronize()
print("Time for 4096x4096:", time.time() - start)

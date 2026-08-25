import torch
x = torch.zeros(10, 8, dtype=torch.uint8)
packed = (x[:, 0::4] | x[:, 1::4])
print("Is contiguous?", packed.is_contiguous())

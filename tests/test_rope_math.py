import torch
x = torch.tensor([1.0, 2.0])
c = torch.tensor([0.8, 0.8])
s = torch.tensor([0.6, 0.6])

# HuggingFace logic
x1 = x[0::2]
x2 = x[1::2]
rh = torch.stack((-x2, x1), dim=-1).flatten(-2)
hf_y = x * c + rh * s

# C++ logic
x0 = x[0]
x1 = x[1]
y0 = x0 * c[0] - x1 * s[0]
y1 = x0 * s[1] + x1 * c[1]
cpp_y = torch.tensor([y0, y1])

print(hf_y)
print(cpp_y)

import torch
rows = 2
cols = 8
block_size = 4
x = torch.arange(16).float().view(rows, cols)
best_m = torch.tensor([[[10], [20]], [[30], [40]]]).float() # [2, 2, 1]
try:
    res = x - best_m
    print("Success:", res.shape)
except Exception as e:
    print("Error:", e)

import torch
rows = 2
cols = 8
block_size = 4
x = torch.arange(16).float().view(rows, cols)
best_m = torch.tensor([[[10], [20]], [[30], [40]]]).float() # [2, 2, 1]
res = x - best_m
print("x shape:", x.shape)
print("best_m shape:", best_m.shape)
print("res shape:", res.shape)

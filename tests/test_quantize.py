import torch

rows = 2
cols = 8
block_size = 4
x = torch.arange(16).float().view(rows, cols)
best_m = torch.tensor([[[10], [20]], [[30], [40]]]).float() # [2, 2, 1]
best_s = torch.ones_like(best_m)

try:
    x_q_final = torch.round((x - best_m) / best_s).clamp(0, 3).to(torch.uint8)
    print("x_q_final shape:", x_q_final.shape)
    x_q_final = x_q_final.view(rows, cols)
    print("Success")
except Exception as e:
    print("Error:", e)

import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Config, DeepseekV4RotaryEmbedding
config = DeepseekV4Config()
emb = DeepseekV4RotaryEmbedding(config)
cos, sin = emb(torch.zeros(1, 15, 64), torch.arange(15).unsqueeze(0), layer_type='main')
print("V4 cos at pos 10:", cos[0, 10, :8])
print("V4 sin at pos 10:", sin[0, 10, :8])

freqs = 1.0 / (10000 ** (torch.arange(0, 64, 2)[: (64 // 2)].float() / 64))
t = torch.arange(15).float()
freqs = torch.outer(t, freqs)
cos_base = torch.cos(freqs)
sin_base = torch.sin(freqs)
print("Base cos at pos 10:", cos_base[10, :4])
print("Base sin at pos 10:", sin_base[10, :4])

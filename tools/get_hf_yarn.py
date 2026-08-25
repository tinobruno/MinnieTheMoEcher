import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
config = type('Config', (), {'rope_scaling': {'beta_fast': 32, 'beta_slow': 1, 'factor': 40, 'mscale': 1.0, 'mscale_all_dim': 1.0, 'original_max_position_embeddings': 4096, 'type': 'yarn', 'rope_theta': 10000, 'rope_type': 'yarn'}, 'max_position_embeddings': 163840, 'rope_theta': 10000})()
try:
    yarn_class = ROPE_INIT_FUNCTIONS['yarn']
    emb = yarn_class(config, 64)
    cos, sin = emb(torch.zeros(1, 100, 64), torch.arange(100).unsqueeze(0))
    print("HF YaRN cos at pos 60:", cos[0, 60, :8])
    
    # Baseline
    freqs = 1.0 / (10000 ** (torch.arange(0, 64, 2)[: (64 // 2)].float() / 64))
    t = torch.arange(100).float()
    freqs = torch.outer(t, freqs)
    cos_base = torch.cos(freqs)
    print("Base cos at pos 60:   ", cos_base[60, :8])
except Exception as e:
    print(e)

import torch
from transformers import AutoConfig
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding, DeepseekV4Config, apply_rotary_pos_emb

config = DeepseekV4Config.from_pretrained("/home/tinobruno/minniethemoecher", local_files_only=True)
rope = DeepseekV4RotaryEmbedding(config)
pos = torch.arange(1).unsqueeze(0)
cos, sin = rope(torch.randn(1, 1, 1, 64), pos, layer_type="main")
print("cos shape from rope():", cos.shape)

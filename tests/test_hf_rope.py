import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Config, DeepseekV4RotaryEmbedding, apply_rotary_pos_emb

config = DeepseekV4Config.from_pretrained("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
rope = DeepseekV4RotaryEmbedding(config)

q = torch.randn(1, 1, 1, 64) # bs=1, seq=1, heads=1, head_dim=64
k = torch.randn(1, 1, 1, 64)
pos = torch.tensor([[0]])
# use compress so it returns cos and sin
cos, sin = rope(q, position_ids=pos, layer_type="main")

print("cos shape:", cos.shape)
print("sin shape:", sin.shape)
print("cos[:5]:", cos[0, 0, :5])

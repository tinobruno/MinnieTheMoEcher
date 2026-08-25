import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from transformers import AutoConfig

config = AutoConfig.from_pretrained("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1", trust_remote_code=True)
emb = DeepseekV4RotaryEmbedding(config)
pos = torch.arange(0, 10).unsqueeze(0)
x = torch.zeros(1, 10, 4096)
cos, sin = emb(x, pos, layer_type="main")
print("HF cos[0, 1, :8]:", cos[0, 1, :8].tolist())
print("HF sin[0, 1, :8]:", sin[0, 1, :8].tolist())
print("HF cos[0, 9, :8]:", cos[0, 9, :8].tolist())
print("HF sin[0, 9, :8]:", sin[0, 9, :8].tolist())

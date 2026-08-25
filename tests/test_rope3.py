from transformers import AutoConfig
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding, DeepseekV4Config

config = DeepseekV4Config.from_pretrained("/home/tinobruno/minniethemoecher", local_files_only=True)
rope = DeepseekV4RotaryEmbedding(config)
inv_freq = rope.main_inv_freq
print("inv_freq shape:", inv_freq.shape)

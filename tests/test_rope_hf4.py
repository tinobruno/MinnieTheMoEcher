import torch
import math
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from transformers import AutoConfig

config = AutoConfig.from_pretrained("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1", trust_remote_code=True)
emb = DeepseekV4RotaryEmbedding(config)
print("compress_attention_scaling:", getattr(emb, "compress_attention_scaling", None))
print("main_attention_scaling:", getattr(emb, "main_attention_scaling", None))

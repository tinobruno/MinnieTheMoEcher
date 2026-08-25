from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Config, DeepseekV4Attention
config = DeepseekV4Config.from_pretrained("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
attn = DeepseekV4Attention(config, layer_idx=0)
print(dir(attn))

from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Config, DeepseekV4RotaryEmbedding
config = DeepseekV4Config.from_pretrained("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
rope = DeepseekV4RotaryEmbedding(config)
print("main_attention_scaling:", rope.main_attention_scaling)
if hasattr(rope, 'compress_attention_scaling'):
    print("compress_attention_scaling:", rope.compress_attention_scaling)

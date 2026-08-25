from transformers import AutoConfig
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Config
config = DeepseekV4Config.from_pretrained("/home/tinobruno/minniethemoecher", local_files_only=True)
print(config.rope_parameters)

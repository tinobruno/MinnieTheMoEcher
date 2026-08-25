from transformers.models.deepseek_v3.modeling_deepseek_v3 import ROPE_INIT_FUNCTIONS
import inspect
print(inspect.getsource(ROPE_INIT_FUNCTIONS["yarn"]))

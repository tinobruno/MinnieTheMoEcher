from transformers.models.deepseek_v4.modeling_deepseek_v4 import ROPE_INIT_FUNCTIONS
import inspect
print(inspect.getsource(ROPE_INIT_FUNCTIONS["yarn"]))

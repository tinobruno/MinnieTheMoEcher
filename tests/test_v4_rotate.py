import inspect
from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb, rotate_half
print("APPLY_ROTARY:")
print(inspect.getsource(apply_rotary_pos_emb))
print("ROTATE_HALF:")
print(inspect.getsource(rotate_half))

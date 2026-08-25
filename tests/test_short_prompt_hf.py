import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu")

raw_text = "The quick brown fox jumps over the lazy dog. What is the animal that jumps?"
prompt_text = "<\xef\xbd\x9cbegin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser\xef\xbd\x9c>" + raw_text + "<\xef\xbd\x9cAssistant\xef\xbd\x9c>"

inputs = tokenizer(prompt_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print("Response:", tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:]))

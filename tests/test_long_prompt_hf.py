import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto", offload_folder="offload")

raw_text = "The capital of Italy is Rome. It is not only the capital of the country but also one of the most historically and culturally significant cities in the world, known for its ancient landmarks like the Colosseum, the Roman Forum, and the Vatican City which is a separate entity within it. Rome has been the capital since 1871, after the unification of Italy as a kingdom with the unification of the various states and territories that were part of the country at that time under the rule of the King of Italy. Based on this text, what is the capital of Belgium?"
prompt_text = "<\xef\xbd\x9cbegin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser\xef\xbd\x9c>" + raw_text + "<\xef\xbd\x9cAssistant\xef\xbd\x9c>"

inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
print("Input IDs length:", inputs.input_ids.shape[1])

# Generate
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:])
print("Response:", response)

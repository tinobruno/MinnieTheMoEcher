from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

messages = [
    {"role": "user", "content": "What is the capital of Italy?"},
    {"role": "assistant", "content": "The capital of Italy is Rome, which is also the largest city of the country. It is located in the central part of the country, on the banks of the River Tiber. As the capital of Italy, Rome is not only the political center of the country, but also a famous historical and cultural city in the world."},
    {"role": "user", "content": "And the capital of Belgium?"}
]

# Manual prompt string construction to avoid chat template issues
prompt = "<｜begin▁of▁sentence｜>"
for msg in messages:
    if msg["role"] == "user":
        prompt += "<｜User｜>" + msg["content"]
    elif msg["role"] == "assistant":
        prompt += "<｜Assistant｜>" + msg["content"] + "<｜end▁of▁sentence｜>"
prompt += "<｜Assistant｜>"

print("Prompt:", repr(prompt))
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print(inputs.input_ids)
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print("Response:", tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:]))

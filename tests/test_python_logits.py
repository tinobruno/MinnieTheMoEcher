from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")

input_ids = tokenizer("Tell me a very long story about the history of Rome. Tell me a very long story about the history of Rome.", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]
    top_logits, top_indices = torch.topk(logits, 5)
    print("Python Top-5 Logits:")
    for i in range(5):
        print(f"  token={top_indices[i].item()} logit={top_logits[i].item():.4f} ({tokenizer.decode([top_indices[i].item()])})")

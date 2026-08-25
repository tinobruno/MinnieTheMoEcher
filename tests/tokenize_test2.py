from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")

prompt = "<｜begin▁of▁sentence｜>You are a helpful, friendly, and knowledgeable AI assistant. Answer clearly and concisely.<｜User｜>What is the capital of Italy?<｜Assistant｜>"
print(tokenizer.encode(prompt, add_special_tokens=False))

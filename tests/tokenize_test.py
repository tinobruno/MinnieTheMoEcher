from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3')
prompt = "<｜begin▁of▁sentence｜><｜User｜>what is the commodore Amiga?<｜Assistant｜>"
print("My prompt tokens:", tokenizer.encode(prompt))
prompt2 = "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>what is the commodore Amiga?<\uff5cAssistant\uff5c>"
print("User prompt tokens:", tokenizer.encode(prompt2))

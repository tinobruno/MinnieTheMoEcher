from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base")
print("hello<｜Assistant｜>:", t.encode("hello<｜Assistant｜>", add_special_tokens=False))
print("hello\\n<｜Assistant｜>:", t.encode("hello\n<｜Assistant｜>", add_special_tokens=False))
print("hello<｜Assistant｜> decodes to:", t.decode(t.encode("hello<｜Assistant｜>", add_special_tokens=False)))

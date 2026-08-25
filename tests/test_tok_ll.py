from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base")
print("ll:", t.encode("ll", add_special_tokens=False))
print("Okay:", t.encode("Okay", add_special_tokens=False))
print("<think>:", t.encode("<think>", add_special_tokens=False))
print("<think>\\n:", t.encode("<think>\n", add_special_tokens=False))

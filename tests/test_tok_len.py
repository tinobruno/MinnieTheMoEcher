from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base", trust_remote_code=True)
text = """Com Am** is not a thing. I think you meant the **Commod**... wait, even that's wrong.

You probably mean the **Comstom... no—let me just tell you:

---

### 🛜 It’s the **Commod**? No — It's the ...

## **Commodary?**
No. Let’s be clear right away: There is no such thing as “the Commity”. What you are likely referring to with this spelling/tyme is one of these two famous things from computing history:

---

### 1. The **Comal (or C?"""
print(len(tokenizer.encode(text)))

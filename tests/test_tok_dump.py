import sys
sys.path.append("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062', trust_remote_code=True)

with open('build/prompt_dump.txt', 'r') as f:
    text = f.read()

ids = tokenizer.encode(text)
print("HF Token count:", len(ids))
# print the first 10 and last 20
print("HF Tokens start:", ids[:10])
print("HF Tokens end:", ids[-20:])

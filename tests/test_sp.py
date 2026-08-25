import json
with open('/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json') as f:
    t = json.load(f)
for at in t.get('added_tokens', []):
    if 'end' in at['content'].lower():
        print(f"{at['id']}: {at['content']}")

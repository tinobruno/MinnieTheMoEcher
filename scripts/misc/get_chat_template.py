import json

tokenizer = json.load(open("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer_config.json"))
print(tokenizer.get("chat_template"))

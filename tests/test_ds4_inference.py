import json

tokenizer = json.load(open("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json"))
model_vocab = tokenizer["model"]["vocab"]

def get_id(t):
    return model_vocab.get(t, -1)

print("</think>", get_id("</think>"))
print("</think>\\n", get_id("</think>\n"))

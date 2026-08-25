from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
tokens = [0, 3476, 477, 260, 11502, 14, 17608]
print(tokenizer.decode(tokens))

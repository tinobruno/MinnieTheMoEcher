from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer.json")
print(tok.encode("<｜end▁of▁sentence｜>").ids)
print(tok.encode("<｜User｜>").ids)

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
t1 = enc.encode("system\n") + enc.encode("You are a helpful assistant")
t2 = enc.encode("system\nYou are a helpful assistant")
print(t1 == t2)
print(t1)
print(t2)

content = open('src/server_single.cpp').read()
idx = content.find("Compute logits")
if idx != -1:
    print(content[idx:idx+500])

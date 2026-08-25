content = open('src/server_single.cpp').read()
idx = content.find("Top-k on CPU")
if idx != -1:
    print(content[idx:idx+2500])

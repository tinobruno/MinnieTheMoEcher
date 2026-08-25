content = open('src/server_single.cpp').read()
idx = content.find("void forward_moe")
if idx != -1:
    print(content[idx:idx+2500])

content = open('src/server_single.cpp').read()
idx = content.find("void decode_step")
if idx != -1:
    print(content[idx:idx+2500])

content = open('src/server_single.cpp').read()
idx = content.find("void forward_layer")
if idx != -1:
    print(content[idx:idx+2000])

content = open('src/server_single.cpp').read()
idx = content.find("head_weight_")
if idx != -1:
    print(content[idx-200:idx+200])

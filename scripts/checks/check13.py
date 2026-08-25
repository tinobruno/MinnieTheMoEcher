content = open('src/server_single.cpp').read()
idx = content.find("void hc_pre")
if idx != -1:
    print(content[idx:idx+1500])

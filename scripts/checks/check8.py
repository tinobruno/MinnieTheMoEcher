content = open('src/server_single.cpp').read()
idx = content.find("Execute shared expert")
if idx != -1:
    print(content[idx:idx+2500])

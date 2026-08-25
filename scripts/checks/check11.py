content = open('src/server_single.cpp').read()
idx = 0
while True:
    idx = content.find("forward_layer", idx + 1)
    if idx == -1: break
    print("MATCH AT", idx)
    print(content[idx-100:idx+200])
    print("-" * 50)

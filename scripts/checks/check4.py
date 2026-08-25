content = open('src/server_single.cpp').read()
idx = 0
while True:
    idx = content.find("head_weight_", idx + 1)
    if idx == -1: break
    print("MATCH AT", idx)
    print(content[idx-200:idx+200])
    print("-" * 50)

content = open('src/server_single.cpp').read()
idx = content.find("while (n_past_ < max_tokens)")
if idx != -1:
    print(content[idx:idx+2500])

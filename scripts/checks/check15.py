content = open('src/server_single.cpp').read()
idx = content.find("t_entire_loop=")
if idx != -1:
    print(content[idx-500:idx+500])

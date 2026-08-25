content = open('src/server_single.cpp').read()
idx = content.find("forward_moe_shared_experts")
if idx != -1:
    print(content[idx-100:idx+400])

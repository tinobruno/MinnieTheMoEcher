import sys
content = open('src/server_single.cpp').read()
target = """    void hc_head_reduce() {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_dim;
        int hc_dim = cfg_.hidden_size;"""
repl = """    void hc_head_reduce() {
        int dim = cfg_.hidden_size;
        int hc = cfg_.hc_mult;
        int hc_dim = hc * dim;"""
if target not in content:
    print("Could not find target")
    sys.exit(1)
open('src/server_single.cpp', 'w').write(content.replace(target, repl))
print("Patched!")

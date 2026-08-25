import sys
content = open('src/server_single.cpp').read()
content = content.replace("if (layer_id < 2 && position < 5) fprintf", "if (layer_id < 2) fprintf")
open('src/server_single.cpp', 'w').write(content)

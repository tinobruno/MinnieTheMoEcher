import re

with open('src/server_single.cpp', 'r') as f:
    content = f.read()

content = content.replace('"<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9ctool_calls>\\n"', '"<\\xef\\xbd\\x9c" "DSML\\xef\\xbd\\x9c" "tool_calls>\\n"')
content = content.replace('"<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9ctool_calls>"', '"<\\xef\\xbd\\x9c" "DSML\\xef\\xbd\\x9c" "tool_calls>"')
content = content.replace('"<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9cinvoke name=\\""', '"<\\xef\\xbd\\x9c" "DSML\\xef\\xbd\\x9c" "invoke name=\\""')
content = content.replace('"<\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9cparameter name=\\""', '"<\\xef\\xbd\\x9c" "DSML\\xef\\xbd\\x9c" "parameter name=\\""')
content = content.replace('"</\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9cparameter>"', '"</\\xef\\xbd\\x9c" "DSML\\xef\\xbd\\x9c" "parameter>"')
content = content.replace('"</\\xef\\xbd\\x9cDSML\\xef\\xbd\\x9cinvoke>"', '"</\\xef\\xbd\\x9c" "DSML\\xef\\xbd\\x9c" "invoke>"')

with open('src/server_single.cpp', 'w') as f:
    f.write(content)


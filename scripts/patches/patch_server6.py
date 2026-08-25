with open('src/server_single.cpp', 'r') as f:
    content = f.read()

# Replace all occurrences of "<\xef\xbd\x9cDSML" with "<\xef\xbd\x9c" "DSML"
content = content.replace('"<\\xef\\xbd\\x9cDSML', '"<\\xef\\xbd\\x9c" "DSML')
content = content.replace('DSML\\xef\\xbd\\x9c', 'DSML" "\\xef\\xbd\\x9c')
content = content.replace('</\\xef\\xbd\\x9cDSML', '"</\\xef\\xbd\\x9c" "DSML')

with open('src/server_single.cpp', 'w') as f:
    f.write(content)


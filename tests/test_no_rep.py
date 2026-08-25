import os

with open("src/server_single.cpp", "r") as f:
    cpp = f.read()

cpp = cpp.replace(
    "float rep_penalty = 1.15f;",
    "float rep_penalty = 1.0f;"
)

with open("src/server_single.cpp", "w") as f:
    f.write(cpp)

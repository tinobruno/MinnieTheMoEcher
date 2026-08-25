reply = ""
chunks = ["hello", " world"]
WHITE = "\033[97m"
RESET = "\033[0m"

for chunk in chunks:
    chunk_str = chunk
    if "<tool_call>" in reply + chunk_str:
        pass
    else:
        print(f"{WHITE}{chunk_str}{RESET}", end="", flush=True)
    reply += chunk_str
print()

import json
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else "tools_out_6.txt"

with open(filename) as f:
    for line in f:
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            try:
                data = json.loads(line[6:])
                content = data["choices"][0]["delta"].get("content", "")
                if content:
                    print(content, end="", flush=True)
            except Exception:
                pass
print()

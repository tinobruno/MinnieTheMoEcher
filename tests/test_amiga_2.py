import sys
import chat

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
]

print("First turn:")
reply = ""
reasoning = ""
for chunk in chat.chat_completion_stream("http://localhost:8001", messages, 512, 0.6):
    if isinstance(chunk, dict):
        if chunk.get("type") == "reasoning":
            reasoning += chunk["content"]
            sys.stdout.write("\033[2m" + chunk["content"] + "\033[0m")
    else:
        reply += chunk
        sys.stdout.write(chunk)
sys.stdout.flush()

print("\n\nUpdating messages...")
full_reply = reply
if reasoning:
    full_reply = f"<think>\n{reasoning}\n</think>\n" + reply
messages.append({"role": "assistant", "content": full_reply})
messages.append({"role": "user", "content": "what is commodore Amiga ?"})

print("Second turn:")
for chunk in chat.chat_completion_stream("http://localhost:8001", messages, 512, 0.6):
    if isinstance(chunk, dict):
        if chunk.get("type") == "reasoning":
            sys.stdout.write("\033[2m" + chunk["content"] + "\033[0m")
    else:
        sys.stdout.write(chunk)
sys.stdout.flush()
print()

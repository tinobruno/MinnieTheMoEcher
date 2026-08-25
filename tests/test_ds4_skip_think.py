import urllib.request
import json

default_system = "You are a helpful, concise AI assistant. To read a URL, output EXACTLY the following syntax: <call_url>http://example.com</call_url> and wait for the response. Do not output anything else on that line."
messages = [{"role": "system", "content": default_system}]

def chat(prompt):
    messages.append({"role": "user", "content": prompt})
    payload = json.dumps({"model": "deepseek", "messages": messages, "max_tokens": 150, "temperature": 0.6}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=payload)
    
    reply = ""
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if b"content" in line:
                chunk = json.loads(line)
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    reply += chunk["choices"][0].get("message", {}).get("content", "")
    
    print(f"Assistant: {reply}")
    messages.append({"role": "assistant", "content": reply})

chat("Hello")
chat("what is commodore Amiga?")

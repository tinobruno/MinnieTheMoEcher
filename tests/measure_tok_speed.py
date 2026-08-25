import urllib.request
import json
import time

url = "http://localhost:8001/v1/chat/completions"
prompt = "Explain in 3 concise bullet points the architectural innovations of DeepSeek V3 compared to dense LLMs."

payload = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 128,
    "temperature": 0.6,
    "stream": False,
    "thinking": {"type": "enabled", "budget_tokens": 4096}
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

t0 = time.time()
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    tokens = data['usage']['completion_tokens']
    print("\n" + "="*50)
    print(f"Total Tokens:   {tokens}")
    print(f"Elapsed Time:   {elapsed:.3f} s")
    print(f"Throughput:     {tokens / elapsed:.2f} tok/s")
    print("="*50)
    print("\n[CONTENT]:\n" + content)

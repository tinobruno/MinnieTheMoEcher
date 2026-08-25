import urllib.request
import json
import time

url = "http://localhost:8001/v1/chat/completions"

def run_test(temp):
    print(f"\n=======================================================")
    print(f"  TESTING 5 RUNS OF 'hello' AT TEMPERATURE = {temp}")
    print(f"=======================================================")
    for trial in range(1, 6):
        payload = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "hello"}
            ],
            "max_tokens": 512,
            "temperature": temp,
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
            print(f"\n--- [Run {trial}/5] (Elapsed: {elapsed:.2f}s, Tokens: {data['usage']['completion_tokens']}) ---")
            print(f"[REASONING]:\n{reasoning if reasoning else '(None / inlined)'}")
            print(f"[FINAL CONTENT]:\n{content}")

print("Running 5x 'hello' test...")
run_test(1.0)
run_test(0.6)


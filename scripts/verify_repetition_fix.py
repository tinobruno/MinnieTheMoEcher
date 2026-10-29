import urllib.request
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

url = "http://127.0.0.1:8001/v1/chat/completions"

def run_test(title, payload):
    print("=" * 70, flush=True)
    print(f"RUNNING: {title}", flush=True)
    print("=" * 70, flush=True)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            msg = res["choices"][0]["message"]
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            finish = res["choices"][0].get("finish_reason", "unknown")
            if reasoning:
                print(f"[REASONING ({len(reasoning)} chars)]:\n{reasoning[:400]}...\n", flush=True)
            print(f"[CONTENT ({len(content)} chars, finish={finish})]:\n{content}\n", flush=True)
            
            # Repetition detection check
            lines = [l.strip() for l in content.split('\n') if len(l.strip()) > 10]
            for l in set(lines):
                if lines.count(l) > 3:
                    print(f"--> [WARNING] Line repeated {lines.count(l)} times: {l[:60]}", flush=True)
    except Exception as e:
        print(f"[ERROR]: {e}", flush=True)

# Test 1: Python Quicksort (non-thinking)
run_test("Test 1: Python Quicksort (Non-Thinking, temp=0.6, max_tokens=300)", {
    "model": "deepseek-v4.1-flash",
    "messages": [
        {"role": "user", "content": "Write a clean Python script implementing quicksort."}
    ],
    "temperature": 0.6,
    "max_tokens": 300,
    "enable_thinking": False,
    "reasoning_effort": "none"
})

# Test 2: HTML Calculator (non-thinking)
run_test("Test 2: HTML Calculator (Non-Thinking, temp=0.6, max_tokens=500)", {
    "model": "deepseek-v4.1-flash",
    "messages": [
        {"role": "user", "content": "Create a simple HTML calculator with basic arithmetic."}
    ],
    "temperature": 0.6,
    "max_tokens": 500,
    "enable_thinking": False,
    "reasoning_effort": "none"
})

# Test 3: HTML Calculator (Thinking, thinking_budget=200, max_tokens=500)
run_test("Test 3: HTML Calculator (Thinking, thinking_budget=200, max_tokens=500)", {
    "model": "deepseek-v4.1-flash",
    "messages": [
        {"role": "user", "content": "Create a simple HTML calculator with basic arithmetic."}
    ],
    "temperature": 0.6,
    "max_tokens": 500,
    "enable_thinking": True,
    "thinking_budget": 200,
    "reasoning_effort": "low"
})

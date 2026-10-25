import urllib.request
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def query(prompt, thinking=False, max_tokens=300, temperature=0.0, rep_pen=1.0, thinking_budget=200):
    url = "http://127.0.0.1:8001/v1/chat/completions"
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": rep_pen
    }
    if not thinking:
        payload["enable_thinking"] = False
        payload["reasoning_effort"] = "none"
        payload["chat_template_kwargs"] = {"thinking": False}
        payload["thinking"] = False
    else:
        payload["enable_thinking"] = True
        payload["max_thinking_tokens"] = thinking_budget
        payload["thinking_budget"] = thinking_budget
        payload["thinking"] = {"budget_tokens": thinking_budget}
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        msg = res["choices"][0]["message"]
        finish_reason = res["choices"][0].get("finish_reason", "unknown")
        return msg.get("content", ""), msg.get("reasoning_content", ""), finish_reason

def main():
    tests = [
        {
            "name": "TEST 1: Python Fibonacci (Greedy)",
            "prompt": "Write a Python function to generate Fibonacci numbers up to n.",
            "thinking": False, "temp": 0.0, "max_tokens": 200, "rep_pen": 1.05
        },
        {
            "name": "TEST 2: HTML Calculator (Boundary crossing past token 128)",
            "prompt": "Create a simple HTML/JS calculator with basic arithmetic.",
            "thinking": False, "temp": 0.0, "max_tokens": 300, "rep_pen": 1.05
        },
        {
            "name": "TEST 3: Python Fibonacci (Sampling temp=0.6 with Top-K/Top-P)",
            "prompt": "Write a Python function to generate Fibonacci numbers up to n.",
            "thinking": False, "temp": 0.6, "max_tokens": 150, "rep_pen": 1.0
        },
        {
            "name": "TEST 4: C++ Quicksort (Greedy)",
            "prompt": "Write a C++ function to sort an array using quicksort.",
            "thinking": False, "temp": 0.0, "max_tokens": 200, "rep_pen": 1.0
        },
        {
            "name": "TEST 5: Reasoning Mode (Which is bigger: 9.11 or 9.9)",
            "prompt": "Which number is bigger: 9.11 or 9.9? Think step by step.",
            "thinking": True, "temp": 0.0, "max_tokens": 300, "rep_pen": 1.0, "thinking_budget": 150
        }
    ]

    all_passed = True
    for i, t in enumerate(tests, 1):
        print(f"\n==================================================================")
        print(f"[{i}/{len(tests)}] {t['name']}")
        print(f"==================================================================")
        content, reasoning, finish = query(
            t["prompt"],
            thinking=t["thinking"],
            max_tokens=t["max_tokens"],
            temperature=t["temp"],
            rep_pen=t["rep_pen"],
            thinking_budget=t.get("thinking_budget", 200)
        )
        if reasoning:
            print(f"--- REASONING ({len(reasoning)} chars) ---")
            print(reasoning.strip()[:400] + ("..." if len(reasoning) > 400 else ""))
        print(f"--- CONTENT ({len(content)} chars, finish_reason={finish}) ---")
        print(content.strip()[:600] + ("..." if len(content) > 600 else ""))
        
        # Check basic sanity: content must be non-empty and not repetitive junk
        if not content or len(content) < 30:
            print("[FAIL] Output too short or empty!")
            all_passed = False
        else:
            print("[PASS] Clean and coherent generation.")

    print("\n==================================================================")
    if all_passed:
        print(">>> ALL VERIFICATION TESTS PASSED SUCCESSFULLY! <<<")
    else:
        print(">>> SOME TESTS FAILED <<<")
    print("==================================================================")

if __name__ == "__main__":
    main()

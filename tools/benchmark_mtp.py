import urllib.request
import json
import time

def run_test(prompt="Write a short poem about the sea in exactly 4 lines", max_tokens=200):
    url = "http://127.0.0.1:9999/v1/chat/completions"
    payload = {
        "model": "qwen",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode('utf-8')
    t1 = time.perf_counter()
    
    res = json.loads(body)
    total_time = t1 - t0
    content = res["choices"][0]["message"]["content"]
    usage = res.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    tok_s = comp_tokens / total_time if total_time > 0 else 0
    
    print("=" * 60)
    print(f"Prompt: {prompt[:50]}...")
    print(f"Time: {total_time:.3f}s | Completion Tokens: {comp_tokens} | Throughput: {tok_s:.2f} tok/s")
    print(f"Response sample:\n{content[:200]}...")
    print("=" * 60)
    return tok_s

def run_suite():
    prompts = [
        "Write a Python function to compute the Levenshtein distance between two strings with full type annotations and docstrings.",
        "Explain the key differences between TCP and UDP in three clear bullet points.",
        "Write a short poem about the sea in exactly 4 lines",
    ]
    results = []
    for p in prompts:
        tok_s = run_test(p, max_tokens=250)
        results.append((p[:40], tok_s))
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUITE SUMMARY:")
    for name, s in results:
        print(f"  - {name}...: {s:.2f} tok/s")
    avg_s = sum(s for _, s in results) / len(results)
    print(f"Average Throughput: {avg_s:.2f} tok/s")
    print("=" * 60)

if __name__ == "__main__":
    print("Running benchmark suite on port 9999...")
    run_suite()

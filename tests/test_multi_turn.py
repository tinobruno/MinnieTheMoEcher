import requests

def test_chat():
    print("=== Turn 1 ===")
    prompt = "What is the capital of Italy?"
    response = requests.post(
        "http://127.0.0.1:8001/v1/chat/completions",
        json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.0
        }
    )
    print(response.json())
    
    print("\n=== Turn 2 ===")
    prompt2 = "And what is the capital of France?"
    response2 = requests.post(
        "http://127.0.0.1:8001/v1/chat/completions",
        json={
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "The capital of Italy is Rome."},
                {"role": "user", "content": prompt2}
            ],
            "max_tokens": 10,
            "temperature": 0.0
        }
    )
    print(response2.json())

test_chat()

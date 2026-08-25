import requests

url = "http://127.0.0.1:8080/chat/completions"

def run_chat(messages):
    data = {
        "model": "deepseek",
        "messages": messages,
        "max_tokens": 10
    }
    response = requests.post(url, json=data).json()
    msg = response["choices"][0]["message"]
    print("Turn ->", msg["content"])
    return msg

print("--- Turn 1 ---")
msgs = [{"role": "user", "content": "Hi."}]
ast = run_chat(msgs)

print("--- Turn 2 ---")
msgs.append(ast)
msgs.append({"role": "user", "content": "Hello."})
run_chat(msgs)

import requests

prompt_messages = [
    {"role": "user", "content": "Please explain the history of the Roman Empire in great detail, focusing on the transition from the Republic to the Empire, the role of Julius Caesar, the establishment of the Principate by Augustus, and the subsequent expansion under the Five Good Emperors. This should be a very long and detailed response."}
]

data = {
    "model": "minniethemoecher",
    "messages": prompt_messages,
    "max_tokens": 100,
    "temperature": 0.0,
    "stream": True
}

print("Sending request...")
response = requests.post("http://127.0.0.1:8001/v1/chat/completions", json=data)
if response.status_code == 200:
    print(response.json()['choices'][0]['message']['content'])
else:
    print("Error:", response.status_code, response.text)

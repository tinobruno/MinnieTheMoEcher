import sys
sys.path.append("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1/encoding")
from encoding_dsv4 import encode_messages
import json

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Tokyo?"}
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

prompt = encode_messages(messages=messages, thinking_mode="chat")
with open("python_prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

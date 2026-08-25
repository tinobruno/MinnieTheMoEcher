import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/tinobruno/minniethemoecher/models/deepseek-v4-flash")
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather"}}]
messages = [{"role": "user", "content": "What is the weather?"}]

prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
print("PROMPT:\n", prompt)

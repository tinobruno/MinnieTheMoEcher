from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3-Base', trust_remote_code=True)
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]
print(t.apply_chat_template([{"role": "user", "content": "hello"}], tools=tools, tokenize=False))

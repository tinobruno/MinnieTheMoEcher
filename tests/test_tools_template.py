from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)
messages = [{"role": "user", "content": "fetch the URL https://example.com"}]
tools = [{
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch URL",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"}
            }
        }
    }
}]
prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
print("PROMPT:")
print(prompt)

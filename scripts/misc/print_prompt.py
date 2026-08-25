import json
with open("curl_tools.json") as f:
    req = json.load(f)

# emulate apply_chat_template
tools = req.get("tools", [])
msgs = req.get("messages", [])

prompt = "<｜begin▁of▁sentence｜>"
if tools:
    prompt += "You have access to the following functions:\n"
    prompt += json.dumps(tools, indent=2)
    prompt += "\n\nTo invoke a function, output a JSON array of tool calls inside a <tool_call> block like the following:\n"
    prompt += "<tool_call>\n[\n  {\"name\": \"function_name\", \"arguments\": {\"arg_name\": \"value\"}}\n]\n</tool_call>"

for msg in msgs:
    if msg["role"] == "user":
        prompt += "<｜User｜>" + msg["content"]
        
prompt += "<｜Assistant｜><think>\n"
print(prompt)

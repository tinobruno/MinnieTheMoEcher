import re

with open('chat.py', 'r') as f:
    content = f.read()

# Update chat_completion_stream to yield finish_reason
content = re.sub(
    r'if "usage" in chunk:\n\s*yield {"usage": chunk\["usage"\]}',
    r'if "usage" in chunk:\n                            yield {"usage": chunk["usage"]}\n                        \n                        if "choices" in chunk and len(chunk["choices"]) > 0:\n                            finish_reason = chunk["choices"][0].get("finish_reason")\n                            if finish_reason:\n                                yield {"finish_reason": finish_reason}',
    content
)

# Update main loop to handle finish_reason
main_loop_patch = """
                    if isinstance(chunk, dict) and "finish_reason" in chunk:
                        if chunk["finish_reason"] == "length":
                            print(f"\\n{YELLOW}[Warning: Response truncated due to max-tokens limit ({max_tokens}). Use /tokens N to increase it.]{RESET}", flush=True)
                        continue
"""
content = re.sub(
    r'if isinstance\(chunk, dict\) and "usage" in chunk:\n\s*continue',
    r'if isinstance(chunk, dict) and "usage" in chunk:\n                        continue' + main_loop_patch,
    content
)

with open('chat.py', 'w') as f:
    f.write(content)

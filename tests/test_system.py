import re

with open("chat.py", "r") as f:
    code = f.read()

new_system = """
    default_system = (
        "You are a helpful, concise AI assistant. "
        "Always think step-by-step and wrap your thoughts inside <think> and </think> tags. "
        "Answer questions directly after your thoughts. "
        "If the user asks you to read or fetch a URL, output the URL inside <call_url> tags, "
        "for example: <call_url>http://example.com</call_url>"
    )
"""

old_system = """    default_system = (
        "You are a helpful, concise AI assistant. "
        "Answer questions directly. "
        "If the user asks you to read or fetch a URL, output the URL inside <call_url> tags, "
        "for example: <call_url>http://example.com</call_url>"
    )"""

code = code.replace(old_system, new_system.strip())

with open("chat.py", "w") as f:
    f.write(code)

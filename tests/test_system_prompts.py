"""Test with the strong TOPIC AUTONOMY system prompt to prevent history bleeding."""
import requests, json

url = "http://127.0.0.1:8001/v1/chat/completions"

# The user's stronger system prompt from chat.py (commented out)
strong_system = ("You are a highly capable, adaptive, and precise AI assistant. "
    "CORE OPERATIONAL RULES: "
    "1. TOPIC AUTONOMY: Treat every user prompt as a potentially distinct domain. "
    "Never lock into a subject-matter pattern or force prior turn domains onto new, unrelated questions. "
    "2. CONCISE & FACTUAL: Deliver direct, clear, and accurate answers immediately. "
    "Avoid fluff, unnecessary conversational filler, robotic pleasantries, and trailing meta-commentary. "
    "3. ACCURATE REASONING: Analyze input semantics carefully before responding. "
    "If a term is unfamiliar within the immediate context, evaluate it as an independent entity "
    "rather than assuming it is a typo or a misstatement of previous topics.")

# Also test with a simpler but effective system prompt  
simple_system = "You are a helpful, friendly, and knowledgeable AI assistant. Answer clearly and concisely."

for sys_name, sys_prompt in [("SIMPLE", simple_system), ("STRONG", strong_system)]:
    print("="*60)
    print(f"TEST with {sys_name} system prompt, temp=0.6")
    print("="*60)
    
    messages = [{"role": "system", "content": sys_prompt}]
    
    questions = [
        "Hello",
        "what is the capital of Italy?",
        "what is the capital of Belgium?",
        "what is commodore Amiga?",
    ]
    
    for i, q in enumerate(questions):
        messages.append({"role": "user", "content": q})
        
        resp = requests.post(url, json={
            "model": "deepseek-v4-flash",
            "messages": messages,
            "max_tokens": 200,
            "temperature": 0.6,
            "repetition_penalty": 1.1,
            "stream": False
        })
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": answer})
        
        if i == 3:  # Only print the Amiga answer in detail
            has_bleed = any(w in answer.lower() for w in ["italy", "rome", "roma", "belgium", "brussels", "confusion", "mistake", "clarif", "capital"])
            status = "✗ BLEEDING" if has_bleed else "✓ CLEAN"
            print(f"\nTurn 4 (Amiga): {status}")
            print(f"Answer: {answer}")
        else:
            print(f"Turn {i+1}: {answer[:80]}...")
    print()

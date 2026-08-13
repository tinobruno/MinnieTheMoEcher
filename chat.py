#!/usr/bin/env python3
"""
moecher-chat — Interactive CLI client for the Moecher inference engine.

Usage:
    python3 chat.py                    # default: localhost:8001
    python3 chat.py --url http://host:port
    python3 chat.py --max-tokens 200 --temperature 0.7
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
import readline  # enables arrow-key history in input()
import textwrap
import os
import time

# ── ANSI colours ─────────────────────────────────────────────────────────────

BOLD      = "\033[1m"
DIM       = "\033[2m"
ITALIC    = "\033[3m"
CYAN      = "\033[36m"
GREEN     = "\033[32m"
YELLOW    = "\033[33m"
MAGENTA   = "\033[35m"
RED       = "\033[31m"
RESET     = "\033[0m"

BANNER = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════╗
║          🎵  moecher-chat  🎵                    ║
║      DeepSeek V4-Flash • bare-metal engine       ║
╚══════════════════════════════════════════════════╝{RESET}
{DIM}Type your message and press Enter. Commands:{RESET}
  {YELLOW}/clear{RESET}    — reset conversation
  {YELLOW}/system{RESET}   — set system prompt
  {YELLOW}/temp N{RESET}   — change temperature (current: {{temp}})
  {YELLOW}/tokens N{RESET} — change max tokens  (current: {{max_tokens}})
  {YELLOW}/quit{RESET}     — exit
"""


def chat_completion_stream(url: str, messages: list, max_tokens: int,
                           temperature: float, model: str = "deepseek-v4-flash"):
    """Send a chat completion request and yield assistant reply chunks."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                if line.startswith(b"data: "):
                    line = line[6:].strip()
                    if line == b"[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        pass
    except urllib.error.URLError as e:
        yield f"\n{RED}[Connection error]{RESET} {e.reason}"
    except Exception as e:
        yield f"\n{RED}[Error]{RESET} {e}"


def wrap_text(text: str, width: int = 80) -> str:
    """Word-wrap text while preserving existing newlines and code blocks."""
    lines = text.split("\n")
    wrapped = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            wrapped.append(line)
        elif in_code:
            wrapped.append(line)  # don't wrap code blocks
        else:
            wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(wrapped)


def main():
    parser = argparse.ArgumentParser(description="Chat with the Moecher engine")
    parser.add_argument("--url", default="http://localhost:8001",
                        help="Base URL of the Moecher API (default: http://localhost:8001)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per response (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (default: 0.6)")
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="Model name (default: deepseek-v4-flash)")
    args = parser.parse_args()

    temperature = args.temperature
    max_tokens = args.max_tokens
    #default_system = "You are a highly capable, adaptive, and precise AI assistant. CORE OPERATIONAL RULES: 1. TOPIC AUTONOMY: Treat every user prompt as a potentially distinct domain. Never lock into a subject-matter pattern or force prior turn domains onto new, unrelated questions. 2. CONCISE & FACTUAL: Deliver direct, clear, and accurate answers immediately. Avoid fluff, unnecessary conversational filler, robotic pleasantries, and trailing meta-commentary (e.g., "Feel free to ask more"). 3. ACCURATE REASONING: Analyze input semantics carefully before responding. If a term is unfamiliar within the immediate context, evaluate it as an independent entity rather than assuming it is a typo or a misstatement of previous topics.  "
    default_system = "You are a helpful, friendly, and knowledgeable AI assistant. Answer clearly and concisely."
    messages: list[dict] = [{"role": "system", "content": default_system}]

    # Try to get terminal width
    try:
        term_width = min(os.get_terminal_size().columns, 100)
    except OSError:
        term_width = 80

    print(BANNER.format(temp=temperature, max_tokens=max_tokens))

    # Quick connectivity check
    print(f"{DIM}Connecting to {args.url} ...{RESET}", end=" ", flush=True)
    try:
        urllib.request.urlopen(f"{args.url}/v1/models", timeout=3)
        print(f"{GREEN}✓ connected{RESET}\n")
    except Exception:
        print(f"{YELLOW}⚠ server may not be ready yet (will retry on first message){RESET}\n")

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}You ❯ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not user_input:
            continue

        # ── Slash commands ────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            if cmd[0] in ("/quit", "/exit", "/q"):
                print(f"{DIM}Goodbye!{RESET}")
                break
            elif cmd[0] == "/clear":
                messages = [{"role": "system", "content": default_system}]
                print(f"{YELLOW}Conversation cleared.{RESET}\n")
                continue
            elif cmd[0] == "/system" and len(cmd) > 1:
                system_text = user_input[len("/system "):].strip()
                # Remove existing system message if any
                messages = [m for m in messages if m["role"] != "system"]
                messages.insert(0, {"role": "system", "content": system_text})
                print(f"{YELLOW}System prompt set: {DIM}{system_text}{RESET}\n")
                continue
            elif cmd[0] == "/temp" and len(cmd) > 1:
                try:
                    temperature = float(cmd[1])
                    print(f"{YELLOW}Temperature set to {temperature}{RESET}\n")
                except ValueError:
                    print(f"{RED}Invalid temperature value{RESET}\n")
                continue
            elif cmd[0] == "/tokens" and len(cmd) > 1:
                try:
                    max_tokens = int(cmd[1])
                    print(f"{YELLOW}Max tokens set to {max_tokens}{RESET}\n")
                except ValueError:
                    print(f"{RED}Invalid token count{RESET}\n")
                continue
            elif cmd[0] == "/help":
                print(BANNER.format(temp=temperature, max_tokens=max_tokens))
                continue
            else:
                print(f"{RED}Unknown command. Type /help for options.{RESET}\n")
                continue

        # ── Send message ──────────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})

        print(f"{CYAN}{BOLD}Assistant ❯{RESET} ", end="", flush=True)

        reply = ""
        token_count = 0
        start_time = time.time()
        for chunk in chat_completion_stream(
            args.url, messages, max_tokens, temperature, args.model
        ):
            print(chunk, end="", flush=True)
            reply += chunk
            token_count += 1
            
        elapsed = time.time() - start_time
        tps = token_count / elapsed if elapsed > 0 else 0
        print(f"\n{DIM}[{token_count} tokens in {elapsed:.2f}s, {tps:.2f} tok/s]{RESET}\n")

        messages.append({"role": "assistant", "content": reply})
        print()

if __name__ == "__main__":
    main()

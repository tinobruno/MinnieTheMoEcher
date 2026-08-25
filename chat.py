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
from html.parser import HTMLParser

# ── URL Fetching & Parsing ───────────────────────────────────────────────────

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.in_script_or_style = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.in_script_or_style = True

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.in_script_or_style = False

    def handle_data(self, data):
        if not self.in_script_or_style and data.strip():
            self.text.append(data.strip())

    def get_text(self):
        return ' '.join(self.text)

def fetch_and_parse_url(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            parser = TextExtractor()
            parser.feed(html)
            text = parser.get_text()
            # truncate to avoid blowing up context limits
            return text[:4000] if len(text) > 4000 else text
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

# ── ANSI colours ─────────────────────────────────────────────────────────────

BOLD      = "\033[1m"
DIM       = "\033[2m"
ITALIC    = "\033[3m"
CYAN      = "\033[36m"
GREEN     = "\033[32m"
YELLOW    = "\033[33m"
MAGENTA   = "\033[35m"
RED       = "\033[31m"
WHITE     = "\033[97m"
RESET     = "\033[0m"

BANNER = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════╗
║          🎵  moecher-chat  🎵                    ║
║      DeepSeek V4-Flash • bare-metal engine       ║
╚══════════════════════════════════════════════════╝{RESET}
{DIM}Type your message and press Enter. Commands:{RESET}
  {YELLOW}/clear{RESET}       — reset conversation
  {YELLOW}/system{RESET}      — set system prompt
  {YELLOW}/temp N{RESET}      — change temperature (current: {{temp}})
  {YELLOW}/tokens N{RESET}    — change max tokens  (current: {{max_tokens}})
  {YELLOW}/budget N{RESET}    — change thinking budget (current: {{budget}})
  {YELLOW}/reasoning X{RESET} — set reasoning effort (current: {{reasoning}})
  {YELLOW}/quit{RESET}        — exit
"""


def chat_completion_stream(url: str, messages: list, max_tokens: int,
                           temperature: float, model: str = "deepseek-v4-flash", tools: list = None,
                           thinking: str = "enabled", reasoning_effort: str = "high",
                           thinking_budget: int = 4096):
    """Send a chat completion request and yield assistant reply chunks."""
    payload_dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "thinking": {
            "type": thinking,
            "budget_tokens": thinking_budget
        },
        "max_thinking_tokens": thinking_budget,
        "reasoning_effort": reasoning_effort
    }
    #if tools:
    #    payload_dict["tools"] = tools
    
    payload = json.dumps(payload_dict).encode("utf-8")

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
                        with open("chat_response.log", "a", encoding="utf-8") as f:
                            json.dump(chunk, f, indent=2, ensure_ascii=False)
                            f.write("\n")
                        if "usage" in chunk:
                            yield {"usage": chunk["usage"]}
                        
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content

                            reasoning = delta.get("reasoning_content", "")
                            if reasoning:
                                yield {"type": "reasoning", "content": reasoning}
                                
                            tool_calls = delta.get("tool_calls")
                            if tool_calls:
                                yield {"type": "tool_calls", "tool_calls": tool_calls}
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
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="Model name (default: deepseek-v4-flash)")
    parser.add_argument("--show-reasoning", action="store_true",
                        help="Show the model's reasoning process in gray")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled",
                        help="Enable or disable the thinking block (default: enabled)")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default="high",
                        help="Reasoning effort level (default: high)")
    parser.add_argument("--thinking-budget", type=int, default=4096,
                        help="Thinking token budget (default: 4096)")
    args = parser.parse_args()

    temperature = args.temperature
    max_tokens = args.max_tokens
    thinking = args.thinking
    reasoning_effort = args.reasoning_effort
    thinking_budget = args.thinking_budget
    default_system = "You are a helpful assistant"

    messages: list[dict] = [{"role": "system", "content": default_system}]

    tools = [{
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and parse text content from a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."}
                },
                "required": ["url"]
            }
        }
    }]

    # Try to get terminal width
    try:
        term_width = min(os.get_terminal_size().columns, 100)
    except OSError:
        term_width = 80

    print(BANNER.format(temp=temperature, max_tokens=max_tokens, reasoning=reasoning_effort, budget=thinking_budget))

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
            elif (cmd[0] in ("/budget", "/thinking-budget")) and len(cmd) > 1:
                try:
                    thinking_budget = int(cmd[1])
                    print(f"{YELLOW}Thinking budget set to {thinking_budget}{RESET}\n")
                except ValueError:
                    print(f"{RED}Invalid budget value{RESET}\n")
                continue
            elif cmd[0] == "/reasoning" and len(cmd) > 1:
                level = cmd[1].lower()
                if level in ["low", "medium", "high", "xhigh", "max"]:
                    reasoning_effort = level
                    print(f"{YELLOW}Reasoning effort set to {reasoning_effort}{RESET}\n")
                else:
                    print(f"{RED}Invalid reasoning effort. Use: low, medium, high, xhigh, max{RESET}\n")
                continue
            elif cmd[0] == "/help":
                print(BANNER.format(temp=temperature, max_tokens=max_tokens, reasoning=reasoning_effort, budget=thinking_budget))
                continue
            else:
                print(f"{RED}Unknown command. Type /help for options.{RESET}\n")
                continue

        # ── Send message ──────────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})

        while True:
            print(f"{CYAN}{BOLD}Assistant ❯{RESET} ", end="", flush=True)

            reply = ""
            reasoning_text = ""
            tool_calls = []
            token_count = 0
            prompt_tokens = 0
            start_time = time.time()
            first_token_time = None
            was_reasoning = False

            try:
                for chunk in chat_completion_stream(
                    args.url, messages, max_tokens, temperature, args.model, tools=tools,
                    thinking=thinking, reasoning_effort=reasoning_effort,
                    thinking_budget=thinking_budget
                ):
                    is_reasoning = False
                    if isinstance(chunk, dict):
                        if "usage" in chunk:
                            prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                            if "completion_tokens" in chunk["usage"]:
                                token_count = chunk["usage"]["completion_tokens"]
                            continue
                        elif chunk.get("type") == "reasoning":
                            is_reasoning = True
                        elif chunk.get("type") == "tool_calls":
                            # Server sent structured tool_calls (not used in streaming currently)
                            continue

                    if first_token_time is None:
                        first_token_time = time.time()

                    token_count += 1

                    if is_reasoning:
                        was_reasoning = True
                        if args.show_reasoning:
                            print(f"{DIM}{chunk['content']}{RESET}", end="", flush=True)
                        reasoning_text += chunk["content"]
                        continue

                    if was_reasoning:
                        if args.show_reasoning:
                            print()
                        was_reasoning = False

                    chunk_str = chunk if isinstance(chunk, str) else chunk.get("content", "")
                    
                    # Prevent printing <tool_call> to the user
                    if "<tool_call>" in reply + chunk_str:
                        pass # Hide tool call XML from user
                    else:
                        print(f"{WHITE}{chunk_str}{RESET}", end="", flush=True)
                    reply += chunk_str
            except Exception as e:
                pass
                
            end_time = time.time()
            ttft = (first_token_time - start_time) if first_token_time else 0.0
            decode_time = (end_time - first_token_time) if first_token_time else 0.0
            
            prefill_tps = prompt_tokens / ttft if ttft > 0 else 0.0
            decode_tps = (token_count - 1) / decode_time if decode_time > 0 and token_count > 1 else 0.0
            
            print(f"\n{DIM}[TTFT: {ttft:.2f}s ({prefill_tps:.2f} tok/s) | Decode: {token_count} tokens in {decode_time:.2f}s, {decode_tps:.2f} tok/s]{RESET}\n")

            # Extract <tool_call> from reply or reasoning_text if present
            import re
            
            def extract_and_remove_tool(text):
                tc = None
                if text and "<tool_call>" in text:
                    match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
                    if match:
                        try:
                            tc_data = json.loads(match.group(1))
                            tc = {
                                "id": "call_" + str(int(time.time())),
                                "type": "function",
                                "function": {
                                    "name": tc_data.get("name", ""),
                                    "arguments": json.dumps(tc_data.get("arguments", {}))
                                }
                            }
                            text = text[:match.start()] + text[match.end():]
                        except Exception as e:
                            print(f"{RED}Failed to parse tool call JSON: {e}{RESET}")
                return text, tc

            reply, tc1 = extract_and_remove_tool(reply)
            reasoning_text, tc2 = extract_and_remove_tool(reasoning_text)
            
            if tc1: tool_calls.append(tc1)
            if tc2: tool_calls.append(tc2)

            # Save assistant reply
            assistant_msg = {"role": "assistant"}
            if reply: assistant_msg["content"] = reply
            if reasoning_text: assistant_msg["reasoning_content"] = reasoning_text
            if tool_calls: assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            
            # Execute tool calls if any
            if tool_calls:
                for tc in tool_calls:
                    if tc["function"]["name"] == "fetch_url":
                        try:
                            args_json = tc["function"]["arguments"]
                            url = json.loads(args_json).get("url")
                            if not url: raise ValueError("No URL provided")
                            
                            print(f"{MAGENTA}{BOLD}[System] Fetching URL: {url} ...{RESET}")
                            content = fetch_and_parse_url(url)
                            print(f"{DIM}[System] Retrieved {len(content)} characters. Sending back to model...{RESET}\n")
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "name": "fetch_url",
                                "content": content
                            })
                        except Exception as e:
                            print(f"{RED}[System] Tool execution error: {e}{RESET}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "name": "fetch_url",
                                "content": f"Error executing tool: {e}"
                            })
                # Loop continues, assistant generates again
            else:
                break # No tool call, wait for next user input
                
        print()

if __name__ == "__main__":
    main()

# MinnieTheMoEcher — Tooling, Proxy, Impersonation & Preview Suite

Welcome to the **MinnieTheMoEcher Tooling Suite** (`feature/tooling`). This subsystem turns **MinnieTheMoEcher** (`moecher.exe`) into an autonomous, high-speed agentic powerhouse with native filesystem manipulation, universal web search, dual forward/reverse proxying, session impersonation, cookie persistence, and interactive multimedia previews.

---

## 🌟 Key Highlights & Features

### 1. 🧰 Native Agentic Tool Suite
The local LLM engine (DeepSeek V4-Flash / Qwen 3.8) can autonomously inspect, synthesize, and modify resources using strict schema-driven tool calling:
- **`read_file`**: Read local filesystem files with 1-indexed line numbers and slice ranges.
- **`write_file`**: Create new files or overwrite existing files with automatic parent directory generation.
- **`edit_file`**: Perform surgical search-and-replace edits on code chunks with single-match safety verification.
- **`execute_command`**: Execute terminal and shell commands with strict safety guardrails and configurable timeouts.
- **`fetch_url`**: Fetch, parse, strip, and extract text/HTML from web pages, APIs, search engines, and YouTube streams.

### 2. 🔍 Universal Multi-Provider Web Search
Seamlessly switch between leading web search backends from the Web UI or via API:
- **Tavily Search**: Fast, LLM-optimized web search results.
- **Brave Search**: High-quality, independent web index API.
- **SearXNG**: Zero-key, self-hosted or public privacy-first meta-search engine instances.
- **Serper.dev**: Google Search results via structured JSON API.
- **Google Custom Search JSON API**: Official Google CSE integration with Programmable Search Engine ID (`cx`).
- **Direct Google Scrape**: Zero-config fallback with automated `SOCS` consent cookie injection to bypass EU consent walls and `gbv=1` rendering.

### 3. 🌐 Dual Forward & Reverse Proxy Engine
- **Rewriting Reverse Proxy (`/api/proxy?url=...`)**:
  - Live cross-origin unblocking with CORS headers (`Access-Control-Allow-Origin: *`, `X-Frame-Options: ALLOWALL`).
  - Automatic CSP and restrictive `<meta>` tag neutralization.
  - Injected JavaScript network shim intercepting `window.fetch`, `XMLHttpRequest`, `navigator.sendBeacon`, `window.open`, and form submissions.
  - Resolves `<base href>` collisions to guarantee that all relative links, styles, and assets route back through the Moecher proxy.
  - Telemetry and analytics suppression (`/li/track`, `googletagmanager`, `doubleclick`, etc.) to eliminate background infinite loops.
- **Forward PAC Proxy (Port 8002)**:
  - Built-in forward proxy with dynamic Proxy Auto-Configuration (`http://localhost:8001/proxy.pac`).
  - One-click Windows System Proxy scripts (`/api/proxy/setup.bat` and `/api/proxy/restore.bat`).
  - Automated clean-up on server shutdown.

### 4. 🍪 Session Impersonation & Multi-Hop Cookie Jar
- **Persistent Cookie Storage**: Thread-safe persistence to `.moecher_cookies.json` and `cookie_jar.json`.
- **Redirect Interception Loop**: Native WinHTTP client uses step-by-step redirect handling (`WINHTTP_OPTION_REDIRECT_POLICY_NEVER` for up to 7 hops) to capture `Set-Cookie` headers from intermediate responses (`301`, `302`, `303`, `307`, `308`).
- **Interactive In-Preview Authentication**: Log in to services (e.g. LinkedIn, GitHub, private portals) directly in the preview panel; session tokens (`li_at`, `bcookie`, `JSESSIONID`) are automatically saved to disk and reused by the LLM crawler.
- **64 MB Payload Capacity**: Server configured with 64 MB form URL-encoded payload limit to eliminate `413 (Payload Too Large)` errors during complex login submissions.
- **Authwall Bypass**: Automatic challenge detection for LinkedIn Error 999 with redirect resolution.

### 5. 🎬 Interactive Preview & Media Player
- **Decoupled Search vs Preview**: Search results scraping runs in a non-disruptive background context so ongoing video or audio playback is never interrupted.
- **Instant YouTube Autoplay**: YouTube URLs automatically resolve oEmbed metadata and launch an interactive HTML5 video player in the preview panel.
- **Static Search Result Cards**: Rich search results formatted with responsive cards, domain badges, direct links, and snippets.
- **Live Code & DOM Views**: Switch between rendered interactive preview, raw HTML code editor, and external browser tabs.

---

## 🏛 Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Moecher Client / Web UI                       │
│  - Chat Interface & System Prompt                                      │
│  - Agentic Settings Tab (Search Providers, Timeouts, Guardrails)       │
│  - HTML Preview Panel & YouTube Player                                 │
│  - Proxy Control Panel (PAC Download, System Proxy Setup)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / Server-Sent Events (SSE)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     moecher.exe Server Engine                          │
│                                                                        │
│   1. ChatML Template + <tools> XML Schema Injection                    │
│   2. High-Speed Inference (CUDA 13.3 + Tensor Core Kernels)           │
│   3. Tool Extraction: <tool_call> JSON blocks                          │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                     Tool Execution Dispatcher                  │   │
│   │                                                                │   │
│   │   [Security Guardrails]                                        │   │
│   │   ├─ Command Blacklist (format, del, rmdir, shutdown, etc.)    │   │
│   │   └─ Workspace Boundary Check (is_path_inside_workspace)       │   │
│   │                                                                │   │
│   │   [Local System Tools]                                         │   │
│   │   ├─ read_file       (std::filesystem + 1-indexed lines)       │   │
│   │   ├─ write_file      (recursive directory creation)            │   │
│   │   ├─ edit_file       (exact match search-and-replace)          │   │
│   │   └─ execute_command (Win32 CreateProcessW / POSIX fork-exec)  │   │
│   │                                                                │   │
│   │   [Web & Search Suite]                                         │   │
│   │   ├─ Universal Search Dispatcher (Tavily, Brave, SearX, etc.)  │   │
│   │   ├─ WinHTTP Step-by-Step Redirect & Cookie Interceptor        │   │
│   │   └─ Persistent Cookie Jar (cookie_jar.json)                   │   │
│   │                                                                │   │
│   │   [Network Proxies]                                            │   │
│   │   ├─ Rewriting Reverse Proxy (/api/proxy) + JS Client Shim     │   │
│   │   └─ Forward PAC Proxy (port 8002, /proxy.pac)                 │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│   4. Stream Inline Activity Badges & Retrieved Documents to UI        │
│   5. Feed <tool_response> back to LLM for multi-turn execution         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Build the Binary
Ensure you have Visual Studio 2019/2022 (with MSVC C++17) and CUDA 13.x installed:
```bat
build_cuda13.bat
```
*(This automatically runs `scripts/embed_web.py` and compiles `build/moecher.exe` with Ninja).*

### 2. Start the Engine
```bat
start.bat
```
The server will start listening on:
- **Web UI & API**: `http://localhost:8001`
- **Forward Proxy**: `http://localhost:8002`

### 3. Configure Search & Proxy
Open `http://localhost:8001` in your browser:
1. Navigate to **Agentic Settings**.
2. Select your preferred **Search Provider** (Tavily, Brave, SearXNG, Serper, or Google) and paste your API key.
3. Click **"Enable Windows System Proxy"** or download `setup.bat` to route all browser traffic through the authenticated Moecher proxy.

---

## 📚 Documentation Reference

For in-depth guides, code samples, API endpoint specifications, and troubleshooting details, please see:
- 📖 [**`TOOLING-MANUAL.md`**](file:///f:/dev/MinnieTheMoEcher/TOOLING-MANUAL.md) — Comprehensive technical reference and operational manual.
- 📖 [**`README-AGENTIC.md`**](file:///f:/dev/MinnieTheMoEcher/README-AGENTIC.md) — Core security model and workspace boundaries.
- 📖 [**`README.md`**](file:///f:/dev/MinnieTheMoEcher/README.md) — General project overview and engine architecture.

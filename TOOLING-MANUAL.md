# MinnieTheMoEcher — Tooling & Impersonation Manual

This document is the complete operational and technical reference manual for the **MinnieTheMoEcher** (`moecher.exe`) Agentic Tooling, Proxy, Cookie Jar, and DOM Preview subsystems.

---

# Table of Contents
1. [Core Tool Suite Specification](#1-core-tool-suite-specification)
2. [Universal Web Search Engine & Providers](#2-universal-web-search-engine--providers)
3. [Reverse Proxy Architecture (`/api/proxy`)](#3-reverse-proxy-architecture-apiproxy)
4. [Forward PAC Proxy & Windows System Integration](#4-forward-pac-proxy--windows-system-integration)
5. [Session Impersonation & Cookie Persistence](#5-session-impersonation--cookie-persistence)
6. [Interactive HTML & Media Preview Panel](#6-interactive-html--media-preview-panel)
7. [Security Guardrails & Safety Architecture](#7-security-guardrails--safety-architecture)
8. [API & Protocol Reference](#8-api--protocol-reference)
9. [Troubleshooting & Gotchas](#9-troubleshooting--gotchas)

---

## 1. Core Tool Suite Specification

All tools are exposed to the DeepSeek V4-Flash / Qwen 3.8 inference engine via function calling schemas injected into the system prompt. The model outputs `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` blocks that are executed synchronously by the engine.

### `read_file`
Reads the content of a local file with 1-indexed line numbers.
- **Parameters**:
  - `path` *(string, required)*: Relative or absolute path to the file.
  - `start_line` *(integer, optional, default: 1)*: Line number to start reading from.
  - `end_line` *(integer, optional, default: -1)*: Line number to stop reading at (-1 reads to end of file).
- **Output Format**:
  ```text
  1: // file contents line 1
  2: // file contents line 2
  ```
- **Constraints**: Enforces workspace boundary checks unless external path authorization is granted. Truncates output exceeding 64 KB.

### `write_file`
Creates a new file or completely replaces the content of an existing file.
- **Parameters**:
  - `path` *(string, required)*: Target file path.
  - `content` *(string, required)*: Complete text to write into the file.
  - `overwrite` *(boolean, optional, default: true)*: Whether to overwrite existing files.
- **Behavior**: Automatically creates any missing parent directories (`std::filesystem::create_directories`).

### `edit_file`
Performs surgical search-and-replace on a targeted block of text.
- **Parameters**:
  - `path` *(string, required)*: Target file path.
  - `target_content` *(string, required)*: Exact existing text to replace (including indentation and newlines).
  - `replacement_content` *(string, required)*: New text to put in place of `target_content`.
- **Safety**: Fails with an informative error if `target_content` is not found or occurs multiple times in the file.

### `execute_command`
Runs a terminal command or system script.
- **Parameters**:
  - `command` *(string, required)*: Shell command to execute.
- **Behavior**:
  - On Windows: Uses Win32 `CreateProcessW` with standard handles and security checks.
  - On Linux: Uses POSIX `fork()` and `execvp()`.
  - Captures stdout/stderr up to 8 KB before truncating.
  - Enforces configurable execution timeout (default: 60 seconds).

### `fetch_url`
Retrieves and parses text or HTML content from external web pages, APIs, and media endpoints.
- **Parameters**:
  - `url` *(string, required)*: Target URL or search prefix (e.g. `youtube: <query>`, `search: <query>`, `https://...`).
  - `timeout_ms` *(integer, optional, default: 10000)*: Network request timeout in milliseconds.
  - `max_chars` *(integer, optional, default: 4000)*: Maximum character length of clean text returned.
  - `mode` *(string, optional, default: "text")*: Retrieval mode: `"text"`, `"raw"`, or `"scripts"`.
  - `pattern` *(string, optional)*: Keyword pattern to filter contextual snippets from HTML.
- **Behavior**:
  - Automatically applies cookie jar and browser impersonation headers.
  - Intercepts YouTube video links to render instant interactive HTML5 video players.
  - Automatically decodes search engine redirects (DuckDuckGo `uddg=`, Google search cards).

---

## 2. Universal Web Search Engine & Providers

MinnieTheMoEcher integrates a **Universal Search Dispatcher** with support for multiple backends configurable via the Web UI (Agentic Settings tab) or through REST API (`/api/settings/search_provider`).

### Supported Providers

| Provider | Backend Mechanism | API Key / Configuration | Capabilities |
|---|---|---|---|
| **Tavily** | Direct JSON API (`https://api.tavily.com/v1/search`) | `tavily_api_key` (`tvly-...`) | Clean synthesized snippets, raw content, high relevancy. |
| **Brave Search** | Brave Web Search API (`https://api.brave.com/res/v1/web/search`) | `brave_api_key` (`BSA...`) | Independent privacy index, comprehensive web coverage. |
| **SearXNG** | Open-source meta-search (`<searxng_url>/search?format=json`) | `searxng_url` (e.g. `https://searx.be`) | Zero API keys required; self-hostable or public instances. |
| **Serper.dev** | Google Search API wrapper (`https://google.serper.dev/search`) | `serper_api_key` | Real-time Google web, image, and news search in JSON format. |
| **Google CSE** | Official Google Custom Search JSON API | `google_search_api_key` + `google_search_cx` | Exact Google search results within configured search engines. |
| **Direct Google Fallback** | Native WinHTTP / headless scrape (`https://google.com/search?q=...&gbv=1`) | No key required (Automatic `SOCS` consent cookies) | Zero-config fallback with EU cookie wall auto-bypass. |

### Search Configuration API

```http
POST /api/settings/search_provider HTTP/1.1
Content-Type: application/json

{
  "provider": "tavily",
  "tavily_api_key": "tvly-YOUR_KEY",
  "brave_api_key": "",
  "serper_api_key": "",
  "searxng_url": "https://searx.be"
}
```

---

## 3. Reverse Proxy Architecture (`/api/proxy`)

The embedded Rewriting Reverse Proxy unblocks cross-origin web browsing inside the Web UI preview panel while maintaining active session context.

```
Browser Client (Iframe) ────► GET/POST /api/proxy?url=<TargetURL>
                                         │
                                         ▼
                            moecher.exe Backend
                                         │
                        1. Inject Impersonation Headers
                        2. Fetch via WinHTTP / Redirect Loop
                        3. Extract & Store Set-Cookie Headers
                        4. Strip CSP & X-Frame-Options
                        5. Inject JS Network & Navigation Shim
                        6. Inject <base href="...">
                                         │
                                         ▼
                                   Rendered HTML
```

### Shim Injections & Features
1. **Network Interception**:
   - Overrides `window.fetch` and `XMLHttpRequest.prototype.open` to automatically transform all requests into `/api/proxy?url=...` calls.
   - Overrides `window.open` and click listeners on `<a>` elements (`target="_self"`).
2. **Form Submissions**:
   - Captures form `submit` events and rewrites `form.action` into `/api/proxy?url=<FullActionURL>` so that login POST requests route through the backend proxy.
3. **Base URL Resolution**:
   - Generates absolute URLs (`http://<host>:8001/api/proxy?url=...`) to avoid collision with `<base href>` tags.
4. **Telemetry Neutralization**:
   - Identifies telemetry tracking endpoints (`/li/track`, `litms`, `doubleclick`, `googletagmanager`, etc.) and returns instant synthetic `200 OK` responses to prevent infinite telemetry loops.

---

## 4. Forward PAC Proxy & Windows System Integration

MinnieTheMoEcher includes a built-in **Forward Proxy Server** on port **8002**.

### Dynamic PAC (Proxy Auto-Configuration)
Available at `http://localhost:8001/proxy.pac`:
```javascript
function FindProxyForURL(url, host) {
    if (shExpMatch(host, "localhost") || shExpMatch(host, "127.0.0.1") || isPlainHostName(host)) {
        return "DIRECT";
    }
    return "PROXY <server_host>:8002; DIRECT";
}
```

### One-Click Windows System Setup
- **Enable System Proxy**: Navigate to `http://localhost:8001/api/proxy/setup.bat` or run:
  ```bat
  reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v AutoConfigURL /t REG_SZ /d "http://localhost:8001/proxy.pac" /f
  ```
- **Restore Normal Internet**: Navigate to `http://localhost:8001/api/proxy/restore.bat` or run:
  ```bat
  reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v AutoConfigURL /f
  ```
- **Remote Client Launch**: Run `installer/launch_remote_client.bat <server_ip>` to launch a sandboxed Edge/Chrome instance routed directly through the remote server's proxy.

---

## 5. Session Impersonation & Cookie Persistence

MinnieTheMoEcher allows the LLM and browser preview to impersonate the user's authenticated session across modern web services.

### Storage & Files
- **Files**: `.moecher_cookies.json` and `cookie_jar.json` in the root workspace.
- **Structure**:
  ```json
  {
    "linkedin.com": {
      "li_at": "AQED...",
      "bcookie": "\"v=2&...\"",
      "JSESSIONID": "\"ajax:...\""
    },
    "google.com": {
      "SOCS": "CAESHAgBEhJnd3NfMjAyNDA4MjAtMF9SQzIaAmVuIAEaBgiA_L20Bg",
      "CONSENT": "PENDING+999"
    }
  }
  ```

### Redirect Chain Cookie Capture
When authenticating via `/api/proxy` (e.g. submitting a login form):
1. WinHTTP executes the request with `WINHTTP_OPTION_REDIRECT_POLICY_NEVER`.
2. On receiving `301`, `302`, `303`, `307`, or `308`, all `Set-Cookie` headers are immediately extracted and stored to disk.
3. The `Location` header is parsed and resolved to an absolute URL via `resolve_redirect_url()`.
4. Method switches to `GET` for standard redirects (`301`/`302`/`303`) and the next request attaches all newly set cookies.
5. The process repeats for up to 7 hops, guaranteeing that intermediate session tokens are never lost.

### 64 MB Payload Capacity (Fixing HTTP 413)
The engine defines:
```cpp
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH (64 * 1024 * 1024)
#define CPPHTTPLIB_PAYLOAD_MAX_LENGTH (64 * 1024 * 1024)
#define CPPHTTPLIB_HEADER_MAX_LENGTH (1024 * 1024)
#define CPPHTTPLIB_REQUEST_URI_MAX_LENGTH (1024 * 1024)
```
This prevents `413 (Payload Too Large)` when submitting complex authentication forms with large encrypted tokens and state parameters.

---

## 6. Interactive HTML & Media Preview Panel

The Web UI features a split-pane layout with an interactive preview:

1. **Decoupled Search vs Preview**:
   - Intermediate search results scraping executes in the background without affecting the active preview iframe.
   - Ongoing music or video playback in the preview panel is never interrupted by AI search operations.
2. **Instant YouTube Autoplay**:
   - Direct video URLs or music search requests automatically render a responsive YouTube player with instant autoplay.
3. **Static Result Cards**:
   - Web search results generate clean cards with site icons, direct URLs, and text snippets rather than heavy background executing iframes.
4. **View Switching**:
   - **Preview Tab**: Live rendered HTML/DOM.
   - **Code Editor Tab**: Raw HTML source code with edit capabilities.
   - **External Popout**: Opens the current proxied document in a standalone browser tab.

---

## 7. Security Guardrails & Safety Architecture

MinnieTheMoEcher enforces defense-in-depth safety checks before executing any command or filesystem modification:

### 1. Command Blacklist Filter
The following commands are blocked unconditionally:
- Partitioning & Format: `format`, `diskpart`, `fdisk`, `mkfs`, `dd`
- Destructive File Ops: `del`, `erase`, `rmdir`, `rd`, `rm`, `unlink`, `shred`, `wipe`, `remove-item`
- System Power: `shutdown`, `reboot`, `poweroff`, `stop-computer`, `restart-computer`
- Security & User Accounts: `reg add`, `reg delete`, `net user`, `net localgroup`, `takeown`, `icacls`, `bcdedit`

### 2. Workspace Boundary Confinement
- All file paths in `read_file`, `write_file`, and `edit_file` must resolve inside the workspace directory (`is_path_inside_workspace`).
- Any access attempting directory traversal outside the workspace triggers an interactive authorization modal in the Web UI.

---

## 8. API & Protocol Reference

### Direct Tool Execution Endpoint

```http
POST /api/tool/execute HTTP/1.1
Content-Type: application/json

{
  "name": "fetch_url",
  "arguments": {
    "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "mode": "text"
  }
}
```

### Search Settings Endpoints

- **`GET /api/settings/search_provider`**: Returns current active search provider and credentials status.
- **`POST /api/settings/search_provider`**: Updates search provider and API keys.
- **`GET /api/settings/google_search`**: Returns Google CSE configuration.
- **`POST /api/settings/google_search`**: Updates Google API key and CX search engine ID.

### Proxy Endpoints

- **`GET /api/proxy?url=<TargetURL>`**: Rewriting Reverse Proxy endpoint.
- **`POST /api/proxy?url=<TargetURL>`**: Rewriting Reverse Proxy for form submissions.
- **`GET /api/proxy/status`**: Returns PAC URL, system proxy status, and setup script links.
- **`GET /api/proxy/setup.bat`**: Generates customized one-click system proxy setup batch file.
- **`GET /api/proxy/restore.bat`**: Generates one-click system proxy cleanup batch file.
- **`GET /proxy.pac`**: Dynamic Proxy Auto-Configuration script.

### Inference & Stop Endpoints

- **`POST /v1/chat/completions`**: OpenAI-compatible chat completions endpoint supporting SSE streaming (`event: tool_call`, `event: reasoning_content`).
- **`POST /v1/chat/stop`** and **`POST /v1/stop`**: Aborts current running inference.

---

## 9. Troubleshooting & Gotchas

### 1. `413 (Payload Too Large)` on Form Submissions
- **Symptom**: Submitting a login or large form via `/api/proxy` returns `413`.
- **Fix**: Verify binary is compiled with `-DCPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=67108864` (handled automatically by `build_cuda13.bat`).

### 2. LinkedIn Error 999 or Authwall
- **Symptom**: Proxied LinkedIn page returns status 999 or redirects to `/authwall`.
- **Resolution**: The proxy automatically intercepts `authwall?trk=...`, follows the challenge, and stores the session cookies. Log in via the preview panel to save persistent cookies to `cookie_jar.json`.

### 3. Links Pointing to `https://www.target.com/api/proxy?...`
- **Symptom**: Relative links clicked inside the preview fail because of `<base href>`.
- **Resolution**: The proxy shim script automatically prepends the local server origin (`window.location.origin`) so all clicks stay on the local proxy.

### 4. Stopping Background Proxy
- If system proxy was enabled via `setup.bat`, simply run `restore.bat` or disable the proxy in Windows Settings > Network & Internet > Proxy.

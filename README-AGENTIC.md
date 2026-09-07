# MinnieTheMoEcher — Agentic Tooling, Security Guardrails & Preview Integration

This document provides a comprehensive technical overview and reference manual for all agentic features, file manipulation tools, execution guardrails, and Web UI integrations introduced in the `feature/tooling` branch of **MinnieTheMoEcher** (`moecher.exe`).

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Built-In Tool Suite](#2-built-in-tool-suite)
   - [`read_file`](#read_file)
   - [`write_file`](#write_file)
   - [`edit_file`](#edit_file)
   - [`execute_command`](#execute_command)
   - [`fetch_url`](#fetch_url)
3. [Security Guardrails & Safety Architecture](#3-security-guardrails--safety-architecture)
   - [Prohibited & Dangerous Commands](#prohibited--dangerous-commands)
   - [Workspace Boundary Confinement](#workspace-boundary-confinement)
   - [External Path Authorization Model](#external-path-authorization-model)
4. [Configurable Execution Timeouts](#4-configurable-execution-timeouts)
5. [Agentic Settings Tab in Web UI](#5-agentic-settings-tab-in-web-ui)
6. [Interactive HTML Preview & Media Integration](#6-interactive-html-preview--media-integration)
   - [Retrieved Documents Panel](#retrieved-documents-panel)
   - [Continuous YouTube Autoplay](#continuous-youtube-autoplay)
   - [Inline Reasoning Activity Cards](#inline-reasoning-activity-cards)
7. [Cross-Platform Compatibility (Windows & Linux)](#7-cross-platform-compatibility-windows--linux)
8. [API & Protocol Reference](#8-api--protocol-reference)

---

## 1. Overview & Architecture

MinnieTheMoEcher provides an autonomous **Agentic Tooling Loop** powered by localized high-speed Mixture-of-Experts inference (DeepSeek V4-Flash / Qwen 3.8). The model can dynamically decide to inspect files, execute terminal commands, modify source code, search the web, or display multimedia.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Moecher Client / Web UI                       │
│  - Chat Interface & System Prompt                                      │
│  - Agentic Settings Tab (Timeouts, Guardrails, Active Tools)           │
│  - HTML Preview Panel & Retrieved Documents Viewer                     │
│  - External Path Authorization Modal Dialog                            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / Server-Sent Events (SSE)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     moecher.exe Server Engine                          │
│                                                                        │
│   1. Apply ChatML Template + <tools> XML Schema Injection              │
│   2. Inference Generation (CUDA 13.3 + Tensor Core Matrix Kernels)     │
│   3. Tool Extraction: <tool_call> JSON blocks                          │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                     Tool Execution Dispatcher                  │   │
│   │                                                                │   │
│   │   [Security Guardrails]                                        │   │
│   │   ├─ Command Blacklist Check (rmdir, format, icacls, etc.)     │   │
│   │   └─ Workspace Boundary Check (is_path_inside_workspace)       │   │
│   │                                                                │   │
│   │   [Cross-Platform Handlers]                                    │   │
│   │   ├─ read_file       (std::filesystem + std::ifstream)         │   │
│   │   ├─ write_file      (std::filesystem + std::ofstream)        │   │
│   │   ├─ edit_file       (search-and-replace chunk editing)        │   │
│   │   ├─ execute_command (CreateProcessW / POSIX fork-exec)        │   │
│   │   └─ fetch_url       (WinHTTP / POSIX sockets + HTML parser)   │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│   4. Emit Inline Activity Badges & Retrieved Documents to Client       │
│   5. Feed <tool_response> back to LLM for multi-turn synthesis         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Built-In Tool Suite

All tools are exposed to the model via standard function calling schemas injected into the system prompt.

### `read_file`
Reads the text contents of a file on the local filesystem with 1-indexed line numbering.

- **Parameters**:
  - `path` *(string, required)*: Relative or absolute file path to read.
  - `start_line` *(integer, optional, default: 1)*: 1-indexed line number to start reading from.
  - `end_line` *(integer, optional, default: -1)*: 1-indexed line number to stop reading at (-1 reads to end of file).
- **Behavior**:
  - Validates file existence and prevents directory reads.
  - Formats output as `<line_number>: <content>` for easy referencing.
  - Enforces a 64 KB truncation limit to avoid overflowing model context.

### `write_file`
Creates a new file or completely overwrites an existing file with the provided text content.

- **Parameters**:
  - `path` *(string, required)*: Target file path.
  - `content` *(string, required)*: Complete text content to write.
  - `overwrite` *(boolean, optional, default: true)*: Whether to overwrite if the file already exists.
- **Behavior**:
  - Automatically creates missing parent directories (`std::filesystem::create_directories`).
  - Returns byte count confirmation on success.

### `edit_file`
Performs surgical search-and-replace on a unique block of text within an existing file.

- **Parameters**:
  - `path` *(string, required)*: Target file path.
  - `target_content` *(string, required)*: Exact character sequence to replace, including indentation and newlines.
  - `replacement_content` *(string, required)*: Replacement text to substitute in place of `target_content`.
- **Behavior**:
  - Ensures `target_content` matches exactly once.
  - Returns an informative error if `target_content` is not found or if multiple occurrences make the edit ambiguous.

### `execute_command`
Executes a terminal/shell command on the local operating system (e.g. `dir`, `ls`, `git`, `cmake`, `cargo`, tests, compilers).

- **Parameters**:
  - `command` *(string, required)*: The command line to execute.
- **Behavior**:
  - Pre-screened against dangerous command patterns.
  - Enforces the user-configured execution timeout (default: 60 seconds).
  - Truncates oversized terminal output at 8 KB to preserve context efficiency.

### `fetch_url`
Retrieves readable text content from public HTTP/HTTPS URLs, search engines, or multimedia endpoints.

- **Parameters**:
  - `url` *(string, required)*: Target web URL (e.g., DuckDuckGo search, Wikipedia, documentation, GitHub).
- **Behavior**:
  - Automatic URL sanitization and normalization.
  - DuckDuckGo search result decoding (`uddg=` redirects resolved to direct target URLs).
  - HTML tag stripping and entity decoding (`&lt;`, `&gt;`, `&amp;`, `&quot;`, `&nbsp;`).
  - Automatic metadata extraction from `<title>`, `<meta property="og:title">`, and `<meta name="description">`.
  - Generates interactive autoplay HTML players for YouTube URLs.

---

## 3. Security Guardrails & Safety Architecture

### Prohibited & Dangerous Commands
The system pre-screens every command passed to `execute_command`. Any attempt to execute dangerous system alterations is blocked before shell invocation:

| Prohibited Category | Blocked Keywords & Patterns |
|---|---|
| **Disk Partitioning & Formatting** | `format`, `diskpart`, `fdisk`, `mkfs`, `dd` |
| **File / Directory Destruction** | `del`, `erase`, `rmdir`, `rd`, `rm`, `unlink`, `shred`, `wipe`, `remove-item`, `clear-content` |
| **System Power & Reboot** | `shutdown`, `reboot`, `poweroff`, `stop-computer`, `restart-computer` |
| **Registry & Security Modification** | `reg add`, `reg delete`, `net user`, `net localgroup`, `takeown`, `icacls`, `bcdedit` |

### Workspace Boundary Confinement
By default, the engine restricts all file access (`read_file`, `write_file`, `edit_file`) and command operations to the directory where `moecher.exe` was launched.

- **Lexical & Canonical Resolution**:
  Paths are normalized and resolved via `std::filesystem::canonical` to prevent relative traversal attacks (`../../`).
- **Case-Insensitive Normalization**:
  Path comparisons on Windows normalize casing to prevent bypasses.

### External Path Authorization Model
When the model attempts to access a path outside the workspace root and `require_external_authorization` is enabled:

1. **Backend Interception**:
   The engine halts the operation and returns `[Authorization Required: The path '...' is outside the current workspace directory]`.
2. **Event Streaming**:
   An SSE event `authorization_required` is pushed to the client with the tool name, target path, and call ID.
3. **Interactive Modal**:
   The Web UI displays an authorization prompt with three actions:
   - **Deny**: Rejects access and instructs the model to confine operations to the workspace.
   - **Allow Once**: Authorizes the single turn and proceeds.
   - **Always Allow Path**: Whitelists the path for the entire session, adding it to the authorized paths list in the Agentic Settings tab.

---

## 4. Configurable Execution Timeouts

- **Default Timeout**: **60 Seconds (1 Minute)**, upgraded from the previous 15-second default.
- **Configurable Range**: **5 seconds to 600 seconds (10 minutes)**.
- **Client Synchronization**:
  - The UI provides a synchronized slider (5s to 300s) and numeric stepper (up to 600s).
  - Selected timeouts are persisted in browser `localStorage` and sent with every API request via `execution_timeout_sec`.
  - Process watchdog cleanly terminates unresponsive or hung child processes (`TerminateProcess` on Windows, `kill(SIGKILL)` on Linux).

---

## 5. Agentic Settings Tab in Web UI

A dedicated **Agentic Settings** tab (`#tab-agentic`) is accessible in the right-side preview panel:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ HTML Preview Panel ─ Tabs: [Preview] [Code] [Retrieved] [Console] [Agentic] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  📁 Workspace Root Directory                                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ F:\dev\MinnieTheMoEcher                                           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ⏱️ Tool Execution Timeout                                              │
│  [━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━]  [ 60 ] sec                       │
│  Default: 60 seconds (1 minute). Set up to 600s for large compilations. │
│                                                                         │
│  🛡️ Workspace Guardrails & Path Authorization                           │
│  [X] Enforce Workspace Boundary                                         │
│  [X] Prompt for External Authorization                                  │
│                                                                         │
│  Authorized External Paths (Session):                                   │
│  [ C:\Users\tino\Projects (x) ]  [ D:\Datasets (x) ]                    │
│  [ Add new path...             ] [ + Add ]                              │
│                                                                         │
│  🛠️ Active Tools Configuration                                           │
│  [X] read_file        [X] write_file        [X] edit_file               │
│  [X] execute_command  [X] fetch_url                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

All settings are automatically saved to `localStorage` under `moecher_agentic_*` keys.

---

## 6. Interactive HTML Preview & Media Integration

### Retrieved Documents Panel
- All web pages and search queries fetched via `fetch_url` are logged in the **Retrieved Documents** tab (`#tab-retrieved`).
- Each entry displays the document title, source URL, timestamp, text snippet, and an isolated preview thumbnail.
- Clicking any retrieved document card immediately opens the complete document in the live preview canvas.

### Continuous YouTube Autoplay
- When asked to play media or songs, the model searches DuckDuckGo, retrieves the YouTube URL, and initiates playback.
- **Unmuted Autoplay**: Video player initializes with YouTube IFrame API calling `unMute()`, `setVolume(100)`, and `playVideo()`.
- **Seamless Transition**: When fetching new songs, currently playing audio is uninterrupted until the new video is loaded.
- **Thumbnail Isolation**: Miniature document lists use static image thumbnails with play badges rather than active secondary iframes, eliminating audio cross-talk.

### Inline Reasoning Activity Cards
- During thinking generation (`enable_thinking: true`), tool activities ("Searching Web", "Reading file", "Writing file", "Running command") stream directly inside the active thought block (`reasoning_content`).
- Cards transition in-place from active spinners to completed badges (`#tool-act-...`), avoiding UI duplication.
- The thought block remains open and continuous until final answer synthesis starts.

---

## 7. Cross-Platform Compatibility (Windows & Linux)

The agentic codebase is designed with zero platform lock-in:

| Functionality | Windows Implementation | Linux Implementation |
|---|---|---|
| **Filesystem & Paths** | `std::filesystem` with case-insensitive normalization | `std::filesystem` POSIX case-sensitive canonical paths |
| **File I/O** | `std::ifstream` / `std::ofstream` binary modes | `std::ifstream` / `std::ofstream` binary modes |
| **Command Execution** | `CreateProcessW` with anonymous pipes & `WaitForSingleObject` | `pipe()` + `fork()` + `execvp()` + `poll()` |
| **HTTP Retrieval** | WinHTTP API (`WinHttpOpen`, `WinHttpSendRequest`) | POSIX socket / httplib backend |
| **Process Termination** | `TerminateProcess(hProcess, 1)` | `kill(pid, SIGKILL)` + `waitpid()` |

---

## 8. API & Protocol Reference

### `POST /v1/chat/completions`

#### Request Payload Extensions:
```json
{
  "model": "deepseek-v4-flash",
  "messages": [
    { "role": "user", "content": "Read src/main.cpp and replace version with 2.1" }
  ],
  "stream": true,
  "execution_timeout_sec": 60,
  "workspace_boundary_enforced": true,
  "require_external_authorization": true,
  "authorized_paths": [
    "C:\\Users\\tino\\workspace",
    "*"
  ],
  "tools": [
    { "type": "function", "function": { "name": "read_file", ... } },
    { "type": "function", "function": { "name": "write_file", ... } },
    { "type": "function", "function": { "name": "edit_file", ... } },
    { "type": "function", "function": { "name": "execute_command", ... } },
    { "type": "function", "function": { "name": "fetch_url", ... } }
  ]
}
```

#### SSE Delta Event Types:
- Standard tokens: `delta.content` and `delta.reasoning_content`
- Retrieved web documents:
  ```json
  "delta": {
    "retrieved_document": {
      "id": "tc-1234",
      "url": "https://...",
      "title": "Example",
      "html": "<!DOCTYPE html>...",
      "snippet": "..."
    }
  }
  ```
- External authorization challenge:
  ```json
  "delta": {
    "authorization_required": {
      "tool": "read_file",
      "path": "C:\\Windows\\System32\\drivers\\etc\\hosts",
      "id": "tc-5678"
    }
  }
  ```

### `GET /v1/workspace`

Returns the server's current working directory and configuration defaults:
```json
{
  "workspace_directory": "F:\\dev\\MinnieTheMoEcher",
  "default_timeout_sec": 60
}
```

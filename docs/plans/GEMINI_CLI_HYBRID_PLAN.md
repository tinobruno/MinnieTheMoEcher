# Implementation Plan: Self-Contained Sandboxed `@google/gemini-cli` OAuth & Streaming Bridge

> **Rule #1 of MinnieTheMoEcher (`moecher` / `minniemoe`) — Zero User-Installed Dependencies**:
> The end user installs **only** `moecher.exe` and downloads models—period. They must **never** be asked to manually install Node.js, Python, npm packages, or system bloatware. Whenever `moecher.exe` needs an external tool or bridge (such as `@google/gemini-cli`), `moecher.exe` automatically provisions, isolates, and runs it inside a **local, sandboxed workspace directory (`.moecher_runtime/`)** with zero changes to the host OS, Windows Registry, or global `PATH`.
>
> **Live Verification Note**: Google retired individual Google AI Pro/Ultra consumer OAuth access on the legacy Node.js `@google/gemini-cli` client (`IneligibleTierError: This client is no longer supported for Gemini Code Assist for individuals`) and consolidated consumer Google AI Ultra CLI access into the official standalone single-binary **`agy.exe` (Antigravity CLI)**. Because `agy.exe` is a native self-contained Windows executable, `moecher.exe` downloads `agy.exe` directly into `.moecher_runtime/agy/agy.exe` with **zero Node.js or npm required**, streaming responses via `--input-format stream-json --output-format stream-json`.

---

## 1. Architecture Overview

1. **End User (Installs ONLY `moecher.exe`)** selects **Google Ultra OAuth (`gemini-cli`)** under **Settings → Hybrid Cloud APIs**.
2. **Self-Contained Sandboxed Provisioning (`.moecher_runtime/`)**:
   - `moecher.exe` checks if `.moecher_runtime/node/node.exe` (Node v22 LTS) and `.moecher_runtime/gemini-cli/` (`@google/gemini-cli`) exist.
   - If missing, `moecher.exe` automatically downloads & extracts portable `node-v22.14.0-win-x64.zip` and installs `@google/gemini-cli` locally inside `.moecher_runtime/` (no admin rights, no global install, no system `PATH` changes).
3. **OAuth Login (`~/.gemini/oauth_creds.json`)**:
   - If not yet signed in, `moecher.exe` launches the official `@google/gemini-cli` login flow, opening the user's browser to Google's official OAuth consent screen (`Sign in with Google`, unlocking up to 2,000 req/day on Google AI Ultra).
4. **Headless Streaming Subprocess (`--output-format stream-json`)**:
   - On `/v1/chat/completions`, `moecher.exe` spawns `.moecher_runtime/node/node.exe` with `@google/gemini-cli` in headless streaming mode, piping the prompt over `stdin` and streaming real-time JSON events from `stdout` into `moecher.exe`'s SSE and `<tool_call>` execution loop.

---

## 2. Self-Contained Sandboxed Runtime (`.moecher_runtime/`) in `src/tool_exec.hpp`

Even if the end user has **no Node.js installed** (or an incompatible global Node version), `moecher.exe` handles everything transparently inside `.moecher_runtime/`:

1. **Sandboxed Directory Layout**:
   - `.moecher_runtime/node/node.exe` — Official standalone Windows x64 Node.js v22 LTS binary (`https://nodejs.org/dist/v22.14.0/node-v22.14.0-win-x64.zip`).
   - `.moecher_runtime/node/npm.cmd` — Bundled with the portable zip.
   - `.moecher_runtime/gemini-cli/node_modules/@google/gemini-cli/` — Official unmodified `@google/gemini-cli` package installed locally.
2. **Strict Host Isolation (Zero Bloatware)**:
   - No `.msi` installer, no UAC/Administrator prompt, no global `npm -g`, no Windows Registry edits, and no modification to the user's system `PATH`.
   - Deleting `.moecher_runtime/` removes 100% of the tool footprint.
3. **Helper Functions in `src/tool_exec.hpp`**:
   - `bool is_gemini_cli_runtime_installed()`
   - `bool is_gemini_cli_oauth_authenticated(std::string* out_account_email = nullptr)`
   - `bool bootstrap_gemini_cli_runtime(std::function<void(const std::string&)> on_status)`

---

## 3. Configuration & Persistence (`src/tool_exec.hpp` & `.moecher_config.json`)

1. Extend the Google provider settings in `src/tool_exec.hpp`:
   - `inline std::string g_google_auth_mode = "api_key";` — `"api_key"` (AI Studio REST API) or `"gemini_cli"` (Sandboxed `@google/gemini-cli` Google Ultra OAuth).
2. Persist `"google_auth_mode"` in `.moecher_config.json` via `load_config_from_disk()` and `save_config_to_disk()`.
3. Include `"google_auth_mode"`, `"gemini_cli_installed"`, `"gemini_cli_authenticated"`, and `"gemini_cli_account"` in `get_llm_provider_settings()`.

---

## 4. Backend REST Endpoints (`src/agentic.cpp`)

Add three endpoints in `register_agentic_routes()`:

1. **`GET /api/settings/gemini_cli/status`**:
   - Returns runtime installation status, OAuth login status, detected Google account email, and active `google_auth_mode`.
2. **`POST /api/settings/gemini_cli/setup`**:
   - Automatically downloads & extracts portable `node-v22.14.0-win-x64.zip` into `.moecher_runtime/node/` and installs `@google/gemini-cli` into `.moecher_runtime/gemini-cli/`.
3. **`POST /api/settings/gemini_cli/login`**:
   - Auto-bootstraps `.moecher_runtime/` if needed, then launches `.moecher_runtime/node/node.exe` with `@google/gemini-cli` to open the official browser **"Sign in with Google"** OAuth consent flow.

---

## 5. Headless Streaming Execution (`src/tool_exec.hpp`)

In `stream_external_llm_chat()`:
- When `provider == "google"` and `g_google_auth_mode == "gemini_cli"`:
  1. Serialize the system prompt (including `moecher`'s `<tools>` / `<tool_call>` instructions when tools are active) and multi-turn conversation history into a structured prompt payload.
  2. Spawn `.moecher_runtime/node/node.exe` running `.moecher_runtime/gemini-cli/node_modules/@google/gemini-cli/bundle/gemini.js`:
     - Arguments: `-m <clean_model> --output-format stream-json`
     - Pass the prompt via `stdin` pipe.
  3. Read newline-delimited JSON events (`stream-json`) from the child process's `stdout` pipe in real time and forward text/reasoning deltas directly into `on_delta(text, is_reasoning)`.
  4. Preserve `moecher`'s `<tool_call>` XML tags so `moecher.exe`'s agentic tool execution loop in `src/server.cpp` works identically across Local GPU, Gemini API Key, and `gemini-cli` OAuth modes.

---

## 6. Web UI Integration (`web/index.html` & `web/script.js`)

1. **Auth Mode Segmented Toggle in Google Gemini Card (`web/index.html`)**:
   - **Mode 1: `🔑 AI Studio API Key`** — Uses the direct WinHTTP client with an AI Studio API key.
   - **Mode 2: `🌐 Google Ultra OAuth (gemini-cli)`** — Shows:
     - Runtime & OAuth status badge (`Ready — Signed in` / `Click to Auto-Setup & Sign In`).
     - **[ ⚡ One-Click Sign in with Google Ultra ]** button (auto-provisions `.moecher_runtime/` on first click and opens the Google OAuth browser login).
2. **Topbar & Provenance Badges (`web/script.js`)**:
   - Displays `☁️ CLOUD · Google Gemini Ultra OAuth (<model>)` when `google_auth_mode == "gemini_cli"`.

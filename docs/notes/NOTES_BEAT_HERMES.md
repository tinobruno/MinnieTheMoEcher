# Roadmap & Architecture Notes: Beating Hermes (`hermes-agent`) with MinnieTheMoEcher

> **Core Philosophy of MinnieTheMoEcher (`moecher.exe`)**:
> 1. **Zero User-Installed Dependencies**: Single native C++/CUDA/Metal binary (`moecher.exe`). Any auxiliary runtime (like `agy.exe` in `.moecher_runtime/`) is self-bootstrapped and isolated locally.
> 2. **Engine-Coupled Agentic Execution**: Unlike Python wrappers (Hermes) that re-send full chat histories over HTTP on every tool turn, Moecher's agentic loop runs directly inside the inference server with **0 ms prefill Pinned System KV Snapshots** (`system_kv_snapshot_`) and **incremental KV round continuation** (`turn_kv_snapshot_`).
> 3. **Rich Creative, Visual & Multimodal Workbench**: Out-of-the-box interactive 3D Studio, Image-to-3D reconstruction, live HTML/Canvas/Shader sandbox, authenticated reverse-proxy browsing, and multimodal vision delegation.

---

## Part 1: Features to Implement to Surpass Hermes Completely

### 1. Persistent Cross-Session Context & Episodic Memory (High Priority)
* **Goal**: Match and exceed Hermes's `MEMORY.md` / `USER.md` and SQLite FTS5 `session_search` while leveraging Moecher's GPU KV snapshot advantage.
* **Implementation Plan**:
  * **Curated Persistent Memory (`.moecher_memory.json` / `MEMORY.md`)**:
    * Expose `memory_save`, `memory_update`, and `memory_delete` tools in [`src/agentic.cpp`](src/agentic.cpp).
    * Inject curated user/project facts into [`rebuild_system_prefix()`](src/server.cpp) so persistent memory is baked directly into `system_kv_snapshot_` (**0 ms prefill overhead** on every turn).
  * **Cross-Session Transcript Archive & Search (`session_search`)**:
    * Persist completed conversation turns (`s_last_conv_messages`) to a local `.moecher_sessions/` store indexed by SQLite FTS5 (amalgamation `sqlite3.c` statically linked—zero external DLLs) or a lightweight C++ BM25 inverted index.
    * Add a `session_search(query, limit)` tool so the model can recall past conversations, code fixes, and architectural decisions from weeks ago.

### 2. Isolated Multi-Context Subagents & GPU Swarming (Roadmap — Swarming Engine)
* **Goal**: Beat Hermes's `delegate_task` by running isolated subagent contexts directly against shared GPU weights (plus optional Hybrid Cloud fan-out).
* **Implementation Plan**:
  * **Isolated GPU Context Slots (`ContextSlot` / `TurnKVSnapshot` Pool)**:
    * All subagents share the 100% resident model weights (`attention_dense_layers.bin` / `moe_experts_iq2.bin`) and the read-only `system_kv_snapshot_` in VRAM.
    * When the parent agent calls `delegate_task(prompt, tools_subset, role)`, Moecher snapshots or pauses the parent's `turn_kv_snapshot_`, runs the child loop in an isolated KV slot from the `system_kv_snapshot_` boundary, and returns only the distilled result to the parent's `round_new_tool_msgs`.
  * **Swarming & Hybrid Cloud Parallelism (`origin/features/hybridAI`)**:
    * Allow swarm subagents to execute either sequentially/batched on local GPU KV slots or in parallel across configured Hybrid Cloud endpoints (Google Ultra OAuth via `.moecher_runtime/agy/agy.exe`, AI Studio, Claude, OpenAI) while the local GPU runs another branch.

### 3. Built-In Cron Scheduler & Unattended Background Runs (Roadmap)
* **Goal**: Native C++ scheduled task runner inside `moecher.exe` without external OS cron/Task Scheduler setup.
* **Implementation Plan**:
  * Add a lightweight background scheduler thread in [`src/agentic.cpp`](src/agentic.cpp) persisted to `.moecher_schedules.json`.
  * Expose agent tools: `schedule_create(cron_or_interval, prompt, tools, notify_target)`, `schedule_list()`, `schedule_delete(id)`.
  * When a schedule triggers, the scheduler acquires [`EngineBusyGuard`](src/server.cpp) (or queues until the current interactive turn completes), runs the multi-round agentic loop in [`src/server.cpp`](src/server.cpp), logs the output to `.moecher_schedules/logs/`, and optionally dispatches a notification via MCP (e.g., the custom Microsoft Teams MCP plugin).
  * Add a **Schedules / Automations** card in the Web UI **Agentic & Tools** settings tab.

### 4. Self-Authoring Reusable Skills (`.moecher_skills/` — Optional / Easy Win)
* **Goal**: Allow Moecher to save and reuse procedural workflows (`SKILL.md`) across sessions without bloating the base system prompt.
* **Implementation Plan**:
  * Store skills as markdown files in `.moecher_skills/<skill_name>/SKILL.md` with YAML frontmatter (`name`, `description`, `triggers`).
  * Inject only the 1-line `name: description` directory (~100–200 tokens total) into the system prompt.
  * Add `skill_list()`, `skill_view(name)`, and `skill_save(name, description, markdown_body)` tools in [`src/agentic.cpp`](src/agentic.cpp) so the model can load full instructions on demand or save a newly discovered workflow when instructed.

### 5. Enterprise Messaging via MCP (Teams Plugin) + Windows `stdio` MCP Parity
* **Decision**: Do **not** bloat `moecher.exe` with 20 hardcoded chat bot daemons. Use MCP plugins (such as our custom **Microsoft Teams MCP plugin**) for messaging and enterprise integrations.
* **Action Item — Windows `stdio` MCP Transport in [`src/mcp_client.hpp`](src/mcp_client.hpp)**:
  * Currently [`MCPClientProcess::start()`](src/mcp_client.hpp#L343) and [`send_request_internal()`](src/mcp_client.hpp#L768) implement `stdio` JSON-RPC pipes under `#if !defined(_WIN32)` (with remote HTTP/SSE MCP working cross-platform).
  * Add Win32 `CreatePipe` + `CreateProcessW` + `PeekNamedPipe`/`ReadFile` support in [`src/mcp_client.hpp`](src/mcp_client.hpp) so local `stdio` MCP servers (`npx`, `uvx`, local C++/Python MCP binaries) run natively on Windows identically to Linux.

### 6. Containerization Strategy (Moecher Inside Docker, Not Docker Inside Moecher)
* **Decision**: Never require Docker inside `moecher.exe` for tool execution—keep host execution fast and guarded by workspace path confinement + interactive SSE authorization modals (`authorization_required`) + `.moecher_runtime/` sandboxing.
* **Optional Deployment Asset**: Maintain a clean GPU-enabled `Dockerfile` (`nvidia/cuda` base) so users who want full OS isolation can simply run `moecher` itself inside a container.

---

## Part 2: Where MinnieTheMoEcher Already Beats Hermes Out of the Box

Unlike Hermes (which is a terminal-first Python harness with no built-in inference engine or visual creative studio), **Moecher** already ships with:

1. **Creative & Visual Studio Capabilities (Out of the Box)**
   * **Interactive 3D Studio & Image-to-3D CAD/Mesh Reconstruction (`create_3d_model`, `ThreeStudio` in [`web/script.js`](web/script.js#L7228))**:
     * Analytic 3D modeling from text or uploaded photos (`inputImage`) with semantic part decomposition (`helpers.createPart`), watertight lathe/vessel/tube generators (`createWatertightLathe`, `createHollowVessel`, `createCappedTube`), normalized sub-region photo texture cropping (`helpers.cropTexture`), realistic pixel color sampling (`helpers.sampleColor`), planar/cylindrical/box UV projection, **interactive Vertex Edit Mode**, 3D transform gizmos, full PBR Material & UV Tiling Inspector, and **1-click `.GLB` / `.OBJ` export**.
   * **Live HTML5 / Canvas / WebGL / Shader Workbench (`#tab-preview`, `#tab-code`, `#tab-console`)**:
     * Instant live execution of generated web apps, games, interactive visualizations, and shaders with Desktop/Tablet/Mobile viewport emulation, live code editor, and real-time JS console error interception.
   * **Continuous Unmuted YouTube & Media Engine (`youtube_search`)**:
     * Zero-API-cost YouTube search and unmuted autoplay player decoupled from background agentic web scraping so music/video playback is never interrupted during multi-round tool calls.
   * **Narrated 1080p Video & Neural Voice Generation Pipelines**:
     * Automated slide rendering + Microsoft Azure Neural Voice synthesis (`generate_neural_narration.py`, `generate_bolt_narration_video.py`) producing publication-ready 1080p MP4 technical briefings.
   * **Vertical Feature Suites Across Branches**:
     * **Hybrid Cloud AI Routing (`origin/features/hybridAI`)**: Zero-dependency sandboxed Google AI Ultra OAuth (`agy.exe` in `.moecher_runtime/`), AI Studio, Claude, and OpenAI routing inside the same UI and tool loop.
     * **PDF Ingestion & Deterministic UBL 2.1 Odoo XML Invoicing (`feature/pdf-odoo-invoicing`)**.
     * **Multi-Chain Crypto Tax & On-Chain Ledger Accounting (`origin/features/cryptoaccounting`)**.

2. **Bare-Metal C++/CUDA/Metal Inference & Agentic Speed**
   * **Native Multi-Model Engine**: DeepSeek V4 / V4.1-Flash (up to 1M context, O(640) compacted shared-memory MLA, mixed NVFP4 Hot + IQ2_XXS Cold), NVIDIA Nemotron 3.5 Bolt & Lightning (asynchronous double-buffered PCIe DMA, 3.86x faster than `llama.cpp`), and Qwen 3.8 27B (~98–112 tok/s MTP speculative decoding + Apple Metal MPP tiled engine).
   * **Zero Decode Bypass (ZDB) (`/v1/decide`)**: Sub-100ms (0 decode passes) System 1 classification and tool routing with automatic System 1 -> System 2 confidence arbiter fallback.
   * **Authenticated Rewriting Reverse Proxy (`/api/proxy`) & Forward PAC Proxy (`:8002`)**: 7-hop WinHTTP cookie capture (`cookie_jar.json`) allowing the agent's web crawler to share live authenticated browser sessions.


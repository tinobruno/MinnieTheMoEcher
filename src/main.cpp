#include "common.hpp"
#include "engine.hpp"
#include "server.hpp"
#include "logging.hpp"
#include "version.hpp"
#include "agentic.hpp"
#include "multimodal.hpp"
#include "mcp_client.hpp"
#include "tool_exec.hpp"
#include <filesystem>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#endif

int main(int argc, char** argv) {
    // Low-latency active spin-wait on CUDA stream completions (eliminates OS sleep latency)
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);

#if defined(_WIN32) || defined(_WIN64)
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    timeBeginPeriod(1); // 1ms high-resolution timer period on Windows
#endif
    std::string manifest_path = "moecher_manifest.json";
    int port = 8001;
    float max_vram_gb = 0.0f;
    float dram_cache_gb = 0.0f;
    bool buffered_io = false;
    std::string log_path = "moecher.log";
    std::string expert_dtype_override = "";
    int default_thinking_budget = 4096;
    std::string imatrix_dataset = "";
    std::string imatrix_out = "";
    int imatrix_max_tokens = -1;
#if defined(__APPLE__)
    bool enable_pld = true;
    int pld_draft_tokens = 4;
    bool enable_mtp = true;
    bool enable_tree_mtp = false;
    int tree_nodes = 2; // Default 2 alternative branches (Top-1 and Top-2, M=3 total nodes)
    bool enable_spec = true;
#else
    bool enable_pld = true;
    int pld_draft_tokens = 4;
    bool enable_mtp = true;
    bool enable_tree_mtp = false;
    int tree_nodes = 2;
    bool enable_spec = true;
#endif
    std::string test_prompt = "";
    bool benchmark_mode = false;
    float test_temp = 0.7f;
    int test_max_tokens = 150;
    bool test_thinking = true;
    int proxy_port = 8002;
    bool enable_forward_proxy = true;
    bool enable_system_proxy = false;
    std::string batched_prefill_mode = "auto";
    int prefill_chunk_size = 512;
    int max_seq_len_override = 0;
    int mtp_k = 0;
    bool enable_vision = true;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --manifest, -m <path>       Path to moecher_manifest.json\n");
            printf("  --port, -p <port>           Server port (default: %d)\n", port);
            printf("  --proxy-port, -pp <port>    Proxy port (default: %d)\n", proxy_port);
            printf("  --max-vram, -V <gb>         Max VRAM budget in GB\n");
            printf("  --dram-cache-gb <gb>        DRAM cache budget in GB\n");
            printf("  --ctx, -c <tokens>          Max context length in tokens (default: 65536)\n");
            printf("  --max-seq-len <tokens>      Alias for --ctx\n");
            printf("  --no-mtp, --disable-mtp     Disable MTP self-speculative decoding\n");
            printf("  --no-tree-mtp               Disable Tree Attention on MTP speculative decoding\n");
            printf("  --tree-mtp                  Enable Tree Attention on MTP speculative decoding\n");
            printf("  --tree-nodes <n>            Number of alternative branches for Tree MTP (default: 2, range: 1..4)\n");
            printf("  --mtp-k <n>                 Max MTP draft depth (default: 1 on 16GB GPUs, 4 on >=24GB GPUs)\n");
            printf("  --allow-host-exec           Allow server-side host OS command execution (DANGEROUS)\n");
            printf("  --allow-file-write          Allow file write/edit operations inside workspace sandbox\n");
            printf("  --workspace <path>          Set restricted workspace sandbox root (default: ./workspace)\n");
            printf("  --tools / --no-tools        Enable or disable tool calling schemas (default: disabled)\n");
            printf("  --budget <tokens>           Default thinking budget tokens\n");
            printf("  --max-tool-rounds <n>       Max sequential tool execution rounds per turn (default: %d)\n", g_max_tool_rounds);
            printf("  --help, -h                  Show this help message\n");
            return 0;
        } else if ((std::string(argv[i]) == "--manifest" || std::string(argv[i]) == "-m") && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if ((std::string(argv[i]) == "--port" || std::string(argv[i]) == "-p") && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if ((std::string(argv[i]) == "--proxy-port" || std::string(argv[i]) == "-pp") && i + 1 < argc) {
            proxy_port = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--no-proxy" || std::string(argv[i]) == "--disable-proxy") {
            enable_forward_proxy = false;
        } else if (std::string(argv[i]) == "--system-proxy" || std::string(argv[i]) == "--auto-proxy") {
            enable_system_proxy = true;
        } else if (std::string(argv[i]) == "--log" && i + 1 < argc) {
            log_path = argv[++i];
        } else if ((std::string(argv[i]) == "--max-vram" || std::string(argv[i]) == "-V") && i + 1 < argc) {
            max_vram_gb = std::stof(argv[++i]);
        } else if (std::string(argv[i]) == "--dram-cache-gb" && i + 1 < argc) {
            dram_cache_gb = std::stof(argv[++i]);
        } else if (std::string(argv[i]) == "--expert-dtype" && i + 1 < argc) {
            expert_dtype_override = argv[++i];
        } else if ((std::string(argv[i]) == "--thinking-budget" || std::string(argv[i]) == "--budget" || std::string(argv[i]) == "--max-thinking-tokens") && i + 1 < argc) {
            default_thinking_budget = std::stoi(argv[++i]);
        } else if ((std::string(argv[i]) == "--temp" || std::string(argv[i]) == "--temperature") && i + 1 < argc) {
            test_temp = std::stof(argv[++i]);
        } else if ((std::string(argv[i]) == "--max-tokens" || std::string(argv[i]) == "-n") && i + 1 < argc) {
            test_max_tokens = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--no-think" || std::string(argv[i]) == "--no-thinking") {
            test_thinking = false;
        } else if (std::string(argv[i]) == "--imatrix-dataset" && i + 1 < argc) {
            imatrix_dataset = argv[++i];
        } else if (std::string(argv[i]) == "--imatrix-out" && i + 1 < argc) {
            imatrix_out = argv[++i];
        } else if (std::string(argv[i]) == "--imatrix-max-tokens" && i + 1 < argc) {
            imatrix_max_tokens = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--test-prompt" && i + 1 < argc) {
            test_prompt = argv[++i];
        } else if (std::string(argv[i]) == "--benchmark") {
            benchmark_mode = true;
        } else if (std::string(argv[i]) == "--no-pld" || std::string(argv[i]) == "--disable-pld") {
            enable_pld = false;
        } else if (std::string(argv[i]) == "--no-mtp" || std::string(argv[i]) == "--disable-mtp") {
            enable_mtp = false;
        } else if (std::string(argv[i]) == "--no-tree-mtp" || std::string(argv[i]) == "--disable-tree-mtp") {
            enable_tree_mtp = false;
        } else if (std::string(argv[i]) == "--tree-mtp" || std::string(argv[i]) == "--enable-tree-mtp") {
            enable_tree_mtp = true;
        } else if ((std::string(argv[i]) == "--tree-nodes" || std::string(argv[i]) == "--tree-branches") && i + 1 < argc) {
            tree_nodes = std::clamp(std::stoi(argv[++i]), 1, 4);
        } else if ((std::string(argv[i]) == "--mtp-k" || std::string(argv[i]) == "--mtp-depth") && i + 1 < argc) {
            mtp_k = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--no-vision" || std::string(argv[i]) == "--disable-vision") {
            enable_vision = false;
        } else if (std::string(argv[i]) == "--no-spec" || std::string(argv[i]) == "--disable-spec") {
            enable_spec = false;
        } else if ((std::string(argv[i]) == "--pld-tokens" || std::string(argv[i]) == "--pld-draft-tokens") && i + 1 < argc) {
            pld_draft_tokens = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--log-experts") {
            g_log_experts = true;
        } else if (std::string(argv[i]) == "--no-log-tokens") {
            g_log_tokens = false;
        } else if (std::string(argv[i]) == "--quiet" || std::string(argv[i]) == "-q") {
            g_quiet = true;
            g_log_tokens = false;
        } else if (std::string(argv[i]) == "--buffered-io") {
            buffered_io = true;
        } else if (std::string(argv[i]) == "--track" ||
                   std::string(argv[i]) == "--track-all-moe" ||
                   std::string(argv[i]) == "-track-all-moe" ||
                   std::string(argv[i]) == "--track-moe-all" ||
                   std::string(argv[i]) == "-track-moe-all" ||
                   std::string(argv[i]) == "--track-moe" ||
                   std::string(argv[i]) == "-track-moe" ||
                   std::string(argv[i]) == "-track") {
            g_track_experts = true;
        } else if (std::string(argv[i]) == "--no-tools" || std::string(argv[i]) == "--disable-tools") {
            g_enable_tools = false;
        } else if (std::string(argv[i]) == "--tools" || std::string(argv[i]) == "--enable-tools") {
            g_enable_tools = true;
        } else if (std::string(argv[i]) == "--allow-host-exec" || std::string(argv[i]) == "--unsafe-server-tools" || std::string(argv[i]) == "--server-exec") {
            g_server_exec = true;
            g_enable_tools = true;
        } else if (std::string(argv[i]) == "--no-server-exec" || std::string(argv[i]) == "--disable-server-exec") {
            g_server_exec = false;
        } else if (std::string(argv[i]) == "--allow-file-write" || std::string(argv[i]) == "--enable-file-write") {
            g_allow_file_write = true;
        } else if (std::string(argv[i]) == "--workspace" && i + 1 < argc) {
            g_workspace_root = argv[++i];
        } else if (std::string(argv[i]) == "--headless-browsing" || std::string(argv[i]) == "-headless-browsing" || std::string(argv[i]) == "--enable-headless-browsing") {
            g_headless_browsing = true;
            g_server_exec = true;
            g_enable_tools = true;
        } else if (std::string(argv[i]) == "--track-reset" ||
                   std::string(argv[i]) == "--reset-track" ||
                   std::string(argv[i]) == "-track-reset" ||
                   std::string(argv[i]) == "-reset-track") {
            g_track_reset = true;
            g_track_experts = true;
        } else if ((std::string(argv[i]) == "--batched-prefill") && i + 1 < argc) {
            batched_prefill_mode = argv[++i];
        } else if (std::string(argv[i]) == "--no-batched-prefill") {
            batched_prefill_mode = "off";
        } else if (std::string(argv[i]) == "--prefill-chunk-size" && i + 1 < argc) {
            prefill_chunk_size = std::stoi(argv[++i]);
        } else if ((std::string(argv[i]) == "--ctx" || std::string(argv[i]) == "--max-seq-len" || std::string(argv[i]) == "-c") && i + 1 < argc) {
            max_seq_len_override = std::stoi(argv[++i]);
        } else if ((std::string(argv[i]) == "--max-tool-rounds" || std::string(argv[i]) == "--max-rounds") && i + 1 < argc) {
            g_max_tool_rounds = std::max(1, std::stoi(argv[++i]));
        }
    }

    // Open log file
    g_log_file.open(log_path, std::ios::app);
    LOG_INFO("=== moecher starting ===");
    LOG_INFO("=== v%s ===", MOECHER_VERSION);
    LOG_INFO("Default thinking token budget: %d", default_thinking_budget);
    LOG_INFO("Tool calling support: %s", g_enable_tools ? "enabled" : "disabled");
    LOG_INFO("Max tool execution rounds: %d", g_max_tool_rounds);
    LOG_INFO("Server-side tool execution: %s (headless-browsing: %s)", g_server_exec ? "enabled" : "disabled", g_headless_browsing ? "enabled" : "disabled");
    LOG_INFO("Workspace sandbox root: %s (Write Allowed: %s)", g_workspace_root.c_str(), g_allow_file_write ? "YES" : "NO");

    if (g_server_exec) {
        LOG_WARN("================================================================================");
        LOG_WARN(" [SECURITY WARNING] Server-side host command execution is ENABLED!");
        LOG_WARN(" Commands generated by models or user prompts will execute directly on this host.");
        LOG_WARN(" Workspace Sandbox: %s (File Write: %s)", g_workspace_root.c_str(), g_allow_file_write ? "ENABLED" : "BLOCKED");
        LOG_WARN(" Do NOT expose this server port to untrusted public networks or unauthorized users.");
        LOG_WARN("================================================================================");
    } else {
        LOG_INFO("Security: Host OS execution is DISABLED (Safe Mode). Pure inference active.");
    }

    MoecherEngine engine;
    engine.batched_prefill_mode_ = batched_prefill_mode;
    engine.prefill_chunk_size_ = prefill_chunk_size;
    engine.enable_pld_ = enable_pld;
    engine.pld_draft_tokens_ = pld_draft_tokens;
    engine.enable_mtp_ = enable_mtp;
    engine.default_enable_mtp_ = enable_mtp;
    engine.enable_tree_mtp_ = enable_tree_mtp;
    engine.tree_nodes_ = tree_nodes;
    engine.mtp_k_ = mtp_k;
    engine.default_mtp_k_ = mtp_k;
    engine.enable_spec_ = enable_spec;
    engine.default_enable_tree_mtp_ = enable_tree_mtp;
    engine.default_tree_nodes_ = tree_nodes;
    engine.default_enable_spec_ = enable_spec;
    engine.enable_vision_ = enable_vision;
    LOG_INFO("Speculative decoding: %s (PLD: %s, MTP: %s, Tree MTP: %s, tree_nodes=%d, mtp_k=%d)",
             enable_spec ? "enabled" : "disabled",
             enable_pld ? "enabled" : "disabled",
             enable_mtp ? "enabled" : "disabled",
             enable_tree_mtp ? "enabled" : "disabled",
             tree_nodes,
             mtp_k);

    try {
        if (!engine.load(manifest_path, max_vram_gb, dram_cache_gb, expert_dtype_override, buffered_io, max_seq_len_override)) {
            LOG_ERROR("Failed to load model");
            return 1;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("FATAL EXCEPTION in engine.load: %s", e.what());
        return 1;
    } catch (...) {
        LOG_ERROR("FATAL UNKNOWN EXCEPTION in engine.load");
        return 1;
    }

    if (!imatrix_dataset.empty() && !imatrix_out.empty()) {
        bool ok = engine.collect_imatrix(imatrix_dataset, imatrix_out, imatrix_max_tokens);
        return ok ? 0 : 1;
    }

    if (!test_prompt.empty() || benchmark_mode) {
        std::string prompt_str = test_prompt.empty() ? "Write a short story about an astronaut." : test_prompt;
        json messages = json::array({
            {{"role", "user"}, {"content", prompt_str}}
        });
        std::vector<int> prompt = apply_chat_template(messages, engine.tokenizer_, test_thinking, "high", json(), engine.cfg_.is_nemotron());
        LOG_INFO("Running test generation for prompt: '%s' (%zu prompt tokens, temp=%.2f, max_tokens=%d, thinking=%d)...",
                 prompt_str.c_str(), prompt.size(), test_temp, test_max_tokens, test_thinking ? 1 : 0);
        std::string result = engine.generate(prompt, test_max_tokens, test_temp, nullptr, 1.0f, test_thinking, default_thinking_budget, 0.95f, 0.0f, 1024);
        printf("\n\n--- Output ---\n%s\n--------------\n", result.c_str());
        return 0;
    }

    // Enable expert frequency tracking if --track flag is set
    std::string freq_path = (std::filesystem::path(manifest_path).parent_path() / "expert_freq.bin").string();
    if (g_track_experts) {
        if (g_track_reset) {
            LOG_INFO("Resetting expert frequency & specialization tracking profile...");
            engine.enable_expert_tracking(""); // Start with empty profile
        } else {
            engine.enable_expert_tracking(freq_path);
        }
    }

    // Initialize MCP (Model Context Protocol) Manager before pre-warming
    auto& mcp_mgr = moecher::mcp::MCPManager::instance();
    std::string mcp_config_file = moecher::tooling::get_workspace_directory() + "/mcp_servers.json";
    mcp_mgr.set_config_path(mcp_config_file);
    if (g_server_exec && mcp_mgr.load_config()) {
        mcp_mgr.start_all_enabled();
    }

    // Configure default base prompt and pre-warm into KV Cache
    std::time_t now = std::time(nullptr);
    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &now);
#else
    localtime_r(&now, &tm_now);
#endif
    char date_buf[128];
    std::strftime(date_buf, sizeof(date_buf), "%A, %B %d, %Y", &tm_now);

    if (!g_enable_tools && !g_server_exec) {
        g_base_system_prompt = "You are a helpful assistant.\nCurrent date: " + std::string(date_buf);
    }

    // Pre-warm default system prompt and tooling into KV Cache for 0ms initial prefill latency
    if (g_enable_tools) {
        json default_tools = resolve_canonical_tools("default");
        json mcp_tools = mcp_mgr.get_openai_tools_schema();
        for (const auto& mt : mcp_tools) {
            default_tools.push_back(mt);
        }
        rebuild_system_prefix(engine, g_base_system_prompt, default_tools);
    } else {
        rebuild_system_prefix(engine, g_base_system_prompt, json::array());
    }

    moecher::server::run_server(engine, port, default_thinking_budget, proxy_port, enable_forward_proxy, enable_system_proxy);

    // Save expert frequency & specialization profile on shutdown
    if (g_track_experts && !engine.expert_freq_counts_.empty()) {
        engine.save_expert_freq(freq_path);
        std::string model_dir = std::filesystem::path(manifest_path).parent_path().string();
        engine.save_expert_profile(model_dir);
    }
    return 0;
}

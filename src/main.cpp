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
#include <fstream>
#include <iostream>
#include <set>
#include <algorithm>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#endif

namespace {

struct DiscoveredModel {
    std::string manifest_path;
    std::string display_name;
    std::string architecture;
    std::string expert_dtype;
    int num_layers = 0;
    int n_experts = 0;
    int max_seq_len = 0;
    bool has_vision = false;
    double total_weights_gb = 0.0;
};

static std::string resolve_manifest_bin_path(const std::filesystem::path& base_dir, const std::string& rel_or_abs) {
    if (rel_or_abs.empty()) return "";
    std::error_code ec;
    std::filesystem::path p(rel_or_abs);
    if (std::filesystem::exists(p, ec) && std::filesystem::is_regular_file(p, ec)) return p.string();
    auto full = base_dir / p;
    if (std::filesystem::exists(full, ec) && std::filesystem::is_regular_file(full, ec)) return full.string();
    auto rel_root = base_dir / ".." / ".." / p;
    if (std::filesystem::exists(rel_root, ec) && std::filesystem::is_regular_file(rel_root, ec)) return rel_root.string();
    return "";
}

static bool inspect_model_manifest(const std::filesystem::path& manifest_file, DiscoveredModel& out) {
    std::error_code ec;
    if (!std::filesystem::exists(manifest_file, ec) || !std::filesystem::is_regular_file(manifest_file, ec)) {
        return false;
    }
    std::string fname = manifest_file.filename().string();
    if (fname.rfind("moecher_manifest", 0) != 0 || manifest_file.extension() != ".json") {
        return false;
    }

    std::ifstream ifs(manifest_file);
    if (!ifs.is_open()) return false;
    json j;
    try {
        ifs >> j;
    } catch (...) {
        return false;
    }
    if (!j.is_object() || !j.contains("model_config") || !j["model_config"].is_object()) {
        return false;
    }

    auto base_dir = manifest_file.parent_path();
    const auto& mc = j["model_config"];

    std::string dense_bin = j.value("dense_bin", "");
    std::string dense_resolved = resolve_manifest_bin_path(base_dir, dense_bin);
    uint64_t total_bytes = 0;
    if (!dense_resolved.empty()) {
        total_bytes = std::filesystem::file_size(dense_resolved, ec);
    } else if (j.contains("gguf_shard1")) {
        std::string s1 = resolve_manifest_bin_path(base_dir, j.value("gguf_shard1", ""));
        if (s1.empty()) return false;
        total_bytes += std::filesystem::file_size(s1, ec);
        std::string s2 = resolve_manifest_bin_path(base_dir, j.value("gguf_shard2", ""));
        if (!s2.empty()) total_bytes += std::filesystem::file_size(s2, ec);
    } else {
        return false; // Weight file not present on disk
    }
    int n_routed = mc.value("n_routed_experts", 0);
    std::string edtype = mc.value("expert_dtype", "");

    if (n_routed > 0 && edtype == "mixed_nvfp4_iq2") {
        std::string cold_p, hot_p;
        if (j.contains("expert_bins") && j["expert_bins"].is_object()) {
            if (j["expert_bins"].contains("cold") && j["expert_bins"]["cold"].is_object()) {
                cold_p = resolve_manifest_bin_path(base_dir, j["expert_bins"]["cold"].value("path", ""));
            }
            if (j["expert_bins"].contains("hot") && j["expert_bins"]["hot"].is_object()) {
                hot_p = resolve_manifest_bin_path(base_dir, j["expert_bins"]["hot"].value("path", ""));
            }
        }
        if (cold_p.empty() && hot_p.empty()) return false;
        if (!cold_p.empty()) total_bytes += std::filesystem::file_size(cold_p, ec);
        if (!hot_p.empty()) total_bytes += std::filesystem::file_size(hot_p, ec);
    } else if (n_routed > 0) {
        std::string exp_bin = j.value("expert_bin", "");
        if (!exp_bin.empty()) {
            std::string exp_resolved = resolve_manifest_bin_path(base_dir, exp_bin);
            if (exp_resolved.empty()) return false;
            total_bytes += std::filesystem::file_size(exp_resolved, ec);
        }
    }

    std::string name = mc.value("model_name", "");
    if (name.empty()) name = mc.value("model_id", "");
    if (name.empty()) {
        name = base_dir.filename().string();
        if (name.empty() || name == ".") name = "DeepSeek-V4-Flash";
    }

    // Distinguish manifest variants in the same folder (e.g., moecher_manifest_mixed.json, moecher_manifest_24gb.json)
    std::string stem = manifest_file.stem().string();
    const std::string prefix = "moecher_manifest_";
    if (stem.rfind(prefix, 0) == 0 && stem.size() > prefix.size()) {
        std::string variant = stem.substr(prefix.size());
        if (name.find(variant) == std::string::npos) {
            name += " [" + variant + "]";
        }
    }

    out.manifest_path = manifest_file.lexically_normal().string();
    out.display_name = name;
    out.architecture = mc.value("architecture", "deepseek_v4");
    out.expert_dtype = edtype.empty() ? (n_routed > 0 ? "iq2" : "dense") : edtype;
    out.num_layers = mc.value("num_hidden_layers", 0);
    out.n_experts = n_routed;
    out.max_seq_len = mc.value("max_seq_len", mc.value("original_seq_len", 32768));
    out.has_vision = mc.value("has_vision", false) || j.contains("visual_config") || mc.contains("vision_config") ||
                     std::filesystem::exists(base_dir / "vision" / "moecher_manifest.json", ec);
    out.total_weights_gb = (double)total_bytes / (1024.0 * 1024.0 * 1024.0);
    return true;
}

static std::vector<DiscoveredModel> discover_available_models(const char* argv0) {
    std::vector<std::filesystem::path> search_roots;
    std::error_code ec;

    auto add_root = [&](const std::filesystem::path& p) {
        if (!p.empty() && std::filesystem::exists(p, ec) && std::filesystem::is_directory(p, ec)) {
            search_roots.push_back(p);
        }
    };

    std::filesystem::path cwd = std::filesystem::current_path(ec);
    add_root(cwd);
    add_root(cwd / "models");
    add_root(cwd / ".." / "models");
    add_root(cwd / ".." / ".." / "models");
    add_root(cwd / ".." / "Moecher" / "models");
    add_root(cwd / ".." / ".." / "Moecher" / "models");

    if (argv0 && argv0[0]) {
        std::filesystem::path exe_dir = std::filesystem::path(argv0).parent_path();
        if (!exe_dir.empty()) {
            add_root(exe_dir);
            add_root(exe_dir / "models");
            add_root(exe_dir / ".." / "models");
            add_root(exe_dir / ".." / "Moecher" / "models");
        }
    }

    if (const char* env_dir = std::getenv("MOECHER_MODELS_DIR")) {
        if (env_dir[0]) add_root(std::filesystem::path(env_dir));
    }

#if defined(_WIN32) || defined(_WIN64)
    for (const char* drive : {"C:", "D:", "E:", "F:", "G:"}) {
        add_root(std::filesystem::path(std::string(drive) + "/Moecher/models"));
        add_root(std::filesystem::path(std::string(drive) + "/models"));
    }
#endif

    std::set<std::string> seen_canonical;
    std::vector<DiscoveredModel> models;

    auto try_add_manifest = [&](const std::filesystem::path& mf) {
        std::error_code c_ec;
        auto canon = std::filesystem::weakly_canonical(mf, c_ec);
        std::string key = c_ec ? mf.lexically_normal().string() : canon.string();
#if defined(_WIN32) || defined(_WIN64)
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return (char)std::tolower(c); });
#endif
        if (seen_canonical.count(key)) return;
        DiscoveredModel dm;
        if (inspect_model_manifest(mf, dm)) {
            seen_canonical.insert(key);
            models.push_back(std::move(dm));
        }
    };

    for (const auto& root : search_roots) {
        for (const auto& entry : std::filesystem::directory_iterator(root, std::filesystem::directory_options::skip_permission_denied, ec)) {
            if (entry.is_regular_file(ec)) {
                try_add_manifest(entry.path());
            } else if (entry.is_directory(ec)) {
                std::string dname = entry.path().filename().string();
                if (!dname.empty() && (dname[0] == '.' || dname == "build" || dname == "src" || dname == "web" || dname == "third_party")) {
                    continue;
                }
                for (const auto& sub : std::filesystem::directory_iterator(entry.path(), std::filesystem::directory_options::skip_permission_denied, ec)) {
                    if (sub.is_regular_file(ec)) {
                        try_add_manifest(sub.path());
                    } else if (sub.is_directory(ec)) {
                        for (const auto& sub2 : std::filesystem::directory_iterator(sub.path(), std::filesystem::directory_options::skip_permission_denied, ec)) {
                            if (sub2.is_regular_file(ec)) {
                                try_add_manifest(sub2.path());
                            }
                        }
                    }
                }
            }
        }
    }

    std::sort(models.begin(), models.end(), [](const DiscoveredModel& a, const DiscoveredModel& b) {
        if (a.display_name != b.display_name) return a.display_name < b.display_name;
        return a.manifest_path < b.manifest_path;
    });

    return models;
}

} // namespace

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
    int mtp_k = 0; // 0 = auto (1 on 16GB GPUs, 4 on >=24GB GPUs)
    bool enable_anti_loop = false;
    bool enable_cpu_experts = true;
    bool enable_vision = true;
    bool manifest_explicit = false;
    bool spec_explicit = false;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --manifest, -m <path>       Path to moecher_manifest.json (omit to choose interactively)\n");
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
            printf("  --anti-loop                 Enable n-gram repetition loop detector & breaker (default: disabled)\n");
            printf("  --cpu-experts, --cpu-moe    Enable Hybrid CPU MoE expert evaluation for L2/L2.5 misses (default: enabled)\n");
            printf("  --no-cpu-experts            Disable Hybrid CPU MoE expert evaluation (DMA all misses to GPU)\n");
            printf("  --max-tool-rounds <n>       Max sequential tool execution rounds per turn (default: %d)\n", g_max_tool_rounds);
            printf("  --mode <chat|agentic>       Operational mode: 'chat' (dynamic intent gating) or 'agentic' (unrestricted tools, default: chat)\n");
            printf("  --chat-mode                 Enable chat mode with dynamic intent gating (outputs code in chat by default)\n");
            printf("  --agentic-mode              Enable full autonomous agentic mode (unrestricted tool & workspace access)\n");
            printf("  --intent-gating             Enable dynamic intent gating (chat mode)\n");
            printf("  --no-intent-gating          Disable dynamic intent gating (agentic mode)\n");
            printf("  --help, -h                  Show this help message\n");
            return 0;
        } else if ((std::string(argv[i]) == "--manifest" || std::string(argv[i]) == "-m" || std::string(argv[i]) == "--model") && i + 1 < argc) {
            manifest_path = argv[++i];
            manifest_explicit = true;
        } else if ((std::string(argv[i]) == "--port" || std::string(argv[i]) == "-p" || std::string(argv[i]) == "--server") && i + 1 < argc) {
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
            spec_explicit = true;
        } else if (std::string(argv[i]) == "--pld" || std::string(argv[i]) == "--enable-pld") {
            enable_pld = true;
            spec_explicit = true;
        } else if (std::string(argv[i]) == "--no-mtp" || std::string(argv[i]) == "--disable-mtp") {
            enable_mtp = false;
            spec_explicit = true;
        } else if (std::string(argv[i]) == "--no-tree-mtp" || std::string(argv[i]) == "--disable-tree-mtp") {
            enable_tree_mtp = false;
            spec_explicit = true;
        } else if (std::string(argv[i]) == "--tree-mtp" || std::string(argv[i]) == "--enable-tree-mtp") {
            enable_tree_mtp = true;
            spec_explicit = true;
        } else if ((std::string(argv[i]) == "--tree-nodes" || std::string(argv[i]) == "--tree-branches") && i + 1 < argc) {
            tree_nodes = std::clamp(std::stoi(argv[++i]), 1, 4);
            spec_explicit = true;
        } else if ((std::string(argv[i]) == "--mtp-k" || std::string(argv[i]) == "--mtp-depth") && i + 1 < argc) {
            mtp_k = std::stoi(argv[++i]);
            spec_explicit = true;
        } else if (std::string(argv[i]) == "--no-vision" || std::string(argv[i]) == "--disable-vision") {
            enable_vision = false;
        } else if (std::string(argv[i]) == "--no-spec" || std::string(argv[i]) == "--disable-spec") {
            enable_spec = false;
            spec_explicit = true;
        } else if (std::string(argv[i]) == "--anti-loop" || std::string(argv[i]) == "--enable-anti-loop") {
            enable_anti_loop = true;
        } else if (std::string(argv[i]) == "--no-anti-loop" || std::string(argv[i]) == "--disable-anti-loop") {
            enable_anti_loop = false;
        } else if (std::string(argv[i]) == "--cpu-experts" || std::string(argv[i]) == "--enable-cpu-experts" || std::string(argv[i]) == "--cpu-moe") {
            enable_cpu_experts = true;
        } else if (std::string(argv[i]) == "--no-cpu-experts" || std::string(argv[i]) == "--disable-cpu-experts" || std::string(argv[i]) == "--no-cpu-moe") {
            enable_cpu_experts = false;
        } else if ((std::string(argv[i]) == "--pld-tokens" || std::string(argv[i]) == "--pld-draft-tokens") && i + 1 < argc) {
            pld_draft_tokens = std::stoi(argv[++i]);
            spec_explicit = true;
        } else if (std::string(argv[i]) == "--log-experts") {
            g_log_experts = true;
        } else if (std::string(argv[i]) == "--no-log-tokens") {
            g_log_tokens = false;
        } else if (std::string(argv[i]) == "--quiet" || std::string(argv[i]) == "-q") {
            g_quiet = true;
            g_log_tokens = false;
        } else if (std::string(argv[i]) == "--verbose" || std::string(argv[i]) == "-v" || std::string(argv[i]) == "--no-quiet" || std::string(argv[i]) == "--log-tokens") {
            g_quiet = false;
            g_log_tokens = true;
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
        } else if (std::string(argv[i]) == "--tools" || std::string(argv[i]) == "--enable-tools" || std::string(argv[i]) == "--allow-ReadFile") {
            g_enable_tools = true;
        } else if (std::string(argv[i]) == "--allow-host-exec" || std::string(argv[i]) == "--unsafe-server-tools" || std::string(argv[i]) == "--server-exec" || std::string(argv[i]) == "--allow-Shell") {
            g_server_exec = true;
            g_enable_tools = true;
        } else if (std::string(argv[i]) == "--no-server-exec" || std::string(argv[i]) == "--disable-server-exec") {
            g_server_exec = false;
        } else if (std::string(argv[i]) == "--allow-file-write" || std::string(argv[i]) == "--enable-file-write" || std::string(argv[i]) == "--allow-WriteFile" || std::string(argv[i]) == "--allow-EditFile") { g_enable_tools = true;
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
        } else if (std::string(argv[i]) == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "agentic") {
                g_intent_gating = false;
                g_intent_gating_cli_explicit = true;
            } else if (m == "chat") {
                g_intent_gating = true;
                g_intent_gating_cli_explicit = true;
            } else {
                LOG_WARN("Unknown operational mode '%s' (valid: chat, agentic). Defaulting to chat.", m.c_str());
                g_intent_gating = true;
                g_intent_gating_cli_explicit = true;
            }
        } else if (std::string(argv[i]) == "--agentic-mode" || std::string(argv[i]) == "--agentic") {
            g_intent_gating = false;
            g_intent_gating_cli_explicit = true;
        } else if (std::string(argv[i]) == "--chat-mode" || std::string(argv[i]) == "--chat") {
            g_intent_gating = true;
            g_intent_gating_cli_explicit = true;
        } else if (std::string(argv[i]) == "--no-intent-gating" || std::string(argv[i]) == "--disable-intent-gating") {
            g_intent_gating = false;
            g_intent_gating_cli_explicit = true;
        } else if (std::string(argv[i]) == "--intent-gating" || std::string(argv[i]) == "--enable-intent-gating") {
            g_intent_gating = true;
            g_intent_gating_cli_explicit = true;
        } else if (argv[i][0] != '-') {
            manifest_path = argv[i];
            manifest_explicit = true;
        }
    }

    // If a directory was passed to -m, look for moecher_manifest.json inside it
    {
        std::error_code ec;
        std::filesystem::path mp(manifest_path);
        if (manifest_explicit && std::filesystem::exists(mp, ec) && std::filesystem::is_directory(mp, ec)) {
            if (std::filesystem::exists(mp / "moecher_manifest.json", ec)) {
                manifest_path = (mp / "moecher_manifest.json").string();
            } else {
                for (const auto& entry : std::filesystem::directory_iterator(mp, ec)) {
                    if (entry.is_regular_file(ec) && entry.path().filename().string().rfind("moecher_manifest", 0) == 0 && entry.path().extension() == ".json") {
                        manifest_path = entry.path().string();
                        break;
                    }
                }
            }
        }
    }

    // Interactive model selector when launched without --manifest / -m
    if (!manifest_explicit) {
        auto models = discover_available_models(argc > 0 ? argv[0] : nullptr);
        if (models.empty()) {
            struct DownloadablePkg {
                const char* key;
                const char* title;
                const char* size_label;
            };
            static const DownloadablePkg kPackages[] = {
                {"qwen_13g",      "Qwen 3.8 27B Vision (13GB/16GB GPU & Apple Silicon)", "~13.8 GB"},
                {"nemotron",      "NVIDIA Nemotron 3.5 Lightning (30B-A3B NVFP4 + Vision)", "~21.6 GB"},
                {"nemotron_bolt", "NVIDIA Nemotron 3.5 Bolt (74B-A9B NVFP4 16GB/24GB)", "~67.5 GB"},
                {"qwen",          "Qwen 3.8 27B INT4 with MTP Speculative Decoding", "~19.1 GB"},
                {"deepseek",      "DeepSeek V4 Flash IQ2 Standard", "~81.4 GB"},
                {"deepseek_q4",   "DeepSeek V4 Flash Q4 (8GB GPU Mode)", "~78.8 GB"},
            };
            constexpr size_t kNumPkgs = sizeof(kPackages) / sizeof(kPackages[0]);
            printf("\n");
            printf("====================================================================================================\n");
            printf("  MinnieTheMoEcher v%s — First-Run Setup (No Local Models Found)\n", MOECHER_VERSION);
            printf("====================================================================================================\n");
            printf("  Select a pre-quantized model package to download from Hugging Face into ./models/:\n");
            printf("----------------------------------------------------------------------------------------------------\n");
            for (size_t idx = 0; idx < kNumPkgs; ++idx) {
                printf(" [%2zu]  %-58s %s%s\n",
                       idx + 1,
                       kPackages[idx].title,
                       kPackages[idx].size_label,
                       idx == 0 ? "  [Recommended]" : "");
            }
            printf("====================================================================================================\n");
            printf("Select a model to download [1-%zu] (press Enter for [1], or 'q' to quit): ", kNumPkgs);
            fflush(stdout);

            std::string dl_line;
            if (!std::getline(std::cin, dl_line)) {
                return 0;
            }
            while (!dl_line.empty() && ((unsigned char)dl_line.back() <= ' ')) dl_line.pop_back();
            size_t sp = 0;
            while (sp < dl_line.size() && ((unsigned char)dl_line[sp] <= ' ')) sp++;
            dl_line = dl_line.substr(sp);
            if (dl_line == "q" || dl_line == "Q" || dl_line == "quit" || dl_line == "exit") {
                return 0;
            }
            size_t dl_idx = 0;
            if (!dl_line.empty()) {
                try {
                    int v = std::stoi(dl_line);
                    if (v >= 1 && (size_t)v <= kNumPkgs) dl_idx = (size_t)(v - 1);
                } catch (...) {}
            }
            const char* model_key = kPackages[dl_idx].key;
            printf("\n[INFO] Downloading model package '%s' (%s)...\n", model_key, kPackages[dl_idx].title);
            std::error_code ec;
            std::string dl_cmd;
#if defined(_WIN32) || defined(_WIN64)
            if (std::filesystem::exists("scripts/download_model.py", ec)) {
                dl_cmd = std::string("python scripts\\download_model.py --model ") + model_key;
            } else if (std::filesystem::exists("download_model.ps1", ec)) {
                dl_cmd = std::string("powershell -ExecutionPolicy Bypass -File download_model.ps1 -Model ") + model_key;
            } else if (std::filesystem::exists("installer/download_model.ps1", ec)) {
                dl_cmd = std::string("powershell -ExecutionPolicy Bypass -File installer\\download_model.ps1 -Model ") + model_key + " -DestDir models";
            }
#else
            if (std::filesystem::exists("scripts/download_model.py", ec)) {
                dl_cmd = std::string("python3 scripts/download_model.py --model ") + model_key;
            }
#endif
            if (!dl_cmd.empty()) {
                int rc = std::system(dl_cmd.c_str());
                if (rc != 0) {
                    LOG_ERROR("Model download command exited with code %d", rc);
                    return 1;
                }
                models = discover_available_models(argc > 0 ? argv[0] : nullptr);
            }
        }
        if (!models.empty()) {
            printf("\n");
            printf("====================================================================================================\n");
            printf("  MinnieTheMoEcher v%s — Available Local Models\n", MOECHER_VERSION);
            printf("====================================================================================================\n");
            printf("  #    %-44s %-28s %-10s %s\n", "Model Name", "Architecture / Quant", "Weights", "Vision");
            printf("----------------------------------------------------------------------------------------------------\n");
            for (size_t idx = 0; idx < models.size(); ++idx) {
                const auto& m = models[idx];
                std::string arch_quant = m.architecture + " (" + m.expert_dtype + ")";
                printf(" [%2zu]  %-44.44s %-28.28s %6.1f GB   %s\n",
                       idx + 1,
                       m.display_name.c_str(),
                       arch_quant.c_str(),
                       m.total_weights_gb,
                       m.has_vision ? "Yes" : "No");
                printf("       -> %s\n", m.manifest_path.c_str());
            }
            printf("====================================================================================================\n");
            size_t chosen_idx = 0;
            while (true) {
                printf("Select a model to launch [1-%zu] (press Enter for [1], or 'q' to quit): ", models.size());
                fflush(stdout);

                std::string line;
                if (!std::getline(std::cin, line)) {
                    return 0;
                }
                // Strip UTF-8 BOM if present and trim whitespace
                if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF) {
                    line.erase(0, 3);
                }
                while (!line.empty() && ((unsigned char)line.back() <= ' ')) line.pop_back();
                size_t start_pos = 0;
                while (start_pos < line.size() && ((unsigned char)line[start_pos] <= ' ')) start_pos++;
                line = line.substr(start_pos);

                if (line == "q" || line == "Q" || line == "quit" || line == "exit") {
                    printf("Exiting.\n");
                    return 0;
                }

                if (line.empty()) {
                    chosen_idx = 0;
                    break;
                }
                try {
                    int val = std::stoi(line);
                    if (val >= 1 && (size_t)val <= models.size()) {
                        chosen_idx = (size_t)(val - 1);
                        break;
                    }
                } catch (...) {}
                printf("[WARN] Invalid selection '%s' (please enter a number between 1 and %zu, or 'q' to quit).\n", line.c_str(), models.size());
            }

            const auto& selected = models[chosen_idx];
            manifest_path = selected.manifest_path;
            printf("\n[INFO] Selected [%zu]: %s (%s)\n\n", chosen_idx + 1, selected.display_name.c_str(), manifest_path.c_str());

            // Apply model-specific recommended speculative decoding defaults unless explicitly passed on CLI
            if (!spec_explicit && selected.architecture == "deepseek_v41") {
                enable_spec = false;
                enable_pld = false;
                enable_mtp = false;
            }
        }
    }

    // Open log file
    g_log_file.open(log_path, std::ios::app);
    LOG_INFO("=== moecher starting ===");
    LOG_INFO("=== v%s ===", MOECHER_VERSION);
    LOG_INFO("Default thinking token budget: %d", default_thinking_budget);
    LOG_INFO("Operational mode: %s (Dynamic intent gating: %s)", g_intent_gating ? "Chat" : "Agentic", g_intent_gating ? "enabled" : "disabled");
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
    engine.default_enable_pld_ = enable_pld;
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
    engine.enable_anti_loop_ = enable_anti_loop;
    engine.default_enable_anti_loop_ = enable_anti_loop;
    engine.enable_cpu_experts_ = enable_cpu_experts;
    engine.default_enable_cpu_experts_ = enable_cpu_experts;
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
        std::vector<int> prompt = engine.apply_chat_template(messages, test_thinking, "high", json());
        LOG_INFO("Running test generation for prompt: '%s' (%zu prompt tokens, temp=%.2f, max_tokens=%d, thinking=%d)...",
                 prompt_str.c_str(), prompt.size(), test_temp, test_max_tokens, test_thinking ? 1 : 0);
        float rep_pen = engine.cfg_.default_repetition_penalty > 1.0f ? engine.cfg_.default_repetition_penalty : 1.0f;
        std::string result = engine.generate(prompt, test_max_tokens, test_temp, nullptr, rep_pen, test_thinking, default_thinking_budget, 0.95f, 0.0f, 1024);
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

    g_base_system_prompt = engine.get_model_default_system_prompt();
    if (!g_base_system_prompt.empty() && !engine.cfg_.is_deepseek()) {
        g_base_system_prompt += "\nCurrent date: " + std::string(date_buf);
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

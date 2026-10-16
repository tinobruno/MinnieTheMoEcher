#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <filesystem>
#include "logging.hpp"
#include "tool_exec.hpp"

namespace fs = std::filesystem;

void test_default_runtime_security() {
    std::cout << "[TEST] 1. Checking Secure-by-Default Runtime Flags..." << std::endl;
    assert(g_server_exec == false);
    assert(g_enable_tools == false);
    assert(g_allow_file_write == false);
    assert(g_workspace_root == "./workspace");
    std::cout << "  ✓ All security flags default to OFF (Safe Mode)!" << std::endl;
}

void test_compile_time_gating() {
    std::cout << "[TEST] 2. Checking Compile-Time Host Execution Gating..." << std::endl;
    std::string res = moecher::tooling::execute_system_command("whoami");
    std::cout << "  Result: " << res << std::endl;
#ifndef ENABLE_UNSAFE_HOST_EXEC
    assert(res.find("Host system command execution is disabled at compile time") != std::string::npos);
    std::cout << "  ✓ execute_system_command is physically disabled at compile time!" << std::endl;
#else
    assert(res.find("Server-side host command execution is disabled by runtime") != std::string::npos);
    std::cout << "  ✓ execute_system_command is blocked by runtime configuration!" << std::endl;
#endif
}

void test_path_sandboxing() {
    std::cout << "[TEST] 3. Checking Strict Workspace Path Sandboxing..." << std::endl;
    fs::path safe_p;
    std::string err_msg;

    // A. Path traversal escaping workspace
    bool r_traversal = moecher::tooling::validate_workspace_path("../../secret.txt", false, safe_p, err_msg);
    assert(!r_traversal);
    assert(err_msg.find("Path traversal blocked") != std::string::npos || err_msg.find("resolves outside") != std::string::npos);
    std::cout << "  ✓ Directory traversal (../../secret.txt) blocked: " << err_msg << std::endl;

    // B. Drive root escaping workspace
#if defined(_WIN32) || defined(_WIN64)
    bool r_abs = moecher::tooling::validate_workspace_path("C:\\Windows\\System32\\cmd.exe", false, safe_p, err_msg);
#else
    bool r_abs = moecher::tooling::validate_workspace_path("/etc/passwd", false, safe_p, err_msg);
#endif
    assert(!r_abs);
    assert(err_msg.find("outside the designated workspace sandbox") != std::string::npos);
    std::cout << "  ✓ Absolute drive root escape blocked: " << err_msg << std::endl;

    // C. Sensitive credential files
    bool r_env = moecher::tooling::validate_workspace_path(".env", false, safe_p, err_msg);
    assert(!r_env);
    assert(err_msg.find("sensitive file") != std::string::npos);
    std::cout << "  ✓ Sensitive file (.env) access blocked: " << err_msg << std::endl;

    bool r_ssh = moecher::tooling::validate_workspace_path("id_rsa", false, safe_p, err_msg);
    assert(!r_ssh);
    assert(err_msg.find("sensitive file") != std::string::npos);
    std::cout << "  ✓ SSH key (id_rsa) access blocked: " << err_msg << std::endl;

    bool r_cookie = moecher::tooling::validate_workspace_path("cookie_jar.json", false, safe_p, err_msg);
    assert(!r_cookie);
    assert(err_msg.find("sensitive file") != std::string::npos);
    std::cout << "  ✓ Credential (cookie_jar.json) access blocked: " << err_msg << std::endl;

    // D. Safe workspace relative path
    bool r_safe = moecher::tooling::validate_workspace_path("safe_report.txt", false, safe_p, err_msg);
    assert(r_safe);
    std::cout << "  ✓ Safe workspace path validated: " << safe_p.string() << std::endl;
}

void test_file_write_gating() {
    std::cout << "[TEST] 4. Checking File Write Permission Gating..." << std::endl;

    // A. When g_allow_file_write is false
    g_allow_file_write = false;
    std::string w_blocked = moecher::tooling::write_file("test_write.txt", "data");
    assert(w_blocked.find("File write/edit operations are disabled") != std::string::npos);
    std::cout << "  ✓ Write attempt blocked when --allow-file-write is absent: " << w_blocked << std::endl;

    std::string e_blocked = moecher::tooling::edit_file("test_write.txt", "a", "b");
    assert(e_blocked.find("File write/edit operations are disabled") != std::string::npos);
    std::cout << "  ✓ Edit attempt blocked when --allow-file-write is absent: " << e_blocked << std::endl;

    // B. When g_allow_file_write is true (opted-in)
    g_allow_file_write = true;
    std::string w_ok = moecher::tooling::write_file("test_write.txt", "Hello Secure Sandbox!\n");
    assert(w_ok.find("Successfully wrote") != std::string::npos);
    std::cout << "  ✓ Write succeeded with explicit permission: " << w_ok << std::endl;

    std::string r_ok = moecher::tooling::read_file("test_write.txt");
    assert(r_ok.find("Hello Secure Sandbox!") != std::string::npos);
    std::cout << "  ✓ Read back sandboxed file content: " << r_ok;

    std::string e_ok = moecher::tooling::edit_file("test_write.txt", "Sandbox!", "Sandbox (Verified)!");
    assert(e_ok.find("Successfully replaced") != std::string::npos);
    std::cout << "  ✓ Edit succeeded inside sandbox: " << e_ok << std::endl;

    std::string r_edited = moecher::tooling::read_file("test_write.txt");
    assert(r_edited.find("Sandbox (Verified)!") != std::string::npos);

    // Clean up inside workspace sandbox
    fs::path ws_file = fs::path(moecher::tooling::get_workspace_directory()) / "test_write.txt";
    std::error_code ec;
    fs::remove(ws_file, ec);
    g_allow_file_write = false;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << " MinnieTheMoEcher Safety Architecture Verification Suite  " << std::endl;
    std::cout << "==========================================================" << std::endl;
    test_default_runtime_security();
    test_compile_time_gating();
    test_path_sandboxing();
    test_file_write_gating();
    std::cout << "==========================================================" << std::endl;
    std::cout << " ALL SECURITY AND HARDENING TESTS PASSED SUCCESSFULLY!    " << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}

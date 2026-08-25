#include <iostream>
int main() {
    size_t vram_free = 1024 * 1024 * 1024; // 1 GB
    size_t headroom = 2ULL * 1024 * 1024 * 1024; // 2 GB
    size_t cache_budget = vram_free - headroom;
    std::cout << cache_budget << std::endl;
    return 0;
}

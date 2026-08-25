#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

int main() {
    std::vector<float> scores = {1.0, 5.0, 2.0, 8.0, 3.0};
    std::vector<int> indices(5);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + 2, indices.end(),
                      [&](int a, int b) { return scores[a] > scores[b]; });
    for (int i = 0; i < 2; i++) {
        std::cout << indices[i] << " (" << scores[indices[i]] << ")\n";
    }
    return 0;
}

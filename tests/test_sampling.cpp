#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

int main() {
    std::vector<float> logits = {10.0, 9.5, 9.0, -10.0, -20.0};
    int vocab = logits.size();
    float temperature = 0.6f;

    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0;
    for (auto& l : logits) {
        l = expf((l - max_logit) / temperature);
        sum_exp += l;
    }
    for (auto& l : logits) l /= sum_exp;

    for (int i = 0; i < vocab; i++) {
        std::cout << "p[" << i << "] = " << logits[i] << "\n";
    }

    float top_p = 0.9f;
    std::vector<std::pair<float, int>> probs;
    for (int i = 0; i < vocab; i++) probs.push_back({logits[i], i});
    std::sort(probs.begin(), probs.end(), [](auto& a, auto& b) { return a.first > b.first; });

    float cumsum = 0.0f;
    std::vector<float> filtered_probs(vocab, 0.0f);
    for (auto& p : probs) {
        filtered_probs[p.second] = p.first;
        cumsum += p.first;
        if (cumsum > top_p) break;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(filtered_probs.begin(), filtered_probs.end());
    
    std::vector<int> counts(vocab, 0);
    for (int i=0; i<1000; i++) {
        counts[dist(gen)]++;
    }
    for (int i=0; i<vocab; i++) {
        std::cout << "counts[" << i << "] = " << counts[i] << "\n";
    }
    return 0;
}

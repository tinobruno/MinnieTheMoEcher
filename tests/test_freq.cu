#include <stdio.h>
#include <math.h>

int main() {
    float base = 10000.0f;
    float factor = 16.0f;
    int rope_dim = 64;
    int max_seq_len = 100;
    int original_seq_len = 0; // 0 in moecher
    
    // What moecher does
    for (int pair = 0; pair < rope_dim/2; pair++) {
        float freq = 1.0f / powf(base, (float)(2 * pair) / (float)rope_dim);
        if (original_seq_len > 0) {
            // YaRN
        }
        printf("moecher pair %d freq %f\n", pair, freq);
    }
    return 0;
}

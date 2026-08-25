#include <stdio.h>
#include <math.h>

int main() {
    int rope_dim = 64;
    float base = 10000.0f;
    float factor = 40.0f;
    int original_seq_len = 4096;
    int beta_fast = 32;
    int beta_slow = 1;

    for (int pair = 0; pair < rope_dim / 2; pair++) {
        float freq = 1.0f / powf(base, (float)(2 * pair) / (float)rope_dim);

        float dim_f = (float)rope_dim;
        float low_f = (dim_f * logf((float)original_seq_len / ((float)beta_fast * 2.0f * 3.14159265f))) / (2.0f * logf(base));
        float high_f = (dim_f * logf((float)original_seq_len / ((float)beta_slow * 2.0f * 3.14159265f))) / (2.0f * logf(base));
        float low = floorf(low_f);
        float high = ceilf(high_f);
        low = fmaxf(low, 0.0f);
        high = fminf(high, (float)(rope_dim / 2) - 1.0f);

        float diff = high - low;
        if (diff == 0.0f) diff = 0.001f;
        
        float linear_func = ((float)pair - low) / diff;
        float ramp_func = fmaxf(fminf(linear_func, 1.0f), 0.0f);
        float inv_freq_mask = 1.0f - ramp_func;
        
        float freq_inter = freq / factor;
        freq = freq_inter * (1.0f - inv_freq_mask) + freq * inv_freq_mask;

        if (pair < 8 || pair >= 24) {
            printf("Pair %d: %e\n", pair, freq);
        }
    }
    return 0;
}

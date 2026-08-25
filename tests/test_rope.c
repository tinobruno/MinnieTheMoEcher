#include <stdio.h>
#include <math.h>

void compute(float* inv_freq) {
    int rope_dim = 64;
    float base = 10000.0f;
    float factor = 40.0f;
    int original_seq_len = 4096;
    int beta_fast = 32;
    int beta_slow = 1;

    float dim_f = (float)rope_dim;
    float low_f = dim_f * logf((float)original_seq_len / ((float)beta_fast * 2.0f * 3.14159265f)) / (2.0f * logf(base));
    float high_f = dim_f * logf((float)original_seq_len / ((float)beta_slow * 2.0f * 3.14159265f)) / (2.0f * logf(base));
    
    int low = fmaxf(floorf(low_f), 0.0f);
    int high = fminf(ceilf(high_f), rope_dim - 1);

    for (int pair = 0; pair < rope_dim / 2; pair++) {
        float freq = 1.0f / powf(base, (float)(2 * pair) / (float)rope_dim);
        float ramp;
        if (low == high) {
            ramp = (pair >= low) ? 1.0f : 0.0f;
        } else {
            ramp = fminf(fmaxf(((float)pair - (float)low) / ((float)high - (float)low), 0.0f), 1.0f);
        }
        float smooth = 1.0f - ramp;
        freq = freq / factor * (1.0f - smooth) + freq * smooth;
        inv_freq[pair] = freq;
    }
}

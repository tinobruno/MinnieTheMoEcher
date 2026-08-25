import sys
cuh = open('src/cuda/activations.cuh').read()
new_cuh_code = """
void hc_pre_weighted_add_cuda(
    __nv_bfloat16* hidden, const __nv_bfloat16* hc_state, const float* pre_weights,
    int dim, int hc, cudaStream_t stream = 0);

void hc_post_update_cuda(
    __nv_bfloat16* hc_state, const __nv_bfloat16* hidden, const __nv_bfloat16* hc_residual,
    const float* post_weights, const float* comb_weights,
    int dim, int hc, cudaStream_t stream = 0);

void hc_head_reduce_cuda(
    __nv_bfloat16* hidden, const __nv_bfloat16* hc_state,
    const float* mixes, const float* scale, const float* base,
    int dim, int hc, cudaStream_t stream = 0);
"""
cuh += new_cuh_code
open('src/cuda/activations.cuh', 'w').write(cuh)

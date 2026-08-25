import sys

new_code = """
// ── HC Pre Weighted Add Kernel ──────────────────────────────────────────────────
__global__ void hc_pre_weighted_add_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ hc_state,
    const float* __restrict__ pre_weights,
    int dim, int hc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim) return;

    float sum = 0.0f;
    for (int h = 0; h < hc; h++) {
        float w = pre_weights[h];
        if (fabsf(w) >= 1e-10f) {
            float val = __bfloat162float(hc_state[h * dim + tid]);
            sum += w * val;
        }
    }
    hidden[tid] = __float2bfloat16(sum);
}

void hc_pre_weighted_add_cuda(
    __nv_bfloat16* hidden, const __nv_bfloat16* hc_state, const float* pre_weights,
    int dim, int hc, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    hc_pre_weighted_add_kernel<<<blocks, threads, 0, stream>>>(hidden, hc_state, pre_weights, dim, hc);
}

// ── HC Post Update Kernel ───────────────────────────────────────────────────────
__global__ void hc_post_update_kernel(
    __nv_bfloat16* __restrict__ hc_state,
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ hc_residual,
    const float* __restrict__ post_weights,
    const float* __restrict__ comb_weights,
    int dim, int hc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y; // Which HC head we are computing
    if (tid >= dim || h >= hc) return;

    // Start with post[h] * hidden
    float p_w = post_weights[h];
    float sum = p_w * __bfloat162float(hidden[tid]);

    // Add sum_j(comb[j * hc + h] * residual[j])
    for (int j = 0; j < hc; j++) {
        float c_w = comb_weights[j * hc + h];
        if (fabsf(c_w) >= 1e-10f) {
            float r_val = __bfloat162float(hc_residual[j * dim + tid]);
            sum += c_w * r_val;
        }
    }
    
    hc_state[h * dim + tid] = __float2bfloat16(sum);
}

void hc_post_update_cuda(
    __nv_bfloat16* hc_state, const __nv_bfloat16* hidden, const __nv_bfloat16* hc_residual,
    const float* post_weights, const float* comb_weights,
    int dim, int hc, cudaStream_t stream)
{
    int threads = 256;
    int blocks_x = (dim + threads - 1) / threads;
    dim3 blocks(blocks_x, hc);
    hc_post_update_kernel<<<blocks, threads, 0, stream>>>(
        hc_state, hidden, hc_residual, post_weights, comb_weights, dim, hc);
}

// ── HC Head Reduce Kernel ───────────────────────────────────────────────────────
__global__ void hc_head_reduce_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ hc_state,
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    int dim, int hc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim) return;

    float s = scale[0];
    float sum = 0.0f;
    for (int h = 0; h < hc; h++) {
        float mix = mixes[h];
        float b = base[h];
        // sigmoid(mix * s + b) + eps
        float arg = mix * s + b;
        float w = 1.0f / (1.0f + expf(-arg)) + 1e-6f;
        
        float val = __bfloat162float(hc_state[h * dim + tid]);
        sum += w * val;
    }
    hidden[tid] = __float2bfloat16(sum);
}

void hc_head_reduce_cuda(
    __nv_bfloat16* hidden, const __nv_bfloat16* hc_state,
    const float* mixes, const float* scale, const float* base,
    int dim, int hc, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    hc_head_reduce_kernel<<<blocks, threads, 0, stream>>>(
        hidden, hc_state, mixes, scale, base, dim, hc);
}
"""

with open('src/cuda/activations.cu', 'a') as f:
    f.write(new_code)

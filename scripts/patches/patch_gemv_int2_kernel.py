import sys
content = open('src/cuda/activations.cu').read()
old_code = """    // INT2 gemv kernel optimized
    for (int cp_chunk = tid; cp_chunk < K_packed / 16; cp_chunk += blockDim.x) {
        int logical_col = cp_chunk * 64;
        int bc = logical_col / block_size;
        
        float s = __bfloat162float(row_scales[bc]);
        float m = __bfloat162float(row_mins[bc]);

        uint4 w4 = w_vec4[cp_chunk];
        uint32_t w_chunks[4] = {w4.x, w4.y, w4.z, w4.w};
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t chunk = w_chunks[i]; 
            
            uint4 a0 = a_vec4[cp_chunk * 8 + i * 2];
            uint4 a1 = a_vec4[cp_chunk * 8 + i * 2 + 1];
            
            uint32_t a_arr[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            
            #pragma unroll
            for (int j = 0; j < 8; j++) { 
                uint32_t a_val = a_arr[j];
                __nv_bfloat162 bf2 = *reinterpret_cast<__nv_bfloat162*>(&a_val);
                float2 f2 = __bfloat1622float2(bf2);
                
                int byte_idx = j / 2;
                int nibble_idx = j % 2;
                uint8_t b = (chunk >> (byte_idx * 8)) & 0xFF;
                
                float v0, v1;
                if (nibble_idx == 0) {
                    v0 = (float)(b & 0x03) * s + m;
                    v1 = (float)((b >> 2) & 0x03) * s + m;
                } else {
                    v0 = (float)((b >> 4) & 0x03) * s + m;
                    v1 = (float)((b >> 6) & 0x03) * s + m;
                }
                
                sum = fmaf(v0, f2.x, sum);
                sum = fmaf(v1, f2.y, sum);
            }
        }
    }"""
new_code = """    // INT2 gemv kernel optimized
    for (int cp_chunk = tid; cp_chunk < K_packed / 16; cp_chunk += blockDim.x) {
        int logical_col = cp_chunk * 64;
        int bc = logical_col / block_size;
        
        float s = __bfloat162float(row_scales[bc]);
        float m = __bfloat162float(row_mins[bc]);

        uint4 w4 = w_vec4[cp_chunk];
        uint32_t w_chunks[4] = {w4.x, w4.y, w4.z, w4.w};
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t chunk = w_chunks[i]; 
            
            uint4 a0 = a_vec4[cp_chunk * 8 + i * 2];
            uint4 a1 = a_vec4[cp_chunk * 8 + i * 2 + 1];
            
            uint32_t a_arr[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            
            #pragma unroll
            for (int j = 0; j < 8; j++) { 
                uint32_t a_val = a_arr[j];
                __nv_bfloat162 bf2 = *reinterpret_cast<__nv_bfloat162*>(&a_val);
                float2 f2 = __bfloat1622float2(bf2);
                
                int byte_idx = j / 2;
                int nibble_idx = j % 2;
                uint8_t b = (chunk >> (byte_idx * 8)) & 0xFF;
                
                float v0, v1;
                if (nibble_idx == 0) {
                    v0 = (float)(b & 0x03) * s + m;
                    v1 = (float)((b >> 2) & 0x03) * s + m;
                } else {
                    v0 = (float)((b >> 4) & 0x03) * s + m;
                    v1 = (float)((b >> 6) & 0x03) * s + m;
                }
                
                sum = fmaf(v0, f2.x, sum);
                sum = fmaf(v1, f2.y, sum);
            }
        }
    }"""
content = content.replace(old_code, new_code)
open('src/cuda/activations.cu', 'w').write(content)

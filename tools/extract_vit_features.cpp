#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "image_loader.hpp"
#include "vision_tower.hpp"

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image_path> <output_bin_path> [output_dim]" << std::endl;
        return 1;
    }
    std::string img_path = argv[1];
    std::string out_path = argv[2];
    int out_dim = (argc >= 4) ? std::atoi(argv[3]) : 4096;

    std::string manifest_path = "models/frankenstin/moecher_manifest.json";
    std::ifstream mf_file(manifest_path);
    if (!mf_file.is_open()) {
        std::cerr << "Cannot open " << manifest_path << std::endl;
        return 1;
    }
    json manifest;
    mf_file >> manifest;

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(cublas_handle, stream);

    moecher::vision::QwenVisionTower tower;
    std::string vision_bin = "models/frankenstin/qwen_vision_tower.bin";
    std::string bridge_bin = "models/frankenstin/bridge_fc2.bin";
    std::string embed_mean_bin = "models/frankenstin/deepseek_embed_mean.bin";

    bool ok = tower.load_from_manifest(
        "", manifest["dense_tensors"], cublas_handle, stream, out_dim,
        vision_bin, bridge_bin, embed_mean_bin);

    if (!ok) {
        std::cerr << "Failed to load vision tower!" << std::endl;
        return 1;
    }

    // Read input image file
    std::ifstream img_file(img_path, std::ios::binary | std::ios::ate);
    if (!img_file.is_open()) {
        std::cerr << "Cannot open input image: " << img_path << std::endl;
        return 1;
    }
    std::streamsize size = img_file.tellg();
    img_file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (!img_file.read((char*)buffer.data(), size)) {
        std::cerr << "Failed to read image bytes: " << img_path << std::endl;
        return 1;
    }

    moecher::vision::ProcessedImage pimg;
    if (!moecher::vision::preprocess_image(buffer.data(), buffer.size(), pimg, 768)) {
        std::cerr << "Failed to preprocess image: " << img_path << std::endl;
        return 1;
    }

    // Forward pass through ViT and spatial merger
    tower.forward(pimg, stream);
    cudaStreamSynchronize(stream);

    // Extract d_merge_fc1_ (shape [576, 4608] in BF16)
    const __nv_bfloat16* d_fc1 = tower.merge_fc1_output();
    size_t fc1_bytes = 576 * 4608 * sizeof(__nv_bfloat16);
    std::vector<__nv_bfloat16> h_fc1(576 * 4608);
    cudaMemcpy(h_fc1.data(), d_fc1, fc1_bytes, cudaMemcpyDeviceToHost);

    std::ofstream out_file(out_path, std::ios::binary);
    out_file.write(reinterpret_cast<const char*>(h_fc1.data()), fc1_bytes);
    out_file.close();

    std::cout << "[Extract] Successfully extracted " << 576 << " patch vectors x 4608 dim (" 
              << fc1_bytes << " bytes) to " << out_path << std::endl;

    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);
    return 0;
}

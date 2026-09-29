#!/usr/bin/env python3
import json

def main():
    ds_manifest_path = "models/deepseek_v4_flash_q4/moecher_manifest_mixed.json"
    vit_tensors_path = "models/frankenstin/qwen_vision_tensors.json"
    out_manifest_path = "models/frankenstin/moecher_manifest.json"

    print("Loading DeepSeek mixed manifest...")
    with open(ds_manifest_path, "r") as f:
        manifest = json.load(f)

    # Update model_config
    manifest["model_config"]["model_name"] = "FrankensTin-Vision-V4"
    manifest["model_config"]["has_vision"] = True
    manifest["model_config"]["visual_hidden_size"] = 4096

    # Update file paths to local relative files in models/frankenstin/
    manifest["dense_bin"] = "attention_dense_layers_q4.bin"
    manifest["vision_bin"] = "qwen_vision_tower.bin"
    manifest["bridge_bin"] = "bridge_fc2.bin"
    manifest["expert_bin"] = "moe_experts_iq2.bin"
    manifest["moe_experts_sparse_nvfp4"] = "moe_experts_sparse_nvfp4.bin"
    manifest["mixed_expert_map"] = "mixed_expert_map.bin"

    print("Loading ViT tensor definitions...")
    with open(vit_tensors_path, "r") as f:
        vit_tensors = json.load(f)

    # In dense_tensors:
    # DeepSeek tensors keep their offsets in dense_bin
    # ViT tensors are annotated with "file": "vision"
    for k, v in vit_tensors.items():
        manifest["dense_tensors"][k] = {
            "file": "vision",
            "offset": v["offset"],
            "nbytes": v["nbytes"],
            "dtype": v["dtype"],
            "shape": v["shape"]
        }

    # Add bridge projector tensors annotated with "file": "bridge"
    manifest["dense_tensors"]["model.visual.merger.linear_fc2.weight"] = {
        "file": "bridge",
        "offset": 0,
        "nbytes": 37748736,
        "dtype": "BF16",
        "shape": [4096, 4608]
    }
    manifest["dense_tensors"]["model.visual.merger.linear_fc2.bias"] = {
        "file": "bridge",
        "offset": 37748736,
        "nbytes": 8192,
        "dtype": "BF16",
        "shape": [4096]
    }

    print(f"Total dense & visual tensors in FrankensTin manifest: {len(manifest['dense_tensors'])}")

    with open(out_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved FrankensTin manifest to {out_manifest_path}")

if __name__ == "__main__":
    main()

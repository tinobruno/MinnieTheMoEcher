import json
from pathlib import Path

def main():
    p = Path("models/nemotron_3.5_bolt_24gb/moecher_manifest.json")
    if not p.exists():
        print("Manifest not found")
        return
    with open(p, "r", encoding="utf-8") as f:
        m = json.load(f)

    dt = m.get("dense_tensors", {})
    updated = 0
    for name, info in dt.items():
        if "shared_experts" in name:
            info["dtype"] = "iq2_xxs"
            updated += 1
        elif "fc1_latent_proj" in name or "fc2_latent_proj" in name:
            info["dtype"] = "q8_0"
            updated += 1
        elif name == "lm_head.weight":
            info["dtype"] = "q8_0"
            updated += 1
        elif "in_proj.weight" in name or "out_proj.weight" in name:
            info["dtype"] = "nvfp4"
            updated += 1

    with open(p, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

    print(f"Updated {updated} tensor dtypes in {p}!")

if __name__ == "__main__":
    main()

import json
with open("moecher_manifest_q2.json", "r") as f:
    m = json.load(f)
m["expert_bin"] = "moe_experts_q2.bin"
with open("moecher_manifest_q2.json", "w") as f:
    json.dump(m, f, indent=2)

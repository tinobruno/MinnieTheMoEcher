# Hugging Face Model Upload Guide

This guide walks you through uploading your model weights to Hugging Face under your account **`TinoBruno`**.

---

## 1. Quick 1-Command Upload (Recommended)

Run the Python upload script:

```bash
python tools/upload_to_hf.py all
```

Or upload an individual model:

```bash
# Upload Qwen 3.8 27B INT4 (~18.7 GB)
python tools/upload_to_hf.py qwen

# Upload DeepSeek V4 Flash IQ2 (~81.4 GB)
python tools/upload_to_hf.py deepseek
```

When prompted, paste your Hugging Face **Write Access Token** (generated at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

The script will automatically:
1. Check / create the repository on Hugging Face.
2. Upload all weights, manifests, tokenizer, and `README.md` files.
3. Show progress bars with automatic resume support if interrupted.

---

## 2. Alternative: Using the Hugging Face CLI

If you prefer using the official CLI:

### Step A: Login
```bash
huggingface-cli login
```
Paste your Write Token and press Enter.

### Step B: Upload Qwen Model
```bash
huggingface-cli upload TinoBruno/moecher-qwen-3.8-27b-q4 models/qwen3_8_27b_q4 . --repo-type model
```

### Step C: Upload DeepSeek Model
```bash
huggingface-cli upload TinoBruno/moecher-deepseek-v4-flash-iq2 models/deepseek_v4_flash_iq2 . --repo-type model
```

---

## 3. Repositories on Hugging Face

Once uploaded, your models will be publicly accessible at:
- **Qwen**: [https://huggingface.co/TinoBruno/moecher-qwen-3.8-27b-q4](https://huggingface.co/TinoBruno/moecher-qwen-3.8-27b-q4)
- **DeepSeek**: [https://huggingface.co/TinoBruno/moecher-deepseek-v4-flash-iq2](https://huggingface.co/TinoBruno/moecher-deepseek-v4-flash-iq2)

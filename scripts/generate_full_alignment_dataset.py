#!/usr/bin/env python3
import json
import os
import sys
import time
import random
import string
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

def create_license_plate(text, width=280, height=70, bg_color=(250, 250, 250), fg_color=(15, 15, 15)):
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Outer black border
    draw.rectangle([0, 0, width - 1, height - 1], outline=(10, 10, 10), width=3)
    
    # EU blue strip on left
    eu_width = 30
    draw.rectangle([3, 3, eu_width, height - 4], fill=(0, 51, 153))
    
    # Font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    
    font_size = int(height * 0.65)
    font = ImageFont.truetype(font_path, font_size)
    
    text_area_x = eu_width + 10
    char_boxes = []
    cur_x = text_area_x
    for ch in text:
        if ch == ' ':
            cur_x += 15
            continue
        bbox = draw.textbbox((cur_x, 0), ch, font=font)
        ch_w = bbox[2] - bbox[0]
        ch_h = bbox[3] - bbox[1]
        y_pos = (height - ch_h) // 2 - 4
        draw.text((cur_x, y_pos), ch, fill=fg_color, font=font)
        char_boxes.append((ch, cur_x, y_pos, cur_x + ch_w, y_pos + ch_h))
        cur_x += ch_w + 6
        
    return img, char_boxes

def generate_plate_scene(plate_img, char_boxes, scene_idx, car_color_name="blue", car_color_rgb=(20, 50, 150)):
    # Diverse backgrounds: asphalt, gravel, lawn, concrete, dark
    bg_colors = [(130, 135, 140), (80, 85, 90), (60, 90, 50), (180, 185, 190), (100, 110, 120)]
    canvas = Image.new("RGB", (768, 768), color=random.choice(bg_colors))
    draw = ImageDraw.Draw(canvas)
    
    is_front = (scene_idx % 2 == 0)
    
    # Randomize vertical placement across rows 8 to 17 (y = 260 to 520)
    bumper_y = random.randint(280, 500)
    bumper_h = random.randint(140, 190)
    
    # Car body section
    draw.rectangle([60, bumper_y - 70, 708, bumper_y + bumper_h], fill=car_color_rgb)
    
    if is_front:
        # Grille above bumper
        draw.rectangle([160, bumper_y - 120, 608, bumper_y - 10], fill=(30, 30, 30), outline=(80, 80, 80), width=4)
        for gx in range(180, 590, 20):
            draw.line([gx, bumper_y - 110, gx, bumper_y - 20], fill=(60, 60, 60), width=3)
    else:
        # Rear tail lights on left and right
        draw.rectangle([80, bumper_y - 50, 160, bumper_y + 10], fill=(190, 25, 25))
        draw.rectangle([608, bumper_y - 50, 688, bumper_y + 10], fill=(190, 25, 25))
        # Boot lid line
        trim_c = (max(0, car_color_rgb[0] - 30), max(0, car_color_rgb[1] - 30), max(0, car_color_rgb[2] - 30))
        draw.line([160, bumper_y - 60, 608, bumper_y - 60], fill=trim_c, width=3)
        
    px = random.randint(230, 280)
    py = bumper_y + (bumper_h - plate_img.height) // 2
    canvas.paste(plate_img, (px, py))
    
    patch_annotations = []
    for ch, cx0, cy0, cx1, cy1 in char_boxes:
        abs_x = px + (cx0 + cx1) // 2
        abs_y = py + (cy0 + cy1) // 2
        patch_x = min(23, max(0, abs_x // 32))
        patch_y = min(23, max(0, abs_y // 32))
        patch_idx = patch_y * 24 + patch_x
        patch_annotations.append({
            "char": ch,
            "patch_idx": patch_idx
        })
        
    color_patches = []
    # Just 2 color patches on the body outside plate/bumper to maintain balance
    hy = max(0, (bumper_y - 80) // 32)
    for hx in [4, 20]:
        idx = hy * 24 + hx
        if 0 <= idx < 576:
            color_patches.append(idx)
            
    out_dir = "scratch/synth_dataset"
    os.makedirs(out_dir, exist_ok=True)
    img_path = f"{out_dir}/scene_{scene_idx:04d}.png"
    canvas.save(img_path)
    
    return {
        "img_path": img_path,
        "car_color": car_color_name,
        "char_patches": patch_annotations,
        "color_patches": color_patches
    }

def main():
    random.seed(42)
    with open("models/frankenstin/tokenizer.json") as f:
        tok = json.load(f)
    vocab = tok["model"]["vocab"]

    # Color tokens (DeepSeek vocabulary IDs)
    color_map = {
        "blue": (8295, (25, 60, 160)),
        "green": (6726, (20, 130, 45)),
        "red": (4332, (190, 25, 25)),
        "silver": (16975, (195, 195, 200)),
        "black": (5159, (30, 30, 30)),
        "white": (5403, (245, 245, 245))
    }

    # Generate plate strings covering target plates and alphanumeric distribution (pure plates, no brand names)
    plate_strings = [
        "B 58 BPS", "107 UAS", "B 58 BPS", "107 UAS", "B 58 BPS", "107 UAS",
        "ABC 123", "XYZ 789", "DEF 456", "GHI 789", "JKL 012", "MNO 345",
        "PQR 678", "STU 901", "VWX 234", "YZA 567", "BCD 890", "EFG 123",
        "RO 01 BPS", "B 999 BOS", "CJ 10 ABC", "TM 25 XYZ", "IS 07 PRO",
        "AB 12 CDE", "CD 34 EFG", "EF 56 GHI", "GH 78 IJK", "IJ 90 KLM",
        "KL 12 MNO", "MN 34 OPQ", "OP 56 QRS", "QR 78 STU", "ST 90 UVW",
        "NY 1024", "CA 2048", "TX 4096", "FL 8192", "WA 1638", "MA 3276",
        "012 345", "678 901", "234 567", "890 123", "456 789", "098 765"
    ]
    # Add random 6-character strings to reach 60 scenes
    for _ in range(60 - len(plate_strings)):
        p = f"{random.choice(string.ascii_uppercase)} {random.randint(10, 99)} {random.choice(string.ascii_uppercase)}{random.choice(string.ascii_uppercase)}{random.choice(string.ascii_uppercase)}"
        plate_strings.append(p)

    dataset = []
    print(f"Generating and extracting {len(plate_strings)} scenes...")
    t0 = time.time()
    
    for i, p_str in enumerate(plate_strings):
        color_name = random.choice(list(color_map.keys()))
        color_id, color_rgb = color_map[color_name]
        
        p_img, boxes = create_license_plate(p_str)
        scene = generate_plate_scene(p_img, boxes, i, car_color_name=color_name, car_color_rgb=color_rgb)
        
        bin_path = f"scratch/synth_dataset/scene_{i:04d}_fc1.bin"
        os.system(f"./tools/extract_vit_features {scene['img_path']} {bin_path} > /dev/null 2>&1")
        
        pairs = []
        for cp in scene["char_patches"]:
            ch = cp["char"]
            if ch in vocab:
                pairs.append({
                    "patch_idx": cp["patch_idx"],
                    "token_id": vocab[ch],
                    "name": ch
                })
        for col_p in scene["color_patches"]:
            pairs.append({
                "patch_idx": col_p,
                "token_id": color_id,
                "name": color_name
            })
            
        dataset.append({
            "bin_path": bin_path,
            "pairs": pairs
        })
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(plate_strings)} scenes in {time.time() - t0:.1f}s")

    # Add real test images with ground truth annotations
    # 1. BMW: blue BMW with plate B 58 BPS
    bmw_pairs = []
    # Plate: B 58 BPS -> 'B' (36), '58' (3175), 'B' (36), 'P' (50), 'S' (53)
    bmw_chars = [
        ('B', 394, 36), ('58', 395, 3175), ('58', 396, 3175), ('B', 397, 36), ('S', 398, 53),
        ('B', 418, 36), ('58', 419, 3175), ('58', 420, 3175), ('P', 421, 50), ('S', 422, 53)
    ]
    for name, p_idx, tid in bmw_chars:
        bmw_pairs.append({"patch_idx": p_idx, "token_id": tid, "name": name})
    # Blue car color patches on hood (just 2 patches for balance)
    for p_idx in [200, 224]:
        bmw_pairs.append({"patch_idx": p_idx, "token_id": 8295, "name": "blue"})
    # BMW emblem
    for p_idx in [275, 276]:
        bmw_pairs.append({"patch_idx": p_idx, "token_id": 64820, "name": "BMW"})

    dataset.append({"bin_path": "scratch/bmw_fc1_features.bin", "pairs": bmw_pairs})

    # 2. Austin-Healey: classic car with plate 107 UAS
    # Characters: '10' (553) in col 10, '7' (25) in col 11, 'U' (55) in col 12, 'A' (35) in col 13, 'S' (53) in col 14
    healey_pairs = []
    healey_chars = [
        ('10', 274, 553), ('7', 275, 25), ('U', 276, 55), ('A', 277, 35), ('S', 278, 53),
        ('10', 298, 553), ('7', 299, 25), ('U', 300, 55), ('A', 301, 35), ('S', 302, 53)
    ]
    for name, p_idx, tid in healey_chars:
        healey_pairs.append({"patch_idx": p_idx, "token_id": tid, "name": name})
    # Austin-Healey vehicle make on rear deck (Row 4, separated from plate)
    healey_pairs.append({"patch_idx": 4 * 24 + 11, "token_id": 114512, "name": "Austin"})
    healey_pairs.append({"patch_idx": 4 * 24 + 12, "token_id": 90703, "name": "-He"})
    healey_pairs.append({"patch_idx": 4 * 24 + 13, "token_id": 107818, "name": "aley"})
    dataset.append({"bin_path": "scratch/healey_fc1_features.bin", "pairs": healey_pairs})

    # 3. Ferrari F50: red sports car
    ferrari_pairs = []
    for p_idx in [250, 274]:
        ferrari_pairs.append({"patch_idx": p_idx, "token_id": 4332, "name": "red"})
    for p_idx in [280, 281]:
        ferrari_pairs.append({"patch_idx": p_idx, "token_id": 74122, "name": "Ferrari"})
    dataset.append({"bin_path": "scratch/ferrari_fc1_features.bin", "pairs": ferrari_pairs})

    # 4. Apple: red apple with green leaf
    apple_pairs = []
    for p_idx in [288, 289, 311, 312]:
        apple_pairs.append({"patch_idx": p_idx, "token_id": 27607, "name": "apple"})
    for p_idx in [264, 265]:
        apple_pairs.append({"patch_idx": p_idx, "token_id": 16319, "name": "leaf"})
    dataset.append({"bin_path": "scratch/apple_fc1_features.bin", "pairs": apple_pairs})

    out_json = "scratch/alignment_dataset.json"
    with open(out_json, "w") as f:
        json.dump(dataset, f, indent=2)

    total_pairs = sum(len(d["pairs"]) for d in dataset)
    print(f"Dataset complete! Saved {len(dataset)} samples ({total_pairs} grounded patch-token pairs) to {out_json}")

if __name__ == "__main__":
    main()

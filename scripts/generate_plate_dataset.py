#!/usr/bin/env python3
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_license_plate(text, width=280, height=70, bg_color=(255, 255, 255), fg_color=(0, 0, 0)):
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Border
    draw.rectangle([0, 0, width - 1, height - 1], outline=(0, 0, 0), width=3)
    
    # Blue EU band on left (optional)
    eu_width = 30
    draw.rectangle([3, 3, eu_width, height - 4], fill=(0, 51, 153))
    
    # Font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    
    font_size = int(height * 0.65)
    font = ImageFont.truetype(font_path, font_size)
    
    # Measure characters and place them
    # Text area starts after EU band
    text_area_x = eu_width + 10
    avail_width = width - text_area_x - 10
    
    # Calculate spacing
    total_len = len(text)
    char_boxes = []
    
    # Draw text centered vertically
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

def generate_plate_scene(plate_img, char_boxes, scene_idx, bg_color=(120, 130, 140)):
    # 768x768 canvas
    canvas = Image.new("RGB", (768, 768), color=bg_color)
    draw = ImageDraw.Draw(canvas)
    
    # Add car bumper texture
    bumper_y = random.randint(400, 520)
    bumper_h = random.randint(160, 220)
    car_color = random.choice([
        (20, 50, 150),  # Blue
        (20, 120, 40),  # Green
        (180, 20, 20),  # Red
        (200, 200, 200),# Silver
        (30, 30, 30),   # Black
        (240, 240, 240) # White
    ])
    draw.rectangle([50, bumper_y, 718, bumper_y + bumper_h], fill=car_color)
    
    # Grille above bumper
    draw.rectangle([150, bumper_y - 120, 618, bumper_y - 10], fill=(40, 40, 40), outline=(100, 100, 100), width=4)
    for gx in range(180, 600, 25):
        draw.line([gx, bumper_y - 110, gx, bumper_y - 20], fill=(70, 70, 70), width=4)
        
    # Place license plate on bumper
    px = random.randint(220, 300)
    py = bumper_y + (bumper_h - plate_img.height) // 2
    canvas.paste(plate_img, (px, py))
    
    # Compute patch positions for each character
    patch_annotations = []
    for ch, cx0, cy0, cx1, cy1 in char_boxes:
        abs_x = px + (cx0 + cx1) // 2
        abs_y = py + (cy0 + cy1) // 2
        patch_x = min(23, max(0, abs_x // 32))
        patch_y = min(23, max(0, abs_y // 32))
        patch_idx = patch_y * 24 + patch_x
        patch_annotations.append({
            "char": ch,
            "abs_pos": (abs_x, abs_y),
            "patch_pos": (patch_x, patch_y),
            "patch_idx": patch_idx
        })
        
    # Add car body color annotations (e.g. hood and bumper patches)
    # Hood is around y = 200..350, x = 200..560
    color_patches = []
    for hy in range(bumper_y // 32, (bumper_y + bumper_h) // 32):
        for hx in [3, 4, 19, 20]: # bumper sides away from plate
            idx = hy * 24 + hx
            color_patches.append(idx)
            
    out_dir = "scratch/synth_plates"
    os.makedirs(out_dir, exist_ok=True)
    img_path = f"{out_dir}/scene_{scene_idx:04d}.png"
    canvas.save(img_path)
    
    return {
        "img_path": img_path,
        "car_color": car_color,
        "char_patches": patch_annotations,
        "color_patches": color_patches
    }

if __name__ == "__main__":
    p_img, boxes = create_license_plate("B 58 BPS")
    scene = generate_plate_scene(p_img, boxes, 0)
    print("Generated sample scene:", scene["img_path"])
    print("Annotated character patches:", scene["char_patches"])

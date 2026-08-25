#!/usr/bin/env python3
"""
package_windows.py — Creates a standalone distribution package of MinnieTheMoECher for Windows.

Usage:
    python scripts/package_windows.py [--output dist_windows] [--zip]
"""

import os
import sys
import shutil
import argparse
import zipfile
from pathlib import Path

def package_windows(output_dir="dist_windows", make_zip=False):
    root_dir = Path(__file__).resolve().parent.parent
    out_path = Path(output_dir).resolve()
    
    print(f"[1/4] Preparing output directory: {out_path}")
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Essential files to copy
    files_to_copy = [
        "start.bat",
        "install.ps1",
        "chat.py",
        "CMakeLists.txt",
        "moecher_manifest_iq2.json",
        "moecher_manifest.json",
        "README.md",
        "README-Windows.md",
        "README-3090.md",
        "QUANTIZATION.md",
        "LICENSE.txt",
        ".gitignore",
    ]
    
    print("[2/4] Copying essential scripts, manifests, and documentation...")
    for f in files_to_copy:
        src = root_dir / f
        if src.exists():
            shutil.copy2(src, out_path / f)
            print(f"  + {f}")
        else:
            print(f"  - (optional) {f} not found, skipping")
            
    # Copy compiled executable if present
    for exe_loc in ["build/Release/moecher.exe", "build/moecher.exe", "moecher.exe"]:
        src_exe = root_dir / exe_loc
        if src_exe.exists():
            shutil.copy2(src_exe, out_path / "moecher.exe")
            print(f"  + moecher.exe (from {exe_loc})")
            break
            
    # 2. Copy directories
    dirs_to_copy = [
        ("web", out_path / "web"),
        ("src", out_path / "src"),
        ("scripts", out_path / "scripts"),
    ]
    
    print("[3/4] Copying source code, web UI, and utilities...")
    for src_name, dst_dir in dirs_to_copy:
        src_dir = root_dir / src_name
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.o", "*.a", "*.bin"))
            print(f"  + {src_name}/")

    # 3. Create Windows Quickstart README
    quickstart_txt = out_path / "QUICKSTART.txt"
    quickstart_txt.write_text(
        "MinnieTheMoECher (Windows Native v2.05)\n"
        "======================================\n\n"
        "QUICK START:\n"
        "1. Place your model weight binaries in this folder:\n"
        "   - attention_dense_layers.bin (9.44 GB)\n"
        "   - moe_experts_iq2.bin (72.56 GB)\n\n"
        "2. If you haven't built the binary yet, right click 'install.ps1' -> Run with PowerShell.\n\n"
        "3. Double-click 'start.bat' to launch the inference server!\n\n"
        "4. Open your web browser at: http://localhost:8001\n"
    )

    # 4. Optional Zip Archive
    if make_zip:
        zip_file = root_dir / f"{out_path.name}.zip"
        print(f"[4/4] Creating ZIP archive: {zip_file}...")
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out_path):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(out_path.parent)
                    zf.write(file_path, arcname)
        print(f"[DONE] Created standalone archive: {zip_file}")
    else:
        print(f"[4/4] Package directory ready at: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package MinnieTheMoECher for Windows")
    parser.add_argument("--output", default="dist_windows", help="Output directory name")
    parser.add_argument("--zip", action="store_true", help="Create a .zip archive")
    args = parser.parse_args()
    package_windows(args.output, args.zip)

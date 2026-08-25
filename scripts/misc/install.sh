#!/bin/bash
set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}     MinnieTheMoECher Interactive Installer${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""

# 1. WSL Check
if grep -q microsoft /proc/version 2>/dev/null; then
    echo -e "${YELLOW}Detected WSL environment.${NC}"
    if [[ "$PWD" == /mnt/* ]]; then
        echo -e "${RED}ERROR: You are running this script from a mounted Windows drive ($PWD).${NC}"
        echo -e "${RED}The engine requires O_DIRECT which is not supported on drvfs.${NC}"
        echo -e "${RED}Please move this repository to the Linux filesystem (e.g. ~/minniethemoecher) and run again.${NC}"
        exit 1
    fi
fi

# 2. Check Disk Space
echo -e "${GREEN}[1/5] Checking disk space...${NC}"
FREE_SPACE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$FREE_SPACE_GB" -lt 400 ]; then
    echo -e "${YELLOW}WARNING: Only ${FREE_SPACE_GB}GB free space detected in the current directory.${NC}"
    echo -e "${YELLOW}The model weights and converted binaries require approximately 400GB.${NC}"
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Installation aborted.${NC}"
        exit 1
    fi
else
    echo -e "Available space: ${FREE_SPACE_GB}GB (sufficient)."
fi

# 3. Prerequisites
echo -e "\n${GREEN}[2/5] Checking and installing prerequisites...${NC}"

if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}nvcc (CUDA Toolkit) not found. MinnieTheMoECher requires CUDA Toolkit 12.0+.${NC}"
    read -p "Would you like to automatically install the NVIDIA CUDA Toolkit 12.x? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Installing CUDA Toolkit...${NC}"
        source /etc/os-release
        OS_VERSION=$(echo $VERSION_ID | tr -d '.')
        
        REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OS_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb"
        
        if grep -q microsoft /proc/version 2>/dev/null; then
            REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
        fi
        
        wget "$REPO_URL" -O /tmp/cuda-keyring.deb
        sudo dpkg -i /tmp/cuda-keyring.deb
        sudo apt-get update
        sudo apt-get install -y cuda-toolkit
        rm /tmp/cuda-keyring.deb
        
        export PATH=/usr/local/cuda/bin:$PATH
        if ! command -v nvcc &> /dev/null; then
             echo -e "${RED}CUDA installation may have failed or nvcc is not in PATH. Please install manually.${NC}"
             exit 1
        fi
    else
        echo -e "${YELLOW}Please ensure CUDA Toolkit 12.0+ is installed manually. The build may fail.${NC}"
        read -p "Do you want to continue anyway? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${RED}Installation aborted.${NC}"
            exit 1
        fi
    fi
else
    echo -e "CUDA Toolkit detected."
fi

MISSING_PKGS=""
if ! command -v cmake &> /dev/null; then MISSING_PKGS="$MISSING_PKGS cmake"; fi
if ! command -v python3 &> /dev/null; then MISSING_PKGS="$MISSING_PKGS python3"; fi
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then MISSING_PKGS="$MISSING_PKGS python3-pip"; fi

if [ -n "$MISSING_PKGS" ]; then
    echo -e "${YELLOW}Missing packages:${MISSING_PKGS}${NC}"
    read -p "Would you like to install them via apt-get? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt-get update
        sudo apt-get install -y $MISSING_PKGS
    else
        echo -e "${RED}Please install missing packages manually to continue.${NC}"
        exit 1
    fi
else
    echo -e "All basic prerequisites (cmake, python3, pip) are installed."
fi

# 4. Download Model
echo -e "\n${GREEN}[3/5] Downloading model from HuggingFace...${NC}"
echo -e "This will download deepseek-ai/DeepSeek-V4-Flash-0731 to ~/.cache/huggingface/hub/"
read -p "Start download? (Y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    python3 -m pip install --upgrade huggingface_hub
    huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731
else
    echo -e "${YELLOW}Skipping download. Assuming model is already downloaded.${NC}"
fi

# 5. Generate Manifest & Binaries
echo -e "\n${GREEN}[4/5] Generating manifest and processing binaries...${NC}"
echo -e "This step converts the HuggingFace safetensors into O_DIRECT aligned binaries."
echo -e "It may take ~30 minutes depending on your CPU and NVMe speed, and needs ~380GB of space."
read -p "Continue? (Y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    DEFAULT_MODEL_DIR="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731"
    read -p "Enter model directory [$DEFAULT_MODEL_DIR]: " MODEL_DIR
    MODEL_DIR=${MODEL_DIR:-$DEFAULT_MODEL_DIR}
    
    if [ ! -d "$MODEL_DIR" ]; then
        echo -e "${RED}Error: Directory $MODEL_DIR not found.${NC}"
        exit 1
    fi

    python3 scripts/build_manifest.py --model-dir "$MODEL_DIR" --output-dir .
else
    echo -e "${YELLOW}Skipping manifest generation.${NC}"
fi

# 6. Build Engine
echo -e "\n${GREEN}[5/5] Building the C++ Inference Engine...${NC}"
read -p "Start build? (Y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    mkdir -p build && cd build
    cmake ..
    make -j$(nproc)
    cd ..
else
    echo -e "${YELLOW}Skipping build.${NC}"
fi

# 7. Final Instructions
echo -e "\n${BLUE}======================================================${NC}"
echo -e "${GREEN}               Installation Complete!                 ${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "\n${YELLOW}How to launch MinnieTheMoECher:${NC}"
echo -e "  ./build/moecher --manifest moecher_manifest.json --port 8001"
echo ""
echo -e "${YELLOW}Optional Parameters:${NC}"
echo -e "  --max-vram <GB>       Limit VRAM usage (e.g., --max-vram 20 for RTX 3090)"
echo -e "  --dram-cache-gb <GB>  Allocate system RAM as an L2 cache for MoE experts"
echo -e "  --quiet               Disable INFO logging for higher performance"
echo ""
echo -e "${YELLOW}To interact via chat client (in a separate terminal):${NC}"
echo -e "  python3 chat.py"
echo ""

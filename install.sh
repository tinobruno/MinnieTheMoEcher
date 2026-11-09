#!/usr/bin/env bash
# ==============================================================================
#  MinnieTheMoEcher: Unified 1-Click Automated Installer (Linux CUDA & macOS Metal)
# ==============================================================================
set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RED="\033[0;31m"
RESET="\033[0m"

clear || true
echo -e "${CYAN}${BOLD}"
echo " ╔════════════════════════════════════════════════════════════════════╗"
echo " ║             MinnieTheMoEcher 1-Click Installer                     ║"
echo " ║     Bare-Metal MoE & Hybrid Attention Fast Inference Engine        ║"
echo " ╚════════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

OS_TYPE="$(uname -s)"
ARCH_TYPE="$(uname -m)"

has_sudo() {
    command -v sudo >/dev/null 2>&1
}

# ── macOS (Apple Silicon Metal) Path ──────────────────────────────────────────
if [[ "$OS_TYPE" == "Darwin" ]]; then
    echo -e "${BLUE}[1/4] Detecting macOS Apple Silicon Hardware...${RESET}"
    CHIP_NAME=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
    RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 17179869184)
    RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))
    echo -e "  ${GREEN}✔${RESET} Detected Chip: ${BOLD}${CHIP_NAME}${RESET} (${ARCH_TYPE})"
    echo -e "  ${GREEN}✔${RESET} Unified Memory: ${BOLD}${RAM_GB} GB${RESET}"
    echo ""

    echo -e "${BLUE}[2/4] Checking Xcode Command Line Tools & CMake...${RESET}"
    if ! xcode-select -p >/dev/null 2>&1; then
        echo -e "${YELLOW}Installing Xcode Command Line Tools...${RESET}"
        xcode-select --install || true
    fi
    if ! command -v cmake >/dev/null 2>&1; then
        if command -v brew >/dev/null 2>&1; then
            echo -e "${YELLOW}Installing CMake via Homebrew...${RESET}"
            brew install cmake
        else
            echo -e "${RED}[ERROR] CMake not found. Install Homebrew (https://brew.sh) or CMake (https://cmake.org).${RESET}"
            exit 1
        fi
    fi
    echo -e "  ${GREEN}✔${RESET} Xcode Clang & CMake ready."
    echo ""

    echo -e "${BLUE}[3/4] Building MinnieTheMoEcher with Apple Metal Backend...${RESET}"
    if command -v python3 >/dev/null 2>&1 && [ -f "scripts/embed_web.py" ]; then
        python3 scripts/embed_web.py
    fi
    mkdir -p build
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_METAL=ON
    NPROCS=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)
    cmake --build build --config Release -j"${NPROCS}"
    cp -f build/moecher ./moecher
    echo -e "  ${GREEN}✔${RESET} Build successful! Binary ready at: ${BOLD}./moecher${RESET}"
    echo ""

    echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════${RESET}"
    echo -e "${GREEN}${BOLD}       macOS Metal Installation Completed Successfully!             ${RESET}"
    echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════${RESET}"
    echo -e "Run ${CYAN}${BOLD}./moecher${RESET} to select a model interactively and open ${BOLD}http://localhost:8001/${RESET}"
    exit 0
fi

# ── Linux (NVIDIA CUDA) Path ──────────────────────────────────────────────────
echo -e "${BLUE}[1/4] Detecting Linux System Hardware...${RESET}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "${RED}[ERROR] NVIDIA GPU / nvidia-smi not found. Please install NVIDIA drivers first.${RESET}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
VRAM_GB=$((VRAM_MB / 1024))
RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
RAM_GB=$((RAM_KB / 1024 / 1024))

echo -e "  ${GREEN}✔${RESET} Detected GPU:  ${BOLD}${GPU_NAME}${RESET} (${VRAM_GB} GB VRAM)"
echo -e "  ${GREEN}✔${RESET} Detected RAM:  ${BOLD}${RAM_GB} GB System Memory${RESET}"

CUDA_ARCH="86;89;90;120a"
if [[ "$GPU_NAME" =~ "3090" ]] || [[ "$GPU_NAME" =~ "3080" ]] || [[ "$GPU_NAME" =~ "A5000" ]] || [[ "$GPU_NAME" =~ "A6000" ]]; then
    CUDA_ARCH="86"
elif [[ "$GPU_NAME" =~ "4090" ]] || [[ "$GPU_NAME" =~ "4080" ]] || [[ "$GPU_NAME" =~ "4060" ]] || [[ "$GPU_NAME" =~ "Ada" ]]; then
    CUDA_ARCH="89"
elif [[ "$GPU_NAME" =~ "5090" ]] || [[ "$GPU_NAME" =~ "5080" ]] || [[ "$GPU_NAME" =~ "5070" ]] || [[ "$GPU_NAME" =~ "5060" ]] || [[ "$GPU_NAME" =~ "PRO 6000" ]]; then
    CUDA_ARCH="120a"
elif [[ "$GPU_NAME" =~ "H100" ]] || [[ "$GPU_NAME" =~ "H200" ]]; then
    CUDA_ARCH="90"
elif [[ "$GPU_NAME" =~ "A100" ]]; then
    CUDA_ARCH="80"
fi
echo -e "  ${GREEN}✔${RESET} Target CUDA Arch: ${BOLD}sm_${CUDA_ARCH}${RESET}"
echo ""

echo -e "${BLUE}[2/4] Checking Build Prerequisites & Dependencies...${RESET}"
MISSING_PKGS=()
for cmd in cmake g++ git curl python3; do
    if ! command -v $cmd >/dev/null 2>&1; then
        MISSING_PKGS+=($cmd)
    fi
done

if ! command -v nvcc >/dev/null 2>&1; then
    if [ -d "/usr/local/cuda/bin" ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    fi
fi

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing packages: ${MISSING_PKGS[*]}${RESET}"
    if has_sudo; then
        sudo apt-get update && sudo apt-get install -y build-essential cmake git curl python3 python3-pip libcurl4-openssl-dev
    else
        apt-get update && apt-get install -y build-essential cmake git curl python3 python3-pip libcurl4-openssl-dev
    fi
else
    echo -e "  ${GREEN}✔${RESET} All essential build tools and libraries are present."
fi
echo ""

echo -e "${BLUE}[3/4] Building MinnieTheMoEcher Engine...${RESET}"
if command -v python3 >/dev/null 2>&1 && [ -f "scripts/embed_web.py" ]; then
    python3 scripts/embed_web.py
fi
mkdir -p build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
NPROCS=$(nproc 2>/dev/null || echo 8)
cmake --build build --config Release -j"${NPROCS}"
cp -f build/moecher ./moecher
echo -e "  ${GREEN}✔${RESET} Build successful! Binary ready at: ${BOLD}./moecher${RESET}"
echo ""

echo -e "${BLUE}[4/4] Ready!${RESET}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}       Linux CUDA Installation Completed Successfully!              ${RESET}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════${RESET}"
echo -e "Run ${CYAN}${BOLD}./moecher${RESET} to select a model interactively and open ${BOLD}http://localhost:8001/${RESET}"

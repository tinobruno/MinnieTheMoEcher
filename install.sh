#!/usr/bin/env bash
# ==============================================================================
#  MinnieTheMoEcher: 1-Click Automated Installer & Setup
# ==============================================================================
set -e

# ANSI Color formatting
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
echo " ║         Bare-Metal DeepSeek-V4 MoE Fast Inference Engine           ║"
echo " ╚════════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── 1. Check Root / Sudo Availability ─────────────────────────────────────────
has_sudo() {
    command -v sudo >/dev/null 2>&1
}

# ── 2. Hardware Detection ─────────────────────────────────────────────────────
echo -e "${BLUE}[1/5] Detecting System Hardware...${RESET}"

# Detect GPU
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "${RED}[ERROR] NVIDIA GPU / nvidia-smi not found. Please install NVIDIA drivers first.${RESET}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
VRAM_GB=$((VRAM_MB / 1024))

# Detect System RAM
RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
RAM_GB=$((RAM_KB / 1024 / 1024))

echo -e "  ${GREEN}✔${RESET} Detected GPU:  ${BOLD}${GPU_NAME}${RESET} (${VRAM_GB} GB VRAM)"
echo -e "  ${GREEN}✔${RESET} Detected RAM:  ${BOLD}${RAM_GB} GB System Memory${RESET}"

# Detect CUDA Compute Capability
CUDA_ARCH="86" # Default Ampere / Ada / Blackwell fallback
if [[ "$GPU_NAME" =~ "3090" ]] || [[ "$GPU_NAME" =~ "3080" ]] || [[ "$GPU_NAME" =~ "A5000" ]] || [[ "$GPU_NAME" =~ "A6000" ]]; then
    CUDA_ARCH="86"
elif [[ "$GPU_NAME" =~ "4090" ]] || [[ "$GPU_NAME" =~ "4080" ]] || [[ "$GPU_NAME" =~ "Ada" ]]; then
    CUDA_ARCH="89"
elif [[ "$GPU_NAME" =~ "H100" ]] || [[ "$GPU_NAME" =~ "H200" ]] || [[ "$GPU_NAME" =~ "Blackwell" ]] || [[ "$GPU_NAME" =~ "PRO 6000" ]]; then
    CUDA_ARCH="90;89"
elif [[ "$GPU_NAME" =~ "A100" ]]; then
    CUDA_ARCH="80"
fi
echo -e "  ${GREEN}✔${RESET} Target CUDA Arch: ${BOLD}sm_${CUDA_ARCH}${RESET}"

# Auto-calculate optimal cache allocation
MAX_VRAM_FLAG=$((VRAM_GB > 4 ? VRAM_GB - 2 : VRAM_GB))
DRAM_CACHE_FLAG=$((RAM_GB > 24 ? RAM_GB - 16 : RAM_GB / 2))
if [ "$DRAM_CACHE_FLAG" -lt 0 ]; then DRAM_CACHE_FLAG=0; fi

echo -e "  ${GREEN}✔${RESET} Recommended VRAM budget: ${BOLD}${MAX_VRAM_FLAG} GB${RESET}"
echo -e "  ${GREEN}✔${RESET} Recommended DRAM cache:  ${BOLD}${DRAM_CACHE_FLAG} GB${RESET}"
echo ""

# ── 3. Install System Dependencies ───────────────────────────────────────────
echo -e "${BLUE}[2/5] Checking Build Prerequisites & Dependencies...${RESET}"

MISSING_PKGS=()
for cmd in cmake g++ git curl python3; do
    if ! command -v $cmd >/dev/null 2>&1; then
        MISSING_PKGS+=($cmd)
    fi
done

if ! command -v nvcc >/dev/null 2>&1; then
    echo -e "${YELLOW}[!] nvcc (CUDA Compiler) was not found in PATH.${RESET}"
    if [ -d "/usr/local/cuda/bin" ]; then
        echo -e "  Found CUDA at /usr/local/cuda/bin. Adding to PATH..."
        export PATH="/usr/local/cuda/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    fi
fi

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing packages: ${MISSING_PKGS[*]}${RESET}"
    read -p "Install missing packages automatically with apt? [Y/n]: " INSTALL_APT
    INSTALL_APT=${INSTALL_APT:-Y}
    if [[ "$INSTALL_APT" =~ ^[Yy]$ ]]; then
        if has_sudo; then
            sudo apt-get update
            sudo apt-get install -y build-essential cmake git curl python3 python3-pip libcurl4-openssl-dev
        else
            apt-get update && apt-get install -y build-essential cmake git curl python3 python3-pip libcurl4-openssl-dev
        fi
    fi
else
    echo -e "  ${GREEN}✔${RESET} All essential build tools and libraries are present."
fi
echo ""

# ── 4. Compile Engine ─────────────────────────────────────────────────────────
echo -e "${BLUE}[3/5] Building MinnieTheMoEcher Engine...${RESET}"
read -p "Compile high-performance C++/CUDA binaries now? [Y/n]: " DO_BUILD
DO_BUILD=${DO_BUILD:-Y}

if [[ "$DO_BUILD" =~ ^[Yy]$ ]]; then
    mkdir -p build
    cd build
    echo -e "  Configuring CMake with CUDA architecture: ${BOLD}${CUDA_ARCH}${RESET}..."
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" >/dev/null
    
    NPROCS=$(nproc)
    echo -e "  Compiling with ${NPROCS} CPU cores..."
    cmake --build . -j"${NPROCS}"
    cd ..
    echo -e "  ${GREEN}✔${RESET} Build successful! Binary ready at: ${BOLD}./build/moecher${RESET}"
fi
echo ""

# ── 5. Setup 1-Click Launch Script ────────────────────────────────────────────
echo -e "${BLUE}[4/5] Generating 1-Click Launcher (start.sh)...${RESET}"

DEFAULT_MANIFEST="moecher_manifest_iq2.json"
if [ ! -f "$DEFAULT_MANIFEST" ] && [ -f "moecher_manifest.json" ]; then
    DEFAULT_MANIFEST="moecher_manifest.json"
fi

cat << 'EOF' > start.sh
#!/usr/bin/env bash
# Auto-generated MinnieTheMoEcher Launch Script
cd "$(dirname "$0")"

MANIFEST="${1:-__MANIFEST__}"
MAX_VRAM="${2:-__MAX_VRAM__}"
DRAM_CACHE="${3:-__DRAM_CACHE__}"
PORT="${4:-8001}"

echo "================================================================="
echo " Starting MinnieTheMoEcher DeepSeek-V4 Server"
echo " Manifest:    ${MANIFEST}"
echo " VRAM Budget: ${MAX_VRAM} GB"
echo " DRAM Cache:  ${DRAM_CACHE} GB"
echo " Port:        ${PORT}"
echo " Web UI:      http://localhost:${PORT}/"
echo "================================================================="

exec ./build/moecher \
  --manifest "${MANIFEST}" \
  --max-vram "${MAX_VRAM}" \
  --dram-cache-gb "${DRAM_CACHE}" \
  --port "${PORT}" \
  --quiet
EOF

sed -i "s/__MANIFEST__/${DEFAULT_MANIFEST}/g" start.sh
sed -i "s/__MAX_VRAM__/${MAX_VRAM_FLAG}/g" start.sh
sed -i "s/__DRAM_CACHE__/${DRAM_CACHE_FLAG}/g" start.sh
chmod +x start.sh

echo -e "  ${GREEN}✔${RESET} Created ${BOLD}./start.sh${RESET} configured for your hardware."
echo ""

# ── 6. Completed Summary ──────────────────────────────────────────────────────
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}       Installation & Configuration Completed Successfully!         ${RESET}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "To start the model server at any time, simply run:"
echo -e "  ${CYAN}${BOLD}./start.sh${RESET}"
echo ""
echo -e "To chat with the model:"
echo -e "  1. Web Interface:   Open ${BOLD}http://localhost:8001/${RESET} in your browser"
echo -e "  2. Terminal Chat:   Run ${BOLD}python3 chat.py${RESET}"
echo -e "  3. OpenAI API:      Connect tools to ${BOLD}http://localhost:8001/v1${RESET}"
echo ""

read -p "Would you like to start the server now? [y/N]: " RUN_NOW
if [[ "$RUN_NOW" =~ ^[Yy]$ ]]; then
    ./start.sh
fi

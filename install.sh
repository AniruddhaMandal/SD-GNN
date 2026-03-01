#!/usr/bin/env bash
# install.sh — reproducible environment setup for SD-GNN
# Usage: bash install.sh [--cuda cu128|cu126|cu121|cpu] [--python python3.12]
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-}"                   # auto-detected below if not set
CUDA_TAG="${CUDA_TAG:-cu128}"          # override: bash install.sh --cuda cu121
TORCH_VERSION="2.10.0"
VENV_DIR="venvSD"

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)   CUDA_TAG="$2";   shift 2 ;;
        --python) PYTHON="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Auto-detect Python 3.11 or 3.12 if not specified
if [[ -z "$PYTHON" ]]; then
    for candidate in python3.11 python3.12 python3; do
        if command -v "$candidate" &>/dev/null; then
            VERSION=$("$candidate" -c "import sys; print(sys.version_info[:2])")
            if [[ "$VERSION" == "(3, 11)" || "$VERSION" == "(3, 12)" ]]; then
                PYTHON="$candidate"
                break
            fi
        fi
    done
    if [[ -z "$PYTHON" ]]; then
        echo "ERROR: Python 3.11 or 3.12 not found. Install one or specify with --python."
        exit 1
    fi
fi

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

# ── Prerequisite checks ────────────────────────────────────────────────────────
echo "==> Checking prerequisites..."

if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. Install Python 3.11 or 3.12 first."
    exit 1
fi
DETECTED_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "    Using $PYTHON ($DETECTED_VER)"

if ! command -v g++ &>/dev/null; then
    echo "ERROR: g++ not found. Install build tools:"
    echo "  Ubuntu/Debian: sudo apt install build-essential"
    exit 1
fi

if [[ "$CUDA_TAG" != "cpu" ]]; then
    if ! command -v nvcc &>/dev/null; then
        echo "WARNING: nvcc not found. CUDA toolkit may not be installed."
        echo "  If you want CPU-only mode, rerun with: bash install.sh --cuda cpu"
    else
        SYSTEM_CUDA=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        echo "    System CUDA: $SYSTEM_CUDA | Target: $CUDA_TAG"
    fi
fi

# ── Virtual environment ────────────────────────────────────────────────────────
echo "==> Creating virtual environment in ./$VENV_DIR ..."
"$PYTHON" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip --quiet

# ── PyTorch ───────────────────────────────────────────────────────────────────
echo "==> Installing PyTorch ${TORCH_VERSION}+${CUDA_TAG} ..."
pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX" --quiet

# ── torch-scatter (must be built from source — no pre-built wheels for torch 2.10) ──
echo "==> Building torch-scatter from source (this takes ~10 minutes) ..."
MAX_JOBS="${MAX_JOBS:-$(nproc)}" pip install torch-scatter --no-binary torch-scatter -q

# ── Python dependencies ────────────────────────────────────────────────────────
echo "==> Installing Python dependencies from requirements.txt ..."
pip install -r requirements.txt --quiet

# ── C++ sampler extensions ────────────────────────────────────────────────────
echo "==> Building graphlet sampler ..."
pip install -e src/samplers/graphlet_sampler --quiet

# ── GXL (main library) ────────────────────────────────────────────────────────
echo "==> Installing GXL (editable) ..."
pip install -e "src/gxl[gnn]" --quiet

# ── Smoke test ────────────────────────────────────────────────────────────────
echo "==> Running smoke test ..."
python - <<'EOF'
import torch, torch_geometric, torch_scatter
print(f"  torch:          {torch.__version__}")
print(f"  torch-geometric:{torch_geometric.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
import gxl
print(f"  gxl:            OK")
EOF

echo ""
echo "==> Setup complete."
echo "    Activate the environment with: source $VENV_DIR/bin/activate"
echo "    Run an experiment with:        gxl -c configs/sd_gnn/TUData/mutag-gin.json"

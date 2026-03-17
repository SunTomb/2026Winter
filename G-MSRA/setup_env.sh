#!/bin/bash
# ============================================================
# G-MSRA Environment Setup Script
# Target: USTC LDS Lab Cluster (Song/Tang/Sui nodes)
# ============================================================

set -e

# --- Proxy (required for nodes without direct internet) ---
# export http_proxy=http://192.168.1.130:7890
# export https_proxy=http://192.168.1.130:7890

ENV_NAME="gmsra"
PYTHON_VERSION="3.10"

echo "============================================"
echo "  G-MSRA Environment Setup"
echo "  Target: USTC LDS Cluster"
echo "============================================"

# --- Step 1: Create conda environment ---
echo "[1/4] Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# --- Step 2: Install PyTorch (CUDA 12.1) ---
echo "[2/4] Installing PyTorch with CUDA 12.1"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- Step 3: Install project dependencies ---
echo "[3/4] Installing project dependencies"
pip install -r requirements.txt

# --- Step 4: Install G-MSRA as editable package ---
echo "[4/4] Installing G-MSRA in development mode"
pip install -e .

# --- Verify ---
echo ""
echo "============================================"
echo "  Verification"
echo "============================================"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)')
import transformers, peft, trl
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT: {peft.__version__}')
print(f'  TRL: {trl.__version__}')
print('  ✓ All dependencies verified!')
"

echo ""
echo "Setup complete! Activate with: conda activate ${ENV_NAME}"

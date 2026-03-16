#!/bin/bash
# ============================================================
# G-MSRA Environment Setup Script
# Target: USTC LDS Lab Cluster (Song/Tang/Sui nodes)
# ============================================================

set -e

# [修复1]：注释掉失效的内网代理，直接走中科大校园网的镜像源
# export http_proxy=http://192.168.1.130:7890
# export https_proxy=http://192.168.1.130:7890

ENV_NAME="gmsra"
PYTHON_VERSION="3.10"

echo "============================================"
echo "  G-MSRA Environment Setup (Fixed Version)"
echo "  Target: USTC LDS Cluster"
echo "============================================"

# --- Step 1: Create conda environment ---
echo "[1/4] Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# [修复2]：faiss-gpu 必须走 conda 渠道安装，放在 pip 之前执行
echo "[1.5/4] Installing faiss-gpu via conda to prevent pip errors"
conda install -n ${ENV_NAME} -c pytorch -c nvidia faiss-gpu -y

# [修复3]：全部使用 `conda run -n ${ENV_NAME}` 前缀
# 这样可以 100% 保证包被装进 gmsra 环境，彻底避免环境变量串台导致装错地方
# --- Step 2: Install PyTorch (CUDA 12.1) ---
echo "[2/4] Installing PyTorch with CUDA 12.1"
conda run -n ${ENV_NAME} pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- Step 3: Install project dependencies ---
echo "[3/4] Installing project dependencies"
conda run -n ${ENV_NAME} pip install -r requirements.txt

# --- Step 4: Install G-MSRA as editable package ---
echo "[4/4] Installing G-MSRA in development mode"
conda run -n ${ENV_NAME} pip install -e .

# --- Verify ---
echo ""
echo "============================================"
echo "  Verification"
echo "============================================"
conda run -n ${ENV_NAME} python -c "
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
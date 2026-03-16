#!/bin/bash
# ============================================================
# G-MSRA: Job script for Tang nodes (8×A40 45G)
# Suitable for Phase 0 (SFT) and evaluation
# For RL training on 7B models, prefer Song nodes (A100)
# ============================================================

PHASE=${1:-"phase0"}
NUM_GPUS=${2:-2}
MODEL_NAME=${3:-"Qwen/Qwen2.5-7B-Instruct"}

export http_proxy=http://192.168.1.130:7890
export https_proxy=http://192.168.1.130:7890
export NCCL_IB_DISABLE=1

eval "$(conda shell.bash hook)"
conda activate gmsra

PROJECT_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
cd ${PROJECT_DIR}

export WANDB_PROJECT="gmsra"
export WANDB_RUN_NAME="${PHASE}_tang_$(date +%Y%m%d_%H%M%S)"

echo "============================================"
echo "  G-MSRA Training on Tang Node (A40)"
echo "  Phase: ${PHASE} | GPUs: ${NUM_GPUS}"
echo "============================================"

# NOTE: A40 has 45G VRAM. For 7B models, use QLoRA (4-bit) to fit.
EXTRA_ARGS="--use_qlora --load_in_4bit"

case ${PHASE} in
    "phase0")
        python scripts/train_phase0_sft.py \
            --model_name ${MODEL_NAME} \
            --output_dir outputs/phase0_tang \
            ${EXTRA_ARGS}
        ;;
    "eval")
        python scripts/eval_locomo.py \
            --checkpoint outputs/phase3/best \
            --output_dir results/ \
            ${EXTRA_ARGS}
        ;;
    *)
        echo "For RL training (Phase 1-3), use Song nodes (A100 80G)."
        echo "Tang nodes support: phase0, eval"
        exit 1
        ;;
esac

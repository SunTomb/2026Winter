#!/bin/bash
# ============================================================
# G-MSRA: Job script for Tang nodes (8×A40 48G)
# Supports Phase 0 (SFT), Phase 1 (RL, multi-GPU), and evaluation
# ============================================================

PHASE=${1:-"phase0"}
NUM_GPUS=${2:-4}
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

case ${PHASE} in
    "phase0")
        # A40 Phase 0: use QLoRA to fit single GPU
        python scripts/train_phase0_sft.py \
            --model_name ${MODEL_NAME} \
            --output_dir outputs/phase0_tang \
            --use_qlora --load_in_4bit
        ;;
    "phase1")
        # A40 Phase 1: multi-GPU RL training with accelerate
        # 4× A40 (48GB each), bf16 full precision, data-parallel
        # --gpu_preset a40 auto-configures: per_device_bs=4, num_gens=8,
        #   max_completion_length=192, gradient_accumulation_steps=4
        accelerate launch \
            --config_file cluster/accelerate_a40.yaml \
            --num_processes ${NUM_GPUS} \
            scripts/train_phase1_rl.py \
            --model_name ${MODEL_NAME} \
            --output_dir outputs/phase1 \
            --num_episodes 5000 \
            --no_qlora \
            --gpu_preset a40
        ;;
    "phase1_ds")
        # A40 Phase 1 with DeepSpeed ZeRO-2 for extra memory savings
        accelerate launch \
            --config_file cluster/accelerate_a40.yaml \
            --num_processes ${NUM_GPUS} \
            scripts/train_phase1_rl.py \
            --model_name ${MODEL_NAME} \
            --output_dir outputs/phase1 \
            --num_episodes 5000 \
            --no_qlora \
            --gpu_preset a40 \
            --deepspeed cluster/ds_zero2_a40.json
        ;;
    "eval")
        python scripts/eval_locomo.py \
            --checkpoint outputs/phase3/best \
            --output_dir results/ \
            --use_qlora --load_in_4bit
        ;;
    *)
        echo "Supported phases: phase0, phase1, phase1_ds, eval"
        echo "Usage: bash cluster/run_tang.sh <phase> [num_gpus] [model_name]"
        echo ""
        echo "Examples:"
        echo "  bash cluster/run_tang.sh phase1 4     # 4× A40 RL training"
        echo "  bash cluster/run_tang.sh phase1_ds 4  # with DeepSpeed ZeRO-2"
        exit 1
        ;;
esac

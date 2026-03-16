#!/bin/bash
# ============================================================
# G-MSRA: Job script for Song nodes (8×A100 80G)
# Usage: Reserve GPUs via http://210.45.70.34:9768/
#        then: bash cluster/run_song.sh <phase> <num_gpus>
# ============================================================

PHASE=${1:-"phase1"}
NUM_GPUS=${2:-4}
MODEL_NAME=${3:-"Qwen/Qwen2.5-7B-Instruct"}

# --- Proxy for downloading models ---
export http_proxy=http://192.168.1.130:7890
export https_proxy=http://192.168.1.130:7890

# --- NCCL settings for multi-GPU ---
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# --- Activate environment ---
eval "$(conda shell.bash hook)"
conda activate gmsra

# --- Project directory ---
PROJECT_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
cd ${PROJECT_DIR}

# --- W&B logging ---
export WANDB_PROJECT="gmsra"
export WANDB_RUN_NAME="${PHASE}_$(date +%Y%m%d_%H%M%S)"

echo "============================================"
echo "  G-MSRA Training on Song Node (A100)"
echo "  Phase: ${PHASE}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Model: ${MODEL_NAME}"
echo "============================================"

case ${PHASE} in
    "phase0")
        python scripts/train_phase0_sft.py \
            --model_name ${MODEL_NAME} \
            --output_dir outputs/phase0 \
            --num_epochs 3 \
            --batch_size 4 \
            --learning_rate 2e-5
        ;;
    "phase1")
        accelerate launch --num_processes ${NUM_GPUS} \
            scripts/train_phase1_rl.py \
            --model_name ${MODEL_NAME} \
            --output_dir outputs/phase1 \
            --num_episodes 5000 \
            --batch_size 16 \
            --learning_rate 1.41e-5
        ;;
    "phase2")
        accelerate launch --num_processes ${NUM_GPUS} \
            scripts/train_phase2_transition.py \
            --checkpoint outputs/phase1/best \
            --output_dir outputs/phase2 \
            --anneal_steps 3000 \
            --tau_threshold 0.5
        ;;
    "phase3")
        accelerate launch --num_processes ${NUM_GPUS} \
            scripts/train_phase3_full.py \
            --checkpoint outputs/phase2/best \
            --output_dir outputs/phase3 \
            --max_episodes 10000 \
            --consolidation_enabled
        ;;
    "eval")
        python scripts/eval_locomo.py \
            --checkpoint outputs/phase3/best \
            --output_dir results/
        python scripts/eval_agent_tasks.py \
            --checkpoint outputs/phase3/best \
            --env alfworld \
            --output_dir results/
        ;;
    "ablation")
        python scripts/run_ablations.py \
            --base_checkpoint outputs/phase1/best \
            --output_dir results/ablations/
        ;;
    *)
        echo "Unknown phase: ${PHASE}"
        echo "Usage: bash run_song.sh <phase0|phase1|phase2|phase3|eval|ablation> [num_gpus] [model_name]"
        exit 1
        ;;
esac

echo "Done! Results saved to outputs/ or results/"

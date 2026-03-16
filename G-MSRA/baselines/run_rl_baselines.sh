#!/bin/bash
#============================================================
#  Overnight RL Baseline Training + Evaluation Script
#  
#  Runs Memory-R1 and Mem0+Memory-R1 baselines sequentially.
#  Designed for unattended execution on USTC LDS cluster.
#
#  Usage:
#    # On Tang2 (recommended):
#    cd /NAS/yesh/G-MSRA
#    nohup bash baselines/run_rl_baselines.sh > results/baselines/overnight.log 2>&1 &
#
#    # Monitor progress:
#    tail -f results/baselines/overnight.log
#
#    # Check if still running:
#    ps aux | grep train_and_eval_rl_baselines
#============================================================

set -e  # Exit on error

# ---- Configuration ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

CONDA_ENV="gmsra"
DATA_DIR="data"
OUTPUT_DIR="results/baselines"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_EPOCHS=3
LR="5e-5"

# ---- Setup ----
echo "============================================================"
echo "  G-MSRA RL Baseline Overnight Pipeline"
echo "  Start time: $(date)"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Project: $PROJECT_DIR"
echo "============================================================"
echo ""

# Activate conda
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" 2>/dev/null || echo "Warning: conda env '$CONDA_ENV' not found, using current env"
fi

# Ensure data exists
if [ ! -f "$DATA_DIR/locomo_train.json" ]; then
    echo "[Setup] Preparing data..."
    python scripts/prepare_data.py --output_dir "$DATA_DIR"
fi

mkdir -p "$OUTPUT_DIR"

# ---- Run Memory-R1 ----
echo ""
echo "============================================================"
echo "  [1/2] Memory-R1: Train + Evaluate"
echo "  Start: $(date)"
echo "============================================================"

python baselines/train_and_eval_rl_baselines.py \
    --agent memory_r1 \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --train_epochs "$TRAIN_EPOCHS" \
    --lr "$LR" \
    --eval_benchmark locomo

MEMORY_R1_STATUS=$?
echo ""
echo "  Memory-R1 exit code: $MEMORY_R1_STATUS"
echo "  Finished: $(date)"

# ---- Run Mem0+Memory-R1 ----
echo ""
echo "============================================================"
echo "  [2/2] Mem0+Memory-R1: Train + Evaluate"
echo "  Start: $(date)"
echo "============================================================"

python baselines/train_and_eval_rl_baselines.py \
    --agent mem0_memory_r1 \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --train_epochs "$TRAIN_EPOCHS" \
    --lr "$LR" \
    --eval_benchmark locomo

MEM0_R1_STATUS=$?
echo ""
echo "  Mem0+Memory-R1 exit code: $MEM0_R1_STATUS"
echo "  Finished: $(date)"

# ---- Summary ----
echo ""
echo "============================================================"
echo "  OVERNIGHT PIPELINE COMPLETE"
echo "  End time: $(date)"
echo "============================================================"
echo ""
echo "  Memory-R1:        $([ $MEMORY_R1_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "  Mem0+Memory-R1:   $([ $MEM0_R1_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo ""
echo "  Results:     $OUTPUT_DIR/"
echo "  Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "  Log files:   $OUTPUT_DIR/rl_baselines_*.log"
echo ""

# List result files
echo "  Generated files:"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (no JSON files found)"
echo ""

# ---- Previously computed baselines (for reference) ----
echo "  Previously computed baselines (from earlier runs):"
if [ -f "$OUTPUT_DIR/baseline_results.json" ]; then
    python -c "
import json
with open('$OUTPUT_DIR/baseline_results.json') as f:
    data = json.load(f)
for name, benchmarks in data.items():
    for bm, res in benchmarks.items():
        if 'avg_f1' in res:
            print(f'    {name:25s} {bm}: F1={res[\"avg_f1\"]:.4f}')
" 2>/dev/null || echo "    (could not parse)"
fi

exit $(( MEMORY_R1_STATUS + MEM0_R1_STATUS ))

# Baseline Implementations for G-MSRA

This directory contains reproduction implementations of all baseline methods
compared against in the G-MSRA paper.

## Baselines

| Agent | Source | Code Availability | Key Features |
|-------|--------|-------------------|-------------|
| **Reflexion** | Shinn 2023, NeurIPS | Open source | Verbal reflection + episodic buffer, no weight updates |
| **Memory-R1** | Chen 2025 | Partial code | RL CRUD with QA F1 reward, no self-reward |
| **Self-Consolidation** | Zhang 2026 | Paper repro | Contrastive reflection + LoRA, fixed trigger, no RL |
| **EvolveR** | 2025 | Paper repro | Experience lifecycle, principle distillation, no RL |
| **Mem0 + Memory-R1** | Combined | Our composition | Multi-level memory + RL CRUD |

## Architecture

All baselines implement the `BaseAgent` interface:

```python
class BaseAgent(ABC):
    def process_event(event, context) -> dict    # Handle new event
    def answer_question(question) -> str          # Answer using memory
    def reset()                                   # Reset for new episode
    def train_step(reward, **kwargs) -> dict      # RL update (optional)
```

## Usage

### Evaluation Only (Reflexion, Self-Consolidation, EvolveR)

These three baselines do not require pre-training — run them directly:

```bash
CUDA_VISIBLE_DEVICES=6 python baselines/eval_baselines.py --agent reflexion --benchmark locomo
CUDA_VISIBLE_DEVICES=6 python baselines/eval_baselines.py --agent self_consolidation --benchmark locomo
CUDA_VISIBLE_DEVICES=6 python baselines/eval_baselines.py --agent evolver --benchmark locomo
```

### Train + Evaluate (Memory-R1, Mem0+Memory-R1)

These two baselines require RL training before evaluation. Use `train_and_eval_rl_baselines.py`:

```bash
# Train + eval a single RL baseline
CUDA_VISIBLE_DEVICES=6 python baselines/train_and_eval_rl_baselines.py --agent memory_r1

# Train + eval the other
CUDA_VISIBLE_DEVICES=6 python baselines/train_and_eval_rl_baselines.py --agent mem0_memory_r1

# Train both sequentially
CUDA_VISIBLE_DEVICES=6 python baselines/train_and_eval_rl_baselines.py

# Customize training
CUDA_VISIBLE_DEVICES=6 python baselines/train_and_eval_rl_baselines.py \
    --train_epochs 5 --lr 5e-5 --eval_benchmark locomo
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--agent` | both | `memory_r1` or `mem0_memory_r1` |
| `--train_epochs` | 3 | Number of training epochs |
| `--lr` | 5e-5 | Learning rate for LoRA REINFORCE |
| `--eval_benchmark` | locomo | Evaluation benchmark |
| `--data_dir` | data | Dataset directory |
| `--output_dir` | results/baselines | Output for results and checkpoints |

### Overnight Unattended Execution

Use the shell script to run both RL baselines back-to-back with `nohup`:

```bash
cd /NAS/yesh/G-MSRA
nohup bash baselines/run_rl_baselines.sh > results/baselines/overnight.log 2>&1 &

# Monitor progress
tail -f results/baselines/overnight.log

# Check if still running
ps aux | grep train_and_eval_rl_baselines
```

**Output files** (in `results/baselines/`):
- `memory_r1_eval_results.json` — Memory-R1 evaluation scores
- `mem0_memory_r1_eval_results.json` — Mem0+R1 evaluation scores
- `rl_baselines_combined.json` — Combined train metrics + eval results
- `checkpoints/memory_r1_best/` — Best LoRA checkpoint
- `checkpoints/mem0_memory_r1_best/` — Best LoRA checkpoint
- `rl_baselines_*.log` — Detailed training log

## Key Differences from G-MSRA

| Feature | Reflexion | Memory-R1 | Self-Consol | EvolveR | Mem0+R1 | **G-MSRA** |
|---------|:---------:|:---------:|:-----------:|:-------:|:-------:|:----------:|
| RL weight updates | - | GRPO | - | - | GRPO | GRPO |
| Self-reward | - | - | - | - | - | **R_mem** |
| Env grounding | - | QA F1 | - | - | QA F1 | **R_env** |
| Consolidation | - | - | LoRA (fixed) | - | - | **LoRA (adaptive)** |
| Confidence filter | - | - | - | - | - | **Yes** |
| Curriculum | - | - | - | - | - | **4-phase** |

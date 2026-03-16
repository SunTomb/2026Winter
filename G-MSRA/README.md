# G-MSRA: Grounded Memory-Guided Self-Rewarding Agents

> **Closing the Loop: Environment-Grounded Self-Reward for Autonomous Memory Management in Lifelong LLM Agents**

## Overview

G-MSRA is a unified framework that integrates:
1. **RL-based Memory Manager** — learns ADD/UPDATE/DELETE/NOOP policies
2. **Environment-Grounded Self-Reward** — dual-layer composite reward to avoid reward hacking
3. **Adaptive Consolidation** — 3D trigger + semantic LoRA distillation

## Project Structure

```
G-MSRA/
├── gmsra/                     # Core library
│   ├── memory/
│   │   ├── entry.py           # MemoryEntry dataclass
│   │   └── store.py           # MemoryStore with FAISS retrieval
│   ├── reward/
│   │   ├── grounded_reward.py # Dual-layer composite reward
│   │   └── env_signals.py     # Environment signal extractors
│   ├── manager/
│   │   └── memory_manager.py  # RL-based Memory Manager
│   ├── consolidation/
│   │   ├── trigger.py         # Adaptive 3D consolidation trigger
│   │   └── distiller.py       # Semantic LoRA distillation
│   ├── agent.py               # Main G-MSRA agent orchestrator
│   ├── config.py              # Centralized configuration
│   └── utils.py               # Shared utilities
├── scripts/                   # Training & evaluation scripts
│   ├── train_phase0_sft.py
│   ├── train_phase1_rl.py
│   ├── train_phase2_transition.py
│   ├── train_phase3_full.py
│   ├── eval_locomo.py
│   ├── eval_agent_tasks.py
│   └── run_ablations.py
├── cluster/                   # USTC LDS cluster job scripts
│   ├── run_song.sh            # A100 nodes
│   └── run_tang.sh            # A40 nodes
├── paper/                     # LaTeX paper draft
│   ├── main.tex
│   └── references.bib
├── setup_env.sh               # Conda environment setup
└── requirements.txt
```

## Quick Start

### 1. Environment Setup (on USTC LDS Cluster)

```bash
# SSH to Han5 jump server, then to Song1 (A100)
ssh -p 622 <username>@210.45.70.34
ssh song1

# Setup environment
cd /data/<username>
git clone <repo_url> G-MSRA
cd G-MSRA
bash setup_env.sh
```

### 2. Training Pipeline

```bash
conda activate gmsra

# Phase 0: SFT warmup (~1-2 hours on 1x A100)
python scripts/train_phase0_sft.py --model_name Qwen/Qwen2.5-7B-Instruct

# Phase 1: RL + external reward (~2-3 days on 4x A100)
python scripts/train_phase1_rl.py --model_name Qwen/Qwen2.5-7B-Instruct --num_gpus 4

# Phase 2: Self-reward transition (~2-3 days on 4x A100)
python scripts/train_phase2_transition.py --checkpoint outputs/phase1/best

# Phase 3: Full closed-loop (continuous)
python scripts/train_phase3_full.py --checkpoint outputs/phase2/best
```

### 3. Evaluation

```bash
# Conversational memory
python scripts/eval_locomo.py --checkpoint outputs/phase3/best

# Agent tasks
python scripts/eval_agent_tasks.py --checkpoint outputs/phase3/best --env alfworld

# Full ablation suite
python scripts/run_ablations.py --config configs/ablations.yaml

# Project-local baseline reproductions
python scripts/run_baselines.py --max_train_episodes 100
```

## Hardware Requirements

| Phase | Min GPUs | Recommended | Est. Time |
|-------|----------|-------------|-----------|
| Phase 0 (SFT) | 1× A100 80G | 1× A100 | 1-2 hours |
| Phase 1 (RL) | 2× A100 80G | 4× A100 | 2-3 days |
| Phase 2 (Transition) | 2× A100 80G | 4× A100 | 2-3 days |
| Phase 3 (Full Loop) | 2× A100 80G | 4× A100 | 3-5 days |
| Evaluation | 1× A100 80G | 1× A100 | 2-4 hours |

## Citation

```bibtex
@article{gmsra2026,
  title={Closing the Loop: Environment-Grounded Self-Reward for Autonomous Memory Management in Lifelong LLM Agents},
  year={2026}
}
```

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

```bash
# Evaluate all baselines on LoCoMo
python baselines/eval_baselines.py --data_dir data --benchmark locomo

# Evaluate single baseline
python baselines/eval_baselines.py --agent reflexion --benchmark locomo --max_episodes 50

# Evaluate on all benchmarks
python baselines/eval_baselines.py --data_dir data --output_dir results/baselines
```

## Key Differences from G-MSRA

| Feature | Reflexion | Memory-R1 | Self-Consol | EvolveR | Mem0+R1 | **G-MSRA** |
|---------|:---------:|:---------:|:-----------:|:-------:|:-------:|:----------:|
| RL weight updates | - | GRPO | - | - | GRPO | GRPO |
| Self-reward | - | - | - | - | - | **R_mem** |
| Env grounding | - | QA F1 | - | - | QA F1 | **R_env** |
| Consolidation | - | - | LoRA (fixed) | - | - | **LoRA (adaptive)** |
| Confidence filter | - | - | - | - | - | **Yes** |
| Curriculum | - | - | - | - | - | **4-phase** |

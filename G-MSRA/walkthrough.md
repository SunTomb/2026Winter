# G-MSRA Project Walkthrough

## Deliverables Summary

Created a **30-file project** at [G-MSRA/](file:///f:/USTC/2026Winter/G-MSRA) with:

### 📦 Code Framework (18 Python files)

#### Core Library (`gmsra/`)

| Module | Key File | Role |
|--------|----------|------|
| Memory | [entry.py](file:///f:/USTC/2026Winter/G-MSRA/gmsra/memory/entry.py) | Zettelkasten-style memory card with confidence scoring (Eq. 3 in paper) |
| Memory | [store.py](file:///f:/USTC/2026Winter/G-MSRA/gmsra/memory/store.py) | FAISS-backed store with graph links, confidence-filtered retrieval, subgraph extraction |
| Reward | [grounded_reward.py](file:///f:/USTC/2026Winter/G-MSRA/gmsra/reward/grounded_reward.py) | **Core innovation**: dual-layer composite reward $R_{env} + \lambda \cdot R_{mem}$ with drift monitoring |
| Reward | [env_signals.py](file:///f:/USTC/2026Winter/G-MSRA/gmsra/reward/env_signals.py) | 3 extractors: AgentTask, Dialogue (user reaction proxy), ExternalQA |
| Manager | [memory_manager.py](file:///f:/USTC/2026Winter/G-MSRA/gmsra/manager/memory_manager.py) | RL-based CRUD decisions with prompt construction and SFT data generation |
| Consolidation | [trigger.py](file:///f:/USTC/2026Winter/G-MSRA/gmsra/consolidation/trigger.py) | 3D adaptive trigger (conflict + variance + growth) |
| Consolidation | [distiller.py](file:///f:/USTC/2026Winter/G-MSRA/gmsra/consolidation/distiller.py) | Semantic triple generation → LoRA SFT with EWC regularization |
| Orchestrator | [agent.py](file:///f:/USTC/2026Winter/G-MSRA/gmsra/agent.py) | Main agent with online/offline phases, checkpointing, diagnostics |
| Config | [config.py](file:///f:/USTC/2026Winter/G-MSRA/gmsra/config.py) | All hyperparameters in structured dataclasses |

#### Training Scripts (`scripts/`)

| Script | Phase | Description |
|--------|-------|-------------|
| [train_phase0_sft.py](file:///f:/USTC/2026Winter/G-MSRA/scripts/train_phase0_sft.py) | 0 | SFT warmup with synthetic CRUD examples |
| [train_phase1_rl.py](file:///f:/USTC/2026Winter/G-MSRA/scripts/train_phase1_rl.py) | 1 | RL + external QA F1 reward via TRL PPO |
| [train_phase2_transition.py](file:///f:/USTC/2026Winter/G-MSRA/scripts/train_phase2_transition.py) | 2 | Curriculum annealing with Kendall τ monitoring |
| [train_phase3_full.py](file:///f:/USTC/2026Winter/G-MSRA/scripts/train_phase3_full.py) | 3 | Full closed-loop with adaptive consolidation |
| [eval_locomo.py](file:///f:/USTC/2026Winter/G-MSRA/scripts/eval_locomo.py) | Eval | LoCoMo/LongMemEval with category breakdown |
| [eval_agent_tasks.py](file:///f:/USTC/2026Winter/G-MSRA/scripts/eval_agent_tasks.py) | Eval | ALFWorld with FRR tracking |
| [run_ablations.py](file:///f:/USTC/2026Winter/G-MSRA/scripts/run_ablations.py) | Eval | All 7 ablations (A1-A7) with config mods |

#### Cluster Scripts (`cluster/`)
- [run_song.sh](file:///f:/USTC/2026Winter/G-MSRA/cluster/run_song.sh) — A100 nodes (all phases)
- [run_tang.sh](file:///f:/USTC/2026Winter/G-MSRA/cluster/run_tang.sh) — A40 nodes (SFT + eval)

---

### 📄 Paper Skeleton (ICLR format)

- [main.tex](file:///f:/USTC/2026Winter/G-MSRA/paper/main.tex) — Complete paper structure with:
  - All 6 sections + appendix
  - 4 equations (composite reward, confidence scoring, trigger function, MDP formulation)
  - 4 result table stubs (main results, agent results, ablations, hyperparameters)
  - 4 figure placeholders (efficiency, reward calibration, trigger visualization, growth)
  - TODO markers for experimental results
- [references.bib](file:///f:/USTC/2026Winter/G-MSRA/paper/references.bib) — 27 BibTeX entries covering all referenced papers

---

### 🔧 Module ↔ Paper Section Mapping

| Code Module | Paper Section |
|-------------|--------------|
| `reward/grounded_reward.py` | §3.2 Environment-Grounded Self-Rewarding |
| `manager/memory_manager.py` | §3.3 RL-Based Memory Manager |
| `consolidation/trigger.py` | §3.4 Adaptive Consolidation Trigger |
| `consolidation/distiller.py` | §3.5 Semantic Distillation Consolidation |
| `scripts/train_phase*.py` | §3.6 Four-Phase Curriculum Training |
| `scripts/eval_*.py` | §4 Experiments |
| `scripts/run_ablations.py` | §4.3 Ablation Studies |

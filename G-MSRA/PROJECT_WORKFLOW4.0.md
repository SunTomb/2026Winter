# G-MSRA 项目工作流程 v4.0 — 全量实验阶段指南

> **Closing the Loop: Environment-Grounded Self-Reward for Autonomous Memory Management in Lifelong LLM Agents**
>
> 更新日期：2026 年 3 月 17 日 · 基于 [v3.0](file:///f:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW3.0.md) 更新

---

## 〇、与前序版本的关系

| 版本 | 侧重 | 状态 |
|------|------|------|
| [v1.0](file:///f:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW.md) | 研究问题 + 排期规划 + 代码骨架 | ✅ 背景资料 |
| [v2.0](file:///f:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW2.0.md) | 训练脚本补全（Phase 0-3 + 消融 + 数据准备） | ✅ 背景资料 |
| [v3.0](file:///f:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW3.0.md) | Baseline 复现（5 个 Agent + 评测框架） | ✅ 背景资料 |
| **v4.0（本文档）** | **合成数据验证完成 → 全量数据实验执行指南** | ✅ 当前版本 |

---

## 一、已完成的全部工作

### 1.1 代码资产清单（40+ 个文件）

```
G-MSRA/
├── gmsra/                              # 核心库 (10 个文件, v1.0)
│   ├── config.py                       # 超参数 (dataclass + YAML)
│   ├── utils.py                        # F1/EM/τ/模型加载/文本生成
│   ├── agent.py                        # ⭐ 主编排器 (Online/Offline 闭环)
│   ├── memory/entry.py                 # MemoryEntry 数据类
│   ├── memory/store.py                 # ⭐ MemoryStore (FAISS + 图链接)
│   ├── reward/env_signals.py           # 3 种环境信号提取器
│   ├── reward/grounded_reward.py       # ⭐ 双层复合奖励
│   ├── manager/memory_manager.py       # RL CRUD 决策模块
│   ├── consolidation/trigger.py        # 3D 自适应触发器
│   └── consolidation/distiller.py      # ⭐ LoRA 语义蒸馏
│
├── scripts/                            # 训练/评测脚本 (10 个文件, v2.0)
│   ├── __init__.py
│   ├── prepare_data.py                 # 数据集下载/格式化/合成回退
│   ├── train_phase0_sft.py             # Phase 0: SFT (115 条数据)
│   ├── train_phase1_rl.py              # Phase 1: RL (GRPO/PPO/REINFORCE)
│   ├── train_phase2_transition.py      # Phase 2: 课程退火
│   ├── train_phase3_full.py            # Phase 3: 全闭环
│   ├── eval_locomo.py                  # LoCoMo/LongMemEval 评测
│   ├── eval_agent_tasks.py             # ALFWorld 评测
│   ├── run_ablations.py                # 消融实验 (A1-A7)
│   └── smoke_test.py                   # 冒烟测试 (27/27 ✅)
│
├── baselines/                          # Baseline 复现 (11 个文件, v3.0)
│   ├── __init__.py
│   ├── base_agent.py                   # 抽象接口
│   ├── reflexion_agent.py              # Baseline 1: Reflexion
│   ├── memory_r1_agent.py              # Baseline 2: Memory-R1
│   ├── self_consolidation_agent.py     # Baseline 3: Self-Consolidation
│   ├── evolver_agent.py                # Baseline 4: EvolveR
│   ├── mem0_memoryr1_agent.py          # Baseline 5: Mem0 + Memory-R1
│   ├── eval_baselines.py              # 统一评测框架
│   ├── train_and_eval_rl_baselines.py  # RL 基线训练管线
│   ├── run_rl_baselines.sh             # 过夜自动化脚本
│   └── README.md                       # 使用文档
│
├── cluster/                            # 集群脚本
│   ├── run_song.sh
│   └── run_tang.sh
│
├── paper/                              # 论文
│   ├── main.tex                        # ICLR 格式草稿
│   └── references.bib                  # 27 篇参考文献
│
├── results/baselines/                  # ⭐ 实验结果 (v4.0 新增)
│   ├── baseline_results.json           # 3 个无需训练 Baseline 的评测结果
│   ├── memory_r1_eval_results.json     # Memory-R1 训练后评测结果
│   ├── mem0_memory_r1_eval_results.json
│   ├── rl_baselines_combined.json      # RL 基线训练指标 + 评测
│   └── rl_baselines_*.log              # 训练日志
│
├── data/                               # 数据集 (当前为合成数据)
│   ├── locomo_train.json               # 44 train (合成, 16 KB)
│   ├── locomo_test.json                # 11 test (合成, 4 KB)
│   ├── longmemeval_test.json           # 11 test (合成)
│   ├── alfworld_tasks.json             # 200 tasks (合成, 109 KB)
│   └── evomemory_test.json             # 100 examples (合成, 64 KB)
│
├── setup_env.sh / setup.py / requirements.txt
└── README.md
```

### 1.2 合成数据 Baseline 验证结果

以下结果来自 Tang2 集群，使用 Qwen2.5-7B-Instruct，**在合成数据上**运行：

| Baseline | 类型 | LoCoMo F1 | EM | 训练时间 | 说明 |
|----------|:----:|:---------:|:--:|:--------:|------|
| **Reflexion** | 无训练 | 0.046 | 0.0 | ~3 min | 反思缓冲区机制验证 ✅ |
| **EvolveR** | 无训练 | 0.069 | 0.0 | ~4 min | 原则蒸馏机制验证 ✅ |
| **Self-Consolidation** | 无训练 | 0.130 | 0.0 | ~1.5 min | 启发式 CRUD + 对比反思 ✅ |
| **Memory-R1** | RL训练 | 0.014 | 0.0 | ~87 min (3 epoch) | RL CRUD + QA F1 奖励 ✅ |
| **Mem0 + Memory-R1** | RL训练 | 0.031 | 0.0 | ~88 min (3 epoch) | 多级记忆 + RL CRUD ✅ |

> **注意**：这些 F1 值较低是因为 **合成数据仅有 44 条训练 / 11 条测试**，且 LLM 生成的回答格式往往比 gold answer 冗长。这些数字仅验证了管线跑通——不代表最终论文结果。

### 1.3 里程碑总览

| 里程碑 | 状态 | 完成日期 |
|--------|:----:|:--------:|
| M0: 核心库完成 | ✅ | 2026-03-15 |
| M0.5: 训练脚本补全 + 冒烟测试 27/27 | ✅ | 2026-03-16 |
| M0.8: Baseline 代码完成 | ✅ | 2026-03-16 |
| **M0.9: 合成数据 Baseline 全通** | **✅** | **2026-03-17** |
| M1: 全量数据 Pipeline 跑通 | ☐ | — |
| M2: G-MSRA 全量训练完成 | ☐ | — |
| M3: 全量 Baseline 评测完成 | ☐ | — |
| M4: 主表 (Table 1+2) 填入 | ☐ | — |
| M5: 消融实验 (Table 3) 完成 | ☐ | — |
| M6: 分析图完成 | ☐ | — |
| M7-M9: 论文初稿 → 投递 | ☐ | — |

---

## 二、下一步：从合成数据到全量数据

### 2.1 总体路线图

```
当前位置                                                          论文投递
    │                                                                │
    ▼                                                                ▼
[合成数据验证 ✅] → [全量数据准备] → [G-MSRA 训练] → [Baseline 评测] → [消融] → [论文]
                    ~~~~~~~~~~      ~~~~~~~~~~~~~~    ~~~~~~~~~~~~~~
                     你需要做          最耗时           可并行
```

### 2.2 Step 1: 获取全量数据集

这是**最高优先级**的工作。当前 `data/` 下的合成数据太小（44 train / 11 test），无法产生有意义的 F1。

#### LoCoMo（核心 Benchmark）

```bash
# 方法 A: 从 GitHub 直接下载 (推荐)
cd /NAS/yesh/G-MSRA
git clone https://github.com/snap-research/locomo.git /tmp/locomo
cp /tmp/locomo/data/locomo10.json data/locomo_raw.json

# 然后用 prepare_data.py 的格式化逻辑转换：
python -c "
import json
with open('data/locomo_raw.json') as f:
    raw = json.load(f)
# 根据实际 LoCoMo 格式调整解析逻辑
print(f'Raw entries: {len(raw)}')
print(f'Sample keys: {list(raw[0].keys()) if isinstance(raw, list) else list(raw.keys())}')
"
```

```bash
# 方法 B: 从 HuggingFace 下载 (需要网络)
python -c "
from datasets import load_dataset
ds = load_dataset('locomo-bench/locomo', trust_remote_code=True)
print(ds)
"
```

```bash
# 方法 C: 如果以上均失败，扩充合成数据
# 修改 scripts/prepare_data.py 中场景数量 (当前 11 个场景 × 5 = 55)
# 增加到 100+ 个场景 × 10 = 1000+ 条数据
```

#### LongMemEval

```bash
# HuggingFace
python -c "
from datasets import load_dataset
ds = load_dataset('LongMemEval/LongMemEval', trust_remote_code=True)
print(ds)
"
```

#### ALFWorld & Evo-Memory

当前合成数据（200 tasks / 100 examples）已足够用于初步实验。如果需要真实数据：
- ALFWorld: `pip install alfworld` 后从环境中动态生成
- Evo-Memory: 查看 arXiv:2511.20857 是否有开源数据集

#### 数据格式要求（所有脚本均假设此格式）

```json
// locomo_train.json / locomo_test.json
[
  {
    "events": ["User says: I live in Beijing.", "User says: I work at Alibaba."],
    "question": "Where does the user live?",
    "answer": "Beijing",
    "category": "information_extraction"   // 可选
  },
  ...
]
```

> **⚠️ 重要**：替换数据后请确认格式兼容性：
> ```bash
> python -c "import json; d=json.load(open('data/locomo_train.json')); print(f'{len(d)} entries, keys={list(d[0].keys())}')"
> ```

### 2.3 Step 2: 全量 G-MSRA 训练

数据就绪后，按以下顺序执行 G-MSRA 主实验：

```bash
# 1. Phase 0: SFT 热启动 (Tang 节点即可, ~30 min)
CUDA_VISIBLE_DEVICES=6 python scripts/train_phase0_sft.py
# 预期输出: outputs/phase0/best/

# 2. Phase 1: RL + 外部奖励 (Song 节点, 2-3 天)
#    这是最耗时的步骤
bash cluster/run_song.sh phase1
# 预期输出: outputs/phase1/best/

# 3. Phase 2: 课程退火 (Song 节点, 1-2 天)
bash cluster/run_song.sh phase2
# 预期输出: outputs/phase2/best/ + calibration.json

# 4. Phase 3: 全闭环 (Song 节点, 3-5 天)
bash cluster/run_song.sh phase3
# 预期输出: outputs/phase3/best/ + metrics.json + diagnostics.json

# 5. 评测
CUDA_VISIBLE_DEVICES=6 python scripts/eval_locomo.py \
    --checkpoint outputs/phase3/best --benchmark locomo
```

### 2.4 Step 3: 全量 Baseline 评测

可以与 G-MSRA 训练**并行执行**（用不同 GPU）：

```bash
# 无需训练的 Baseline (直接跑, 每个 ~10 min)
CUDA_VISIBLE_DEVICES=7 python baselines/eval_baselines.py --agent reflexion --benchmark locomo
CUDA_VISIBLE_DEVICES=7 python baselines/eval_baselines.py --agent self_consolidation --benchmark locomo
CUDA_VISIBLE_DEVICES=7 python baselines/eval_baselines.py --agent evolver --benchmark locomo

# 需要训练的 Baseline (过夜跑)
CUDA_VISIBLE_DEVICES=7 nohup bash baselines/run_rl_baselines.sh > results/baselines/fulldata_overnight.log 2>&1 &
```

### 2.5 Step 4: 消融实验

在 G-MSRA Phase 1 checkpoint 可用后启动：

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/run_ablations.py \
    --base_checkpoint outputs/phase1/best \
    --num_episodes 1000
```

---

## 三、执行策略建议

### 3.1 GPU 并行策略

如果有多块 GPU 可用，建议如下分配：

| GPU | 任务 | 预估时间 |
|:---:|------|:--------:|
| GPU 0-3 | G-MSRA Phase 1 RL 训练 (4×A100) | 2-3 天 |
| GPU 6 | Baseline 评测 (逐个) | 数小时 |
| GPU 7 | RL Baseline 训练 (过夜) | 1 天 |

### 3.2 结果收集 → 论文填充

所有结果汇总到 `results/` 目录后，填入 `paper/main.tex`：

| 论文表格 | 数据来源 | main.tex 行号 |
|----------|---------|:------------:|
| Table 1 (主表) | `results/baselines/` + G-MSRA eval | L266-272 |
| Table 2 (ALFWorld) | ALFWorld 评测结果 | L287-292 |
| Table 3 (消融) | `results/ablations/` | L316-323 |
| Fig 3 (Reward Drift) | Phase 2 `calibration.json` | L349-355 |
| Fig 4 (Trigger 3D) | Phase 3 `diagnostics.json` | L361-366 |

### 3.3 预期论文结果趋势

基于方法设计，全量数据上的预期排序：

```
G-MSRA (Full) > Memory-R1 ≈ Mem0+R1 > Self-Consolidation > EvolveR > Reflexion
```

- **G-MSRA vs Memory-R1**：两者 RL 策略相同，但 G-MSRA 有自奖励 + 巩固 → 长期提升更大
- **Self-Consolidation 排第三**：有 LoRA 巩固但无 RL → 中等水平
- **EvolveR vs Reflexion**：均无权重更新，但 EvolveR 的抽象原则比 Reflexion 的具体反思泛化性更好

---

## 四、当前 F1 低的原因与改善方向

### 4.1 合成数据局限

| 因素 | 当前状态 | 全量数据预期 |
|------|---------|------------|
| 训练集大小 | 44 条 | 1000+ 条 |
| 测试集大小 | 11 条 | 100+ 条 |
| 场景多样性 | 11 个模板 × 5 | 真实多轮对话 |
| 答案格式 | 短答案 ("Beijing") | 自然语言 |

### 4.2 F1 计算特性

当前的 `compute_f1` 是 token-level F1。LLM 倾向于生成完整句子（如 "The user currently lives in Beijing"），而 gold answer 是简短的（如 "Beijing"）。这导致 precision 很低但 recall 可能不错。

**改善方向**：
1. 在 `answer_question` 的 prompt 中加强"be very concise, answer in as few words as possible"
2. 后处理：提取核心实体
3. 使用 Judge 评分作为补充指标

### 4.3 RL Baseline 训练不充分

Memory-R1 和 Mem0+R1 仅训练了 3 个 epoch × 44 条 = 132 步。原论文 Memory-R1 使用 152 条 QA pairs 但训练 5000+ steps。全量数据后建议增加 `--train_epochs 10`。

---

## 五、项目文件完整索引

### 核心库 (`gmsra/`)

| 文件 | 类/函数 | 作用 |
|------|--------|------|
| `config.py` | `GMSRAConfig` | 7 个子配置，支持 YAML |
| `utils.py` | `compute_f1`, `load_model_and_tokenizer`, `generate_text` | 通用工具 |
| `agent.py` | `GMSRAAgent.step()` | 主编排：CRUD → 奖励 → 触发 → 巩固 |
| `memory/entry.py` | `MemoryEntry` | 记忆卡片 + 置信度 |
| `memory/store.py` | `MemoryStore` | FAISS 检索 + 图链接 + 子图提取 |
| `reward/env_signals.py` | `AgentTask/Dialogue/ExternalQA SignalExtractor` | R_env 三种来源 |
| `reward/grounded_reward.py` | `GroundedRewardGenerator` | R_total = R_env + λ·R_mem |
| `manager/memory_manager.py` | `MemoryManager.decide()` | RL CRUD prompt + 解析 |
| `consolidation/trigger.py` | `ConsolidationTrigger` | 3D: Conflict + Variance + Growth |
| `consolidation/distiller.py` | `SemanticDistiller` | 子图 → 三元组 → LoRA SFT + EWC |

### 训练脚本 (`scripts/`)

| 文件 | 作用 | 关键参数 |
|------|------|---------|
| `prepare_data.py` | 下载/格式化 4 个数据集 | `--output_dir data` |
| `train_phase0_sft.py` | SFT 热启动 (115 条) | `--output_dir outputs/phase0` |
| `train_phase1_rl.py` | RL 训练 (GRPO→PPO→REINFORCE) | `--num_episodes 5000` |
| `train_phase2_transition.py` | 课程退火 + Kendall τ | `--anneal_steps 3000` |
| `train_phase3_full.py` | 全闭环 + 巩固 | `--num_episodes 10000` |
| `eval_locomo.py` | 对话记忆评测 | `--checkpoint outputs/...` |
| `eval_agent_tasks.py` | Agent 任务评测 | `--env alfworld` |
| `run_ablations.py` | A1-A7 消融 | `--base_checkpoint ...` |
| `smoke_test.py` | 管线验证 (27/27 ✅) | 直接运行 |

### Baseline (`baselines/`)

| 文件 | Baseline | 关键特性 |
|------|---------|---------|
| `reflexion_agent.py` | Reflexion | 反思缓冲区，无权重更新 |
| `memory_r1_agent.py` | Memory-R1 | RL CRUD + QA F1，复用 MemoryStore |
| `self_consolidation_agent.py` | Self-Consolidation | 对比反思 + 固定 LoRA |
| `evolver_agent.py` | EvolveR | 轨迹分析 → 抽象原则 |
| `mem0_memoryr1_agent.py` | Mem0 + Memory-R1 | 三级记忆 + RL CRUD |
| `eval_baselines.py` | 统一评测 | `--agent X --benchmark Y` |
| `train_and_eval_rl_baselines.py` | RL 训练管线 | `--train_epochs N --lr X` |
| `run_rl_baselines.sh` | 过夜自动化 | `nohup bash ... &` |

---

## 六、风险与应对

| 风险 | 概率 | 应对 |
|------|:----:|------|
| ~~代码骨架不完整~~ | — | ✅ 已解决 |
| ~~Baseline 缺失~~ | — | ✅ 已解决 |
| ~~管线跑不通~~ | — | ✅ 合成数据验证通过 |
| 全量数据下载失败 | 中 | 扩充合成数据到 1000+；或手动从论文仓库下载 |
| Phase 1 RL 训练发散 | 中 | 先用小 lr (5e-6)，batch_size 从 8 开始 |
| 全量 F1 仍然低 | 中 | 检查 prompt 格式、答案后处理、增加 Judge 指标 |
| 训练太慢 | 中 | 用 QLoRA 4-bit 减少显存；或减少 episodes |
| Self-Consolidation 复现不准 | 低 | 论文中标注 "our reproduction" |

---

> **最终提醒**：项目的技术基建已经 100% 完成。现在唯一剩下的就是：
> 1. **拿到全量数据** → 换掉 `data/` 目录下的合成数据
> 2. **在集群上跑**  → G-MSRA 四阶段训练 + Baseline 评测 + 消融实验
> 3. **填入论文**    → 把数字填入 `paper/main.tex` 的占位符
>
> 做完这三步，论文的实验部分就齐了。加油！

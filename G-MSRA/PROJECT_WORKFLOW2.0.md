# G-MSRA 项目工作流程 v2.0 — 代码补全后更新

> **Closing the Loop: Environment-Grounded Self-Reward for Autonomous Memory Management in Lifelong LLM Agents**
>
> 更新日期：2026 年 3 月 16 日 · 基于 [PROJECT_WORKFLOW.md](file:///f:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW.md) v1.0 更新

---

## 〇、本文档与 v1.0 的关系

v1.0（`PROJECT_WORKFLOW.md`）定义了项目的**研究问题、排期规划和技术蓝图**，仍然有效。

本文档（v2.0）侧重于：

1. **截至 2026-03-16 的代码补全进度**——哪些 v1.0 标记为 ⚠️ 的骨架已被真正实现
2. **每个文件的详细代码说明**——比 v1.0 更细粒度的模块解读
3. **接续完成任务的人应当怎么做**——从代码到集群，step by step

> **一条原则不变**：所有实验在 USTC LDS 实验室集群上完成。Song 节点（A100 80G）用于 RL 训练，Tang 节点（A40 45G）用于 SFT 和评测。

---

## 一、当前进度总览

### 1.1 完成度对比（v1.0 → v2.0）

| 模块 | v1.0 状态 | v2.0 状态 | 变更说明 |
|------|:---------:|:---------:|---------|
| `gmsra/config.py` | ✅ 可用 | ✅ 未变 | — |
| `gmsra/utils.py` | ✅ 可用 | ✅ 未变 | — |
| `gmsra/memory/entry.py` | ✅ 可用 | ✅ 未变 | — |
| `gmsra/memory/store.py` | ✅ 可用 | ✅ 未变 | — |
| `gmsra/reward/env_signals.py` | ✅ 可用 | ✅ 未变 | — |
| `gmsra/reward/grounded_reward.py` | ✅ 可用 | ✅ 未变 | — |
| `gmsra/manager/memory_manager.py` | ✅ 可用 | ✅ 未变 | — |
| `gmsra/consolidation/trigger.py` | ✅ 可用 | ✅ 未变 | — |
| `gmsra/consolidation/distiller.py` | ✅ 可用 | ✅ 未变 | — |
| `gmsra/agent.py` | ✅ 可用 | ✅ 未变 | — |
| `scripts/train_phase0_sft.py` | ⚠️ 骨架 | ✅ **已完成** | 8→115 条多样化 SFT 数据，加入 cosine LR scheduler |
| `scripts/train_phase1_rl.py` | ⚠️ 骨架 | ✅ **已完成** | 3 层 RL 集成：GRPOTrainer → PPOTrainer → REINFORCE |
| `scripts/train_phase2_transition.py` | ⚠️ 骨架 | ✅ **已完成** | 退火过程中加入 REINFORCE 策略更新 |
| `scripts/train_phase3_full.py` | ⚠️ 骨架 | ✅ **已完成** | 闭环训练中加入 REINFORCE 策略更新 |
| `scripts/eval_locomo.py` | ⚠️ 骨架 | ✅ 未变 | 评测循环本身已完整，等待真实数据 |
| `scripts/eval_agent_tasks.py` | ⚠️ 骨架 | ✅ 未变 | 同上 |
| `scripts/run_ablations.py` | ⚠️ 骨架 | ✅ **已完成** | 连通训练管线 + 评估 + Table 3 汇总 |
| `scripts/prepare_data.py` | ❌ 不存在 | ✅ **新增** | 4 个数据集的下载/格式化脚本 |
| `scripts/smoke_test.py` | ❌ 不存在 | ✅ **新增** | 端到端冒烟测试（27/27 通过） |
| `scripts/__init__.py` | ❌ 不存在 | ✅ **新增** | 跨脚本导入支持 |
| `paper/main.tex` | ⚠️ 骨架 | ⚠️ 未变 | 等待实验数据填入 |
| `cluster/*.sh` | ✅ 可用 | ✅ 未变 | — |

### 1.2 一句话总结

> **所有训练脚本已从"骨架"升级为"可运行"。** 核心库 `gmsra/` 不变（本身已完整）。下一步是在集群上安装环境、准备真实数据、逐 Phase 运行训练。

---

## 二、代码架构——每个文件做什么

### 2.1 整体架构图（更新版）

```
G-MSRA/
├── gmsra/                        # 核心库（全部可运行）
│   ├── config.py                 # 超参数集中管理（dataclass + YAML 加载）
│   ├── utils.py                  # 工具函数：模型加载(QLoRA)、F1/EM/τ、文本生成
│   ├── agent.py                  # ⭐ 主编排器：Online/Offline 闭环
│   ├── memory/
│   │   ├── entry.py              # MemoryEntry：内容+关键词+标签+图链接+置信度
│   │   └── store.py              # ⭐ MemoryStore：FAISS检索 + 图链接 + 子图提取
│   ├── reward/
│   │   ├── env_signals.py        # 3种环境信号提取器
│   │   └── grounded_reward.py    # ⭐ 双层复合奖励 + Judge + 退火
│   ├── manager/
│   │   └── memory_manager.py     # RL 记忆管理器：CRUD 决策
│   └── consolidation/
│       ├── trigger.py            # 3D 触发器：Conflict + Variance + Growth
│       └── distiller.py          # ⭐ LoRA 蒸馏：子图 → 三元组 → SFT + EWC
│
├── scripts/                      # 训练/评测脚本（v2.0 全部可运行）
│   ├── __init__.py               # [NEW] 跨脚本导入
│   ├── prepare_data.py           # [NEW] 数据集下载与格式化
│   ├── train_phase0_sft.py       # [UPDATED] SFT 热启动（115条多样化数据）
│   ├── train_phase1_rl.py        # [REWRITTEN] RL训练（GRPO/PPO/REINFORCE）
│   ├── train_phase2_transition.py# [UPDATED] 课程退火 + 策略更新
│   ├── train_phase3_full.py      # [UPDATED] 全闭环 + 策略更新
│   ├── eval_locomo.py            # 对话记忆评测
│   ├── eval_agent_tasks.py       # Agent任务评测
│   ├── run_ablations.py          # [REWRITTEN] 消融实验（连通训练管线）
│   └── smoke_test.py             # [NEW] 端到端冒烟测试
│
├── cluster/                      # 集群脚本
│   ├── run_song.sh               # Song (A100 80G)
│   └── run_tang.sh               # Tang (A40 45G)
│
├── paper/                        # 论文
│   ├── main.tex                  # ICLR 格式论文骨架
│   └── references.bib            # 参考文献
│
├── setup_env.sh                  # 环境安装
├── setup.py                      # 可编辑安装
├── requirements.txt              # 依赖
└── README.md                     # 文档
```

### 2.2 核心库详解 (`gmsra/`)

#### `config.py` — 超参数中心

定义了 7 个子配置 dataclass，通过 `GMSRAConfig` 聚合：

| 配置类 | 关键参数 | 说明 |
|--------|----------|------|
| `ModelConfig` | `model_name`, `max_length` | 基座模型选择 |
| `MemoryConfig` | `embed_dim`, `max_entries`, `confidence_topk` | 记忆库容量与检索 |
| `RewardConfig` | `lambda_mem`, `anneal_steps`, `tau_threshold` | 奖励函数超参 |
| `RLConfig` | `learning_rate`, `batch_size`, `ppo_epochs` | RL 训练超参 |
| `LoRAConfig` | `r`, `alpha`, `target_modules` | LoRA 蒸馏参数 |
| `TriggerConfig` | `alpha`, `beta`, `gamma`, `theta` | 3D 触发器权重 |
| `GMSRAConfig` | 聚合以上所有 + `seed` | 支持 `from_yaml()` 加载 |

#### `utils.py` — 5 个工具函数

| 函数 | 用途 | 在哪用 |
|------|------|--------|
| `set_seed(seed)` | 设置随机种子 | 所有脚本入口 |
| `compute_f1(pred, gold)` | Token-level F1 | Phase 1 RL 奖励 |
| `compute_exact_match(pred, gold)` | 精确匹配 | 评测 |
| `compute_kendall_tau(x, y)` | 排序相关性 | Phase 2 退火监控 |
| `load_model_and_tokenizer(name)` | 加载模型（支持 QLoRA 4-bit） | 所有训练/评测脚本 |
| `generate_text(model, tokenizer, prompt)` | 文本生成 | Judge 评分 & 三元组生成 |

#### `agent.py` — 主编排器（最重要的类）

`GMSRAAgent` 是连接所有模块的枢纽。核心方法：

```python
class GMSRAAgent:
    def initialize(model, tokenizer, env_type):
        # 初始化 MemoryStore, MemoryManager, RewardGenerator,
        # ConsolidationTrigger, SemanticDistiller, EnvironmentSignalExtractor

    def step(event, task_context, agent_response, env_signal_kwargs):
        # 1. MemoryManager.decide() → 决定 ADD/UPDATE/DELETE/NOOP
        # 2. MemoryManager.execute_operation() → 修改 MemoryStore
        # 3. RewardGenerator.compute_reward() → R_env + λ·R_mem
        # 4. MemoryEntry.update_confidence() → 更新置信度
        # 5. ConsolidationTrigger.should_trigger() → 是否巩固
        # 6. (如果触发) SemanticDistiller.consolidate() → LoRA 蒸馏
        # 返回 {"operation": ..., "reward": ..., "consolidation": ...}

    def answer_question(question):
        # 用记忆增强回答问题（检索相关记忆 + 生成）

    def save_checkpoint(path) / load_checkpoint(path):
        # 保存/加载 MemoryStore + LoRA 权重 + 统计数据
```

**调用关系图**：

```
event (用户对话 / 环境观测)
         │
         ▼
┌─ MemoryManager.decide(event) ──→ "ADD: user prefers coffee"
│       │
│       ▼
│  MemoryManager.execute_operation() ──→ MemoryStore.add()
│       │
│       ▼
│  GroundedRewardGenerator.compute_reward()
│       ├── R_env ← EnvironmentSignalExtractor.extract()       ← 外部世界信号
│       └── R_mem ← LLM-as-Judge + MemoryStore.retrieve()     ← 内部一致性评估
│       │
│       ▼  R_total = R_env + λ·R_mem
│       │
│       ▼
│  ConsolidationTrigger.should_trigger()
│       │ (如果三维信号超过阈值)
│       ▼
│  SemanticDistiller.consolidate()
│       ├── MemoryStore.extract_high_frequency_subgraph()
│       ├── 生成语义三元组 (用 LLM)
│       ├── 训练 LoRA (SFT + EWC 正则化)
│       └── 清理已蒸馏的低置信记忆
│
└── 循环 → 下一个事件
```

#### `memory/entry.py` — 记忆条目

每张记忆卡片包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | UUID 自动生成 |
| `content` | str | 记忆内容文本 |
| `keywords` | list[str] | 关键词（用于检索增强） |
| `tags` | list[str] | 标签分类（fact / preference / procedure） |
| `links` | list[str] | 指向关联记忆的 ID（Zettelkasten 风格） |
| `confidence` | float | 置信度分数 [0, 1]，受奖励和命中率影响 |
| `hit_total / hit_success` | int | 被检索次数 / 检索成功次数 |

置信度更新公式：`confidence = sigmoid(w₁·R_env_write + w₂·hit_rate + w₃·log_age)`

#### `memory/store.py` — 记忆数据库

| 方法 | 功能 |
|------|------|
| `add(entry)` | 添加记忆，自动建立 FAISS 索引 |
| `update(entry_id, new_content)` | 更新内容 + 重建向量索引 |
| `delete(entry_id)` | 删除记忆 + 清理图链接 |
| `retrieve(query, top_k)` | FAISS 向量检索 |
| `retrieve_confident(query, top_k, min_confidence)` | 仅返回高置信度记忆 |
| `add_link(id_a, id_b)` | 建立双向图链接 |
| `extract_high_frequency_subgraph(min_degree)` | 提取高度连接的知识子图 |
| `save(path) / load(path)` | JSON + FAISS 持久化 |

#### `reward/grounded_reward.py` — 核心创新

双层复合奖励 $R_{\text{total}} = R_{\text{env}} + \lambda \cdot R_{\text{mem}}$

- **R_env**：来自 `EnvironmentSignalExtractor`（外部世界的真实反馈）
- **R_mem**：来自 LLM-as-Judge + 高置信记忆检索（内部一致性评估）
- `lambda`：权重系数，默认 < 1，确保环境信号占主导地位
- 支持退火：Phase 2 中 α 从 1.0 线性退至 0.0，混合 R_ext 和 R_total
- 支持 Drift 监控：追踪 R_total 的滑动均值变化

#### `consolidation/trigger.py` — 三维触发器

三维信号函数：`S(t) = α·Conflict(t) + β·Variance(t) + γ·Growth(t)`

| 维度 | 含义 | 计算方式 |
|------|------|---------|
| Conflict | 记忆冲突程度 | 近期 UPDATE/DELETE 操作占比 |
| Variance | 奖励方差 | 近期 R_total 的标准差 |
| Growth | 记忆增长率 | 近期 ADD 操作占比 |

当 `S(t) > θ` 且距上次巩固间隔 > `min_interval` 时触发。

#### `consolidation/distiller.py` — LoRA 蒸馏

1. **提取子图**：从 MemoryStore 的图结构中提取高频连接的知识子图
2. **生成三元组**：用 LLM 将记忆卡片转化为 `(subject, relation, object)` 语义三元组
3. **LoRA SFT**：用三元组做监督微调，将知识蒸馏进 LoRA 权重
4. **EWC 正则化**：用 Fisher 信息矩阵保护已学知识不被遗忘
5. **双 LoRA 架构**：长期知识 LoRA（EWC 保护）+ 短期知识 LoRA（可自由更新）

---

### 2.3 训练脚本详解 (`scripts/`)

#### `prepare_data.py` — 数据准备（v2.0 新增）

| 数据集 | 下载源 | 输出文件 | 合成数据规模 |
|--------|--------|---------|-------------|
| LoCoMo | HuggingFace `locomo-bench/locomo` | `data/locomo_train.json`, `data/locomo_test.json` | 44 train + 11 test |
| LongMemEval | HuggingFace `LongMemEval/LongMemEval` | `data/longmemeval_test.json` | 复用 LoCoMo 格式 |
| ALFWorld | 合成生成 | `data/alfworld_tasks.json` | 200 tasks |
| Evo-Memory | 合成生成 | `data/evomemory_test.json` | 100 examples |

> **注意**：首次运行会尝试从 HuggingFace 下载真实数据。如果下载失败（网络/权限），会自动退回到合成数据（用于开发和调试）。在集群上运行时，请确保配置了代理（`run_song.sh` 已包含代理设置）。

#### `train_phase0_sft.py` — Phase 0: SFT 热启动

**v2.0 核心改进**：

- 训练数据从 8 条重复 × 10 = 80 条 → **115 条唯一、多样化的** prompt-completion 对
- 覆盖场景：
  - 中英文混合（"用户来自合肥" / "User lives in NYC"）
  - 多轮上下文（3-5 条历史记忆）
  - 实体消歧（"User's colleague Alice" vs "User's friend Alice"）
  - 时间推理（"上个月搬到深圳"）
  - 边缘情况（空输入、重复信息、无关闲聊）
- 加入 `CosineAnnealingLR` 学习率调度
- 每 epoch 重新 shuffle 训练数据

#### `train_phase1_rl.py` — Phase 1: RL + 外部奖励（v2.0 重写）

**v1.0 问题**：创建了 `PPOConfig` 但从未调用 `ppo_trainer.step()`，RL 训练是空循环。

**v2.0 实现**：三层降级策略，确保在任何 TRL 版本下都能运行：

```
优先级 1: GRPOTrainer（推荐，无需 Value Head）
    ├── 使用 reward_funcs 回调函数
    ├── HuggingFace Dataset 格式
    └── 自动多 GPU 分布式

优先级 2: PPOTrainer（如果 GRPO 不可用）
    ├── 需要 AutoModelForCausalLMWithValueHead
    ├── 手动 tokenize query → generate response → compute reward
    └── 调用 ppo_trainer.step(queries, responses, rewards)

优先级 3: REINFORCE（本地无 TRL 时的后备）
    ├── 手动前向传播 + reward * log_prob 梯度
    ├── Running baseline 方差缩减
    └── 支持 LoRA 微调
```

**奖励函数设计** (`compute_rl_reward`)：

| 奖励成分 | 权重 | 说明 |
|----------|------|------|
| 格式奖励 | +0.2 / -0.3 | 输出是否为合法的 ADD/UPDATE/DELETE/NOOP 格式 |
| 内容质量 | ±0.1 | ADD/UPDATE 的内容是否非空且有意义（>10 字符） |
| QA F1 | ×0.7 | 通过 Agent 回答问题的 F1 分数 |

#### `train_phase2_transition.py` — Phase 2: 课程退火

**v1.0 问题**：只计算了退火奖励，但从未更新模型权重。

**v2.0 改进**：

- 加入 REINFORCE 策略更新：每步用 `advantage = R_annealed - baseline` 更新策略
- Running baseline 使用指数移动平均（`0.95 * baseline + 0.05 * reward`）
- `CosineAnnealingLR` 学习率调度（LR = Phase 1 的 50%）
- 每 500 步保存 checkpoint
- 输出 `calibration.json` 包含完整的 `(R_ext, R_self, R_annealed)` 序列，用于论文 §5.1 图

#### `train_phase3_full.py` — Phase 3: 全闭环

**v1.0 问题**：同 Phase 2，只跑了 Agent 循环但未更新权重。

**v2.0 改进**：

- 加入 REINFORCE 策略更新（LR = Phase 1 的 30%，更保守）
- 优势过滤：`|advantage| > 0.01` 才更新，避免噪声梯度
- 支持 `--env_type dialogue|agent_task` 双模式
- 输出 `metrics.json`（每 50 步记录成功率、记忆大小、巩固次数）
- 输出 `diagnostics.json`（完整诊断信息，用于论文分析图）

#### `run_ablations.py` — 消融实验（v2.0 重写）

**v1.0 问题**：7 组配置已定义，但 `run_ablation()` 里只有 `# TODO` 注释。

**v2.0 实现**：

- 每个消融 = **修改配置 → 训练（缩短版 Phase 3）→ 评估 → 保存结果**
- A1：`agent.env_extractor = None`（禁用环境信号）
- A5：`monkey-patch` distiller 使用随机采样替代子图提取
- A6：不启用 `consolidation_enabled`
- A7：`anneal_start_alpha = 0.0`（跳过课程退火）
- 按优先级排序执行（A1 > A2 = A6 > A3 = A4 = A7 > A5）
- 输出 Table 3 汇总表

#### `smoke_test.py` — 冒烟测试（v2.0 新增）

27 项测试覆盖：

| 测试组 | 通过数 | 覆盖内容 |
|--------|:------:|---------|
| 模块导入 | 10/10 | 所有 `gmsra.*` 子模块 |
| 配置系统 | 2/2 | 默认值验证 + YAML 加载 |
| 记忆系统 | 4/4 | 创建、置信度、序列化、文本表示 |
| 工具函数 | 2/2 | F1/EM + Kendall τ |
| 奖励系统 | 5/5 | 3 种提取器 × 多种输入 |
| SFT 数据 | 1/1 | 115 条数据、4 种操作类型 |
| 数据准备 | 3/3 | LoCoMo + ALFWorld + Evo-Memory |

> 运行方式：`cd G-MSRA && python scripts/smoke_test.py`

---

## 三、接续完成任务的人应当怎么做

### 3.1 你的第一天——环境搭建 + 冒烟测试

```bash
# 1. SSH 到集群
ssh song2  # 或 tang2 / tang3 / sui3

# 2. 克隆代码（如果还没有的话）
cd /NAS/<your_username>/
git clone <repo_url> G-MSRA
cd G-MSRA

# 3. 安装环境
bash setup_env.sh
# 如果 setup_env.sh 报错，手动执行：
# conda create -n gmsra python=3.11 -y
# conda activate gmsra
# pip install -r requirements.txt
# pip install -e .

# 4. 冒烟测试
conda activate gmsra
python scripts/smoke_test.py
# 应该看到 27/27 passed

# 5. 准备数据集
python scripts/prepare_data.py --output_dir data
# 首次会尝试从 HuggingFace 下载。
# 如果成功，data/ 下会有真实数据；
# 如果失败，会生成合成数据用于流程调试。
```

### 3.2 Phase 0: SFT 热启动

```bash
# Tang 节点即可（A40 45G，QLoRA 模式）
bash cluster/run_tang.sh phase0

# 或者在 Song 节点（更快，不需要 QLoRA）
bash cluster/run_song.sh phase0

# 预期输出：
# outputs/phase0/best/  ← LoRA 权重
# 训练日志中 avg_loss 应逐 epoch 下降
```

**验证**：训练完成后，检查 `outputs/phase0/best/` 是否包含 `adapter_model.safetensors`。

### 3.3 Phase 1: RL + 外部奖励（最关键步骤）

```bash
# 需要 Song 节点（A100 80G × 4）
bash cluster/run_song.sh phase1

# 关键参数（可调）：
#   --num_episodes 5000    ← 训练集大小
#   --batch_size 16        ← 根据 GPU 内存调整
#   --learning_rate 1.41e-5
```

**注意事项**：

1. **TRL 版本兼容性**：脚本会自动检测 TRL 版本并选择合适的 Trainer：
   - TRL ≥ 0.12：使用 `GRPOTrainer`（推荐）
   - TRL ≥ 0.8：使用 `PPOTrainer`
   - 无 TRL：使用手动 `REINFORCE`
   
2. **如果 GRPOTrainer 报错**：可能是 TRL API 变化。解决方案：
   ```bash
   pip install trl==0.12.0  # 固定版本
   ```
   或者修改 `_train_with_grpo()` 中的 `GRPOConfig` 参数以匹配当前版本。

3. **如果内存不足**：降低 `batch_size` 到 8 或 4，增加 `gradient_accumulation_steps`。

4. **如何判断训练是否正常**：
   - 日志中 `avg_reward` 应逐渐上升
   - `mem_size` 应稳定增长（不会无限增长）
   - 格式错误率应下降（NOOP/ADD 的格式越来越标准）

### 3.4 Phase 2: 课程退火

```bash
bash cluster/run_song.sh phase2

# 关键参数：
#   --anneal_steps 3000    ← 退火总步数
#   --tau_threshold 0.5    ← Kendall τ 阈值（低于则暂停退火）
```

**监控指标**：

| 指标 | 正常范围 | 异常信号 |
|------|---------|---------|
| α | 1.0 → 0.0 线性下降 | 长时间 paused（τ 低于阈值） |
| Kendall τ | ≥ 0.5 | 持续 < 0.3 → 降低 τ 阈值到 0.3 |
| R_ext vs R_self | 趋势相似 | 完全不相关 → 需重新检查 Phase 1 |

如果退火始终暂停：

```python
# 在 config 中修改：
config.reward.tau_threshold = 0.3  # 降低阈值
# 或者：
config.reward.anneal_end_alpha = 0.2  # 不完全退火
```

### 3.5 Phase 3: 全闭环

```bash
bash cluster/run_song.sh phase3
# 预计 3-5 天（10000 episodes）

# 监控：
tail -f outputs/phase3/metrics.json
# 看 success_rate 是否稳定上升
# 看 consolidation_count 是否在增长
```

### 3.6 评测

```bash
bash cluster/run_song.sh eval
# 自动运行 eval_locomo.py + eval_agent_tasks.py

# 手动运行单个评测：
python scripts/eval_locomo.py --checkpoint outputs/phase3/best --benchmark locomo
python scripts/eval_locomo.py --checkpoint outputs/phase3/best --benchmark longmemeval
python scripts/eval_agent_tasks.py --checkpoint outputs/phase3/best --env alfworld
```

### 3.7 消融实验

```bash
# 运行全部 7 组消融（按优先级排序）：
python scripts/run_ablations.py --base_checkpoint outputs/phase1/best --num_episodes 1000

# 只运行特定消融：
python scripts/run_ablations.py --ablations A1_no_env_anchor,A6_no_consolidation
```

---

## 四、仍需手工完成的工作

以下工作不能完全自动化，需要接手人根据实际情况完成：

### 4.1 ⭐ ALFWorld 真实环境对接（可选但加分）

当前 `eval_agent_tasks.py` 使用模拟数据。如果要对接真实 ALFWorld：

```bash
pip install alfworld
```

然后修改 `gmsra/reward/env_signals.py` 的 `AgentTaskSignalExtractor`，使其调用 ALFWorld 的 API：

```python
# 需要新增的包装器
import alfworld.agents.environment as env

class ALFWorldWrapper:
    def __init__(self):
        self.env = env.AlfredTWEnv(config={"env_type": "TWEnv"})
    
    def step(self, action: str) -> dict:
        obs, reward, done, info = self.env.step(action)
        return {
            "success": done and reward > 0,
            "partial_score": reward,
            "observation": obs,
        }
```

> **风险评估**：ALFWorld 的 gym API 可能与当前 Python 版本不完全兼容。如果时间紧张，建议先用 LoCoMo + LongMemEval + Evo-Memory 出主表，ALFWorld 放到附录。

### 4.2 Baseline 复现

需要运行的 Baseline 及其来源：

| Baseline | 来源 | 复现难度 |
|----------|------|---------|
| Memory-R1 | 官方代码（如开源） | 低 |
| ReadAgent | 论文描述 | 中 |
| Reflexion | 开源代码 | 低 |
| EvolveR | 开源代码 | 低 |
| Self-Consolidation | 论文描述 | 高 |
| Mem0 / LightMem | 开源框架 | 低 |

如果某个 baseline 无法复现，备选方案是引用原论文数字（需注明评测条件差异）。

### 4.3 论文撰写

`paper/main.tex` 已有完整骨架。填入实验数据后需要：

1. 补充 Table 1（主表）、Table 2（ALFWorld）、Table 3（消融）的实际数字
2. 替换 `--` 占位符为实际值
3. 嵌入分析图（Reward Drift 曲线、Kendall τ 演化等）
4. 撰写 Key Observations 段落

---

## 五、里程碑检查清单（更新版）

| 里程碑 | 目标周 | 判定标准 | 状态 |
|--------|:------:|---------|:----:|
| M0: 代码补全 | W1 | 所有训练脚本从骨架升级为可运行 | ✅ 已完成 |
| M0.5: 冒烟测试 | W1 | `smoke_test.py` 27/27 通过 | ✅ 已完成 |
| M1: Pipeline 跑通 | W3 | 集群上 Phase 0→1→2→3→eval 无 crash | ☐ |
| M2: Phase 1 收敛 | W6 | LoCoMo F1 ≥ Memory-R1 baseline | ☐ |
| M3: Phase 2 退火成功 | W8 | α 退至 0，Kendall τ ≥ 0.5 | ☐ |
| M4: 主表完成 | W12 | Table 1 + Table 2 全部数字填入 | ☐ |
| M5: 消融完成 | W15 | Table 3 全部数字 + A1 证实 Reward Hacking | ☐ |
| M6: 分析图完成 | W18 | 6 张核心图 + Case Study | ☐ |
| M7: 初稿完成 | W22 | 完整论文可读 | ☐ |
| M8: 内审通过 | W24 | 导师 + 同学审稿意见回收 + 修改完毕 | ☐ |
| M9: 最终投递 | W28 | OpenReview 提交成功 | ☐ |

---

## 六、风险与应对（更新版）

| 风险 | 概率 | 影响 | 应对方案 |
|------|:----:|:----:|---------|
| ~~TRL API 对接复杂度超预期~~ | ~~中~~ | ~~高~~ | ✅ **已解决**：3 层降级策略（GRPO→PPO→REINFORCE） |
| TRL 版本不兼容 | 中 | 🟡 中 | 固定 `trl==0.12.0`，或修改 Config 参数匹配新版 API |
| Phase 2 退火过程中 τ 始终低于阈值 | 中 | 🟡 中 | 降低 τ 阈值到 0.3；不完全退火（α 终止于 0.2） |
| ALFWorld 环境对接工程量大 | 中 | 🟡 中 | 先用 3 个文本 benchmark 出主表，ALFWorld 放附录 |
| HuggingFace 数据集下载失败 | 中 | 🟢 低 | `prepare_data.py` 已内建合成数据回退 |
| Baseline 复现困难 | 低 | 🟡 中 | 引用原论文数字 + 注明条件差异 |
| A100 排队等待时间过长 | 中 | 🟡 中 | Phase 0/eval 用 Tang 节点；预约系统避开高峰 |

---

> **最后提醒**：这个项目的学术"杀手锏"是 **消融 A1**（移除环境锚后的 Reward Hacking 现象）。请在实验设计中给 A1 最高优先级和最完整的数据记录。同时，`calibration.json`（Phase 2 输出）是画 §5.1 Reward Drift 曲线的核心数据来源，务必完整保存。

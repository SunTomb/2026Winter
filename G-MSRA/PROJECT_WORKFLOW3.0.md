# G-MSRA 项目工作流程 v3.0 — Baseline 复现完成后更新

> **Closing the Loop: Environment-Grounded Self-Reward for Autonomous Memory Management in Lifelong LLM Agents**
>
> 更新日期：2026 年 3 月 16 日 · 基于 [PROJECT_WORKFLOW2.0.md](file:///f:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW2.0.md) 更新

---

## 〇、与前序版本的关系

| 版本 | 覆盖范围 | 状态 |
|------|---------|------|
| [v1.0](file:///f:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW.md) | 研究问题、代码骨架、排期规划 | ✅ 仍然有效 |
| [v2.0](file:///f:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW2.0.md) | 训练脚本补全（Phase 0-3 + 消融 + 数据准备） | ✅ 仍然有效 |
| **v3.0（本文档）** | **Baseline 复现**（5 个对照方法 + 统一评测框架） | ✅ 当前版本 |

本文档侧重于：
1. 新增 `baselines/` 目录的代码结构与每个 Baseline 的实现逻辑
2. 各 Baseline 与 G-MSRA 的 feature 差异对照
3. 接续完成任务的人如何运行 Baseline 评测和获取论文 Table 1/2 数据

---

## 一、当前进度总览

### 1.1 累计完成度（v1.0 → v2.0 → v3.0）

| 模块 | v1.0 | v2.0 | v3.0 | 说明 |
|------|:----:|:----:|:----:|------|
| `gmsra/` 核心库（10 个文件） | ✅ | ✅ | ✅ | 未变 |
| `scripts/train_phase0_sft.py` | ⚠️ | ✅ | ✅ | 115 条 SFT 数据 |
| `scripts/train_phase1_rl.py` | ⚠️ | ✅ | ✅ | GRPO/PPO/REINFORCE 三层 |
| `scripts/train_phase2_transition.py` | ⚠️ | ✅ | ✅ | 退火 + 策略更新 |
| `scripts/train_phase3_full.py` | ⚠️ | ✅ | ✅ | 闭环 + 策略更新 |
| `scripts/run_ablations.py` | ⚠️ | ✅ | ✅ | 连通训练管线 |
| `scripts/prepare_data.py` | ❌ | ✅ | ✅ | 4 数据集 |
| `scripts/smoke_test.py` | ❌ | ✅ | ✅ | 27/27 通过 |
| `baselines/` 全部 | ❌ | ❌ | ✅ **新增** | 5 个 Baseline + 评测框架 |
| `paper/main.tex` | ⚠️ | ⚠️ | ⚠️ | 等待实验数据 |

### 1.2 一句话总结

> **所有代码均已就绪（核心库 + 训练脚本 + 5 个 Baseline + 评测框架）。** 下一步是在集群上安装环境，逐 Phase 跑训练，同时跑 Baseline 评测，收集 Table 1/2/3 的数字。

---

## 二、Baseline 目录详解 (`baselines/`)

### 2.1 架构图

```
baselines/
├── __init__.py                     # 包初始化
├── base_agent.py                   # 抽象基类（统一接口）
├── reflexion_agent.py              # Baseline 1: Reflexion
├── memory_r1_agent.py              # Baseline 2: Memory-R1
├── self_consolidation_agent.py     # Baseline 3: Self-Consolidation
├── evolver_agent.py                # Baseline 4: EvolveR
├── mem0_memoryr1_agent.py          # Baseline 5: Mem0 + Memory-R1
├── eval_baselines.py               # 统一评测框架
└── README.md                       # 使用文档
```

### 2.2 `BaseAgent` 统一接口

所有 Baseline（包括未来可能新增的方法）都必须实现这个接口：

```python
class BaseAgent(ABC):
    def initialize(model, tokenizer)        # 初始化模型
    def process_event(event, context) → dict  # 处理事件 → 记忆操作
    def answer_question(question) → str       # 用记忆回答问题
    def reset()                               # 重置 episode 状态
    def train_step(reward, **kwargs) → dict   # RL 更新（可选）
    def get_memory_contents() → list[str]     # 返回当前记忆
    def get_stats() → dict                    # 返回统计信息
```

### 2.3 五个 Baseline 详解

#### Baseline 1: Reflexion（`reflexion_agent.py`）

| 属性 | 值 |
|------|-----|
| **论文** | Shinn et al., NeurIPS 2023 |
| **开源** | `github.com/noahshinn/reflexion` |
| **核心思想** | 失败后用 LLM 生成文字"反思"，存入经验缓冲区，下次作为 prompt 前缀 |
| **记忆方式** | 简单列表（`self.memories`）+ 经验缓冲区（`self.reflections`） |
| **权重更新** | ❌ 无（纯 prompt 学习） |
| **自奖励** | ❌ 无 |
| **巩固** | ❌ 无 |

**关键机制**：
- `reflect_on_failure(event, context, reward)`：当奖励低于阈值时，生成反思文本
- 反思缓冲区最多 10 条，prepend 到后续所有 prompt
- 与 EvolveR 的区别：Reflexion 存的是**具体失败反思**，EvolveR 提取的是**抽象策略原则**

#### Baseline 2: Memory-R1（`memory_r1_agent.py`）

| 属性 | 值 |
|------|-----|
| **论文** | Chen et al., ICLR 2026 |
| **开源** | 部分代码 `syr-cn/ReMemR1` |
| **核心思想** | RL 训练 CRUD 策略，奖励 = 外部 QA F1 |
| **记忆方式** | 复用我们的 `MemoryStore`（FAISS + 图链接） |
| **权重更新** | ✅ REINFORCE（完整版用 GRPO） |
| **自奖励** | ❌ 无（仅 QA F1） |
| **巩固** | ❌ 无 |

**与 G-MSRA 的关系**：这实质上是 G-MSRA 去掉自奖励、巩固和课程训练后的简化版。是论文中最重要的直接对比对象（"我们做的比仅靠外部标注好多少"）。

#### Baseline 3: Self-Consolidation（`self_consolidation_agent.py`）

| 属性 | 值 |
|------|-----|
| **论文** | Zhang et al., arXiv:2602.01966, 2026 |
| **开源** | ❌ 无（纯论文复现，**最高风险 Baseline**） |
| **核心思想** | 对比反思（with/without 记忆的输出对比）+ 固定间隔 LoRA 蒸馏 |
| **记忆方式** | 简单列表（启发式 CRUD，无 RL） |
| **权重更新** | ✅ LoRA SFT（仅巩固时） |
| **自奖励** | ❌ 无 |
| **巩固** | ✅ 有，但固定每 50 步触发（非自适应） |

**关键机制**：
- `_contrastive_reflect()`：比较有记忆 vs 无记忆时的回答质量
  - 如果有记忆更好 → 提取为正样本对 `(input, correct_output)`
  - 正样本积累到一定量后用于 LoRA SFT
- `_consolidate()`：每 N 步固定触发，取最近 50 个正样本训练 LoRA

**与 G-MSRA 的区别**：
1. 无 RL → 无法学习最优 CRUD 策略
2. 无自适应触发 → 可能在错误时机巩固
3. 无置信度过滤 → 低质量记忆可能污染蒸馏

#### Baseline 4: EvolveR（`evolver_agent.py`）

| 属性 | 值 |
|------|-----|
| **论文** | arXiv:2510.16079, 2025 |
| **开源** | ❌ 未确认（论文复现，**中等风险**） |
| **核心思想** | 两阶段循环：在线交互录制轨迹 → 离线用 LLM 提取抽象策略原则 |
| **记忆方式** | 简单列表 + 策略原则库 |
| **权重更新** | ❌ 无 |
| **自奖励** | ❌ 无 |
| **巩固** | ❌ 无（但有"原则蒸馏"概念上类似） |

**关键机制**：
- `end_episode(reward)`：结束 episode 时保存轨迹
- `_distill_principles()`：每 20 个 episode，用 LLM 对比高奖励/低奖励轨迹，提取抽象原则
- `_retrieve_principles(query)`：关键词匹配检索相关原则，prepend 到 prompt

**与 Reflexion 的区别**：Reflexion 存"具体教训"（"上次忘了更新地址"），EvolveR 存"抽象原则"（"当用户提到搬迁时，总是更新地址信息"）。

#### Baseline 5: Mem0 + Memory-R1（`mem0_memoryr1_agent.py`）

| 属性 | 值 |
|------|-----|
| **来源** | Mem0（`mem0ai/mem0`）+ Memory-R1 合体 |
| **开源** | ✅ Mem0 开源 + 我们组合 |
| **核心思想** | Mem0 的多级结构化记忆 + Memory-R1 的 RL CRUD |
| **记忆方式** | 三级：User（长期）/ Session（会话）/ Agent（观察）+ 实体追踪 |
| **权重更新** | ✅ REINFORCE |
| **自奖励** | ❌ 无 |
| **巩固** | ❌ 无 |

**Mem0 特色机制**：
- `_extract_entities(event)`：从事件中提取实体（Title Case 词）
- `_deduplicate(event)`：Jaccard 相似度去重（>0.7 跳过）
- `_classify_memory_level(content)`：按内容关键词分配到 User/Session/Agent 层
- 回答时分级检索 + 实体知识整合

**论文意义**：测试"仅靠更好的工程（结构化记忆 + RL）是否能替代我们的理论贡献（自奖励 + 环境锚定 + 自适应巩固）"。

### 2.4 Feature 对比矩阵

| 特性 | Reflexion | Memory-R1 | Self-Consol | EvolveR | Mem0+R1 | **G-MSRA** |
|------|:---------:|:---------:|:-----------:|:-------:|:-------:|:----------:|
| RL 权重更新 | - | REINFORCE | - | - | REINFORCE | **GRPO** |
| 环境锚定 R_env | - | QA F1 | - | - | QA F1 | **环境反馈** |
| 自奖励 R_mem | - | - | - | - | - | **LLM Judge** |
| 参数巩固 | - | - | LoRA(固定) | - | - | **LoRA(自适应)** |
| 置信度过滤 | - | - | - | - | - | **是** |
| 课程训练 | - | - | - | - | - | **4 阶段** |
| 记忆结构 | 列表 | FAISS+图 | 列表 | 列表+原则库 | 三级+实体 | **FAISS+图+置信度** |
| 反思/原则 | 反思缓冲 | - | 对比反思 | 策略原则 | - | **Judge+记忆** |

---

## 三、评测框架详解 (`baselines/eval_baselines.py`)

### 3.1 评测流程

```
python baselines/eval_baselines.py --data_dir data --benchmark locomo
```

```
为每个 Baseline:
  1. load_agent(name) → 动态导入 + 实例化
  2. agent.initialize() → 加载模型
  3. 对每个 episode:
     a. agent.reset()
     b. for event in episode.events:
          agent.process_event(event, context)
     c. prediction = agent.answer_question(question)
     d. 计算 F1、EM
  4. 输出 per-category 分数
  5. 保存 JSON 结果
```

### 3.2 支持的 Benchmark 和 Metrics

| Benchmark | 类型 | Metrics | 对应论文表格 |
|-----------|------|---------|-------------|
| LoCoMo | 对话记忆 | F1, EM, Judge | Table 1 |
| LongMemEval | 对话记忆 | F1, EM, Judge | Table 1 |
| ALFWorld | Agent 任务 | Success Rate, FRR, Token Cost | Table 2 |
| Evo-Memory | 记忆演化 | F1, EM | Table 1 (补充) |

### 3.3 输出格式

结果保存到 `results/baselines/baseline_results.json`：

```json
{
  "reflexion": {
    "locomo": {
      "avg_f1": 0.XXXX,
      "avg_em": 0.XXXX,
      "per_category": {
        "information_extraction": {"f1": ..., "em": ..., "count": ...},
        "temporal_reasoning": {"f1": ..., "em": ..., "count": ...}
      }
    }
  },
  "memory_r1": { ... },
  ...
}
```

---

## 四、接续完成任务的人应当怎么做

### 4.1 完整执行路径（从零到论文 Table 1/2 数据）

```bash
# === 前置准备（已完成，详见 v2.0 §3.1-3.2） ===
ssh song2
cd /NAS/<your_user>/G-MSRA
bash setup_env.sh
conda activate gmsra
python scripts/smoke_test.py        # 确认 27/27 通过
python scripts/prepare_data.py      # 准备数据集

# === G-MSRA 主实验 ===
bash cluster/run_song.sh phase0     # Phase 0: SFT 热启动
bash cluster/run_song.sh phase1     # Phase 1: RL + 外部奖励
bash cluster/run_song.sh phase2     # Phase 2: 课程退火
bash cluster/run_song.sh phase3     # Phase 3: 全闭环
bash cluster/run_song.sh eval       # 评测

# === Baseline 评测 ===
# 方法 A: 运行全部 Baseline（需要 GPU，每个要加载模型）
python baselines/eval_baselines.py \
    --data_dir data \
    --output_dir results/baselines \
    --max_episodes 100  # 先跑少量验证

# 方法 B: 逐个运行（推荐，便于调试）
python baselines/eval_baselines.py --agent reflexion --benchmark locomo
python baselines/eval_baselines.py --agent memory_r1 --benchmark locomo
python baselines/eval_baselines.py --agent self_consolidation --benchmark locomo
python baselines/eval_baselines.py --agent evolver --benchmark locomo
python baselines/eval_baselines.py --agent mem0_memory_r1 --benchmark locomo

# === 消融实验 ===
python scripts/run_ablations.py --base_checkpoint outputs/phase1/best
```

### 4.2 需要特别注意的事项

#### Memory-R1 的训练

Memory-R1 Baseline 有 `train_step()` 方法，需要先训练再评测才有意义：

```python
# 在评测框架中，Memory-R1 和 Mem0+R1 可以通过以下方式训练
# （eval_baselines.py 目前仅评测，不训练）
# 你可能需要写一个短训练循环：

agent = MemoryR1Agent()
agent.initialize()

for episode in train_data:
    agent.reset()
    for event in episode["events"]:
        agent.process_event(event, episode["question"])
    prediction = agent.answer_question(episode["question"])
    reward = compute_f1(prediction, episode["answer"])
    agent.train_step(reward, event=episode["events"][-1],
                     context=episode["question"])
```

如果时间紧张，可以先用未训练的 Baseline 跑基线数据（这些 Baseline 不训练也能工作，只是效果更差——这对我们有利，因为我们的方法有训练）。

#### Self-Consolidation 的验证

Self-Consolidation 是论文复现，没有官方代码可对比。确认方式：
1. 检查 `positive_pairs` 是否在积累
2. 检查 LoRA 巩固是否每 50 步触发（查日志 `[SelfConsolidation] Consolidation #N`）
3. 巩固前后的回答质量应有提升

#### EvolveR 的验证

类似地，检查：
1. `_distill_principles()` 是否在每 20 episode 后提取原则
2. 原则内容是否合理（查 `evolver_state.json` 中的 `principles` 字段）
3. 原则检索是否在实际影响行为

### 4.3 论文数字填充指南

当所有实验完成后，你需要将结果填入 `paper/main.tex`：

**Table 1**（主表，L266-272）：
```latex
Reflexion        & <F1>  & <EM>  & <Judge> & <F1>  & <EM>  & <Judge> \\
EvolveR          & <F1>  & <EM>  & <Judge> & <F1>  & <EM>  & <Judge> \\
Self-Consolidation & <F1>  & <EM>  & <Judge> & <F1>  & <EM>  & <Judge> \\
Memory-R1        & <F1>  & <EM>  & <Judge> & <F1>  & <EM>  & <Judge> \\
Mem0 + Memory-R1 & <F1>  & <EM>  & <Judge> & <F1>  & <EM>  & <Judge> \\
G-MSRA (Ours)    & \textbf{<F1>} & ...
```

**Table 2**（ALFWorld，L287-292）：从 `baseline_results.json` 中提取 `success_rate`, `frr`, `avg_token_cost`。

**Table 3**（消融，L316-323）：从 `results/ablations/ablation_summary.json` 提取。

### 4.4 Judge 分数的计算

论文 Table 1 中有一列 "Judge"（LLM-as-Judge 评分）。目前 `eval_baselines.py` 计算 F1 和 EM，但 **Judge 分数尚未实现**。

实现方式（建议在评测框架中加入）：

```python
def compute_judge_score(question, prediction, answer, model, tokenizer):
    prompt = (
        f"Rate the quality of this answer on a scale of 0 to 10.\n"
        f"Question: {question}\n"
        f"Reference Answer: {answer}\n"
        f"Model Answer: {prediction}\n"
        f"Score (0-10):"
    )
    result = generate_text(model, tokenizer, prompt, max_new_tokens=5)
    try:
        return float(result.strip()) / 10.0
    except ValueError:
        return 0.5
```

---

## 五、代码架构全景（v3.0 更新版）

```
G-MSRA/
├── gmsra/                        # 核心库（v1.0，未变）
│   ├── config.py                 # 超参数
│   ├── utils.py                  # 工具函数
│   ├── agent.py                  # ⭐ 主编排器
│   ├── memory/                   # MemoryEntry + MemoryStore
│   ├── reward/                   # 环境信号 + 双层奖励
│   ├── manager/                  # RL 记忆管理器
│   └── consolidation/            # 触发器 + LoRA 蒸馏器
│
├── scripts/                      # 训练/评测（v2.0 补全）
│   ├── prepare_data.py           # 数据准备
│   ├── train_phase{0,1,2,3}.py   # 四阶段训练
│   ├── eval_*.py                 # 评测脚本
│   ├── run_ablations.py          # 消融实验
│   └── smoke_test.py             # 冒烟测试
│
├── baselines/                    # ⭐ Baseline 复现（v3.0 新增）
│   ├── base_agent.py             # 抽象接口
│   ├── reflexion_agent.py        # Reflexion
│   ├── memory_r1_agent.py        # Memory-R1
│   ├── self_consolidation_agent.py # Self-Consolidation
│   ├── evolver_agent.py          # EvolveR
│   ├── mem0_memoryr1_agent.py    # Mem0 + Memory-R1
│   └── eval_baselines.py         # 统一评测
│
├── cluster/                      # 集群脚本
├── paper/                        # 论文骨架
├── setup_env.sh / setup.py       # 环境安装
└── requirements.txt              # 依赖
```

---

## 六、里程碑检查清单（v3.0 更新）

| 里程碑 | 目标周 | 判定标准 | 状态 |
|--------|:------:|---------|:----:|
| M0: 代码补全 | W1 | 训练脚本可运行 | ✅ |
| M0.5: 冒烟测试 | W1 | 27/27 通过 | ✅ |
| **M0.8: Baseline 复现** | **W1** | **5 个 Baseline 代码就绪 + 评测框架可运行** | **✅ 新增** |
| M1: Pipeline 跑通 | W3 | 集群上 Phase 0→3→eval 无 crash | ☐ |
| M1.5: Baseline 评测 | W3 | `eval_baselines.py` 跑通全部 5 个 Baseline | ☐ |
| M2: Phase 1 收敛 | W6 | LoCoMo F1 ≥ Memory-R1 | ☐ |
| M3: Phase 2 退火 | W8 | α→0, τ ≥ 0.5 | ☐ |
| M4: 主表完成 | W12 | Table 1 + 2 全部数字 | ☐ |
| M5: 消融完成 | W15 | Table 3 + A1 证实 Reward Hacking | ☐ |
| M6: 分析图完成 | W18 | 6 张核心图 | ☐ |
| M7: 初稿完成 | W22 | 可读论文 | ☐ |
| M9: 最终投递 | W28 | OpenReview 提交 | ☐ |

---

## 七、风险与应对（v3.0 更新）

| 风险 | 概率 | 影响 | 应对方案 |
|------|:----:|:----:|---------|
| ~~TRL 对接复杂~~ | ~~中~~ | ~~高~~ | ✅ 已解决（三层降级） |
| ~~Baseline 代码缺失~~ | ~~中~~ | ~~中~~ | ✅ 已解决（5 个全部实现） |
| Self-Consolidation 复现不准 | 中 | 🟡 中 | 核心行为（对比反思 + 固定 LoRA）已实现；如审稿人质疑，可加注"our reproduction" |
| EvolveR 复现不准 | 低 | 🟡 中 | 原则蒸馏的核心循环已实现；与 Reflexion 的 feature 差异明确 |
| Baseline 评测耗时过长 | 中 | 🟡 中 | 用 `--max_episodes 100` 先跑小规模；逐个 Baseline 分批跑 |
| Phase 2 退火 τ 低 | 中 | 🟡 中 | 降低阈值到 0.3 或不完全退火 |
| ALFWorld 对接 | 中 | 🟡 中 | 先用 3 个文本 benchmark 出主表 |

---

> **最后提醒**：论文的"杀手锏"是 **消融 A1**（纯自奖励在长周期后因 Reward Hacking 性能劣化）。但 Baseline 对比同样关键——如果 G-MSRA 能在 **不使用外部 QA 标注** 的情况下匹配甚至超越 Memory-R1（依赖 QA F1 标注），这就直接证明了我们方法的实用价值。请确保 Memory-R1 Baseline 的训练和评测条件公平。

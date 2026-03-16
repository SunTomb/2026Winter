# G-MSRA 项目工作流程与排期规划

> **Closing the Loop: Environment-Grounded Self-Reward for Autonomous Memory Management in Lifelong LLM Agents**
>
> 制订日期：2026 年 3 月 15 日 · 目标会议：ICLR 2027（截稿 2026 年 9 月底）/ NeurIPS 2026（截稿 2026 年 5 月中旬，备选）

---

## 〇、本文档使用指南

本文档面向**项目负责人及所有合作者**，目的是让任何人从零开始也能理解：

1. 这个项目的核心研究问题是什么（第一部分）
2. 目前代码框架已经搭好了什么、每个文件做什么（第二部分）
3. 从现在起到论文投递，每一周要做什么、由谁做、产出是什么（第三到七部分）

> **一条原则**：本项目所有实验都在 USTC LDS 实验室集群上完成（Song 节点 / A100 80G 为主），部署和集群使用方法详见 [README.md](file:///f:/USTC/2026Winter/G-MSRA/README.md) 和 `cluster/` 目录下的脚本。

---

## 一、研究问题一句话回顾

当前 LLM Agent 记忆研究有三条割裂线：

| 线 | 解决什么 | 代表作 | 遗留缺陷 |
|----|---------|--------|---------|
| ① 记忆策略学习 | Agent 学会增删改查记忆 | Memory-R1 | 依赖外部标注 QA F1 作为 RL 奖励 |
| ② 自生成学习信号 | 无需人工标注的自我改进 | Self-Rewarding LMs | 完全没有记忆，每次打分从零开始 |
| ③ 参数巩固 | 把文本经验蒸馏进 LoRA 权重 | Self-Consolidation | 巩固触发靠硬编码阈值，原材料缺乏结构 |

**G-MSRA 的核心贡献**是用一个"环境锚定 + 记忆一致性"的双层复合奖励把这三条线闭合：

$$R_{\text{total}} = R_{\text{env}}\ (\text{环境锚，防 Reward Hacking}) + \lambda \cdot R_{\text{mem}}\ (\text{记忆一致性，精细化引导}), \quad \lambda < 1$$

---

## 二、当前进度——代码框架详解

截至 2026-03-15，项目已搭建了 **30 个文件** 的完整骨架。下面逐模块说明其作用和完成度。

### 2.1 代码框架整体架构图

```
G-MSRA/
├── gmsra/                        # ← 核心库（全部为 Python 骨架/可运行框架）
│   ├── config.py                 # 所有超参数的集中管理
│   ├── utils.py                  # 通用工具：模型加载、F1/EM/Kendall τ 计算、文本生成
│   ├── agent.py                  # ⭐ 主编排器：连接下面四个子模块，驱动 Online/Offline 闭环
│   │
│   ├── memory/
│   │   ├── entry.py              # MemoryEntry 数据类：内容、关键词、标签、图链接、置信度
│   │   └── store.py              # ⭐ MemoryStore：FAISS 向量检索 + 图链接 + 高频子图提取
│   │
│   ├── reward/
│   │   ├── env_signals.py        # 三种环境信号提取器（Agent任务/对话/外部QA）
│   │   └── grounded_reward.py    # ⭐ 核心创新：双层复合奖励 + Judge Prompt + 退火接口
│   │
│   ├── manager/
│   │   └── memory_manager.py     # RL 记忆管理器：CRUD 决策 + Prompt 构建 + 操作解析
│   │
│   └── consolidation/
│       ├── trigger.py            # 3D 自适应巩固触发器：Conflict + Variance + Growth
│       └── distiller.py          # ⭐ LoRA 语义蒸馏：子图筛选 → 三元组生成 → SFT + EWC
│
├── scripts/                      # ← 训练与评测脚本
│   ├── train_phase0_sft.py       # Phase 0: SFT 热启动（含合成训练数据生成）
│   ├── train_phase1_rl.py        # Phase 1: RL + 外部奖励（PPO/GRPO via TRL）
│   ├── train_phase2_transition.py# Phase 2: 课程退火 + Kendall τ 监控
│   ├── train_phase3_full.py      # Phase 3: 全闭环 + 自适应巩固
│   ├── eval_locomo.py            # 评测：LoCoMo / LongMemEval
│   ├── eval_agent_tasks.py       # 评测：ALFWorld / WebArena
│   └── run_ablations.py          # 消融实验运行器（A1-A7 已预配置）
│
├── cluster/                      # ← USTC LDS 集群适配脚本
│   ├── run_song.sh               # Song 节点（A100 80G）运行脚本
│   └── run_tang.sh               # Tang 节点（A40 45G）运行脚本
│
├── paper/                        # ← 论文骨架
│   ├── main.tex                  # ICLR 格式论文：6 节 + 附录，含公式、表格、图片占位
│   └── references.bib            # 27 篇参考文献 BibTeX
│
├── setup_env.sh                  # Conda 环境安装脚本（含集群代理配置）
├── setup.py                      # 可编辑安装配置
├── requirements.txt              # Python 依赖列表
└── README.md                     # 项目文档
```

### 2.2 各模块完成度状态

| 模块 | 状态 | 说明 |
|------|:----:|------|
| `gmsra/config.py` | ✅ 可用 | 所有超参数已设置合理默认值，支持 YAML 加载 |
| `gmsra/utils.py` | ✅ 可用 | 模型加载（含 QLoRA 4-bit）、F1/EM/Kendall τ、文本生成均已实现 |
| `gmsra/memory/entry.py` | ✅ 可用 | MemoryEntry 数据结构完整，置信度公式已实现 |
| `gmsra/memory/store.py` | ✅ 可用 | FAISS 索引、CRUD、图链接、子图提取、持久化均已实现 |
| `gmsra/reward/env_signals.py` | ✅ 可用 | 三种提取器：AgentTask（直接反馈）、Dialogue（用户反应代理）、ExternalQA |
| `gmsra/reward/grounded_reward.py` | ✅ 可用 | Judge Prompt 已设计，退火接口已实现，Drift 监控已实现 |
| `gmsra/manager/memory_manager.py` | ✅ 可用 | CRUD Prompt 构建、操作解析、SFT 数据生成已实现 |
| `gmsra/consolidation/trigger.py` | ✅ 可用 | 3D 触发函数已实现，含诊断接口 |
| `gmsra/consolidation/distiller.py` | ✅ 可用 | 三元组生成 + LoRA SFT + EWC 正则化已实现 |
| `gmsra/agent.py` | ✅ 可用 | 完整 step() 循环、checkpoint 保存/加载、诊断输出已实现 |
| `scripts/train_phase0_sft.py` | ⚠️ 骨架 | 含合成训练数据（~80 例），**需扩充到 200+ 多样化样本** |
| `scripts/train_phase1_rl.py` | ⚠️ 骨架 | RL 循环框架已写，**需对接 TRL 的 PPOTrainer 完整 API** |
| `scripts/train_phase2_transition.py` | ⚠️ 骨架 | 退火和 τ 监控逻辑完整，**需与 Phase 1 的 RL 训练器深度集成** |
| `scripts/train_phase3_full.py` | ⚠️ 骨架 | 全循环框架完整，**需真实数据集和真实环境反馈** |
| `scripts/eval_*.py` | ⚠️ 骨架 | 评测循环完整，**需下载并格式化实际 benchmark 数据集** |
| `scripts/run_ablations.py` | ⚠️ 骨架 | 7 组消融配置已定义，**需连接到完整训练管线** |
| `paper/main.tex` | ⚠️ 骨架 | 结构/公式/表格完整，结果列为 `--`，**需填入实验数据** |
| `cluster/*.sh` | ✅ 可用 | 已适配 USTC LDS 集群，含代理、NCCL、多 GPU 配置 |

> **核心结论**：库代码（`gmsra/`）已基本可用，训练脚本（`scripts/`）需要与真实数据和 TRL API 深度对接后才能跑通端到端流程。这是 **下一步工作的第一优先级**。

---

## 三、总体排期（甘特图视角）

假设目标投递为 **ICLR 2027**（截稿约 2026 年 9 月底），从 2026 年 3 月中旬到提交共约 **28 周**。

```
Week:  W1──W2──W3──W4──W5──W6──W7──W8──W9──W10─W11─W12─W13─W14─...─W24─W25─W26─W27─W28
       ├─ P1: 基建 ──┤
       │  环境部署     │
       │  数据下载     │
       │  Phase0 跑通  │
                       ├── P2: 核心实验 ─────────────────┤
                       │  Phase 1 RL 训练                 │
                       │  Phase 2 课程退火                 │
                       │  Phase 3 全闭环                   │
                       │  主表结果 (LoCoMo+ALFWorld)       │
                                                          ├── P3: 消融+分析 ────┤
                                                          │  A1-A7 消融实验       │
                                                          │  Reward Drift 分析    │
                                                          │  可视化 + Case Study  │
                                                                                  ├── P4: 论文 ──────┤
                                                                                  │  论文初稿           │
                                                                                  │  内审 + 修改         │
                                                                                  │  Camera-ready       │
                                                                                                       └→ 投递
```

---

## 四、分阶段详细规划

### Phase 1：基础设施搭建（W1—W3, 共 3 周）

**目标**：在集群上跑通 Phase 0 SFT，确认全链路无 bug。

#### W1：环境部署 + 数据准备

| # | 任务 | 具体操作 | 产出 | 负责人 |
|---|------|---------|------|--------|
| 1.1 | 集群环境部署 | SSH 到 Song1，执行 `bash setup_env.sh`，验证 PyTorch + CUDA + PEFT 安装 | 可用的 `gmsra` conda 环境 | — |
| 1.2 | 模型下载 | `huggingface-cli download Qwen/Qwen2.5-7B-Instruct` 到 `/NAS/<user>/models/` | 模型本地缓存 | — |
| 1.3 | 数据集下载与格式化 | 下载 LoCoMo、LongMemEval、ALFWorld 数据集，编写格式转换脚本，统一为 JSON 格式 | `data/locomo_train.json`, `data/locomo_test.json`, `data/longmemeval_test.json`, `data/alfworld_tasks.json` | — |
| 1.4 | Evo-Memory 数据集 | 从论文仓库获取 Evo-Memory benchmark 数据 | `data/evomemory_test.json` | — |

**数据格式约定**（所有脚本已假设此格式）：
```json
// locomo_train.json — 每条是一个 episode
[
  {
    "events": ["User says: I moved to Shanghai.", "User says: I work at Alibaba."],
    "question": "Where does the user live?",
    "answer": "Shanghai"
  },
  ...
]
```

#### W2：Phase 0 跑通 + 训练脚本对接

| # | 任务 | 具体操作 | 产出 |
|---|------|---------|------|
| 2.1 | 扩充 SFT 训练数据 | 在 `train_phase0_sft.py` 中扩充 `generate_sft_data()`，从 ~80 条增加到 200+ 条，覆盖更多场景（多轮对话、冲突更新、偏好变化、时间推理等） | 更丰富的 SFT 数据 |
| 2.2 | 跑通 Phase 0 | `bash cluster/run_song.sh phase0`，确认输出 `outputs/phase0/best` | 训练好的 SFT LoRA |
| 2.3 | 对接 TRL PPOTrainer | **关键任务**：`train_phase1_rl.py` 中的 RL 循环目前是伪代码。需要对接 TRL 库的 `PPOTrainer` 或 `GRPOTrainer` 完整 API。参考 [Memory-R1 的开源代码](https://github.com/) 和 [TRL 文档](https://huggingface.co/docs/trl/) | 可运行的 Phase 1 脚本 |

> **⚠️ 对接 TRL 是最关键的工程任务**。当前 `train_phase1_rl.py` 已有完整的数据加载、Agent 初始化和评测循环，但 RL 训练核心需要实例化 `PPOTrainer`，将 Memory Manager 的输出作为 response，将 QA F1 作为 reward，调用 `ppo_trainer.step()` 更新策略。具体需要：
> 1. 将 `memory_manager.decide()` 的输出封装为 TRL 期望的 `response_tensors`
> 2. 将 `compute_f1()` 的结果封装为 `rewards` tensor
> 3. 正确处理 query/response tokenization

#### W3：端到端冒烟测试

| # | 任务 | 具体操作 | 产出 |
|---|------|---------|------|
| 3.1 | 小规模 Phase 1 测试 | 用 50 条 LoCoMo 数据跑 100 个 episode 的 Phase 1，确认 RL 训练不发散 | Phase 1 训练日志 |
| 3.2 | Phase 2 冒烟测试 | 用 Phase 1 的 checkpoint 跑 50 步 Phase 2 退火，确认 Kendall τ 计算正确 | Phase 2 校准日志 |
| 3.3 | Phase 3 冒烟测试 | 确认巩固触发器能正确 fire、LoRA 蒸馏能执行 | Phase 3 巩固日志 |
| 3.4 | 评测脚本测试 | `python scripts/eval_locomo.py --checkpoint outputs/phase0/best`，确认输出格式正确 | 评测结果 JSON |

**W3 结束判定标准**：从 Phase 0 → Phase 1 → Phase 2 → Phase 3 → eval 的完整 pipeline 能在小数据上跑通，无 crash，日志合理。

---

### Phase 2：核心实验（W4—W12, 共 9 周）

**目标**：在全量数据上完成 4 阶段训练，获得主表结果。

#### W4—W6：Phase 1 正式训练

| # | 任务 | 硬件 | 预计时长 |
|---|------|------|---------|
| 4.1 | Phase 1 全量训练 | 4×A100 (Song1) | 2-3 天 |
| | 参数：`--num_episodes 5000 --batch_size 16 --learning_rate 1.41e-5` | | |
| 4.2 | Phase 1 超参搜索 | 4×A100 并行多组 | 3-4 天 |
| | 搜索: lr ∈ {5e-6, 1e-5, 2e-5}, batch ∈ {8, 16, 32} | | |
| 4.3 | Phase 1 评测 | 1×A100 | 2-4 小时 |
| | 在 LoCoMo + LongMemEval 上评测 best checkpoint，作为上界参照 | | |
| 4.4 | **复现 Memory-R1 Baseline** | 4×A100 | 2-3 天 |
| | 如果 Memory-R1 有官方代码则直接跑，否则用本框架跑无自奖励版本 | | |

**W6 交付物**：Phase 1 best checkpoint + Memory-R1 baseline 对比数据。

#### W7—W8：Phase 2 课程退火

| # | 任务 | 关键监控指标 |
|---|------|------------|
| 5.1 | Phase 2 正式退火 | Kendall τ 不低于 0.5 |
| | `--anneal_steps 3000 --tau_threshold 0.5` | 若 τ 下降则自动暂停 |
| 5.2 | 退火速率实验 | 对比 anneal_steps = {1000, 3000, 5000} |
| 5.3 | 校准数据保存 | 保存完整的 `(R_ext, R_self)` 对，用于论文 §5.1 的校准分析图 |

**W8 交付物**：Phase 2 best checkpoint + `calibration.json` 校准数据。

#### W9—W12：Phase 3 全闭环 + 主表评测

| # | 任务 | 说明 |
|---|------|------|
| 6.1 | Phase 3 全闭环训练 | 10000 episodes，预计 3-5 天 |
| 6.2 | 监控巩固行为 | 记录每次 Trigger fire 的三维信号值和蒸馏统计 |
| 6.3 | **全基线对比评测** | 用最终 checkpoint 跑全部 6 个 baseline + G-MSRA 在 3 个 benchmark 上的结果 |
| 6.4 | ALFWorld 实验 | 需要安装 ALFWorld 环境，将 `env_signals.py` 的 `AgentTaskSignalExtractor` 对接到真实环境 |
| 6.5 | Evo-Memory 实验 | 对接 Evo-Memory benchmark API |

**W12 交付物**：论文主表 (Table 1 + Table 2) 的全部数字。

> **⚠️ 关键风险点**：
> - **ALFWorld 对接**（6.4）可能需要额外工程工作（环境安装、action space 适配）。如果时间紧张，可以先用 LoCoMo + LongMemEval + Evo-Memory 三个 benchmark 出主表，ALFWorld 作为补充实验放到附录。
> - **Baseline 复现**：Reflexion 和 EvolveR 有开源代码，Self-Consolidation 需要根据论文复现。如复现成本过高，可引用其论文中报告的数字（需注明评测条件差异）。

---

### Phase 3：消融实验与深度分析（W13—W18, 共 6 周）

**目标**：完成 7 组消融 + 论文核心分析图。

#### W13—W15：消融实验

| ID | 消融内容 | 验证的假说 | 论文价值 |
|----|---------|-----------|---------|
| **A1** | 移除 $R_{\text{env}}$，仅用 $R_{\text{mem}}$ | **Reward Hacking 是否被阻断** | ⭐⭐⭐⭐⭐ 全文最关键消融 |
| A2 | 移除 $R_{\text{mem}}$，仅用 $R_{\text{env}}$ | 记忆一致性引导的精细化价值 | ⭐⭐⭐⭐ |
| A3 | 移除记忆置信度过滤 | 噪声过滤的必要性 | ⭐⭐⭐ |
| A4 | 固定触发阈值 (每 50 步) | 自适应 vs 启发式 | ⭐⭐⭐ |
| A5 | 随机选择蒸馏内容 | 图谱筛选的作用 | ⭐⭐ |
| A6 | 移除 LoRA 巩固 | 参数化巩固的长期增益 | ⭐⭐⭐⭐ |
| A7 | 跳过 Phase 1-2 直接自奖励 | 课程训练的稳定性 | ⭐⭐⭐ |

**执行策略**：
- A1、A2、A6、A7 优先级最高，必须完成
- A3、A4、A5 如时间紧张可缩减数据规模（50% episodes）
- 每个消融用 `scripts/run_ablations.py` 的配置修改机制运行

#### W16—W18：深度分析与可视化

| # | 分析内容 | 对应论文章节 | 输出 |
|---|---------|-------------|------|
| 7.1 | **Reward Drift 曲线** | §5.1 | R_total vs 步数的折线图，对比 Full vs A1(纯自奖励)。预期 A1 在 1000 步后开始漂移 |
| 7.2 | **Kendall τ 演化图** | §5.1 | R_env vs R_mem 的散点图 + τ 随训练的变化曲线 |
| 7.3 | **Trigger 三维信号分解** | §5.2 | Conflict / Variance / Growth 三条线 + 巩固事件标注 |
| 7.4 | **记忆库增长曲线** | §4.4 | 有巩固 vs 无巩固 (A6) 的记忆库大小对比 |
| 7.5 | **Token 成本分析** | §4.4 | 与 Mem0 / LightMem 的每步 Token 开销对比 |
| 7.6 | **置信度分布演化** | §5.3 | 不同训练阶段的置信度直方图 |
| 7.7 | **案例研究** | §5.4 | 2-3 个定性案例：R_env 修正错误 R_mem、R_mem 提供精细指导 |

**W18 交付物**：消融表 (Table 3) 全部数字 + 6 张分析图 + Case Study 文本。

---

### Phase 4：论文撰写与投递（W19—W28, 共 10 周）

#### W19—W22：初稿撰写

| 优先级 | 章节 | 负责人 | 说明 |
|:------:|------|--------|------|
| 1 | §3 Method | — | 已有论文骨架，填充细节和解释。核心：把代码里每个模块的设计动机和数学定义写清楚 |
| 2 | §4 Experiments | — | 填入主表数字、消融表数字。每个表格下面写 2-3 条 Key Observations |
| 3 | §5 Analysis | — | 嵌入分析图，写 Figure Caption + 分析段落。§5.1 (Reward Drift) 是最关键的分析 |
| 4 | §1 Introduction | — | 基本结构已有（三条断裂线 + 第四条矛盾），补充实验结果的 highlight 数字 |
| 5 | §2 Related Work | — | 框架已有，补充与每篇工作的差异化定位 |
| 6 | Abstract | — | 最后写，浓缩全文为 150-200 词 |
| 7 | Appendix | — | 超参表、额外分析图、Case Study 全文 |

**论文骨架参考**：[main.tex](file:///f:/USTC/2026Winter/G-MSRA/paper/main.tex) 已包含完整结构。

#### W23—W24：内审与修改

| # | 任务 | 产出 |
|---|------|------|
| 8.1 | 导师初审 | 反馈意见 |
| 8.2 | 模拟审稿 | 请同学用 ICLR Review Guidelines 打分，重点关注 Reward Hacking 论证链 |
| 8.3 | 针对性修改 | 修改后的论文 v2 |
| 8.4 | 补充实验（如需要） | 针对审稿意见的补实验 |

#### W25—W26：精修与格式化

| # | 任务 |
|---|------|
| 9.1 | 全文 proofread（语法、术语一致性） |
| 9.2 | 图表美化（统一字体、颜色、分辨率 ≥ 300dpi） |
| 9.3 | 参考文献查验（确认每篇引用的年份、会议名称） |
| 9.4 | Appendix 补充（完整超参表、额外消融、代码接口描述） |

#### W27—W28：最终投递

| # | 任务 |
|---|------|
| 10.1 | 论文 PDF 生成 + 多次检查格式 |
| 10.2 | 代码仓库整理（清理实验日志、写 README、确认可复现性） |
| 10.3 | 补充材料整理（如果投递允许 Supplementary） |
| 10.4 | 提交到 OpenReview |

---

## 五、关键技术细节：接续完成代码的人需要做什么

如果你是**接手该项目的合作者**，下面是你需要重点理解和完成的事项：

### 5.1 最优先：对接 TRL PPOTrainer

**文件**：`scripts/train_phase1_rl.py`

**当前状态**：脚本有完整的 Agent 初始化和评测循环，但 RL 训练核心是伪代码。

**你需要做的**：

```python
# 需要修改的核心部分（伪代码 → 真实 TRL API）
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 1. 用 TRL 的 ValueHead 模型替换普通模型
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# 2. 实例化 PPOTrainer
ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    tokenizer=tokenizer,
    dataset=...,  # 需要格式化为 TRL 期望的 Dataset
)

# 3. 在每个 episode 中:
#    a. 用 Memory Manager 生成 response (决策)
#    b. 用 GroundedRewardGenerator 计算 reward
#    c. 调用 ppo_trainer.step(queries, responses, rewards)
for batch in ppo_trainer.dataloader:
    queries = batch["input_ids"]
    responses = ppo_trainer.generate(queries, ...)
    rewards = [compute_reward(r) for r in responses]  # ← 接入 G-MSRA 奖励
    stats = ppo_trainer.step(queries, responses, rewards)
```

参考资源：
- Memory-R1 官方代码（如果开源）
- TRL 官方 PPO 示例：https://github.com/huggingface/trl/tree/main/examples/ppo
- GRPO 可参考 DeepSeek-R1 的开源训练代码

### 5.2 次优先：数据集格式化

**需要创建的文件**：

| 文件 | 来源 | 格式 |
|------|------|------|
| `data/locomo_train.json` | LoCoMo benchmark | `[{"events": [...], "question": "...", "answer": "..."}]` |
| `data/locomo_test.json` | LoCoMo benchmark | 同上 |
| `data/longmemeval_test.json` | LongMemEval | 同上（加 `"category"` 字段） |
| `data/alfworld_tasks.json` | ALFWorld | `[{"instruction": "...", "events": [...], "type": "put", "env_kwargs": {...}}]` |

建议写一个 `scripts/prepare_data.py` 来自动下载和格式化。

### 5.3 第三优先：ALFWorld 环境对接

**文件**：`gmsra/reward/env_signals.py` 的 `AgentTaskSignalExtractor`

**当前状态**：提取器假设输入是一个 `task_result` 字典。需要对接 ALFWorld 的 gym 环境 API。

```python
# 需要新增的 ALFWorldWrapper
import alfworld.agents.environment as env

class ALFWorldWrapper:
    def __init__(self):
        self.env = env.AlfredTWEnv(...)

    def step(self, action: str) -> dict:
        obs, reward, done, info = self.env.step(action)
        return {
            "success": done and reward > 0,
            "partial_score": reward,
            "observation": obs,
        }
```

### 5.4 理解核心模块的调用关系

```
用户请求 / 环境事件
        │
        ▼
┌─ MemoryManager.decide(event) ──→ "ADD: user prefers coffee"
│       │
│       ▼
│  MemoryManager.execute_operation() ──→ MemoryStore.add()
│       │
│       ▼
│  GroundedRewardGenerator.compute_reward()
│       ├── R_env ← EnvironmentSignalExtractor.extract()  (外部世界)
│       └── R_mem ← LLM-as-Judge + MemoryStore.retrieve_confident()  (内部评估)
│       │
│       ▼
│  R_total = R_env + λ·R_mem  ──→ 反馈给 RL 策略更新
│       │
│       ▼
│  ConsolidationTrigger.should_trigger()
│       │ (如果触发)
│       ▼
│  SemanticDistiller.consolidate()
│       ├── MemoryStore.extract_high_frequency_subgraph()
│       ├── 生成语义三元组
│       ├── LoRA SFT + EWC
│       └── 清理已蒸馏记忆
│
└── 循环 ──→ 下一个事件
```

---

## 六、风险与应对

| 风险 | 概率 | 影响 | 应对方案 |
|------|:----:|:----:|---------|
| TRL API 对接复杂度超预期 | 中 | 🔴 高 | 预留 W2-W3 两周缓冲；如果 PPO 太难调试，先换 REINFORCE 验证闭环再优化 |
| Phase 2 退火过程中 τ 始终低于阈值 | 中 | 🟡 中 | 降低 τ 阈值到 0.3；增加 Phase 1 训练数据量；混合使用外部奖励不完全退火（α 终止于 0.2 而非 0） |
| ALFWorld 环境对接工程量大 | 中 | 🟡 中 | 先用 LoCoMo + LongMemEval + Evo-Memory 三个 benchmark 出主表，ALFWorld 作为补充 |
| Baseline 复现困难 | 低 | 🟡 中 | 引用原论文数字 + 注明评测条件差异；优先复现有开源代码的 baseline |
| A100 排队等待时间过长 | 中 | 🟡 中 | Phase 0 和评测可在 Tang 节点 (A40) 运行；合理使用预约系统避开高峰 |
| 实验时间不够冲 NeurIPS 2026 | 高 | 🟢 低 | 以 ICLR 2027 为主目标（充足时间），NeurIPS 2026 为可选冲刺 |

---

## 七、里程碑检查清单

| 里程碑 | 目标周 | 判定标准 | ✓ |
|--------|:------:|---------|:-:|
| M1: Pipeline 跑通 | W3 | 小数据上 Phase 0→1→2→3→eval 无 crash | ☐ |
| M2: Phase 1 收敛 | W6 | LoCoMo F1 ≥ Memory-R1 baseline | ☐ |
| M3: Phase 2 退火成功 | W8 | α 退至 0，Kendall τ ≥ 0.5 | ☐ |
| M4: 主表完成 | W12 | Table 1 + Table 2 全部数字填入 | ☐ |
| M5: 消融完成 | W15 | Table 3 全部数字 + A1 证实 Reward Hacking | ☐ |
| M6: 分析图完成 | W18 | 6 张核心图 + Case Study | ☐ |
| M7: 初稿完成 | W22 | 完整论文可读 | ☐ |
| M8: 内审通过 | W24 | 导师 + 同学审稿意见回收 + 修改完毕 | ☐ |
| M9: 最终投递 | W28 | OpenReview 提交成功 | ☐ |

---

> **最后提醒**：这个项目的学术"杀手锏"是 **消融 A1**。如果能令人信服地证明"移除环境锚后，纯自奖励在长周期中因 Reward Hacking 而性能劣化"，这一张表（配合 Reward Drift 曲线图）就够说服任何审稿人。请在实验设计中给 A1 最高优先级和最完整的数据记录。

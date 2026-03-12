<div align="center">

<h1>

✨ OpenAgentQG：走向开放世界的知识图谱问句生成 ✨

</h1>

<h3>—————— 审稿中 / Under Review ——————</h3>

</div>


<div align="center">

[![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/eduzrh/OpenAgentQG)
[![Language: Python 3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![Made with PyTorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/eduzrh/OpenAgentQG/issues)

[English](README.md) | [简体中文](./README_zh_CN.md)

</div>


<p align="center">
  <a href="#-架构"><b>🏗️ 架构</b></a> |
  <a href="#-安装"><b>⚙️ 安装</b></a> |
  <a href="#-快速上手"><b>🚀 快速上手</b></a> <br>
  <a href="#-数据集"><b>📦 数据集</b></a> |
  <a href="#-使用说明"><b>📖 使用说明</b></a> |
  <a href="#-可复现性"><b>🔬 可复现性</b></a> |
  <a href="#-许可证"><b>📜 许可证</b></a> |
  <a href="#-联系作者"><b>📬 联系作者</b></a> |
  <a href="#-技术报告technical-report"><b>📄 技术报告（Technical Report）</b></a>
</p>

---

## 📰 最新进展

<div align="center">

| 🆕 更新 | 📅 日期 | 📝 描述 |
|:---:|:---:|:---|
| 📄 **TKDE 投稿** | 2026 年 3 月 | 论文已投稿至 IEEE TKDE。根据期刊政策，我们会在投稿后 **7 天内** 更新仓库链接并释出整理后的代码版本。 |

</div>

---

## 🔥 关键特性

<div align="center">

| 特性 | 图标 | 描述 |
|:---|:---:|:---|
| **神经–符号知识融合（Neuro-Symbolic Knowledge Fusion）** | 🔗 | 将大模型的参数知识与不完整 KG 进行融合，构建统一的融合图，从而缓解缺失事实带来的影响 |
| **自适应智能体协作（Adaptive Agentic Collaboration）** | 🤝 | 多智能体按需协作，根据子图动态选择角色进行问句与答案生成 |
| **轻量级自适应通信（Lightweight Adaptive Communication）** | 📡 | 通过结构化接口与高效消息传递，让智能体之间协作成本可控 |
| **零样本 & 资源友好（Zero-Shot & Resource-Efficient）** | ⚡ | 不依赖标注 QA 数据；每对 QA 消耗的 token 和时间较少（如在 WebQuestions-IncKG 上约 1,394 tokens、10.7 秒） |
| **开放世界场景下的 SOTA 表现** | 📈 | 相比 21 个基线在开放世界 KGQG 上最高可提升 30.35%；在更低成本下接近全监督 SOTA 水平 |

</div>

---

## 🏗️ 架构

OpenAgentQG 主要包含两个阶段：

**阶段 1：神经–符号知识融合（Neuro-Symbolic Knowledge Fusion）**  
将基础模型（LLM）的参数知识与不完整 KG 进行融合，构建更丰富的融合图，降低缺失关系与缺失实体的影响。

**阶段 2：自适应智能体协同生成（Adaptive Agentic Collaborative Generation）**  
多个角色（如 Editor-in-Chief、Managing Editor、Contributor、Content Editor、Copy Editor 等）按需协作：  
- 动态决定每个子图需要参与的角色  
- 生成并多轮改写问句  
- 进行质量评估，只保留高质量 QA 对

**通信机制**  
各个智能体通过结构化接口和轻量级消息传递进行交互，在保证协作效果的同时控制通信开销。

---

## ⚙️ 安装

### 📋 环境依赖

```bash
pip install -r requirements.txt
```

### 📦 主要依赖

| 包 | 作用 |
|:---|:---|
| 🐍 **Python** | >= 3.7（在 3.8+ 上测试） |
| 🔥 **PyTorch** | 深度学习框架 |
| 🤗 **OpenAI API / LLM** | 默认使用 GPT-4（`gpt-4-0125-preview`）作为智能体 LLM，可替换为其他基础模型 |
| 📊 **NumPy, Pandas** | 数据处理 |
| ⏳ **Tqdm** | 进度条与日志展示 |

> 💡 **说明：** 完整的基线配置与实验细节请参考技术报告与源码。

---

## 📦 数据集

我们在六种公开的 KGQG 场景上进行评估：

- **WebQuestions**（整合 WebQuestionsSP 与 ComplexWebQuestions）：使用 [mhqg](https://github.com/liyuanfang/mhqg) 提供的数据构造  
- **PathQuestions**：使用 [IRN](https://github.com/zmtkeke/IRN) 提供的数据构造  
- **IncKG / Text 变体**：关于构造方式与下载链接，请参考 [OpenAgentQG 仓库说明](https://github.com/eduzrh/OpenAgentQG)

---

## 🚀 快速上手

### Step 1：克隆仓库 📥

```bash
git clone https://github.com/eduzrh/OpenAgentQG.git
cd OpenAgentQG
```

### Step 2：安装依赖并设置 API Key 📦

```bash
pip install -r requirements.txt
# 设置 OpenAI API key（或你自己的 LLM 推理服务）用于智能体调用
```

### Step 3：准备数据 📂

将对应数据集（如 WebQuestions-IncKG、PathQuestions-IncKG）的文件放置到本仓库预期的目录结构下。  
具体格式与路径请参考本仓库文档与技术报告。

### Step 4：运行 OpenAgentQG ▶️

在目标数据集上运行两阶段流水线（神经–符号融合 + 多智能体协作生成）。例如：

```bash
# 示例（请根据你本地脚本名和路径进行调整）
python run_openagentqg.py --dataset WebQuestions-IncKG --split test
```

### Step 5：评估 📊

| 指标 | 描述 |
|:---|:---|
| **BLEU-4** | 问句生成的 N-gram 重叠度 |
| **ROUGE-L** | 以最长公共子序列衡量文本相似度；Overall = avg(BLEU-4, ROUGE-L) |
| **Hits@1** | 下游 KGQA 模型评估指标（Top-1 命中率） |

---

## 📖 使用说明

### 基本用法

运行完整的开放世界 KGQG 流水线（零样本设置）：

- **输入（Input）**：不完整 KG 子图（subgraph）与目标答案实体（entities）  
- **输出（Output）**：生成的自然语言问句及对应 QA 对

### 关键参数

| 参数 | 描述 |
|:---|:---|
| **LLM** | 用于各角色智能体的基础模型，默认 GPT-4，可替换为其他兼容的 LLM |
| **Quality threshold** | 接受生成 QA 对时的质量阈值，可通过命令行或配置文件调整 |

### 下游 KGQA

生成的 QA 对可用于数据增强。我们在多种 KGQA 模型（如 TransferNet、GRAFT-Net、EmbedKGQA）上验证了 OpenAgentQG 的有效性，表明合成 QA 对能够提升下游问答性能。

---

## 🔬 可复现性

- **运行环境**：基于 PyTorch，详细依赖见 `requirements.txt` 与技术报告。  
- **随机种子与配置**：配置项在代码与技术报告中给出，包含关键超参数与运行模式。  
- **对比基线**：与 21 种配置进行系统对比，覆盖 LLM 零样本方法、Chain-of-Thought、Multi-Agent 框架以及全监督 KGQG 模型等。

---

## 📊 评估指标

- **BLEU-4** 与 **ROUGE-L**：KGQG 任务中常用的文本生成指标。  
- **Overall**：综合指标，定义为 \((\text{BLEU-4} + \text{ROUGE-L}) / 2\)。  
- **Hits@1**：用于下游 KGQA 的 Top-1 准确率。

---

## 📜 许可证

[MIT License](LICENSE) — 详情见仓库中的许可证文件。

---

## 📬 联系作者

- **Email**： [runhaozhao@nudt.edu.cn](mailto:runhaozhao@nudt.edu.cn)  
- **GitHub Issues**：在 [OpenAgentQG Issues](https://github.com/eduzrh/OpenAgentQG/issues) 提交，建议使用 `bug`、`enhancement`、`question` 等标签。

---

## 📄 技术报告（Technical Report）

本节对实现细节、Prompt 模板、关键超参数、附加消融实验以及扩展分析做一个简要说明，完整形式与推导可以参考论文与源码。

### 1. 实现细节（Implementation Details）

OpenAgentQG 实现为一个两阶段流水线：首先在不完整 KG 上执行神经–符号知识融合，然后运行自适应多智能体生成过程得到问答对。代码中将数据加载、图构建、智能体编排与评估拆分为相对独立的模块（如融合模块、角色模块、质量评估模块等），便于单独运行、调试和扩展。更具体的实现（类结构、数据结构、处理流程）可直接参考源码，例如 `agentic/agents.py`、数据处理脚本以及运行脚本。

### 2. Prompt 与模板（Prompts and Templates）

我们为不同角色和子任务设计了模块化的 Prompt 与模板，包括知识聚合、角色协调、问句生成、答案校验和质量评估等。每个角色都有相应的接口模板（输入字段、必需输出字段、风格约束等），同时还设置了系统级与任务级指令，用于控制如：对 KG 的忠实性、问题多样性、对低置信度样本的拒绝等行为。完整的 Prompt 与模板（以及用于消融的小变体）可以在构造 LLM 调用的相关代码中直接查看。

### 3. 超参数与配置（Hyperparameters and Configuration）

为了保持整体流程在 token 与时间上的高效性，我们选取了较为轻量的超参数设置：例如为每次智能体调用设置适中的上下文窗口，使用较小的 beam size 或较低的 temperature / top-p 采样，并设置质量阈值过滤低分问答对。融合迭代次数、子图最大规模以及各数据集的生成预算则通过配置文件和命令行参数进行控制，方便在不同数据集之间扩展与迁移。完整、可复现的超参数列表与默认配置可在仓库中的配置文件与脚本中找到。

### 4. 附加实验与消融（Additional Experiments and Ablations）

我们在 WebQuestions-IncKG 与 PathQuestions-IncKG 上分别对两个阶段进行消融（w/o Stage 1：去掉神经–符号融合；w/o Stage 2：去掉自适应多智能体协作生成），在 BLEU-4 / ROUGE-L 上都观察到相比完整 OpenAgentQG 明显的性能下降；完整的数值表见技术报告。这里简要总结在敏感性分析中变化的关键超参数，这些参数都在当前代码中有明确实现。

| **超参数** | **符号** | **默认值（代码）** | **作用** |
|:---|:---:|:---:|:---|
| 熵阈值（Entropy threshold） | τ | 0.5 | 在神经–符号融合中，当 \(\Delta H \le \tau\) 时接受 meta-neural 虚节点，见 `fusion/neuro_symbolic.py`。 |
| QA 质量阈值（Quality threshold） | – | 4 | 在智能体阶段仅保留自动评分 ≥ 该阈值的 QA 对，见 `run_full_pipeline.py`。 |
| 最大 MDP 迭代轮数 | – | 3 | 限制 Editor–Contributor 等角色之间的多轮回调次数，见 `config.py`。 |
| Meta-neural Top-K | – | 3 | 每个子图最多加入的虚节点数量，见 `config.py`。 |

在我们的实验中，将 τ 在 0.2–0.8 范围内、质量阈值在 3–5 范围内做网格搜索时，BLEU-4 / ROUGE-L 一般会保持在相对稳定的区间（例如在 WebQuestions-IncKG 上，BLEU-4 通常处于二十几到三十出头范围），而表中给出的默认值位于该稳定区间中部。整体趋势与技术报告以及 `config.py` 中的注释是一致的，我们也刻意避免对超参数做过度调参。

### 5. 扩展分析与案例（Extended Analysis and Discussions）

**扩展定性案例分析。** 为了更直观地展示两个阶段的作用，我们构造了一个 PathQuestions 风格的多跳推理示例，其中在不完整 KG 中刻意删除了一条关键关系。该示例遵循与 `mhqg-pq` 数据相同的图模式与字段设计，但用于说明算法行为，并非直接取自数据文件。  
在该示例中，不完整图中包含事实：*(Marie Curie, won, Nobel Prize in Physics)*、*(Pierre Curie, won, Nobel Prize in Physics)*、*(Pierre Curie, profession, Physicist)*，但缺失了关系 *(Marie Curie, spouse, Pierre Curie)*。答案为 {Pierre Curie}。

| 设置 | 示例 / 分析 |
|:---|:---|
| **Ground Truth** | Who is the physicist spouse of Marie Curie who also shared the Nobel Prize in Physics? |
| **w/o Stage 1** | Who won the Nobel Prize in Physics along with Marie Curie? *(无法恢复“spouse”关系，仅捕捉到共现信息；缺少神经–符号融合与参数知识补全。)* |
| **w/o Stage 2** | Who is Pierre Curie to Marie Curie and won the Nobel Prize? *(能够借助参数知识恢复缺失关系，但容易出现答案泄露与语法不够自然的问题，因为缺少多智能体协作式的改写与审校。)* |
| **OpenAgentQG (Full)** | **Which physicist shared the Nobel Prize in Physics with their spouse, Marie Curie?** *(通过 meta-neural 虚节点成功恢复缺失关系，并在多智能体协作下生成既自然又不泄露答案的高质量问句。)* |

可以看到：去掉 Stage 1 时，模型无法在图上恢复缺失的 “spouse” 关系，只能生成较为粗糙的共现问题；去掉 Stage 2 时，即使通过参数知识恢复了隐含关系，生成的问句也容易出现答案泄露或表达不自然的问题。完整的 OpenAgentQG 则通过 “图补全 + 多智能体协作改写” 同时解决这两类问题。更多案例、失败分析与细节可以在技术报告及本仓库的评估脚本中找到。

---

## 🔗 参考文献与致谢

本工作在以下先前成果的基础上进行了扩展与改进：

- **CIKM 2024**：*Zero-shot Knowledge Graph Question Generation via Multi-Agent LLMs and Small Models Synthesis* — [CIKM 2024 论文链接](https://doi.org/10.1145/3637521.3664594)。

我们感谢以下数据集与相关工作：

- **数据集**：  
  - [WebQuestionsSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763) 与 [ComplexWebQuestions](https://www.tau-nlp.org/compwebq)（用于 WebQuestions 场景）；  
  - [PathQuestions](https://github.com/zmtkeke/IRN) 及相关 KGQG 资源（如 [mhqg](https://github.com/liyuanfang/mhqg)、[IRN](https://github.com/zmtkeke/IRN)）。  
- **KGQG 与 QA 方法**：G2S-AE（Chen et al.）、R2DQG（Ren et al., IJCAI 2025）、KQG-COT+（Liang et al., EMNLP 2023）、SGSH（Guo et al., NAACL 2024）以及 L2A、零样本与模板式 KGQG 系列工作等。  
- **多智能体与 LLM 相关工作**：MetaGPT、CAMEL、Chain-of-Thought 与 Self-Consistency 提示方法以及相关的多智能体 LLM 框架。

同时，我们也感谢下游评估中使用的 KGQA 模型（TransferNet、GRAFT-Net、EmbedKGQA）以及开放知识库（Wikidata、YAGO）社区。


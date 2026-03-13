<div align="center">

<h1>

✨ OpenAgentQG: Towards Knowledge Graph Question Generation in the Open World ✨

</h1>



<h3>—————— Under Review——————</h3>

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
  <a href="#-architecture"><b>🏗️ Architecture</b></a> |
  <a href="#-installation"><b>⚙️ Installation</b></a> |
  <a href="#-quick-start"><b>🚀 Quick Start</b></a> <br>
  <a href="#-datasets"><b>📦 Datasets</b></a> |
  <a href="#-usage"><b>📖 Usage</b></a> |
  <a href="#-reproducibility"><b>🔬 Reproducibility</b></a> |
  <a href="#-license"><b>📜 License</b></a> |
  <a href="#-contact"><b>📬 Contact</b></a> |
  <a href="#-technical-report"><b>📄 Technical Report</b></a>
</p>



---

## 📰 Latest News

<div align="center">

| 🆕 Updates | 📅 Date | 📝 Description |
|:---:|:---:|:---|
| 📄 **TKDE Submission** | Mar 2026 | Manuscript submitted to IEEE TKDE. A clean code version will be released and the repository link will be updated within **7 days after submission** in compliance with the policy. |
</div>


## 🔥 Key Features

<div align="center">

| Feature | Icon | Description |
|:---|:---:|:---|
| **Neuro-Symbolic Knowledge Fusion** | 🔗 | Enriches incomplete KGs with knowledge from foundation models into a unified fusion graph |
| **Adaptive Agentic Collaboration** | 🤝 | Multi-agent collaboration with on-demand role selection for QA generation |
| **Lightweight Adaptive Communication** | 📡 | Structured interfaces and efficient message passing across agents |
| **Zero-Shot & Resource-Efficient** | ⚡ | No labeled QA data required; minimal tokens and time per QA pair (e.g., 1,394 tokens, 10.7s on WebQuestions-IncKG) |
| **State-of-the-Art in Open World** | 📈 | Up to 30.35% relative improvement over 21 baselines; approaches full-training SOTA with lower cost |

</div>

---

## 🏗️ Architecture

OpenAgentQG has two main stages:

**Stage 1: Neuro-Symbolic Knowledge Fusion**  
Fuses knowledge from foundation models with the incomplete KG to build a richer graph, reducing the impact of missing facts.

**Stage 2: Adaptive Agentic Collaborative Generation**  
Multiple agents (e.g., Editor-in-Chief, Managing Editor, Contributor, Content Editor, Copy Editor) collaborate on demand: they decide who participates per subgraph, generate and refine questions, and assess quality so that only good QA pairs are kept.

**Communication**  
Agents use structured interfaces and efficient message passing to coordinate without heavy overhead.

---

## ⚙️ Installation

### 📋 Prerequisites

```bash
pip install -r requirements.txt
```

### 📦 Main Dependencies

| Package | Purpose |
|:---|:---|
| 🐍 **Python** | >= 3.7 (tested on 3.8+) |
| 🔥 **PyTorch** | Deep learning framework |
| 🤗 **OpenAI API / LLM** | Default: GPT-4 (gpt-4-0125-preview) for agents |
| 📊 **NumPy, Pandas** | Data handling |
| ⏳ **Tqdm** | Progress bars |

> 💡 **Note:** See the technical report and code for baseline setup and full details.

---

## 📦 Datasets

We evaluate on six public KGQG settings:


- **WebQuestions** (from WebQuestionsSP & ComplexWebQuestions)
- **PathQuestions**
- **IncKG / Text variants**: See [OpenAgentQG](https://github.com/eduzrh/OpenAgentQG/data) for construction and links.

---

## 🚀 Quick Start

### Step 1: Clone the Repository 📥

```bash
git clone https://github.com/eduzrh/OpenAgentQG.git
cd OpenAgentQG
```

### Step 2: Install Dependencies & Set API Keys 📦

```bash
pip install -r requirements.txt
# Set OpenAI API key (or your LLM endpoint) for agent calls
```

### Step 3: Prepare Data 📂

Place dataset files (e.g., WebQuestions-IncKG, PathQuestions-IncKG) under the expected structure. See the repository and technical report for exact formats.

### Step 4: Run OpenAgentQG ▶️

Run the two-stage pipeline (neuro-symbolic fusion + agentic collaborative generation) on your target dataset. Example:

```bash
# Example (adjust script name and paths to your codebase)
python run_openagentqg.py --dataset WebQuestions-IncKG --split test
```

### Step 5: Evaluate 📊

| Metric | Description |
|:---|:---|
| **BLEU-4** | N-gram overlap for question generation |
| **ROUGE-L** | Longest common subsequence; Overall = avg(BLEU-4, ROUGE-L) |
| **Hits@1** | Downstream KGQA evaluation |

---

## 📖 Usage

### Basic Usage

Run the full pipeline for open-world KGQG (zero-shot):

- **Input**: Incomplete KG subgraph(s) and target answer entity(ies).  
- **Output**: Generated natural language questions and QA pairs.

### Key Parameters

| Parameter | Description |
|:---|:---|
| **LLM** | Default: GPT-4; configurable for other foundation models |
| **Quality threshold** | Configurable bar for accepting generated QA pairs |

### Downstream KGQA

Generated QA pairs can be used for data augmentation. OpenAgentQG has been shown to improve KGQA models (e.g., TransferNet, GRAFT-Net, EmbedKGQA) when used for augmentation.

---

## 🔬 Reproducibility

- **Environment**: PyTorch; details in the technical report.  
- **Seeds & configs**: Documented in code and technical report.  
- **Baselines**: Compared against 21 configurations (LLM-based zero-shot, chain-of-thought, multi-agent, and full-training KGQG methods).

---

## 📊 Evaluation Metrics

- **BLEU-4**, **ROUGE-L**: Standard text generation metrics for KGQG.  
- **Overall**: Average of BLEU-4 and ROUGE-L.  
- **Hits@1**: For downstream KGQA evaluation.

---

## 📜 License

[MIT License](LICENSE) — see repository for details.

---

## 📬 Contact

- **Email**: [runhaozhao@nudt.edu.cn](mailto:runhaozhao@nudt.edu.cn)
- **GitHub Issues**: [OpenAgentQG](https://github.com/eduzrh/OpenAgentQG/issues) — labels: `bug`, `enhancement`, `question`.

---

## 📄 Technical Report

### 1. Implementation Details

OpenAgentQG is implemented as a two-stage pipeline that first performs neuro-symbolic knowledge fusion over incomplete KGs and then runs an adaptive multi-agent generation process for question–answer pairs. The codebase organizes data loading, graph construction, agent orchestration, and evaluation into modular modules (e.g., fusion, agent roles, quality assessment) so that each stage can be run, profiled, and extended independently. For concrete implementation details such as class layouts, data schemas, and processing flows, please refer to the source code (e.g., `agentic/agents.py`, data processors, and runner scripts).

### 2. Prompts and Templates

We design modular prompts for different agent roles and sub-tasks, including knowledge aggregation, role coordination, question generation, answer verification, and quality assessment. Role-specific templates define the interface of each agent (e.g., input schema, required outputs, and style constraints), while system- and task-level instructions control behavior such as faithfulness to the KG, diversity of questions, and rejection of low-confidence generations. The full set of prompts and templates, along with their minor variants used in ablations, can be found directly in the code where LLM calls are constructed.

### 3. Hyperparameters and Configuration

We adopt lightweight hyperparameters so that the pipeline is token- and time-efficient: for example, we fix a moderate context window per agent call, use small beam sizes (or low sampling temperature/top-p) for generation, and set a quality threshold to filter out low-scoring QA pairs. The number of fusion iterations, maximum subgraph size, and per-dataset generation budgets are controlled by configuration files and command-line arguments, enabling easy scaling across datasets. A complete, reproducible list of hyperparameters and default settings is documented in the config files and scripts in this repository.

### 4. Additional Experiments and Ablations

We ablate the two main stages (w/o Stage 1: no Neuro-symbolic Unified Knowledge Fusion; w/o Stage 2: no Adaptive Agentic Fusion Graph Collaborative Generation) on both WebQuestions-IncKG and PathQuestions-IncKG, and consistently observe performance drops in BLEU-4 / ROUGE-L compared with the full OpenAgentQG pipeline; full numerical tables are reported in the paper. Here we briefly summarize the key hyperparameters that are varied in our sensitivity study and are explicitly implemented in this codebase.

| **Hyperparameter** | **Symbol** | **Default (code)** | **Role** |
|:---|:---:|:---:|:---|
| Entropy threshold | τ | 0.5 | Accept meta-neural virtual nodes when ΔH ≤ τ in neuro-symbolic fusion (`fusion/neuro_symbolic.py`). |
| QA quality threshold | – | 4 | Keep generated QA pairs whose automatic score ≥ this value in the agentic stage (`run_full_pipeline.py`). |
| Max MDP iterations | – | 3 | Upper bound on Editor–Contributor refinement rounds in Stage 2 (`config.py`). |
| Meta-neural top-k | – | 3 | Max number of virtual nodes added per subgraph (`config.py`). |

Empirically, sweeping τ within a moderate range (e.g., 0.2–0.8) and the QA quality threshold between 3 and 5 keeps BLEU-4 / ROUGE-L in a relatively narrow band (typically mid-20s to low-30s BLEU-4 on WQ-IncKG in our runs), with the defaults above lying near the centre of this stable region. These trends are consistent with the overall performance ranges described in the paper and the comments in `config.py`, while avoiding heavy hyperparameter tuning.

### 5. Extended Analysis and Discussions

**Extended qualitative case analysis.** To illustrate the effect of the two stages, we compare the full OpenAgentQG pipeline with w/o Stage 1 and w/o Stage 2 on a stylized multi-hop example with a **missing relation** in the incomplete KG.  Answer: {Pierre Curie}.

| Setting | Example / Analysis |
|:---|:---|
| **Ground Truth** | Who is the physicist spouse of Marie Curie who also shared the Nobel Prize in Physics? |
| **w/o Stage 1** | Who won the Nobel Prize in Physics along with Marie Curie? *(Fails to infer "spouse"; no parametric fusion.)* |
| **w/o Stage 2** | Who is Pierre Curie to Marie Curie and won the Nobel Prize? *(Infers the relation but has answer leakage and weak grammar without agentic refinement.)* |
| **OpenAgentQG (Full)** | **Which physicist shared the Nobel Prize in Physics with their spouse, Marie Curie?** *(Recovers the missing relation via meta-neural nodes and refines the question without leaking the answer.)* |

Without Stage 1, the model cannot fill the missing "spouse" link and produces a generic co-occurrence question. Without Stage 2, parametric knowledge can restore the relation but the single-shot output leaks the answer and is unpolished. The full framework both completes the graph (Stage 1) and refines the question via multi-agent collaboration (Stage 2).

---

## 🔗 References & Acknowledgments

This work extends our prior conference paper:

- **CIKM 2024**: *Zero-shot Knowledge Graph Question Generation via Multi-Agent LLMs and Small Models Synthesis* — [CIKM 2024](https://doi.org/10.1145/3637521.3664594).

We thank the following datasets and prior work that we build upon or compare with:

- **Datasets**: [WebQuestionsSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763) and [ComplexWebQuestions](https://www.tau-nlp.org/compwebq) for WebQuestions; [PathQuestions](https://github.com/zmtkeke/IRN) and related KGQG resources ([mhqg](https://github.com/liyuanfang/mhqg), [IRN](https://github.com/zmtkeke/IRN)).
- **KGQG & QA**: G2S-AE (Chen et al.), R2DQG (Ren et al., IJCAI 2025), KQG-COT+ (Liang et al., EMNLP 2023), SGSH (Guo et al., NAACL 2024), and the L2A, Zero-shot, and template-based KGQG lines of work.
- **Multi-agent & LLMs**: MetaGPT, CAMEL, Chain-of-Thought and Self-Consistency prompting, and related LLM-based collaboration frameworks.

We also thank the KGQA methods used in downstream evaluation (TransferNet, GRAFT-Net, EmbedKGQA) and the open knowledge bases (Wikidata, YAGO).



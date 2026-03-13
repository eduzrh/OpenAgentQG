# Dataset Description & Download (Datasets)


Data files are **not** included in the release. Please download them yourself and place them under the expected directories. This repo evaluates on **6** KGQG settings: WebQuestions, PathQuestions, **WebQuestions-IncKG**, **PathQuestions-IncKG**, **WebQuestions-Text**, and **PathQuestions-Text**.

---

## 1. Overview of the 6 datasets

| Dataset | Description | Directory / usage in this repo |
|--------|-------------|--------------------------------|
| **WebQuestions** | Based on WebQuestionsSP + ComplexWebQuestions. For downloading the datasets, see (https://github.com/hugochan/Graph2Seq-for-KGQG), (https://github.com/liyuanfang/mhqg) | `data/` |
| **PathQuestions** | Based on PathQuestions. For downloading the datasets, see (https://github.com/hugochan/Graph2Seq-for-KGQG), (https://github.com/liyuanfang/mhqg) | `data/` |
| **WebQuestions-IncKG (WQ-IncKG)** | Incomplete KG: **about 50% of triples are randomly removed** from the WebQuestions-related KG; the construction follows the IncKG setup in works such as TransferNet, GRAFT-Net, and EmbedKGQA | `data/` |
| **PathQuestions-IncKG (PQ-IncKG)** | Incomplete KG: **about 50% of triples are randomly removed** from the PathQuestions-related KG; the same processing as above | `data/` |
| **WebQuestions-Text** | Text/QA variant of WebQuestions; data processing follows the setups in TransferNet, GRAFT-Net, EmbedKGQA, etc. | `data/` (see references and data sources below) |
| **PathQuestions-Text** | Text/QA variant of PathQuestions; data processing also follows the above works | `data/` (see references and data sources below) |

For **WQ-IncKG** and **PQ-IncKG**, the “randomly removing 50% of KG triples” construction follows the incomplete-KG setups in TransferNet, GRAFT-Net, EmbedKGQA, etc. Please refer to their papers and dataset descriptions for details.  
For **WebQuestions-Text** and **PathQuestions-Text**, the splits and formats also follow the public datasets and processing pipelines in those works.

---

## 2. Download & Placement

### 1. METEOR data (for evaluation)

To compute METEOR, you need to download the paraphrase data and place it under the evaluation code directory:

- **Download**: [paraphrase-en.gz](https://github.com/xinyadu/nqg/blob/master/qgevalcap/meteor/data/paraphrase-en.gz) (from [nqg](https://github.com/xinyadu/nqg))
- **Path**: After extraction, put the files under **`src/core/evaluation/meteor/data`** (if you use this repo’s evaluation module, that corresponds to `core/evaluation/meteor/data`; if you use an external Graph2Seq repo, it corresponds to its `vendor/`)

### 2. GloVe word embeddings

- **Download**: [glove.840B.300d.zip](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip)
- **Place**: After extraction, put **`glove.840B.300d.txt`** into this repo’s **`data`** directory (or the `data` directory used by your QG model config)

### 3. Main data (WebQuestions / PathQuestions, etc.)

- **Download**: Get the data package from [here](https://1drv.ms/u/s!AjiSpuwVTt09gVsFilSx0NpJlid-?e=1TKqfG) (same as in [Graph2Seq-for-KGQG](https://github.com/hugochan/Graph2Seq-for-KGQG))
- **Place**: Put the extracted data under this repo’s **`data`** directory, keeping the same subdirectory structure as `mhqg-wq`, `mhqg-pq`, etc.

---

## 4. References

- **IncKG (incomplete KG) and Text variants**: The data processing and the “50% random triple removal” setup follow the public datasets and descriptions in TransferNet, GRAFT-Net, EmbedKGQA, etc.
- **Graph2Seq**: Chen, Yu, Lingfei Wu, and Mohammed J. Zaki. *“Toward Subgraph Guided Knowledge Graph Question Generation with Graph Neural Networks.”* Code and data: [Graph2Seq-for-KGQG](https://github.com/hugochan/Graph2Seq-for-KGQG).
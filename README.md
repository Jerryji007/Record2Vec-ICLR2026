# Record2Vec-ICLR2026

## Can we generate portable representations for clinical time series data using LLMs? (ICLR 2026)

<br>
<p align="center">
<a href="https://arxiv.org/abs/2603.23987"><img src="https://img.shields.io/badge/arXiv-2603.23987-b31b1b.svg?logo=arxiv&logoColor=red" alt="Record2Vec Paper on arXiv"/></a>

## News

- **[Apr 23, 2026]** We will present our poster at Rio Convention Centre!
- 🎉 **[Jan 26, 2026]** Our paper has been accepted to ICLR 2026.

### Transfer Learning Unified

A unified CLI for training and evaluating transfer-learning baselines and representation-driven models on **clinical time-series tasks**.

---

#### Quick Start

You can quickly get started by viewing the help for the main script and its subcommands:

```bash
python transfer_learning_unified.py --help
python transfer_learning_unified.py train --help
python transfer_learning_unified.py test --help
```

---

## Usage

The general command structure is as follows:

```bash
python transfer_learning_unified.py <train|test> [ARGS...]
```

---

#### Examples

1.  **Train — LLM summarization → embedding → MLP for mortality**

<!-- end list -->

```bash
python transfer_learning_unified.py train \
  --pipeline llm --use_sum --sum_model medgemma-27b-text-it \
  --embedding_model Qwen3-Embedding-8B \
  --model_type mlp --task mort
```

2.  **Test — Baseline reps with MLP on forecasting + 16-shot finetune**

<!-- end list -->

```bash
python transfer_learning_unified.py test \
  --pipeline baseline --reps mean,right,interp \
  --model_type mlp --task forecast --finetune --finetune_size 16
```

3.  **Train — TFM embeddings with LSTM on forecasting**

<!-- end list -->

```bash
python transfer_learning_unified.py train \
  --pipeline embedding --embedding_model TFM \
  --model_type lstm --task forecast
```

4.  **Test — TSDE embeddings with LSTM on lab event prediction (temporal)**

<!-- end list -->

```bash
python transfer_learning_unified.py test \
  --pipeline embedding --embedding_model TSDE \
  --model_type lstm --task lab --use_temporal
```

---

#### Key Arguments

#### Core

| Argument                                                              | Description                                     |
| :-------------------------------------------------------------------- | :---------------------------------------------- |
| `train` \| `test`                                                     | Main subcommands for running experiments.       |
| `--pipeline {baseline \| embedding \| llm}`                           | The processing **pipeline** to use.             |
| `--task {mort \| forecast \| lab \| gender \| age \| feature \| los}` | The **prediction target** task.                 |
| `--model_type {mlp \| lstm \| timemixer \| patchtsmixer}`             | The downstream **model family** for prediction. |

#### Representations & Inputs

| Argument                   | Description                                                                                |
| :------------------------- | :----------------------------------------------------------------------------------------- |
| `--reps <comma-list>`      | A comma-separated list of **baseline representations** (e.g., `mean,right,interp`).        |
| `--embedding_model <NAME>` | The name of a pre-trained **embedding model** (e.g., `TFM`, `TSDE`, `Qwen3-Embedding-8B`). |
| `--use_temporal`           | Enable **temporal modeling** where supported by the task and model.                        |

#### LLM Summarization (`pipeline=llm`)

| Argument             | Description                                                              |
| :------------------- | :----------------------------------------------------------------------- |
| `--use_sum`          | Enable the **summarization step** for LLM pipelines.                     |
| `--sum_model <NAME>` | The **LLM model** used for summarization (e.g., `medgemma-27b-text-it`). |

#### Evaluation / Fine-tuning

| Argument              | Description                                                  |
| :-------------------- | :----------------------------------------------------------- |
| `--finetune`          | Enable **few-shot fine-tuning** during the `test` phase.     |
| `--finetune_size <N>` | The number of samples for few-shot fine-tuning (e.g., `16`). |

---

#### Tips

- Start with **baselines** (`--pipeline baseline --reps mean,right,interp`) to sanity-check your data and metrics before trying complex models.
- When comparing different embeddings, keep the **downstream model** (`--model_type`) fixed for a fair comparison of the learned representations.
- For robust results, use **consistent random seeds** and report the mean $\pm$ std across multiple runs.

<!-- end list -->

## Citing Record2Vec

If you use this code or find our paper useful in your research, please consider cite our paper:

```bib
@article{ji2026can,
  title={Can we generate portable representations for clinical time series data using LLMs?},
  author={Ji, Zongliang and Sun, Yifei and Amaral, Andre and Goldenberg, Anna and Krishnan, Rahul G},
  journal={arXiv preprint arXiv:2603.23987},
  year={2026}
}
```

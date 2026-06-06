# LLM Eval Lab – PolicyGPT

A **RAG evaluation harness** and dashboard for benchmarking PolicyGPT across retrieval strategies, models, and prompts, with automatic batch evaluation and observability for **quality, latency, and cost**.

> I built this lab so I could stop guessing which RAG configs were better for my PolicyGPT app and measure retrieval/model tradeoffs in minutes instead of days.

---

## Case study at a glance

- **Problem**  
  I was tuning PolicyGPT’s RAG backend (dense vs hybrid, `top_k`, prompts, models) mostly by feel and a few cherry‑picked questions, with no reliable way to see if a change actually helped.

- **What I built**  
  A config‑driven evaluation harness and Streamlit dashboard that runs PolicyGPT against golden QA and summarization sets, logs metrics per example (recall / faithfulness / correctness / BLEU / ROUGE‑L / latency / cost), and lets me compare runs side by side.

- **Impact**  
  On a 50‑question PolicyGPT QA eval set, hybrid retrieval (BM25 + dense) improved context recall from **0.64 → 0.82** and faithfulness from **0.71 → 0.84**, at about **+120 ms** median latency per query.  
  That evidence is what I used to choose the default config in the main PolicyGPT app.

---

## TL;DR

- Benchmarks different PolicyGPT RAG configs (retrieval + model) on fixed golden QA and summarization datasets.  
- Tracks **recall**, **faithfulness**, **correctness**, **BLEU**, **ROUGE‑L**, **latency**, and **token cost** per example.  
- Streamlit dashboard to compare runs, slice failures, and inspect individual questions or summaries.  
- Designed to plug into any HTTP RAG backend with a small client wrapper.

---

## Overview

LLM Eval Lab is an evaluation and observability layer for my PolicyGPT RAG backend:

- Runs batch evaluations against PolicyGPT’s `/query` API.  
- Compares retrieval strategies (dense vs hybrid BM25 + dense) and different models.  
- Computes:
  - **retrieval recall**  
  - **LLM‑judged faithfulness and correctness** for QA  
  - **BLEU** and **ROUGE‑L** for summarization  
  - **latency**, **tokens**, and **cost** per query  
- Provides a **Streamlit dashboard** to inspect a single run, compare runs side by side, and filter failures.

It’s implemented as a **separate, config‑driven tool** so it can plug into any HTTP RAG backend with minimal changes.

---

## Snapshot of results

On a PolicyGPT QA eval set (50 questions from real policy documents):

- **Dense-only retrieval**  
  - Recall: `0.64`  
  - Faithfulness: `0.71`  
  - Correctness: `0.69`  
  - Latency: `~380 ms` median  

- **Hybrid (BM25 + dense)**  
  - Recall: `0.82`  
  - Faithfulness: `0.84`  
  - Correctness: `0.81`  
  - Latency: `~500 ms` median  

Hybrid retrieval is more accurate and faithful, with roughly **+120 ms** extra latency per query.  
These numbers come from the CSVs in `results/`, produced by the eval runs.

> You can rerun the benchmarks and update these numbers as your datasets and configs evolve.

---

## My role

I designed and built this project end‑to‑end:

- Defined the golden QA and summarization dataset formats and evaluation flow.  
- Implemented the config‑driven batch runner that talks to the PolicyGPT FastAPI backend.  
- Implemented metrics:
  - retrieval recall from gold context snippets  
  - LLM‑judged faithfulness and correctness for QA  
  - BLEU and ROUGE‑L for summarization  
  - latency, token counts, and cost per query  
- Designed the evaluation UX and dashboard:
  - top‑level metric callouts and run comparison  
  - filters for low‑faithfulness / high‑latency slices  
  - detailed per‑example inspection for debugging failures

---

## Features

### Batch evaluation

- YAML configs in `eval_configs/` describe each run:
  - backend URL  
  - retrieval strategy (dense vs hybrid)  
  - `top_k`, weights, etc.  
  - model name  
  - dataset path  
  - task type (`qa` or `summarization`)
- `src/eval_runner.py`:
  - loads the dataset  
  - calls the PolicyGPT `/query` endpoint  
  - computes metrics per example  
  - writes results to `results/<run_name>_<timestamp>.csv`

### Metrics

**QA runs:**

- **Retrieval recall** – `1.0` if any retrieved chunk contains the gold context snippet, else `0.0`.  
- **Faithfulness** – LLM‑judged `[0, 1]` score for how well the answer is grounded in the retrieved context.  
- **Correctness** – LLM‑judged `[0, 1]` score comparing the answer to the ground‑truth answer.  
- **Latency** – client‑side time per `/query` call (ms).  
- **Tokens & cost** – prompt + completion tokens and approximate USD cost per query.

**Summarization runs:**

- **Recall** – optional retrieval recall signal.  
- **BLEU** – SacreBLEU corpus BLEU score.  
- **ROUGE‑L** – ROUGE‑L F1 score.  
- **Latency**, **tokens**, and **cost** per query.

### Dashboard

- Streamlit app in `dashboard/app.py`:
  - Detects task type (QA vs summarization) from the CSV columns.  
  - Shows a primary run with top‑level metric callouts.  
  - Lets you pick a second run for **side‑by‑side comparison**.  
  - Displays total and average cost/tokens per run.  
  - For QA:
    - filters examples by faithfulness range and max latency  
    - shows a per‑example table with metrics and questions  
    - provides a detailed view with question, gold answer, model answer, and metrics  
  - For summarization:
    - shows per‑example rows with source text, gold summary, model summary, BLEU, ROUGE‑L, and latency

This is the view I use to decide whether a new config is worth adopting and to separate **retrieval** failures from **generation** failures.

---

## Project structure

```text
llm-eval-lab/
  eval_configs/
    policygpt_dense.yaml
    policygpt_hybrid.yaml
    policy_summarization_dense.yaml
  data/
    eval_set_policygpt.jsonl
    policy_summaries.csv
  src/
    config.py            # load YAML + env and build RunConfig
    datasets.py          # dataset loaders and dataclasses
    policygpt_client.py  # HTTP client for PolicyGPT backend
    metrics.py           # recall + LLM-judged metrics + BLEU/ROUGE
    eval_runner.py       # orchestrates runs and writes CSV results
  dashboard/
    app.py               # Streamlit dashboard
  results/
  assets/
    overview.png
    comparison.png
    failures.png
  .env.example
  README.md
  pyproject.toml or requirements.txt
```

---

## Setup

### 1. Prerequisites

- Python 3.10+  
- PolicyGPT backend running locally (or remotely)  
- Groq API key (used as the LLM judge and main model)

### 2. Install and configure

```bash
git clone <this-repo-url>
cd llm-eval-lab

python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Using pyproject.toml:
pip install -e .
# or:
# pip install -r requirements.txt
```

Create your `.env`:

```bash
cp .env.example .env
```

Edit `.env`:

```env
POLICYGPT_BASE_URL=http://localhost:8000
GROQ_API_KEY=your_groq_key_here
JUDGE_MODEL=llama-3.3-70b-versatile
RESULTS_DIR=results
```

Make sure your PolicyGPT backend is running:

```bash
cd ../rag-docs-qa/backend
.\.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

---

## Running an evaluation

From `llm-eval-lab`:

```bash
cd llm-eval-lab
.\.venv\Scripts\activate

# Dense-only QA run
python -m src.eval_runner eval_configs/policygpt_dense.yaml

# Hybrid QA run
python -m src.eval_runner eval_configs/policygpt_hybrid.yaml

# Summarization run
python -m src.eval_runner eval_configs/policy_summarization_dense.yaml
```

Each run:

- Loads the configured dataset.  
- Calls PolicyGPT’s `/query` for each example, passing `retrieval_strategy` and `top_k`.  
- Computes the appropriate metrics for the task type.  
- Writes a CSV in `results/`.

---

## Dashboard

Start the Streamlit app:

```bash
cd llm-eval-lab
.\.venv\Scripts\activate

streamlit run dashboard/app.py
```

Open the local URL (e.g. `http://localhost:8501`).

**Dashboard capabilities:**

- Select a primary run and an optional comparison run.  
- See metric callouts (avg recall / faithfulness / correctness / BLEU / ROUGE‑L / latency / cost).  
- Compare two runs on aggregate metrics and cost.  
- For QA:
  - filter examples by faithfulness range and latency threshold  
  - inspect individual examples with question + gold answer + model answer + metrics  
- For summarization:
  - inspect source text, gold summary, model summary, BLEU, ROUGE‑L, and latency per example

---

## Golden datasets

### QA dataset

The QA eval dataset is stored as JSONL in `data/eval_set_policygpt.jsonl`.

Each line is a JSON object:

```jsonl
{"id": "ex1", "question": "What is the company policy on remote work?", "gold_answer": "Employees may work remotely up to three days per week with manager approval.", "gold_context_snippet": "may work remotely up to three days per week", "where": {"source": "employee_handbook"}}
{"id": "ex2", "question": "How many days of paid vacation does a new full-time employee receive?", "gold_answer": "New full-time employees receive 20 days of paid vacation per year.", "gold_context_snippet": "20 days of paid vacation per year", "where": {"source": "benefits_policy"}}
```

Over time, this should grow to ~50–100 examples covering:

- different document sources  
- easy and hard questions  
- known edge cases

### Summarization dataset

The summarization eval dataset is stored in `data/policy_summaries.csv`, with columns for:

- example id  
- source text  
- gold summary  
- metadata used for retrieval and evaluation

---

## Screenshots

(Add actual screenshots from your running app; these filenames are just references.)

### Overview

`assets/overview.png` – top‑level metrics and run selection.

### Run comparison

`assets/comparison.png` – dense vs hybrid comparison on aggregate metrics.

### Failure slice inspection

`assets/failures.png` – low‑faithfulness / high‑latency examples to debug retrieval vs generation issues.

---

## Key findings

From using this lab on PolicyGPT, I learned:

- Hybrid retrieval (BM25 + dense) consistently improved **context recall** and **faithfulness** compared to dense‑only, at a modest latency cost.  
- Some failures are **retrieval failures** (low recall) while others are **generation failures** (good recall but low correctness), which helped separate retrieval tuning from prompt/model changes.  
- Having a golden set and per‑example metrics made subtle regressions visible that wouldn’t show up in a few ad‑hoc tests.  
- Keeping eval code separate and config‑driven makes it easy to add new model/back‑end combinations without touching business logic.

---

## What I’d do next

- Grow the golden datasets and add more task‑specific metrics.  
- Add per‑metric charts over time and wire this into CI to catch regressions per commit.  
- Plug in additional backends (other RAG apps or LLM services) to compare different systems on the same datasets.
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from .config import load_run_config
from .datasets import load_eval_set, load_policy_summarization_dataset
from .metrics import compute_metrics
from .policygpt_client import PolicyGPTClient

load_dotenv()


def run_eval(config_path: str) -> str:
    cfg = load_run_config(config_path)

    if cfg.task_type == "qa":
        dataset = load_eval_set(cfg.eval.dataset_path)
    elif cfg.task_type == "summarization":
        dataset = load_policy_summarization_dataset(cfg.eval.dataset_path)
    else:
        raise ValueError(f"Unknown task_type: {cfg.task_type}")

    client = PolicyGPTClient(base_url=cfg.app.base_url)

    results_dir = os.getenv("RESULTS_DIR", "results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(
        results_dir,
        f"{cfg.name}_{timestamp}.csv",
    )

    if cfg.task_type == "qa":
        fieldnames = [
            "run_name",
            "example_id",
            "question",
            "gold_answer",
            "answer",
            "latency_ms",
            "recall",
            "faithfulness",
            "correctness",
            "retrieval_strategy",
            "top_k",
            "overfetch_factor",
            "w_dense",
            "w_bm25",
            "model_name",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost_usd",
        ]
    else:  # summarization
        fieldnames = [
            "run_name",
            "example_id",
            "source_text",
            "gold_summary",
            "summary",
            "latency_ms",
            "recall",
            "bleu",
            "rouge_l",
            "retrieval_strategy",
            "top_k",
            "overfetch_factor",
            "w_dense",
            "w_bm25",
            "model_name",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost_usd",
        ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ex in dataset:
            print(f"Running {cfg.name} on example {ex.id}...")
            retrieval_strategy = cfg.retrieval.strategy

            answer, context, latency, usage = client.query(
                question=ex.question if cfg.task_type == "qa" else ex.source_text,
                top_k=cfg.retrieval.top_k,
                retrieval_strategy=retrieval_strategy,
            )

            metrics = compute_metrics(
                ex, answer, context, task_type=cfg.task_type
            )

            if cfg.task_type == "qa":
                row: Dict[str, Any] = {
                    "run_name": cfg.name,
                    "example_id": ex.id,
                    "question": ex.question,
                    "gold_answer": ex.gold_answer,
                    "answer": answer,
                    "latency_ms": round(latency * 1000, 2),
                    "recall": metrics["recall"],
                    "faithfulness": metrics["faithfulness"],
                    "correctness": metrics["correctness"],
                    "retrieval_strategy": retrieval_strategy,
                    "top_k": cfg.retrieval.top_k,
                    "overfetch_factor": cfg.retrieval.overfetch_factor,
                    "w_dense": cfg.retrieval.w_dense,
                    "w_bm25": cfg.retrieval.w_bm25,
                    "model_name": cfg.model.name,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cost_usd": usage.get("cost_usd", 0.0),
                }
            else:  # summarization
                row = {
                    "run_name": cfg.name,
                    "example_id": ex.id,
                    "source_text": ex.source_text,
                    "gold_summary": ex.gold_summary,
                    "summary": answer,
                    "latency_ms": round(latency * 1000, 2),
                    "recall": metrics["recall"],
                    "bleu": metrics["bleu"],
                    "rouge_l": metrics["rouge_l"],
                    "retrieval_strategy": retrieval_strategy,
                    "top_k": cfg.retrieval.top_k,
                    "overfetch_factor": cfg.retrieval.overfetch_factor,
                    "w_dense": cfg.retrieval.w_dense,
                    "w_bm25": cfg.retrieval.w_bm25,
                    "model_name": cfg.model.name,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cost_usd": usage.get("cost_usd", 0.0),
                }

            writer.writerow(row)

    print(f"\nSaved results to {out_path}")
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.eval_runner <config_path.yaml>")
        raise SystemExit(1)

    run_eval(sys.argv[1])
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from .config import load_run_config
from .datasets import load_eval_set
from .metrics import compute_metrics
from .policygpt_client import PolicyGPTClient

load_dotenv()


def run_eval(config_path: str) -> str:
    cfg = load_run_config(config_path)
    dataset = load_eval_set(cfg.eval.dataset_path)
    client = PolicyGPTClient(base_url=cfg.app.base_url)

    results_dir = os.getenv("RESULTS_DIR", "results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(
        results_dir,
        f"{cfg.name}_{timestamp}.csv",
    )

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
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ex in dataset:
            print(f"Running {cfg.name} on example {ex.id}...")
            # Map retrieval.strategy -> optional flag if your backend supports it
            retrieval_strategy = cfg.retrieval.strategy

            answer, context, latency = client.query(
                question=ex.question,
                top_k=cfg.retrieval.top_k,
                retrieval_strategy=retrieval_strategy,
            )

            metrics = compute_metrics(ex, answer, context)

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
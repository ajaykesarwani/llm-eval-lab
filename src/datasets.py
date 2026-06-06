from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EvalExample:
    id: str
    question: str
    gold_answer: str
    gold_context_snippet: str
    where: Optional[Dict[str, Any]] = None


def load_eval_set(path: str) -> List[EvalExample]:
    items: List[EvalExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                EvalExample(
                    id=obj["id"],
                    question=obj["question"],
                    gold_answer=obj["gold_answer"],
                    gold_context_snippet=obj["gold_context_snippet"],
                    where=obj.get("where"),
                )
            )
    return items


@dataclass
class SummarizationExample:
    id: str
    source_text: str
    gold_summary: str


def load_policy_summarization_dataset(path: str) -> List[SummarizationExample]:
    import csv

    items: List[SummarizationExample] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(
                SummarizationExample(
                    id=row["example_id"],
                    source_text=row["source_text"],
                    gold_summary=row["gold_summary"],
                )
            )
    return items
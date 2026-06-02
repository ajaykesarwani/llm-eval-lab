from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from .datasets import EvalExample

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")


_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY or None,
)


def compute_retrieval_recall(
    example: EvalExample,
    retrieved_chunks: List[Dict[str, Any]],
) -> float:
    """
    Simple recall: 1.0 if any retrieved chunk contains the gold context snippet (case-insensitive),
    else 0.0.
    """
    snippet = example.gold_context_snippet.lower()
    for c in retrieved_chunks:
        text = (c.get("text") or "").lower()
        if snippet in text:
            return 1.0
    return 0.0


def _judge(
    question: str,
    answer: str,
    gold_answer: str,
    context_chunks: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Ask the LLM to judge faithfulness and correctness.
    Returns:
      { "faithfulness": float, "correctness": float }
    """
    context_texts = []
    for c in context_chunks:
        source = (c.get("metadata") or {}).get("source", "unknown")
        text = c.get("text") or ""
        context_texts.append(f"[source={source}] {text}")
    context_block = "\n\n".join(context_texts)

    prompt = f"""
You are evaluating a Retrieval-Augmented Generation (RAG) answer.

You will be given:
- A user question
- A ground-truth answer
- Retrieved context
- A model-generated answer

Return ONLY a JSON object with two keys:
- "faithfulness": between 0 and 1 (1 = fully supported by context, 0 = not supported)
- "correctness": between 0 and 1 (1 = matches the ground-truth answer, 0 = incorrect)

Question:
{question}

Ground-truth answer:
{gold_answer}

Context:
{context_block}

Model answer:
{answer}
"""

    completion = _client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )

    raw = completion.choices[0].message.content.strip()
    print("JUDGE RAW:\n", raw, "\n")

    raw = raw.replace("```json", "").replace("```", "").strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw_json = raw[start : end + 1]
    else:
        raw_json = raw

    try:
        data = json.loads(raw_json)
        faithfulness = float(data.get("faithfulness", 0.0))
        correctness = float(data.get("correctness", 0.0))
    except Exception as e:
        print("JUDGE PARSE ERROR:", e)
        faithfulness = 0.0
        correctness = 0.0

    faithfulness = max(0.0, min(1.0, faithfulness))
    correctness = max(0.0, min(1.0, correctness))
    return {"faithfulness": faithfulness, "correctness": correctness}


def compute_metrics(
    example: EvalExample,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, float]:
    recall = compute_retrieval_recall(example, retrieved_chunks)
    judge_scores = _judge(
        question=example.question,
        answer=answer,
        gold_answer=example.gold_answer,
        context_chunks=retrieved_chunks,
    )
    return {
        "recall": recall,
        "faithfulness": judge_scores["faithfulness"],
        "correctness": judge_scores["correctness"],
    }
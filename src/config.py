from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    base_url: str


@dataclass
class RetrievalConfig:
    strategy: str
    top_k: int
    overfetch_factor: int
    w_dense: float
    w_bm25: float


@dataclass
class ModelConfig:
    name: str


@dataclass
class EvalConfig:
    dataset_path: str


@dataclass
class RunConfig:
    name: str
    description: str
    app: AppConfig
    retrieval: RetrievalConfig
    model: ModelConfig
    eval: EvalConfig
    task_type: str = "qa"

def _expand_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        key = value[2:-1]
        return os.getenv(key, "")
    return value


def load_run_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Expand ${ENV_VAR} placeholders
    def expand(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand(v) for v in obj]
        return _expand_env(obj)

    cfg = expand(raw)

    return RunConfig(
        name=cfg["name"],
        description=cfg.get("description", ""),
        app=AppConfig(base_url=cfg["app"]["base_url"]),
        retrieval=RetrievalConfig(
            strategy=cfg["retrieval"]["strategy"],
            top_k=int(cfg["retrieval"]["top_k"]),
            overfetch_factor=int(cfg["retrieval"]["overfetch_factor"]),
            w_dense=float(cfg["retrieval"]["w_dense"]),
            w_bm25=float(cfg["retrieval"]["w_bm25"]),
        ),
        model=ModelConfig(name=cfg["model"]["name"]),
        eval=EvalConfig(dataset_path=cfg["eval"]["dataset_path"]),
        task_type=cfg.get("task_type", "qa"),
    )
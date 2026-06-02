from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import httpx


class PolicyGPTClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)

    def query(
        self,
        question: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        retrieval_strategy: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Call the PolicyGPT /query endpoint and return (answer, context_chunks, latency_seconds).
        """
        url = f"{self.base_url}/query"
        payload: Dict[str, Any] = {
            "query": question,
            "top_k": top_k,
        }
        if where:
            payload["where"] = where
        if retrieval_strategy:
            payload["retrieval_strategy"] = retrieval_strategy

        start = time.perf_counter()
        resp = self._client.post(url, json=payload)
        elapsed = time.perf_counter() - start

        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
        context = data.get("context", [])
        return answer, context, elapsed
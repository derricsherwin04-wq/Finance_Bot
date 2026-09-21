"""Minimal client for a local OpenAI-compatible model server."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LocalModelError(RuntimeError):
    pass


class LocalModel:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("FINCH_API_BASE") or "http://127.0.0.1:8080/v1").rstrip("/")
        self.model = model or os.getenv("FINCH_MODEL") or "local-model"

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 450) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:  # noqa: S310 - explicit local URL by default
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise LocalModelError(f"Local model server returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise LocalModelError(
                f"Could not reach the local model server at {self.base_url}. "
                "Start llama.cpp, or use --sources-only."
            ) from error
        try:
            return str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as error:
            raise LocalModelError("The local server returned an unexpected chat response.") from error

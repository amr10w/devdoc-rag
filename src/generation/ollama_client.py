"""
src/generation/ollama_client.py

Robust Ollama client supporting local installations (http://localhost:11434),
Docker network endpoints (http://host.docker.internal:11434), and remote
authenticated endpoints (e.g., https://ollama.com).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import requests
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


class OllamaClient:
    """Client for generating completions and chatting with local or remote Ollama models."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_API_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5")
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY", "")
        self.timeout = timeout

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Sends a completion request to Ollama /api/generate."""
        target_model = model or self.model
        opts = {"temperature": temperature}
        if options:
            opts.update(options)

        payload: dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": opts,
        }
        if system:
            payload["system"] = system

        url = f"{self.base_url}/api/generate"
        response = requests.post(url, json=payload, headers=self._get_headers(), timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Sends a multi-turn or system/user chat request to Ollama /api/chat."""
        target_model = model or self.model
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        url = f"{self.base_url}/api/chat"
        response = requests.post(url, json=payload, headers=self._get_headers(), timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        return message.get("content", "").strip()

    def health_check(self) -> dict[str, Any]:
        """Verifies Ollama connectivity and available models."""
        url = f"{self.base_url}/api/tags"
        start = time.perf_counter()
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=5)
            latency_ms = (time.perf_counter() - start) * 1000.0
            if resp.status_code == 200:
                tags = resp.json().get("models", [])
                model_names = [m.get("name") for m in tags]
                return {
                    "status": "connected",
                    "url": self.base_url,
                    "model": self.model,
                    "available_models": model_names,
                    "latency_ms": round(latency_ms, 2),
                }
            return {"status": "degraded", "code": resp.status_code, "latency_ms": round(latency_ms, 2)}
        except Exception as e:
            return {"status": "disconnected", "error": str(e), "url": self.base_url}


_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client

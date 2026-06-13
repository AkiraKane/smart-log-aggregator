"""LLM client with Ollama primary and OpenAI fallback."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class LLMError(Exception):
    """Raised when LLM calls fail."""


class LLMClient:
    """LLM client with Ollama primary and OpenAI API fallback."""

    def __init__(
        self,
        ollama_url: str | None = None,
        ollama_model: str | None = None,
        openai_api_key: str | None = None,
        openai_model: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL)
        self.ollama_model = ollama_model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.openai_model = openai_model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self.timeout = timeout

    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a response, trying Ollama first, then OpenAI fallback."""
        try:
            return self._call_ollama(prompt, system)
        except LLMError:
            if self.openai_api_key:
                return self._call_openai(prompt, system)
            raise

    def _call_ollama(self, prompt: str, system: str = "") -> str:
        """Call Ollama API at /api/generate."""
        url = f"{self.ollama_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("response", "")
        except urllib.error.URLError as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise LLMError(f"Ollama returned invalid JSON: {e}") from e

    def _call_openai(self, prompt: str, system: str = "") -> str:
        """Call OpenAI-compatible chat completions API."""
        url = "https://api.openai.com/v1/chat/completions"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.openai_model,
            "messages": messages,
            "temperature": 0.3,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise LLMError(f"OpenAI API error {e.code}: {error_body}") from e
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            raise LLMError(f"OpenAI request failed: {e}") from e

    def health_check_ollama(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            url = f"{self.ollama_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

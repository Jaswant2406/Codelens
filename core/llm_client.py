from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import httpx


class FreeLLMClient:
    """Best-effort free LLM client with provider fallbacks."""

    def __init__(self, providers: list[dict[str, Any]] | None = None) -> None:
        self.providers = providers or [
            {"name": "groq", "model": "llama-3.1-8b-instant", "env_key": "GROQ_API_KEY"},
            {
                "name": "openrouter",
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "env_key": "OPENROUTER_API_KEY",
            },
            {"name": "ollama", "model": "codellama", "base_url": "http://localhost:11434"},
        ]

    def complete(self, prompt: str, system: str = "") -> str:
        """Return the first successful completion or a local fallback."""
        for provider in self.providers:
            try:
                if provider["name"] == "groq":
                    return self._complete_openai_compatible(
                        base_url="https://api.groq.com/openai/v1/chat/completions",
                        api_key=os.getenv(provider["env_key"], ""),
                        model=provider["model"],
                        prompt=prompt,
                        system=system,
                    )
                if provider["name"] == "openrouter":
                    return self._complete_openai_compatible(
                        base_url="https://openrouter.ai/api/v1/chat/completions",
                        api_key=os.getenv(provider["env_key"], ""),
                        model=provider["model"],
                        prompt=prompt,
                        system=system,
                        extra_headers={
                            "HTTP-Referer": "https://localhost/codelens",
                            "X-Title": "CodeLens",
                        },
                    )
                if provider["name"] == "ollama":
                    return self._complete_ollama(
                        base_url=provider.get("base_url", "http://localhost:11434"),
                        model=provider.get("model", "codellama"),
                        prompt=prompt,
                        system=system,
                    )
            except httpx.HTTPError:
                continue
        return self._fallback_text(prompt)

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Yield streaming chunks from the first successful provider."""
        for provider in self.providers:
            try:
                if provider["name"] == "groq":
                    yield from self._stream_openai_compatible(
                        base_url="https://api.groq.com/openai/v1/chat/completions",
                        api_key=os.getenv(provider["env_key"], ""),
                        model=provider["model"],
                        prompt=prompt,
                        system=system,
                    )
                    return
                if provider["name"] == "openrouter":
                    yield from self._stream_openai_compatible(
                        base_url="https://openrouter.ai/api/v1/chat/completions",
                        api_key=os.getenv(provider["env_key"], ""),
                        model=provider["model"],
                        prompt=prompt,
                        system=system,
                        extra_headers={
                            "HTTP-Referer": "https://localhost/codelens",
                            "X-Title": "CodeLens",
                        },
                    )
                    return
                if provider["name"] == "ollama":
                    yield from self._stream_ollama(
                        base_url=provider.get("base_url", "http://localhost:11434"),
                        model=provider.get("model", "codellama"),
                        prompt=prompt,
                        system=system,
                    )
                    return
            except httpx.HTTPError:
                continue
        yield self._fallback_text(prompt)

    def _complete_openai_compatible(
        self,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        system: str,
        extra_headers: dict[str, str] | None = None,
    ) -> str:
        if not api_key:
            raise httpx.HTTPError("Missing API key")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]

    def _stream_openai_compatible(
        self,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        system: str,
        extra_headers: dict[str, str] | None = None,
    ) -> Iterator[str]:
        if not api_key:
            raise httpx.HTTPError("Missing API key")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", base_url, json=payload, headers=headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    payload_data = httpx.Response(200, content=data).json()
                    delta = payload_data["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta

    def _complete_ollama(self, base_url: str, model: str, prompt: str, system: str) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("response", "")

    def _stream_ollama(self, base_url: str, model: str, prompt: str, system: str) -> Iterator[str]:
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": True,
        }
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", f"{base_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    payload_data = httpx.Response(200, content=line).json()
                    chunk = payload_data.get("response", "")
                    if chunk:
                        yield chunk

    def _fallback_text(self, prompt: str) -> str:
        return f"Local fallback response for prompt: {prompt}"

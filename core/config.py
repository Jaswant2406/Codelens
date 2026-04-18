from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class CodeLensConfig:
    data_dir: Path = Path(".codelens")
    chroma_path: Path = Path(".codelens/chroma")
    graph_path: Path = Path(".codelens/graph.json")
    functions_path: Path = Path(".codelens/functions.json")
    cache_path: Path = Path(".codelens/embed_cache.json")
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    anthropic_model: str = "claude-sonnet-4-20250514"
    max_depth: int = 3
    llm_providers: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "groq", "model": "llama-3.1-8b-instant", "env_key": "GROQ_API_KEY"},
            {
                "name": "openrouter",
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "env_key": "OPENROUTER_API_KEY",
            },
            {"name": "ollama", "model": "codellama", "base_url": "http://localhost:11434"},
        ]
    )
    multirag_top_k: int = 8
    multirag_mmr_lambda: float = 0.5
    query_expansion_enabled: bool = True

    @classmethod
    def load(cls, path: str | Path = "codelens.yaml") -> "CodeLensConfig":
        config_path = Path(path)
        if not config_path.exists():
            config = cls()
            config.ensure_paths()
            return config

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        llm_section = raw.get("llm", {})
        multirag_section = raw.get("multirag", {})
        config = cls(
            data_dir=Path(raw.get("data_dir", ".codelens")),
            chroma_path=Path(raw.get("chroma_path", ".codelens/chroma")),
            graph_path=Path(raw.get("graph_path", ".codelens/graph.json")),
            functions_path=Path(raw.get("functions_path", ".codelens/functions.json")),
            cache_path=Path(raw.get("cache_path", ".codelens/embed_cache.json")),
            model_name=raw.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            anthropic_model=raw.get("anthropic_model", "claude-sonnet-4-20250514"),
            max_depth=int(raw.get("max_depth", 3)),
            llm_providers=llm_section.get(
                "providers",
                [
                    {"name": "groq", "model": "llama-3.1-8b-instant", "env_key": "GROQ_API_KEY"},
                    {
                        "name": "openrouter",
                        "model": "meta-llama/llama-3.1-8b-instruct:free",
                        "env_key": "OPENROUTER_API_KEY",
                    },
                    {"name": "ollama", "model": "codellama", "base_url": "http://localhost:11434"},
                ],
            ),
            multirag_top_k=int(multirag_section.get("top_k", 8)),
            multirag_mmr_lambda=float(multirag_section.get("mmr_lambda", 0.5)),
            query_expansion_enabled=bool(multirag_section.get("query_expansion", True)),
        )
        config.ensure_paths()
        return config

    def ensure_paths(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.functions_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "chroma_path": str(self.chroma_path),
            "graph_path": str(self.graph_path),
            "functions_path": str(self.functions_path),
            "cache_path": str(self.cache_path),
            "model_name": self.model_name,
            "anthropic_model": self.anthropic_model,
            "max_depth": self.max_depth,
            "llm_providers": self.llm_providers,
            "multirag_top_k": self.multirag_top_k,
            "multirag_mmr_lambda": self.multirag_mmr_lambda,
            "query_expansion_enabled": self.query_expansion_enabled,
        }

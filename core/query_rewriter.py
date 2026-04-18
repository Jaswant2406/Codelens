from __future__ import annotations

import re

from .llm_client import FreeLLMClient


class QueryRewriter:
    """Expand a user question into multiple retrieval queries."""

    def __init__(self, llm_client: FreeLLMClient) -> None:
        self.llm_client = llm_client

    def expand(self, query: str) -> list[str]:
        """Return original query, HyDE text, and focused sub-questions."""
        try:
            hyde_doc = self.llm_client.complete(
                f"Write a short hypothetical code snippet or function that would answer this question: {query}. Be concise."
            )
            subquestions_text = self.llm_client.complete(
                f"Split this into 2 focused sub-questions for code search: {query}. Return only the questions, one per line."
            )
            subquestions = [
                re.sub(r"^\s*[-*\d.]+\s*", "", line).strip()
                for line in subquestions_text.splitlines()
            ]
            candidates = [query, hyde_doc, *subquestions]
            deduped: list[str] = []
            seen: set[str] = set()
            for candidate in candidates:
                normalized = candidate.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                deduped.append(normalized)
            return deduped or [query]
        except Exception:
            return [query]

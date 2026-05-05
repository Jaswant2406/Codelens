from __future__ import annotations

import re
from collections import Counter
from math import log

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:  # type: ignore[no-redef]
        """Minimal BM25 fallback used when rank_bm25 is unavailable."""

        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.doc_lengths = [len(document) for document in corpus]
            self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if corpus else 0.0
            self.doc_freqs = [Counter(document) for document in corpus]
            self.idf: dict[str, float] = {}
            doc_count = len(corpus)
            for token in {token for document in corpus for token in document}:
                frequency = sum(1 for document in corpus if token in document)
                self.idf[token] = log((doc_count - frequency + 0.5) / (frequency + 0.5) + 1)

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            scores: list[float] = []
            for index, document in enumerate(self.corpus):
                score = 0.0
                frequencies = self.doc_freqs[index]
                doc_length = self.doc_lengths[index] or 1
                for token in query_tokens:
                    if token not in frequencies:
                        continue
                    idf = self.idf.get(token, 0.0)
                    tf = frequencies[token]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / (self.avgdl or 1))
                    score += idf * numerator / denominator
                scores.append(score)
            return scores

from .models import FunctionNode


class KeywordRetriever:
    """BM25 keyword retriever over function-level documents."""

    def __init__(self) -> None:
        self.nodes: list[FunctionNode] = []
        self.bm25: BM25Okapi | None = None

    def build(self, function_nodes: list[FunctionNode]) -> None:
        """Build a BM25 index over the supplied function nodes."""
        self.nodes = function_nodes
        tokenized_docs = []
        for node in function_nodes:
            document = (
                f"{node.name} {node.docstring or ''} "
                f"{' '.join(parameter.name for parameter in node.parameters)} {node.code[:200]}"
            )
            tokenized_docs.append(self._tokenize(document))
        self.bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None

    def search(self, query: str, top_k: int = 10) -> list[tuple[FunctionNode, float]]:
        """Return top BM25 matches with normalized scores."""
        if self.bm25 is None:
            return []
        tokenized_query = self._tokenize(query)
        scores = list(self.bm25.get_scores(tokenized_query))
        if not scores:
            return []
        max_score = max(scores)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        results: list[tuple[FunctionNode, float]] = []
        for index, score in ranked:
            normalized = score / max_score if max_score > 0 else 0.0
            results.append((self.nodes[index], normalized))
        return results

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for token in text.lower().split():
            split_token = re.sub(r"([A-Z])", r"_\1", token).lower()
            parts = re.split(r"[_\W]+", split_token)
            tokens.extend(part for part in parts if part)
        return tokens

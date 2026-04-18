from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import chromadb
from sentence_transformers import SentenceTransformer

from .config import CodeLensConfig
from .models import FunctionNode


class EmbeddingEngine:
    """Embedding engine with Chroma and JSON-store fallbacks."""

    def __init__(self, config: CodeLensConfig) -> None:
        self.config = config
        self.model = None
        try:
            self.model = SentenceTransformer(config.model_name)
        except Exception:
            self.model = None
        self.cache = self._load_cache()
        self.vector_store_path = config.data_dir / "embeddings.json"
        self.vector_store = self._load_vector_store()
        self.collection = None
        try:
            self.client = chromadb.PersistentClient(path=str(config.chroma_path))
            self.collection = self.client.get_or_create_collection("functions")
        except Exception:
            self.client = None

    def _load_cache(self) -> dict[str, str]:
        if self.config.cache_path.exists():
            return json.loads(self.config.cache_path.read_text(encoding="utf-8"))
        return {}

    def _load_vector_store(self) -> dict[str, dict[str, object]]:
        if self.vector_store_path.exists():
            return json.loads(self.vector_store_path.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self) -> None:
        self.config.cache_path.write_text(
            json.dumps(self.cache, indent=2),
            encoding="utf-8",
        )

    def _save_vector_store(self) -> None:
        self.vector_store_path.write_text(
            json.dumps(self.vector_store, indent=2),
            encoding="utf-8",
        )

    def _embed_text(self, function: FunctionNode) -> str:
        params = ", ".join(
            f"{parameter.name}:{parameter.type_hint or 'unknown'}"
            for parameter in function.parameters
        )
        return f"{function.name} {function.docstring or ''} {params} {function.code[:500]}"

    def index_functions(self, functions: Iterable[FunctionNode], file_hashes: dict[str, str]) -> None:
        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, str | int]] = []
        embeddings: list[list[float]] = []

        for function in functions:
            file_hash = file_hashes.get(function.file, "")
            if self.cache.get(function.node_id) == file_hash:
                continue
            text = self._embed_text(function)
            documents.append(text)
            ids.append(function.node_id)
            metadata = {
                "file": function.file,
                "language": function.language,
                "start_line": function.start_line,
                "node_id": function.node_id,
            }
            metadatas.append(metadata)
            embedding = self.embed_query(text)
            embeddings.append(embedding)
            self.vector_store[function.node_id] = {
                "embedding": embedding,
                "document": text,
                "metadata": metadata,
            }
            self.cache[function.node_id] = file_hash

        if ids and self.collection is not None:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        self._save_cache()
        self._save_vector_store()

    def embed_query(self, text: str) -> list[float]:
        if self.model is not None:
            return self.model.encode(text).tolist()
        return _hash_embedding(text)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Return node ids ordered by vector similarity."""
        vector = self.embed_query(query)
        if self.collection is not None:
            result = self.collection.query(query_embeddings=[vector], n_results=top_k)
            return result.get("ids", [[]])[0]

        scored = []
        for node_id, payload in self.vector_store.items():
            score = _cosine_similarity(vector, payload["embedding"])
            scored.append((score, node_id))
        scored.sort(reverse=True)
        return [node_id for _, node_id in scored[:top_k]]

    def vector_search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return ranked vector matches with similarity scores."""
        vector = self.embed_query(query)
        if self.collection is not None:
            result = self.collection.query(query_embeddings=[vector], n_results=top_k)
            ids = result.get("ids", [[]])[0]
            distances = result.get("distances", [[]])[0]
            if distances:
                return [
                    (node_id, 1.0 - float(distance))
                    for node_id, distance in zip(ids, distances)
                ]
            return [(node_id, 1.0 / (index + 1)) for index, node_id in enumerate(ids)]

        scored: list[tuple[str, float]] = []
        for node_id, payload in self.vector_store.items():
            score = _cosine_similarity(vector, payload["embedding"])
            scored.append((node_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def get_embeddings_dict(self) -> dict[str, list[float]]:
        """Expose embeddings by node id for MMR reranking."""
        return {
            node_id: payload["embedding"]
            for node_id, payload in self.vector_store.items()
            if "embedding" in payload
        }


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _hash_embedding(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        weight = (digest[3] / 255.0) + 0.5
        vector[index] += sign * weight
    return vector

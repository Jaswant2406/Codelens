from __future__ import annotations

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.id_map = []

    def add(self, embeddings: list[list[float]], ids: list[str]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.id_map.extend(ids)

    def search(self, query_vector: list[float], k: int = 5):
        q = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(q, k)
        return [
            (self.id_map[i], float(distances[0][idx]))
            for idx, i in enumerate(indices[0])
            if i < len(self.id_map)
        ]

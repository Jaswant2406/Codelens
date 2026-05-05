from __future__ import annotations

import math

import numpy as np


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists with reciprocal-rank fusion."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for index, (node_id, _) in enumerate(ranked_list):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + index + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def mmr_rerank(
    candidates: list[tuple[str, float]],
    embeddings: dict[str, list[float]],
    top_k: int = 8,
    lambda_param: float = 0.5,
) -> list[str]:
    """Rerank fused candidates using maximal marginal relevance."""
    if not embeddings or not candidates:
        return [node_id for node_id, _ in candidates[:top_k]]

    max_score = max(score for _, score in candidates) or 1.0
    remaining = [(node_id, score / max_score) for node_id, score in candidates]
    selected: list[str] = []

    while remaining and len(selected) < top_k:
        best_node_id = remaining[0][0]
        best_value = -math.inf
        for node_id, relevance in remaining:
            if node_id not in embeddings:
                score = relevance
            else:
                penalty = 0.0
                if selected:
                    penalty = max(
                        _cosine_similarity(embeddings[node_id], embeddings[selected_id])
                        for selected_id in selected
                        if selected_id in embeddings
                    )
                score = lambda_param * relevance - (1 - lambda_param) * penalty
            if score > best_value:
                best_value = score
                best_node_id = node_id
        selected.append(best_node_id)
        remaining = [(node_id, score) for node_id, score in remaining if node_id != best_node_id]

    return selected


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_array = np.array(left)
    right_array = np.array(right)
    left_norm = np.linalg.norm(left_array)
    right_norm = np.linalg.norm(right_array)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left_array, right_array) / (left_norm * right_norm))

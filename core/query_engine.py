from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import networkx as nx

from .embedder import EmbeddingEngine
from .fusion import mmr_rerank, reciprocal_rank_fusion
from .keyword_retriever import KeywordRetriever
from .models import FunctionNode
from .query_rewriter import QueryRewriter


class MultiRAGQueryEngine:
    """Multi-retriever query engine with fusion and reranking."""

    def __init__(
        self,
        embedder: EmbeddingEngine,
        graph: nx.DiGraph,
        keyword_retriever: KeywordRetriever,
        query_rewriter: QueryRewriter,
    ) -> None:
        self.embedder = embedder
        self.graph = graph
        self.keyword_retriever = keyword_retriever
        self.query_rewriter = query_rewriter
        self.functions_by_id = {
            node_id: FunctionNode.from_dict({"node_id": node_id, **data})
            for node_id, data in graph.nodes(data=True)
        }
        self._last_debug: dict[str, list[dict[str, str | float]]] = {
            "vector": [],
            "graph": [],
            "keyword": [],
            "fused": [],
        }

    def search(self, query: str, top_k: int = 8) -> list[FunctionNode]:
        """Search via vector, graph, and keyword retrieval, then fuse results."""
        expanded_queries = self.query_rewriter.expand(query)
        vector_lists: list[list[tuple[str, float]]] = []
        graph_lists: list[list[tuple[str, float]]] = []
        keyword_lists: list[list[tuple[str, float]]] = []

        with ThreadPoolExecutor() as executor:
            vector_futures = [
                executor.submit(self.embedder.vector_search, expanded_query, 10)
                for expanded_query in expanded_queries
            ]
            keyword_futures = [
                executor.submit(self.keyword_retriever.search, expanded_query, 10)
                for expanded_query in expanded_queries
            ]
            vector_results = [future.result() for future in vector_futures]
            keyword_results = [future.result() for future in keyword_futures]

        for vector_result in vector_results:
            filtered_vector_result = [
                (node_id, score)
                for node_id, score in vector_result
                if node_id in self.functions_by_id
            ]
            if not filtered_vector_result:
                continue
            vector_lists.append(filtered_vector_result)
            entry_node_id = filtered_vector_result[0][0]
            graph_lists.append(
                [
                    (node_id, 1.0 / (index + 1))
                    for index, node_id in enumerate(self._graph_chain(entry_node_id, depth=2))
                ]
            )

        for keyword_result in keyword_results:
            keyword_lists.append(
                [(function.node_id, score) for function, score in keyword_result]
            )

        vector_list = self._normalize_and_flatten(vector_lists)
        graph_list = self._normalize_and_flatten(graph_lists)
        keyword_list = self._normalize_and_flatten(keyword_lists)
        fused = reciprocal_rank_fusion([vector_list, graph_list, keyword_list])
        selected_ids = mmr_rerank(
            fused,
            self.embedder.get_embeddings_dict(),
            top_k=top_k,
            lambda_param=self.embedder.config.multirag_mmr_lambda,
        )
        self._last_debug = {
            "vector": self._debug_entries(vector_list),
            "graph": self._debug_entries(graph_list),
            "keyword": self._debug_entries(keyword_list),
            "fused": self._debug_entries(fused),
        }
        return [
            self.functions_by_id[node_id]
            for node_id in selected_ids
            if node_id in self.functions_by_id
        ]

    def graph_expand(self, entry_nodes: Iterable[str], depth: int = 3) -> nx.DiGraph:
        """Preserve graph expansion behavior from the previous query engine."""
        visited: set[str] = set()
        queue = deque((node, 0) for node in entry_nodes)
        while queue:
            node, level = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if level >= depth:
                continue
            for neighbor in list(self.graph.successors(node)) + list(self.graph.predecessors(node)):
                queue.append((neighbor, level + 1))
        return self.graph.subgraph(visited).copy()

    def gather_code_snippets(self, subgraph: nx.DiGraph) -> list[dict[str, str | int]]:
        """Preserve snippet gathering behavior from the previous query engine."""
        return [
            {
                "node_id": function.node_id,
                "name": function.name,
                "file": function.file,
                "start_line": function.start_line,
                "docstring": function.docstring or "",
                "code": function.code,
            }
            for node_id, function in self.functions_by_id.items()
            if node_id in subgraph.nodes
        ]

    def get_retriever_debug(self, query: str) -> dict[str, list[dict[str, str | float]]]:
        """Return retriever-level debug output for the supplied query."""
        self.search(query)
        return self._last_debug

    def _graph_chain(self, function_id: str, depth: int = 2) -> list[str]:
        visited = {function_id}
        queue = deque([(function_id, 0)])
        ordered: list[str] = []
        while queue:
            current, current_depth = queue.popleft()
            ordered.append(current)
            if current_depth >= depth:
                continue
            for neighbor in self.graph.successors(current):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, current_depth + 1))
        return ordered

    def _normalize_and_flatten(self, ranked_lists: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
        flattened: list[tuple[str, float]] = []
        for ranked_list in ranked_lists:
            if not ranked_list:
                continue
            scores = [score for _, score in ranked_list]
            minimum = min(scores)
            maximum = max(scores)
            for node_id, score in ranked_list:
                normalized = (score - minimum) / (maximum - minimum + 1e-9)
                flattened.append((node_id, normalized))
        flattened.sort(key=lambda item: item[1], reverse=True)
        return flattened

    def _debug_entries(self, ranked_list: list[tuple[str, float]]) -> list[dict[str, str | float]]:
        entries: list[dict[str, str | float]] = []
        for node_id, score in ranked_list[:8]:
            function = self.functions_by_id.get(node_id)
            entries.append(
                {
                    "node_id": node_id,
                    "score": float(score),
                    "name": function.name if function else node_id,
                }
            )
        return entries

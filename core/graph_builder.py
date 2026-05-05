from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Iterable

import networkx as nx

from .models import FunctionNode


class CallGraphBuilder:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._by_name: dict[str, list[str]] = {}

    def build(self, functions: Iterable[FunctionNode]) -> nx.DiGraph:
        functions = list(functions)
        self.graph.clear()
        self._by_name.clear()

        for function in functions:
            self.graph.add_node(function.node_id, **function.to_dict())
            self._by_name.setdefault(function.name, []).append(function.node_id)

        for function in functions:
            for callee_name in function.calls:
                for candidate in self._by_name.get(callee_name, []):
                    self.graph.add_edge(function.node_id, candidate, rel="calls")
        return self.graph

    def get_callees(self, function_id: str) -> list[str]:
        return list(self.graph.successors(function_id))

    def get_callers(self, function_id: str) -> list[str]:
        return list(self.graph.predecessors(function_id))

    def get_chain(self, function_id: str, depth: int = 3) -> list[str]:
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

    def dead_code(self) -> list[str]:
        return [node for node in self.graph.nodes if self.graph.in_degree(node) == 0]

    def export_json(self, path: str | Path) -> None:
        payload = {
            "nodes": [{"id": node, **data} for node, data in self.graph.nodes(data=True)],
            "edges": [
                {"source": source, "target": target, **data}
                for source, target, data in self.graph.edges(data=True)
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load_json(path: str | Path) -> nx.DiGraph:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = nx.DiGraph()
        for node in payload.get("nodes", []):
            node_id = node.pop("id")
            graph.add_node(node_id, **node)
        for edge in payload.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            graph.add_edge(source, target, **edge)
        return graph

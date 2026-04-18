from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from .ast_parser import parse_functions
from .agent import CodeLensAgent
from .config import CodeLensConfig
from .embedder import EmbeddingEngine
from .explainer import Explainer
from .graph_builder import CallGraphBuilder
from .keyword_retriever import KeywordRetriever
from .llm_client import FreeLLMClient
from .models import FunctionNode
from .pr_intelligence import PRAnalysisResult, PRIssueIntelligence
from .query_engine import MultiRAGQueryEngine
from .query_rewriter import QueryRewriter
from .repo_loader import load_repository
from .vector_index import VectorIndex


class CodeLensError(Exception):
    """Raised when the service cannot fulfill a request safely."""


@dataclass(slots=True)
class IndexStats:
    repo_path: str
    file_count: int
    function_count: int
    edge_count: int
    dead_code_count: int
    files: list[dict[str, str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_path": self.repo_path,
            "file_count": self.file_count,
            "function_count": self.function_count,
            "edge_count": self.edge_count,
            "dead_code_count": self.dead_code_count,
            "files": self.files,
        }


class CodeLensService:
    def __init__(self, config_path: str | Path = "codelens.yaml") -> None:
        self.config = CodeLensConfig.load(config_path)
        self.repo_state_path = self.config.data_dir / "repo_path.txt"
        self.graph_builder = CallGraphBuilder()
        self.embedder = EmbeddingEngine(self.config)
        self.graph = self.graph_builder.graph
        self.functions: list[FunctionNode] = []
        self.llm_client = FreeLLMClient(self.config.llm_providers)
        self.query_rewriter = QueryRewriter(self.llm_client)
        self.keyword_retriever = KeywordRetriever()
        self.query_engine = MultiRAGQueryEngine(
            embedder=self.embedder,
            graph=self.graph,
            keyword_retriever=self.keyword_retriever,
            query_rewriter=self.query_rewriter,
        )
        self.explainer = Explainer(self.llm_client)
        self.pr_intelligence = PRIssueIntelligence(self.functions)
        self.agent = CodeLensAgent(self)
        self.vector_index: VectorIndex | None = None
        self.repo_functions_by_id: dict[str, FunctionNode] = {}
        self.embeddings: list[list[float]] = []
        self.id_map: list[str] = []

    def index(self, repo_url_or_path: str) -> IndexStats:
        try:
            repo_path, files = load_repository(repo_url_or_path)
        except Exception as exc:
            raise CodeLensError(f"Failed to load repository: {exc}") from exc
        if not files:
            raise CodeLensError(
                "No supported source files were found. Use a repo that contains .py, .js, .ts, .go, or .java files."
        )
        functions: list[FunctionNode] = []
        parse_failures: list[str] = []
        for file in tqdm(files, desc="Parsing files"):
            try:
                functions.extend(parse_functions([file]))
            except Exception:
                parse_failures.append(file.path)
                continue
        if not functions:
            raise CodeLensError(
                "Files were found, but no functions could be parsed from them. Try a different repo or check parser support for that language mix."
            )

        graph = self.graph_builder.build(tqdm(functions, desc="Building graph"))
        self.graph_builder.export_json(self.config.graph_path)
        self.config.functions_path.write_text(
            json.dumps([function.to_dict() for function in functions], indent=2),
            encoding="utf-8",
        )
        self.repo_state_path.write_text(str(repo_path), encoding="utf-8")

        self.embedder = EmbeddingEngine(self.config)
        self.embedder.index_functions(
            tqdm(functions, desc="Embedding functions"),
            {file.path: file.hash for file in files},
        )
        self.functions = functions
        self.graph = graph
        self.keyword_retriever.build(self.functions)
        self.graph_builder.graph = graph
        self.query_engine = MultiRAGQueryEngine(
            embedder=self.embedder,
            graph=self.graph,
            keyword_retriever=self.keyword_retriever,
            query_rewriter=self.query_rewriter,
        )
        self.pr_intelligence = PRIssueIntelligence(self.functions)
        self.agent = CodeLensAgent(self)
        self.repo_functions_by_id = {function.node_id: function for function in functions}
        self.vector_index = None
        self.embeddings = []
        self.id_map = []

        return IndexStats(
            repo_path=str(repo_path),
            file_count=len(files),
            function_count=len(functions),
            edge_count=graph.number_of_edges(),
            dead_code_count=len(self.graph_builder.dead_code()),
            files=[{"path": file.path, "language": file.language} for file in files],
        )

    def load_state(self) -> tuple[list[FunctionNode], object]:
        if not self.config.functions_path.exists() or not self.config.graph_path.exists():
            raise CodeLensError(
                "No indexed repository found yet. Run indexing first, then retry this action."
            )
        functions = [
            FunctionNode.from_dict(item)
            for item in json.loads(self.config.functions_path.read_text(encoding="utf-8"))
        ]
        graph = self.graph_builder.load_json(self.config.graph_path)
        self.graph_builder.graph = graph
        self.functions = functions
        self.graph = graph
        self.keyword_retriever.build(self.functions)
        self.query_engine = MultiRAGQueryEngine(
            embedder=self.embedder,
            graph=self.graph,
            keyword_retriever=self.keyword_retriever,
            query_rewriter=self.query_rewriter,
        )
        self.pr_intelligence = PRIssueIntelligence(self.functions)
        self.agent = CodeLensAgent(self)
        self.repo_functions_by_id = {function.node_id: function for function in functions}
        return functions, graph

    def ai_query(self, question: str, api_key: str, file_path: str = "") -> dict[str, object]:
        if not question.strip():
            raise CodeLensError("Question is required.")
        if not api_key.strip():
            raise CodeLensError("API key is required for the AI tab.")
        if not self.functions:
            self.load_state()

        client = self._configure_genai(api_key)
        if file_path.strip():
            return self._ai_query_for_file(client, question, file_path.strip())
        self._ensure_vector_index(client)
        query_embedding = self._embed_text(client, question)
        results = self.vector_index.search(query_embedding, k=5) if self.vector_index else []
        nodes = [
            self.repo_functions_by_id[node_id]
            for node_id, _ in results
            if node_id in self.repo_functions_by_id
        ]

        context_text = ""
        for node_id, score in results:
            node = self.repo_functions_by_id.get(node_id)
            if node is None:
                continue
            context_text += (
                f"\nFunction: {node.name}\n"
                f"File: {node.file}\n"
                f"Score: {score:.3f}\n"
                "Code:\n"
                f"{node.code[:1000]}\n"
            )
        if not context_text:
            context_text = "No relevant code found."

        prompt = f"""
You are analyzing a codebase.

Relevant Context:
{context_text}

Question:
{question}

Instructions:

* Use context when relevant
* Do NOT hallucinate missing code
"""
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
        except Exception as exc:
            raise CodeLensError(f"AI generation failed: {exc}") from exc
        return {
            "answer": self._extract_generated_text(response),
            "matches": [node.to_dict() for node in nodes],
            "context": context_text,
        }

    def _ai_query_for_file(self, client: object, question: str, file_path: str) -> dict[str, object]:
        repo_root = self._repo_root()
        if repo_root is None or not repo_root.exists():
            raise CodeLensError("No indexed repository found yet. Run indexing first, then retry this action.")
        target = repo_root / file_path
        if not target.exists() or not target.is_file():
            raise CodeLensError(f"File not found in indexed repository: {file_path}")
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = target.read_text(encoding="utf-8", errors="ignore")

        context_text = (
            f"File: {file_path}\n"
            "Code:\n"
            f"{content[:12000]}"
        )
        prompt = f"""
You are analyzing a codebase.

Relevant Context:
{context_text}

Question:
{question}

Instructions:

* Use context when relevant
* Do NOT hallucinate missing code
"""
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
        except Exception as exc:
            raise CodeLensError(f"AI generation failed: {exc}") from exc

        file_matches = [
            function
            for function in self.functions
            if function.file == file_path
        ]
        return {
            "answer": self._extract_generated_text(response),
            "matches": [node.to_dict() for node in file_matches[:20]],
            "context": context_text,
        }

    def _configure_genai(self, api_key: str):
        try:
            from google import genai
        except Exception as exc:
            raise CodeLensError("google-genai is not installed in the current environment.") from exc
        try:
            return genai.Client(api_key=api_key)
        except Exception as exc:
            raise CodeLensError(f"Failed to initialize Google Gen AI client: {exc}") from exc

    def _ensure_vector_index(self, client: object) -> None:
        if self.vector_index is not None and self.id_map:
            return
        if not self.functions:
            self.load_state()
        self.repo_functions_by_id = {function.node_id: function for function in self.functions}
        embeddings: list[list[float]] = []
        ids: list[str] = []
        node_texts: list[str] = []
        for node in self.functions:
            node_text = f"""
Function: {node.name}
File: {node.file}
Doc: {node.docstring or ""}
Code:
{node.code[:2000]}
"""
            ids.append(node.node_id)
            node_texts.append(node_text)
        embeddings = self._embed_texts(client, node_texts)
        self.embeddings = embeddings
        self.id_map = ids
        if not embeddings:
            self.vector_index = None
            return
        self.vector_index = VectorIndex(dim=len(embeddings[0]))
        self.vector_index.add(embeddings, ids)

    def _embed_text(self, client: object, text: str) -> list[float]:
        embeddings = self._embed_texts(client, [text])
        if embeddings:
            return embeddings[0]
        raise CodeLensError("Failed to generate embeddings for the AI tab.")

    def _embed_texts(self, client: object, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        extracted: list[list[float]] = []
        batch_size = 100
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                )
            except Exception as exc:
                raise CodeLensError(f"Embedding failed: {exc}") from exc

            raw_embeddings = getattr(response, "embeddings", None) or []
            for item in raw_embeddings:
                values = getattr(item, "values", None)
                if values is None and isinstance(item, dict):
                    values = item.get("values")
                if values is not None:
                    extracted.append(list(values))
        if len(extracted) != len(texts):
            raise CodeLensError("Embedding response was incomplete for the AI tab.")
        return extracted

    def _extract_generated_text(self, response: object) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        candidates = getattr(response, "candidates", None) or []
        parts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            candidate_parts = getattr(content, "parts", None) if content is not None else None
            if not candidate_parts and isinstance(candidate, dict):
                content = candidate.get("content", {})
                candidate_parts = content.get("parts", [])
            for part in candidate_parts or []:
                part_text = getattr(part, "text", None)
                if part_text is None and isinstance(part, dict):
                    part_text = part.get("text")
                if part_text:
                    parts.append(str(part_text))
        if parts:
            return "\n".join(parts)
        return "No response generated."

    def ask(self, question: str, depth: int | None = None) -> tuple[list[FunctionNode], object, Iterator[str]]:
        functions, graph = self.load_state()
        def is_test_path(path: str) -> bool:
            normalized = path.replace("\\", "/").lower()
            return "/tests/" in normalized or normalized.startswith("tests/")

        def normalize_path(path: str) -> str:
            return path.replace("\\", "/").lower()

        def is_script_path(path: str) -> bool:
            normalized = normalize_path(path)
            return "/scripts/" in normalized or normalized.startswith("scripts/")

        repo_functions = [function for function in functions if not is_test_path(function.file)]
        query_type = self.explainer._classify_query_type(question)
        lowered_question = question.lower()
        explain_query = lowered_question.startswith("explain ")
        file_query = ".py" in lowered_question or "file" in lowered_question or "module" in lowered_question
        raw_file_names = [token for token in re.findall(r"[a-zA-Z0-9_./-]+\.py", lowered_question)]
        file_names = list(raw_file_names)
        if explain_query and not file_names:
            explain_target = lowered_question[len("explain ") :].strip().strip("?.!,")
            if explain_target and len(explain_target.split()) == 1:
                file_names.append(explain_target)
                file_query = True
        matched_file_path = ""
        if file_names:
            requested_file = file_names[0]
            file_resolution = self._resolve_file_match(requested_file, repo_functions)
            if file_resolution["status"] == "multiple":
                options = ", ".join(f"`{path}`" for path in file_resolution["matches"])
                return ([], {}, iter([f"Multiple files matched: {options}. Please specify."]))
            if file_resolution["status"] == "none":
                return ([], {}, iter([f"No file named {requested_file} found"]))
            matched_file_path = str(file_resolution["path"])
        file_matches = [function for function in repo_functions if function.file == matched_file_path] if matched_file_path else []
        if self._is_entry_query(question):
            if not matched_file_path:
                return ([], {}, iter(["Please specify a function or file to analyze."]))
            return ([], {}, iter([self._analyze_entry_points(matched_file_path)]))
        if file_query:
            if matched_file_path:
                file_nodes = [function for function in repo_functions if function.file == matched_file_path]
                return (file_nodes, {}, iter([self._analyze_file(matched_file_path, file_nodes)]))
        query_tokens = [
            token
            for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_\.]*", lowered_question)
            if token not in {
                "how", "what", "why", "where", "who", "works", "work", "internally",
                "internal", "does", "is", "the", "a", "an", "to", "and", "of", "here",
                "file", "module", "flow", "trace", "explain",
            }
        ]
        if query_type == "DEBUG":
            matches = [function for function in self.query_engine.search(question, top_k=3) if not is_test_path(function.file)]
            suggestions = [function.name for function in matches[:3]]
            response = [
                "No specific function identified. ",
                "Provide a function name or error.\n",
            ]
            if suggestions:
                response.append("Suggestions: " + ", ".join(suggestions))
            return (
                matches,
                {},
                iter(response),
            )
        functions_by_id = {function.node_id: function for function in repo_functions}
        target_id = self.explainer._detect_target_global(question, repo_functions)
        if file_query and not matched_file_path and not target_id:
            return ([], {}, iter(["Please specify a file name to analyze."]))
        if not target_id and not file_query and not explain_query:
            return ([], {}, iter(["Please specify a function or file to analyze."]))
        query_matches = self.query_engine.search(
            question,
            top_k=self.config.multirag_top_k if target_id else 5,
        )
        query_matches = [function for function in query_matches if not is_test_path(function.file)]
        file_hint_tokens = list(query_tokens)
        if not query_matches and not target_id:
            response = ["No relevant functionality found in the current repository."]
            return ([], {}, iter(response))
        file_hint_matches = [
            function
            for function in repo_functions
            if any(token in Path(function.file).stem.lower() or token in function.file.lower() for token in file_hint_tokens)
        ]
        file_hint_matches.sort(
            key=lambda function: (
                0 if function.name.lower() in {"request", "get", "post", "put", "patch", "delete", "head", "options"} else 1,
                function.start_line,
                function.name,
            )
        )
        selected_ids: list[str] = []
        if target_id and target_id in functions_by_id:
            target_function = functions_by_id[target_id]
            target_file = target_function.file
            selected_ids.append(target_id)
            for caller_id in graph.predecessors(target_id):
                if (
                    caller_id in functions_by_id
                    and caller_id not in selected_ids
                    and (not is_script_path(target_file) or functions_by_id[caller_id].file == target_file)
                ):
                    selected_ids.append(caller_id)
            normalized_calls = {call.lower().replace("_", "").replace(" ", "") for call in target_function.calls}
            for function in repo_functions:
                if function.node_id == target_id:
                    continue
                normalized_name = function.name.lower().replace("_", "").replace(" ", "")
                if (
                    normalized_name in normalized_calls
                    and function.node_id not in selected_ids
                    and (not is_script_path(target_file) or function.file == target_file)
                ):
                    selected_ids.append(function.node_id)
            if len(selected_ids) < 8 and not is_script_path(target_file):
                for match in query_matches:
                    if match.node_id in selected_ids:
                        continue
                    if (
                        graph.has_edge(target_id, match.node_id)
                        or graph.has_edge(match.node_id, target_id)
                    ):
                        selected_ids.append(match.node_id)
                    if len(selected_ids) >= 8:
                        break
        elif file_hint_matches:
            for function in file_hint_matches[:8]:
                if function.node_id not in selected_ids:
                    selected_ids.append(function.node_id)
        else:
            if explain_query and query_matches:
                anchor = query_matches[0]
                selected_ids.append(anchor.node_id)
                for caller_id in graph.predecessors(anchor.node_id):
                    if caller_id in functions_by_id and caller_id not in selected_ids:
                        selected_ids.append(caller_id)
                for callee_id in graph.successors(anchor.node_id):
                    if callee_id in functions_by_id and callee_id not in selected_ids:
                        selected_ids.append(callee_id)
            else:
                return ([], {}, iter(["No relevant functionality found in the current repository."]))

        selected_ids = selected_ids[:8]
        matches = [functions_by_id[node_id] for node_id in selected_ids if node_id in functions_by_id]
        if not matches:
            response = ["No relevant functionality found in the current repository."]
            return ([], {}, iter(response))

        file_set = {function.file for function in matches}
        script_only = all(is_script_path(function.file) for function in matches)
        if script_only and len(file_set) > 1:
            primary_file = matches[0].file
            selected_ids = [node_id for node_id in selected_ids if functions_by_id[node_id].file == primary_file]
            matches = [functions_by_id[node_id] for node_id in selected_ids if node_id in functions_by_id]
            file_set = {function.file for function in matches}

        subgraph = graph.subgraph(selected_ids).copy()
        if len(file_set) > 1 and subgraph.number_of_edges() == 0:
            return (matches, {}, iter(["No valid execution chain found in the current context."]))
        if subgraph.number_of_nodes() > 1 and subgraph.number_of_edges() < subgraph.number_of_nodes() - 1:
            return (matches, {}, iter(["No valid execution chain found in the current context."]))
        if len(file_set) > 1 and subgraph.number_of_nodes() > 1:
            root_count = sum(1 for node_id in subgraph.nodes if subgraph.in_degree(node_id) == 0)
            if root_count != 1:
                return (matches, {}, iter(["No valid execution chain found in the current context."]))
        snippets = [
            {
                "node_id": function.node_id,
                "name": function.name,
                "file": function.file,
                "start_line": function.start_line,
                "docstring": function.docstring or "",
                "params": [parameter.to_dict() for parameter in function.parameters],
                "calls": list(function.calls),
                "code": function.code,
            }
            for function in matches
        ]
        callers = {node_id: [caller for caller in graph.predecessors(node_id) if caller in selected_ids] for node_id in selected_ids}
        dead_nodes = self.graph_builder.dead_code()
        metadata_insights = self._metadata_insights(matches, subgraph, graph, dead_nodes)
        raw_keywords = [
            token
            for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_\.]*", question.lower())
            if token not in {
                "why", "how", "what", "is", "the", "a", "an", "to",
                "not", "does", "do", "are", "was", "it", "works", "work", "internally", "internal",
            }
        ]
        keywords: list[str] = []
        for token in raw_keywords:
            for part in token.split("."):
                normalized = part.strip()
                if normalized and normalized not in keywords:
                    keywords.append(normalized)
        relevant = bool(target_id) or bool(file_matches) or bool(file_hint_matches) or any(
            any(
                keyword in str(item["name"]).lower()
                or keyword in str(item.get("docstring", "")).lower()
                or keyword in str(item["file"]).lower()
                for keyword in keywords
            )
            for item in snippets
        )
        if keywords and not relevant:
            return ([], {}, iter(["No relevant functionality found in the current repository."]))
        if not selected_ids:
            return ([], {}, iter(["No relevant functionality found in the current repository."]))
        return matches, subgraph, self.explainer.stream_explain(
            question,
            snippets,
            selected_ids,
            callers=callers,
            dead_nodes=dead_nodes,
            metadata_insights=metadata_insights,
        )

    def _is_entry_query(self, question: str) -> bool:
        """Return whether the question asks about script entry behavior."""
        lowered = question.lower()
        return any(
            phrase in lowered
            for phrase in ["entry point", "how script starts", "how execution begins", "main function", "execution begins"]
        )

    def _analyze_entry_points(self, file_path: str) -> str:
        """Analyze script entry behavior for one file without using graph edges."""
        repo_root = self._repo_root()
        if repo_root is None or not repo_root.exists():
            return "No relevant functionality found in the current repository."
        path = repo_root / file_path
        if not path.exists():
            return "No relevant functionality found in the current repository."
        relative_path, tree, _ = self._load_file_ast(path, repo_root)
        if tree is None:
            return "No relevant functionality found in the current repository."

        for node in tree.body:
            if isinstance(node, ast.If) and self._is_main_guard(node):
                called_name = self._main_called_name(node)
                if called_name:
                    return "\n\n".join(
                        [
                            "## Entry Point",
                            f"`{called_name}()` in `{relative_path}`",
                            "## Execution Behavior",
                            f"The script checks `if __name__ == \"__main__\":` and then calls `{called_name}()`.",
                            "## Notes",
                            "Entry detection is based on this file only.",
                        ]
                    )
                return "\n\n".join(
                    [
                        "## Entry Point",
                        f"`if __name__ == \"__main__\":` in `{relative_path}`",
                        "## Execution Behavior",
                        "The script enters the `__main__` block, but the next callable step is not available in current context.",
                        "## Notes",
                        "Entry detection is based on this file only.",
                    ]
                )

        if self._has_top_level_executable_code(tree):
            return "\n\n".join(
                [
                    "## Entry Point",
                    f"`{relative_path}`",
                    "## Execution Behavior",
                    "Script executes top-to-bottom with no explicit entry function.",
                    "## Notes",
                    "Top-level executable statements were found in this file.",
                ]
            )

        return "\n\n".join(
            [
                "## Entry Point",
                "Not available in current context",
                "## Execution Behavior",
                "Not available in current context",
                "## Notes",
                "No explicit script entry behavior was found in this file.",
            ]
        )

    def _analyze_file(self, file_path: str, file_nodes: list[FunctionNode]) -> str:
        """Analyze one file in isolation."""
        repo_root = self._repo_root()
        if repo_root is None or not repo_root.exists():
            return "No relevant functionality found in the current repository."
        path = repo_root / file_path
        if not path.exists():
            return "No relevant functionality found in the current repository."

        relative_path, tree, source = self._load_file_ast(path, repo_root)
        if tree is None:
            return "No relevant functionality found in the current repository."
        if Path(relative_path).name == "__init__.py":
            imports_summary: list[str] = []
            exported_names: list[str] = []
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = ", ".join(alias.name for alias in node.names)
                    imports_summary.append(f"from {module} import {names}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports_summary.append(f"import {alias.name}")
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "__all__"
                            and isinstance(node.value, (ast.List, ast.Tuple))
                        ):
                            for item in node.value.elts:
                                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                                    exported_names.append(item.value)
            imports_summary = list(dict.fromkeys(imports_summary))
            execution_behavior = (
                "This file is empty. It serves as a namespace marker only."
                if not tree.body
                else ("\n".join(f"- `{line}`" for line in imports_summary[:8]) if imports_summary else "No imports found.")
            )
            notes = (
                "- Exported names: " + ", ".join(f"`{name}`" for name in exported_names)
                if exported_names
                else "No __all__ is defined — all public names are importable by default."
            )
            return "\n\n".join(
                [
                    "## Overview",
                    f"`{relative_path}` is the package initializer. Runs on first import.",
                    "## Purpose",
                    "\n".join(
                        [
                            "- Executed automatically when the package is imported.",
                            "- Can re-export names, define package symbols, or run setup code.",
                            "- An empty __init__.py still marks the directory as a Python package.",
                        ]
                    ),
                    "## Execution Behavior",
                    execution_behavior,
                    "## Notes",
                    notes,
                ]
            )

        imports: list[str] = []
        top_level_exec: list[str] = []
        class_names: list[str] = []
        class_methods: dict[str, dict[str, list[str]]] = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append(module or ".".join(alias.name for alias in node.names))
            elif isinstance(node, ast.ClassDef):
                class_names.append(node.name)
                methods: list[str] = []
                method_names: list[str] = []
                seen_method_names: set[str] = set()
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name not in seen_method_names:
                        seen_method_names.add(child.name)
                        method_names.append(child.name)
                        methods.append(
                            f"  - `{child.name}`: {self._summarize_code_purpose(ast.get_source_segment(source, child) or '', {fn.name for fn in file_nodes})}"
                        )
                class_methods[node.name] = {"lines": methods, "names": method_names}
            elif isinstance(node, ast.If) and self._is_main_guard(node):
                top_level_exec.append('if __name__ == "__main__":')
            elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                snippet = ast.get_source_segment(source, node) or node.__class__.__name__
                snippet = " ".join(snippet.strip().split())
                if snippet:
                    top_level_exec.append(snippet[:120])
        imports = list(dict.fromkeys(imports))

        seen_function_names: set[str] = set()
        function_lines: list[str] = []
        class_method_names = {
            name
            for cls in class_methods.values()
            for name in cls["names"]
        }
        for class_name in class_names:
            methods = class_methods.get(class_name, {}).get("lines", [])
            if methods:
                function_lines.append(f"- Class `{class_name}`")
                function_lines.extend(methods)
        for node in sorted(file_nodes, key=lambda item: item.start_line):
            if node.name in seen_function_names:
                continue
            if node.name in class_method_names:
                continue
            seen_function_names.add(node.name)
            function_lines.append(f"- `{node.name}`: {self._summarize_code_purpose(node.code, {fn.name for fn in file_nodes})}")
        function_lines = function_lines or ["- Not available in current context"]
        import_lines = [f"- `{item}`" for item in imports[:10]] or ["- None visible in this file"]
        exec_lines = [f"- `{item}`" for item in top_level_exec[:10]] or ["- No top-level executable code found"]
        purpose = self._infer_file_purpose(relative_path, file_nodes, imports, top_level_exec, class_names)
        pattern_notes: list[str] = []
        has_execute = any(fn.name.lower().startswith("execute") for fn in file_nodes)
        has_undo = any(fn.name.lower().startswith("undo") for fn in file_nodes)
        if has_execute and has_undo:
            pattern_notes.append("The structure suggests a Command pattern because `execute` and `undo` behaviors are both present.")
        strategy_names = [
            node.name for node in file_nodes
            if "strategy" in node.name.lower() or "handler" in node.name.lower()
        ]
        if len(set(strategy_names)) >= 2:
            pattern_notes.append("The structure suggests a Strategy pattern because multiple strategy-like functions are defined.")
        if pattern_notes:
            purpose = purpose + "\n\nPattern Insight:\n- " + "\n- ".join(pattern_notes)

        return "\n\n".join(
            [
                "## Overview",
                f"`{relative_path}` is analyzed in isolation.",
                "## Functions",
                "\n".join(function_lines),
                "## Imports",
                "\n".join(import_lines),
                "## Top-Level Execution",
                "\n".join(exec_lines),
                "## Purpose",
                purpose,
                "## Notes",
                "This file-level analysis is restricted to the specified file only.",
            ]
        )

    def _repo_root(self) -> Path | None:
        """Load the persisted repository root if available."""
        if not self.repo_state_path.exists():
            return None
        return Path(self.repo_state_path.read_text(encoding="utf-8").strip())

    def _load_file_ast(self, path: Path, repo_root: Path) -> tuple[str, ast.Module | None, str]:
        """Load one file and parse its AST."""
        relative_path = str(path.relative_to(repo_root)).replace("\\", "/")
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return relative_path, None, source
        return relative_path, tree, source

    def _resolve_file_match(self, requested_file: str, repo_functions: list[FunctionNode]) -> dict[str, object]:
        """Resolve a file query using exact and fuzzy filename matching."""
        def normalize_file_name(value: str) -> str:
            name = Path(value).name.lower()
            return re.sub(r"^\d+[_-]*", "", name)

        repo_root = self._repo_root()
        candidate_paths = sorted({function.file for function in repo_functions})
        if repo_root is not None and repo_root.exists():
            for path in repo_root.rglob("*.py"):
                relative_path = str(path.relative_to(repo_root)).replace("\\", "/")
                normalized = relative_path.lower()
                if "/tests/" in normalized or normalized.startswith("tests/"):
                    continue
                if relative_path not in candidate_paths:
                    candidate_paths.append(relative_path)
            candidate_paths.sort()
        requested_name = requested_file.lower()
        normalized_requested = normalize_file_name(requested_file)

        exact_matches = [
            path
            for path in candidate_paths
            if Path(path).name.lower() == requested_name or path.lower().endswith(requested_name)
        ]
        if len(exact_matches) == 1:
            return {"status": "single", "path": exact_matches[0]}
        if len(exact_matches) > 1:
            return {"status": "multiple", "matches": exact_matches}

        fuzzy_matches = []
        for path in candidate_paths:
            normalized_candidate = normalize_file_name(path)
            if (
                normalized_requested in normalized_candidate
                or normalized_candidate in normalized_requested
            ):
                fuzzy_matches.append(path)

        if len(fuzzy_matches) == 1:
            return {"status": "single", "path": fuzzy_matches[0]}
        if len(fuzzy_matches) > 1:
            return {"status": "multiple", "matches": fuzzy_matches}
        return {"status": "none"}

    def _is_main_guard(self, node: ast.If) -> bool:
        """Return whether an if-statement matches __name__ == '__main__'."""
        test = node.test
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        )

    def _main_called_name(self, node: ast.If) -> str:
        """Return the first directly called function name in a __main__ block."""
        for statement in node.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                func = statement.value.func
                if isinstance(func, ast.Name):
                    return func.id
                if isinstance(func, ast.Attribute):
                    return func.attr
            if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
                func = statement.value.func
                if isinstance(func, ast.Name):
                    return func.id
                if isinstance(func, ast.Attribute):
                    return func.attr
        return ""

    def _has_top_level_executable_code(self, tree: ast.Module) -> bool:
        """Return whether a module has top-level executable statements."""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            return True
        return False

    def _summarize_code_purpose(self, code: str, local_function_names: set[str] | None = None) -> str:
        """Summarize visible behavior from code only."""
        stripped = code.strip()
        calls = re.findall(r"([A-Za-z_][A-Za-z0-9_\.]*)\(", stripped)
        ignore_calls = {"if", "for", "while", "return", "print", "super", "range", "len", "open", "dict", "list", "set", "int", "str"}
        local_names = local_function_names or set()
        meaningful_calls = [
            call for call in calls
            if call.split(".")[0] not in ignore_calls
            and not call.startswith("self.")
            and not call.startswith("cls.")
            and call.split(".")[0] in local_names
        ]
        return_matches = re.findall(r"return\s+([^\n]+)", stripped)
        assigns = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", stripped)
        ignore_vars = {"i", "j", "k", "tmp", "self", "cls"}
        assigns = [name for name in assigns if name not in ignore_vars][:3]
        parts: list[str] = []
        if meaningful_calls:
            parts.append("calls " + ", ".join(f"`{call}`" for call in meaningful_calls[:3]))
        if return_matches:
            value = return_matches[0].strip()
            value = re.sub(r"\s+", " ", value)
            parts.append("returns " + value[:30])
        if assigns:
            parts.append("modifies local state via " + ", ".join(f"`{name}`" for name in assigns[:3]))
        if parts:
            return "; ".join(parts)
        return "contains internal logic"

    def _infer_file_purpose(
        self,
        relative_path: str,
        file_nodes: list[FunctionNode],
        imports: list[str],
        top_level_exec: list[str],
        class_names: list[str],
    ) -> str:
        """Infer a conservative file purpose."""
        if class_names and file_nodes:
            return "It defines classes along with supporting functions."
        if top_level_exec:
            return "It behaves like a script with visible top-level execution."
        if imports and file_nodes:
            return "It combines imported dependencies with locally defined functions."
        if file_nodes:
            return "It primarily defines helper functions in this file."
        if imports:
            return "It imports external dependencies but defines no local symbols."
        return "It appears to be a namespace or placeholder file."

    def impact(self, function_name: str) -> list[dict[str, object]]:
        functions, graph = self.load_state()
        functions_by_id = {function.node_id: function for function in functions}
        rows: list[dict[str, object]] = []
        for node_id, data in graph.nodes(data=True):
            if data.get("name") != function_name:
                continue
            for caller in graph.predecessors(node_id):
                caller_fn = functions_by_id[caller]
                rows.append(
                    {
                        "node_id": caller_fn.node_id,
                        "name": caller_fn.name,
                        "file": caller_fn.file,
                        "line": caller_fn.start_line,
                        "risk": _risk_label(graph.in_degree(caller), graph.out_degree(caller)),
                    }
                )
        return rows

    def deadcode(self) -> list[dict[str, object]]:
        _, graph = self.load_state()
        rows: list[dict[str, object]] = []
        for node_id in graph.nodes:
            if graph.in_degree(node_id) == 0:
                data = graph.nodes[node_id]
                rows.append(
                    {
                        "node_id": node_id,
                        "name": data["name"],
                        "file": data["file"],
                        "line": data["start_line"],
                    }
                )
        return rows

    def node_details(self, node_id: str) -> dict[str, object] | None:
        functions, _ = self.load_state()
        for function in functions:
            if function.node_id == node_id:
                return function.to_dict()
        return None

    def nodes_by_name(self, function_name: str) -> list[dict[str, object]]:
        functions, _ = self.load_state()
        lowered = function_name.strip().lower()
        return [
            function.to_dict()
            for function in functions
            if function.name.lower() == lowered
        ]

    def graph_context(self, node_id: str, depth: int = 3) -> dict[str, object]:
        functions, graph = self.load_state()
        functions_by_id = {function.node_id: function for function in functions}
        if node_id not in functions_by_id:
            raise CodeLensError("Function not found")
        call_chain = self.graph_builder.get_chain(node_id, depth=depth)
        edges = [
            {"source": source, "target": target, **data}
            for source, target, data in graph.edges(data=True)
            if source in call_chain and target in call_chain
        ]
        nodes = [
            functions_by_id[current_id].to_dict()
            for current_id in call_chain
            if current_id in functions_by_id
        ]
        return {
            "root": node_id,
            "call_chain": call_chain,
            "edges": edges,
            "nodes": nodes,
        }

    def analyze_pr_or_issue(self, issue_text: str = "", pr_diff: str = "") -> dict[str, object]:
        self.load_state()
        result: PRAnalysisResult = self.pr_intelligence.analyze(issue_text=issue_text, pr_diff=pr_diff)
        return result.to_dict()

    def run_agent(self, query: str) -> dict[str, object]:
        self.load_state()
        return self.agent.run(query).to_dict()

    def _metadata_insights(
        self,
        matches: list[FunctionNode],
        subgraph: object,
        graph: object,
        dead_nodes: list[str],
    ) -> dict[str, object]:
        """Build compact metadata that helps explanation stay grounded."""
        functions_by_id = {function.node_id: function for function in self.functions}
        fan_out = []
        fan_in = []
        for function in matches[:8]:
            fan_out.append(
                {
                    "node_id": function.node_id,
                    "name": function.name,
                    "count": graph.out_degree(function.node_id),
                }
            )
            fan_in.append(
                {
                    "node_id": function.node_id,
                    "name": function.name,
                    "count": graph.in_degree(function.node_id),
                }
            )

        dead_hits = [
            {
                "node_id": node_id,
                "name": functions_by_id[node_id].name,
                "file": functions_by_id[node_id].file,
            }
            for node_id in dead_nodes
            if node_id in functions_by_id and node_id in subgraph.nodes
        ]

        return {
            "retrieved_function_count": len(matches),
            "subgraph_node_count": subgraph.number_of_nodes(),
            "subgraph_edge_count": subgraph.number_of_edges(),
            "top_fan_out": sorted(fan_out, key=lambda item: item["count"], reverse=True)[:3],
            "top_fan_in": sorted(fan_in, key=lambda item: item["count"], reverse=True)[:3],
            "dead_hits": dead_hits[:5],
        }


def _risk_label(callers: int, callees: int) -> str:
    score = callers + callees
    if score >= 5:
        return "HIGH"
    if score >= 2:
        return "MEDIUM"
    return "LOW"

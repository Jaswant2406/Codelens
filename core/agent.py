from __future__ import annotations

from dataclasses import dataclass
import re

from .explainer import LocalExplainer
from .models import FunctionNode


@dataclass(slots=True)
class AgentResponse:
    answer: str
    steps: list[dict[str, object]]
    tool_results: dict[str, object]
    fix_suggestions: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "steps": self.steps,
            "tool_results": self.tool_results,
            "fix_suggestions": self.fix_suggestions,
        }


class CodeLensAgent:
    """A lightweight tool-using agent over indexed codebase data."""

    def __init__(self, service: object) -> None:
        self.service = service
        self.local_explainer = LocalExplainer()

    def run(self, query: str) -> AgentResponse:
        """Execute a multi-step tool flow to answer a repo question."""
        self.service.load_state()
        steps: list[dict[str, object]] = []
        query_type = self.service.explainer._classify_query_type(query)
        diagnostic_query = self._is_diagnostic_query(query)
        target_id = self.service.explainer._detect_target_global(query, self.service.functions)
        search_results = self.search_functions(query, target_id=target_id)
        steps.append({
            "tool": "search_functions",
            "detail": f"Found {len(search_results)} relevant function match(es).",
        })

        chain = self.trace_call_chain(search_results[0]["node_id"]) if search_results else []
        steps.append({
            "tool": "trace_call_chain",
            "detail": f"Traced {len(chain)} function(s) in the primary path.",
        })

        file_hits = self.find_file(query)
        steps.append({
            "tool": "find_file",
            "detail": f"Matched {len(file_hits)} likely related file(s).",
        })

        explanation = (
            self.explain_code(
                search_results[0]["node_id"],
                self._agent_explainer_question(query, search_results[0]["name"], diagnostic_query),
            )
            if search_results
            else "I could not find a strong code match yet."
        )
        steps.append({
            "tool": "explain_code",
            "detail": "Built a focused explanation for the strongest match." if search_results else "Skipped because no code match was available.",
        })

        answer = self._compose_answer(query, query_type, search_results, chain, file_hits, explanation, diagnostic_query)
        fix_suggestions = self._build_fix_suggestions(search_results, chain, file_hits)
        return AgentResponse(
            answer=answer,
            steps=steps,
            tool_results={
                "search_functions": search_results,
                "trace_call_chain": chain,
                "find_file": file_hits,
                "explain_code": explanation,
            },
            fix_suggestions=fix_suggestions,
        )

    def search_functions(self, query: str, target_id: str = "") -> list[dict[str, object]]:
        """Find the most relevant functions for a natural-language query."""
        matches = [function for function in self.service.query_engine.search(query, top_k=6) if not self._is_test_path(function.file)]
        query_tokens = self._query_tokens(query)
        ranked: list[tuple[int, FunctionNode]] = []
        seen_ids: set[str] = set()
        for function in self.service.functions:
            if self._is_test_path(function.file):
                continue
            score = 0
            normalized_name = function.name.lower()
            normalized_file = function.file.lower()
            if any(token == normalized_name for token in query_tokens):
                score += 20
            elif any(token in normalized_name for token in query_tokens):
                score += 5
            if any(token in normalized_file for token in query_tokens):
                score += 1
            if score > 0:
                ranked.append((score, function))
        ranked.sort(key=lambda item: (-item[0], item[1].name, item[1].file))
        merged: list[FunctionNode] = []
        if target_id:
            target = self.service.node_details(target_id)
            if target:
                target_node = FunctionNode.from_dict(target)
                merged.append(target_node)
                seen_ids.add(target_id)
        for _, function in ranked:
            if function.node_id not in seen_ids:
                merged.append(function)
                seen_ids.add(function.node_id)
            if len(merged) >= 6:
                break
        for function in matches:
            if function.node_id not in seen_ids:
                merged.append(function)
                seen_ids.add(function.node_id)
            if len(merged) >= 6:
                break
        return [
            {
                "node_id": function.node_id,
                "name": function.name,
                "file": function.file,
                "start_line": function.start_line,
                "docstring": function.docstring or "",
            }
            for function in merged[:6]
        ]

    def trace_call_chain(self, node_id: str) -> list[dict[str, object]]:
        """Trace a short call chain from a function node."""
        chain_ids = self.service.graph_builder.get_chain(node_id, depth=3)
        results: list[dict[str, object]] = []
        for chain_id in chain_ids:
            details = self.service.node_details(chain_id)
            if not details:
                continue
            results.append(
                {
                    "node_id": chain_id,
                    "name": details["name"],
                    "file": details["file"],
                    "start_line": details["start_line"],
                }
            )
        return results

    def find_file(self, query: str) -> list[str]:
        """Find files whose path looks relevant to the query."""
        query_tokens = {token.lower() for token in query.replace("?", " ").split() if len(token) > 2}
        matches = []
        for function in self.service.functions:
            if self._is_test_path(function.file):
                continue
            haystack = function.file.lower()
            if any(token in haystack for token in query_tokens):
                if function.file not in matches:
                    matches.append(function.file)
        return matches[:8]

    def explain_code(self, node_id: str, question: str) -> str:
        """Explain one function using the local explainer."""
        details = self.service.node_details(node_id)
        if not details:
            return "No code details were found for that function."
        node = FunctionNode.from_dict(details)
        if self._is_test_path(node.file):
            return "No relevant functionality found."
        context: dict[str, FunctionNode] = {node.node_id: node}
        call_chain_ids = self.service.graph_builder.get_chain(node.node_id, depth=3)
        for related_id in self.service.graph_builder.get_callers(node.node_id):
            related_details = self.service.node_details(related_id)
            if related_details:
                related_node = FunctionNode.from_dict(related_details)
                if not self._is_test_path(related_node.file):
                    context[related_id] = related_node
        for related_id in call_chain_ids[1:]:
            related_details = self.service.node_details(related_id)
            if related_details:
                related_node = FunctionNode.from_dict(related_details)
                if not self._is_test_path(related_node.file):
                    context[related_id] = related_node
        chunks = list(
            self.local_explainer.explain(
                question=question,
                context=context,
                call_chain=call_chain_ids,
                callers={related_id: self.service.graph_builder.get_callers(related_id) for related_id in context},
                dead_nodes=self.service.graph_builder.dead_code(),
            )
        )
        return "".join(chunks)

    def _compose_answer(
        self,
        query: str,
        query_type: str,
        search_results: list[dict[str, object]],
        chain: list[dict[str, object]],
        file_hits: list[str],
        explanation: str,
        diagnostic_query: bool,
    ) -> str:
        """Compose an intent-aware assistant answer from tool outputs."""
        primary = search_results[0] if search_results else None
        if not primary:
            return (
                "## Short Answer\n\n"
                f"I could not find a confident code match for `{query}` yet.\n\n"
                "## Key Insights\n\n"
                "Try indexing the target repo again, or ask with a file, module, or function hint so I can narrow it down."
            )

        chain_text = " -> ".join(item["name"] for item in chain[:4]) if chain else primary["name"]
        file_text = ", ".join(file_hits[:3]) if file_hits else primary["file"]
        if diagnostic_query or query_type == "DEBUG":
            details = self.service.node_details(primary["node_id"])
            node = FunctionNode.from_dict(details) if details else None
            problems = self._generate_problem_list(node) if node else []
            likely_fixes = self._generate_fix_list(node, problems)
            if "## Possible Causes" in explanation:
                return "\n\n".join(
                    [
                        explanation,
                        "## Likely Fixes",
                        self._numbered_lines(likely_fixes),
                    ]
                )
            return "\n\n".join(
                [
                    "## Debugging Report",
                    f"`{primary['name']}` is the best grounded place to inspect for this issue.",
                    "## What Is Happening",
                    "\n".join(
                        [
                            f"1. The closest match is `{primary['name']}` in `{primary['file']}`.",
                            f"2. The visible path is `{chain_text}`." if chain else f"2. No wider path was confirmed beyond `{primary['name']}`.",
                        ]
                    ),
                    "## Potential Problems",
                    self._numbered_lines(problems or ["No concrete structural problem was confirmed from the current context."]),
                    "## Likely Fixes",
                    self._numbered_lines(likely_fixes),
                    "## Next Checks",
                    self._numbered_lines(
                        [
                            f"Inspect `{primary['name']}` in `{primary['file']}` first.",
                            "Verify whether the visible callers actually reach this path." if chain else "Verify whether this function is reached at runtime.",
                            f"Check nearby files: {file_text}.",
                        ]
                    ),
                ]
            )
        if query_type == "CALLERS":
            return "\n\n".join(
                [
                    "## Short Answer",
                    f"`{primary['name']}` is the symbol most likely tied to your caller question.",
                    "## What Is Happening",
                    "\n".join(
                        [
                            f"1. The likely target is `{primary['name']}` in `{primary['file']}`.",
                            f"2. The visible path is `{chain_text}`." if chain else "2. No broader path was confirmed in the current context.",
                        ]
                    ),
                    "## Key Insights",
                    f"1. Related files worth checking: {file_text}.",
                ]
            )
        return "\n\n".join(
            [
                "## Short Answer",
                f"`{primary['name']}` is the strongest grounded match for your question.",
                "## What Is Happening",
                "\n".join(
                    [
                        f"1. It is defined in `{primary['file']}`.",
                        f"2. The visible path is `{chain_text}`." if chain else f"2. No broader path was confirmed beyond `{primary['name']}`.",
                        f"3. Related files worth checking next: {file_text}.",
                    ]
                ),
                "## Key Insights",
                "1. This answer is grounded only in the currently retrieved code and call relationships.",
            ]
        )

    def _generate_problem_list(self, node: FunctionNode | None) -> list[str]:
        """Generate grounded debugging heuristics for one function."""
        if node is None:
            return []
        problems: list[str] = []
        if re.search(r"except(?:\s+Exception)?\s*:\s*pass", node.code):
            problems.append("Exceptions are swallowed with `pass`.")
        branch_count = len(re.findall(r"\bif\b|\belif\b", node.code))
        if branch_count > 5:
            problems.append(f"The function has {branch_count} conditional branches.")
        if re.search(r"\bglobal\b", node.code):
            problems.append("The function depends on global state.")
        if not re.search(r"\breturn\b", node.code) and re.search(r"[\+\-\*/=]", node.code):
            problems.append("The function performs work without an explicit return.")
        if any(token in node.code for token in ["request", "session", "current_app", "g."]):
            problems.append("The function depends on external application state.")
        return problems

    def _generate_fix_list(self, node: FunctionNode | None, problems: list[str]) -> list[str]:
        """Generate concise, likely fixes from grounded structural problems."""
        if node is None:
            return ["Include the function name explicitly so I can suggest a targeted fix."]
        fixes: list[str] = []
        if any("explicit return" in problem for problem in problems):
            fixes.append(f"Add an explicit `return` in `{node.name}` after the calculation or final branch.")
        if any("Exceptions are swallowed" in problem for problem in problems):
            fixes.append(f"Replace the silent `pass` in `{node.name}` with logging or a handled error path.")
        if any("global state" in problem for problem in problems):
            fixes.append(f"Pass state into `{node.name}` as parameters instead of relying on `global` values.")
        if any("external application state" in problem for problem in problems):
            fixes.append(f"Ensure the required external state is initialized before `{node.name}` runs.")
        if any("conditional branches" in problem for problem in problems):
            fixes.append(f"Simplify the branching in `{node.name}` or add guards for the failing case.")
        if not fixes:
            fixes.append(f"Start by validating the inputs reaching `{node.name}` and compare them with the expected path.")
        return fixes

    def _is_diagnostic_query(self, query: str) -> bool:
        """Return whether the query is asking for failure/debug reasoning."""
        lowered = query.lower()
        return any(token in lowered for token in ["why", "failing", "fails", "not working", "throwing", "raised", "exception"])

    def _agent_explainer_question(self, query: str, function_name: str, diagnostic_query: bool) -> str:
        """Build the question passed into the local explainer."""
        if diagnostic_query:
            return f"why {function_name} is failing"
        return query

    def _numbered_lines(self, lines: list[str]) -> str:
        """Render a simple numbered list."""
        return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))

    def _build_fix_suggestions(
        self,
        search_results: list[dict[str, object]],
        chain: list[dict[str, object]],
        file_hits: list[str],
    ) -> list[str]:
        """Create copy-ready, grounded next-step suggestions."""
        if not search_results:
            return [
                "Re-index the target repository and rerun the query with a specific function, file, or module name.",
                "Ask a narrower question such as `where is auth implemented?` so the retrieval stage can anchor on concrete code.",
            ]

        primary = search_results[0]
        chain_names = [item["name"] for item in chain[:4]]
        suggestions = [
            f"Review `{primary['file']}:{primary['start_line']}` first and confirm whether `{primary['name']}` is the right place to change behavior.",
            f"Trace the call path `{ ' -> '.join(chain_names) if chain_names else primary['name'] }` before editing so you do not miss upstream or downstream side effects.",
        ]
        if file_hits:
            suggestions.append(
                f"Check related files next: {', '.join(f'`{file_path}`' for file_path in file_hits[:3])}."
            )
        suggestions.append(
            "Add or update a focused regression test around the identified path before changing logic so you can verify the fix safely."
        )
        return suggestions

    def _query_tokens(self, query: str) -> list[str]:
        """Extract meaningful identifier-like tokens from the query."""
        stopwords = {
            "how", "what", "why", "where", "who", "works", "work", "internally",
            "internal", "does", "is", "the", "a", "an", "to", "and", "of",
        }
        tokens: list[str] = []
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_\.]*", query.lower()):
            for part in token.split("."):
                normalized = part.strip()
                if normalized and normalized not in stopwords and normalized not in tokens:
                    tokens.append(normalized)
        return tokens

    def _is_test_path(self, path: str) -> bool:
        """Return whether a path points to test code."""
        normalized = path.replace("\\", "/").lower()
        return "/tests/" in normalized or normalized.startswith("tests/")

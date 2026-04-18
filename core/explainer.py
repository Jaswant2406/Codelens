from __future__ import annotations

import re
import time
from collections.abc import Iterator
from itertools import islice

from .llm_client import FreeLLMClient
from .models import FunctionNode, Parameter


class LocalExplainer:
    """
    Generates natural-language code explanations using only static analysis.
    No LLM, no API, no downloads. Runs entirely on parsed AST data.
    """

    def explain(
        self,
        question: str,
        context: dict[str, FunctionNode],
        call_chain: list[str],
        callers: dict[str, list[str]] | None = None,
        dead_nodes: list[str] | None = None,
        metadata_insights: dict[str, object] | None = None,
    ) -> Iterator[str]:
        """Yield explanation string chunks in a streaming-friendly way."""
        intent = self._classify_intent(question)
        grounded = self._ground_context(
            question,
            intent,
            context,
            call_chain,
            callers or {},
            dead_nodes or [],
        )
        if grounded["message"]:
            rendered = grounded["message"]
        else:
            rendered = self._render(
                intent,
                question,
                grounded["context"],
                grounded["call_chain"],
                grounded["callers"],
                grounded["dead_nodes"],
                metadata_insights or {},
            )
        for start in range(0, len(rendered), 80):
            yield rendered[start : start + 80]
            time.sleep(0.01)

    def _ground_context(
        self,
        question: str,
        intent: str,
        context: dict[str, FunctionNode],
        call_chain: list[str],
        callers: dict[str, list[str]],
        dead_nodes: list[str],
    ) -> dict[str, object]:
        """Ground context around a single target function and its direct relationships."""
        if intent in {"EXPLAIN_DEAD", "EXPLAIN_ENTRY", "GENERAL_SUMMARY"}:
            return {
                "message": "",
                "context": context,
                "call_chain": call_chain,
                "callers": callers,
                "dead_nodes": dead_nodes,
            }

        lowered_question = question.lower()
        normalized_question = re.sub(r"[\s_]+", "", lowered_question)
        dotted_parts = [
            part
            for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_\.]*", lowered_question)
            if "." in token
            for part in token.split(".")
            if part
        ]
        dotted_tokens = set(dotted_parts)
        tail_token = dotted_parts[-1] if dotted_parts else ""
        candidates: list[tuple[int, int, int, int, str]] = []
        for node_id, node in context.items():
            normalized_name = re.sub(r"[\s_]+", "", node.name.lower())
            if not normalized_name:
                continue
            if len(normalized_name) == 1 and node.name.lower() not in dotted_tokens and node.name.lower() != tail_token:
                continue
            if normalized_name in normalized_question:
                exact = 1 if re.search(rf"\b{re.escape(node.name.lower())}\b", lowered_question) else 0
                dotted = 1 if node.name.lower() in dotted_tokens else 0
                tail = 1 if tail_token and node.name.lower() == tail_token else 0
                candidates.append((tail, dotted, exact, len(normalized_name), node_id))
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        target_id = candidates[0][4] if candidates else ""
        if not target_id:
            fallback_ids = self._dedupe_lines(
                [node_id for node_id in call_chain if node_id in context] or list(context.keys())
            )[:8]
            fallback_context = {node_id: context[node_id] for node_id in fallback_ids if node_id in context}
            return {
                "message": "",
                "context": fallback_context,
                "call_chain": fallback_ids,
                "callers": {
                    node_id: [caller_id for caller_id in callers.get(node_id, []) if caller_id in fallback_context]
                    for node_id in fallback_context
                },
                "dead_nodes": [node_id for node_id in dead_nodes if node_id in fallback_context],
            }

        target_node = context[target_id]
        direct_caller_ids = [node_id for node_id in callers.get(target_id, []) if node_id in context]
        callee_name_map = {re.sub(r"[\s_]+", "", node.name.lower()): node_id for node_id, node in context.items()}

        bfs_ids: list[str] = []
        queue: list[tuple[str, int]] = [(target_id, 0)]
        visited = {target_id}
        while queue and len(bfs_ids) < 7:
            node_id, depth = queue.pop(0)
            if depth >= 3:
                continue
            node = context[node_id]
            for callee in node.calls:
                callee_id = callee_name_map.get(re.sub(r"[\s_]+", "", callee.lower()))
                if not callee_id or callee_id in visited:
                    continue
                visited.add(callee_id)
                bfs_ids.append(callee_id)
                if len(bfs_ids) >= 7:
                    break
                queue.append((callee_id, depth + 1))

        if not direct_caller_ids and not bfs_ids:
            return {
                "message": "",
                "context": {target_id: target_node},
                "call_chain": [target_id],
                "callers": {target_id: []},
                "dead_nodes": [node_id for node_id in dead_nodes if node_id == target_id],
            }

        ordered_ids = [target_id, *direct_caller_ids, *bfs_ids]
        filtered_ids = []
        for node_id in ordered_ids:
            if node_id not in filtered_ids:
                filtered_ids.append(node_id)
            if len(filtered_ids) >= 8:
                break
        filtered_context = {node_id: context[node_id] for node_id in filtered_ids}
        filtered_callers = {}
        if intent == "EXPLAIN_CALLERS":
            filtered_callers[target_id] = [node_id for node_id in direct_caller_ids if node_id in filtered_context]
        elif direct_caller_ids:
            filtered_callers[target_id] = [node_id for node_id in direct_caller_ids if node_id in filtered_context]

        if intent == "EXPLAIN_CALLERS":
            filtered_chain = [target_id]
        else:
            filtered_chain = [target_id, *[node_id for node_id in bfs_ids if node_id in filtered_context]]

        return {
            "message": "",
            "context": filtered_context,
            "call_chain": filtered_chain,
            "callers": filtered_callers,
            "dead_nodes": [node_id for node_id in dead_nodes if node_id in filtered_context],
        }

    def _detect_target(self, question: str, context: dict[str, FunctionNode]) -> str:
        """Detect the best matching target function from the question."""
        lowered_question = question.lower()
        normalized_question = self._normalize_name(question)
        dotted_parts = [
            part
            for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_\.]*", lowered_question)
            if "." in token
            for part in token.split(".")
            if part
        ]
        tail_token = dotted_parts[-1] if dotted_parts else ""
        candidates: list[tuple[int, int, int, str]] = []
        for node_id, node in context.items():
            normalized_name = self._normalize_name(node.name)
            if not normalized_name:
                continue
            exact = 1 if re.search(rf"\b{re.escape(node.name.lower())}\b", lowered_question) else 0
            tail = 1 if tail_token and node.name.lower() == tail_token else 0
            partial = 1 if normalized_name in normalized_question else 0
            if exact or partial:
                candidates.append((tail, exact, len(normalized_name), node_id))
        if not candidates:
            return ""
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return candidates[0][3]

    def _extract_requested_name(self, question: str) -> str:
        """Extract a readable function name guess from the question."""
        cleaned = re.sub(
            r"\b(how|where|what|who|does|do|is|the|function|flow|trace|chain|callers|calls|implemented|work)\b",
            " ",
            question.lower(),
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?.")
        return cleaned or "that name"

    def _normalize_name(self, value: str) -> str:
        """Normalize names for fuzzy matching."""
        return re.sub(r"[\s_]+", "", value.lower())

    def _classify_intent(self, question: str) -> str:
        """Classify the user's question by intent."""
        lowered = re.sub(r"\s+", " ", question.lower()).strip()
        tokens = [token for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_\.]*", lowered) if token not in {
            "how", "what", "why", "where", "who", "does", "do", "is", "the", "a", "an", "to", "here",
            "works", "work", "internally", "internal", "trace", "flow", "explain",
        }]
        if ".py" in lowered or " file " in f" {lowered} " or " module " in f" {lowered} ":
            return "FILE"
        if any(token in lowered for token in ["dead code", "unused", "unreachable"]):
            return "EXPLAIN_DEAD"
        if any(token in lowered for token in ["who calls", "what uses", "callers"]):
            return "EXPLAIN_CALLERS"
        if any(token in lowered for token in ["entry point", "entry", "starts", "begins", "main function", "execution begins"]):
            return "EXPLAIN_ENTRY"
        if any(token in lowered for token in ["pattern", "concept", "authentication flow", "routing", "singleton"]):
            return "CONCEPT"
        if any(token in lowered for token in ["where", "located", "which file", "what file", "which module"]):
            return "WHERE_LOCATION"
        if any(token in lowered for token in ["why", "not working", "not closing", "fails", "failing", "throwing", "raised", "exception"]):
            return "WHY"
        if lowered.startswith("explain "):
            return "EXPLAIN_FLOW"
        if any(token in lowered for token in ["flow", "trace", "chain", "path"]) or (
            "how does" in lowered and "call" in lowered
        ):
            return "EXPLAIN_FLOW"
        if lowered.startswith("how ") or "how does" in lowered or "how do" in lowered:
            return "HOW_BEHAVIOR"
        if lowered.startswith("what ") or "what does" in lowered or "explain" in lowered:
            return "WHAT_SUMMARY"
        if len(tokens) <= 1:
            return "UNKNOWN"
        return "GENERAL_SUMMARY"

    def _extract_facts(self, node: FunctionNode) -> dict[str, object]:
        """Extract structural facts from a function node."""
        signature = (
            f"{node.name}({', '.join(self._format_parameter(parameter) for parameter in node.parameters)})"
            if self._is_explicit_function(node)
            else node.name
        )
        return_hint = self._return_hint(node.code)
        complexity_score = sum(node.code.count(keyword) for keyword in ["if ", "for ", "while ", "try "])
        if complexity_score <= 1:
            complexity_hint = "simple"
        elif complexity_score <= 3:
            complexity_hint = "moderate"
        else:
            complexity_hint = "complex"
        return {
            "signature": signature,
            "return_hint": return_hint,
            "calls_list": list(node.calls),
            "docstring": node.docstring or "",
            "line_count": max(node.end_line - node.start_line, 1),
            "complexity_hint": complexity_hint,
            "symbol_kind": self._symbol_kind(node),
        }

    def _derive_purpose(self, node: FunctionNode) -> str:
        """Derive a one-line purpose summary from static structure."""
        docstring = (node.docstring or "").strip()
        if docstring:
            summary = self._docstring_summary(docstring)
            if summary:
                return summary
        return_hint = self._return_hint(node.code)
        if return_hint:
            return self._friendly_return_summary(return_hint)
        if node.calls:
            return f"calls {', '.join(f'`{call}`' for call in node.calls[:2])}"
        return "performs internal operations"

    def _build_flow_narrative(self, call_chain: list[str], context: dict[str, FunctionNode]) -> str:
        """Build a readable narrative from an ordered call chain."""
        visible_nodes = self._visible_nodes(call_chain, context)
        if not visible_nodes:
            return "No direct execution path was confirmed from the retrieved context."
        if len(visible_nodes) > 1 and not any(
            any(re.sub(r"[\s_]+", "", callee.lower()) == re.sub(r"[\s_]+", "", next_node.name.lower()) for callee in node.calls)
            for node, next_node in zip(visible_nodes, visible_nodes[1:])
        ):
            return "No direct execution relationships found between retrieved functions."
        if len(visible_nodes) == 1:
            node = visible_nodes[0]
            kind = self._symbol_kind(node)
            return f"1. `{node.name}` is the only confirmed {kind} in the current flow."
        lines = []
        for index, node in enumerate(visible_nodes, start=1):
            facts = self._extract_facts(node)
            branch_note = ""
            if len(node.calls) > 1:
                branch_note = " Branches to " + ", ".join(f"`{call}`" for call in node.calls[:4]) + "."
            lines.append(f"{index}. `{facts['signature']}` {self._derive_purpose(node)}.{branch_note}")
        return "\n".join(lines)

    def _render(
        self,
        intent: str,
        question: str,
        context: dict[str, FunctionNode],
        call_chain: list[str],
        callers: dict[str, list[str]],
        dead_nodes: list[str],
        metadata_insights: dict[str, object],
    ) -> str:
        """Render the final local explanation."""
        functions = list(context.values())
        primary = functions[0] if functions else None
        visible_nodes = self._visible_nodes(call_chain, context)
        sections: list[str] = []

        if intent == "EXPLAIN_FLOW":
            sections = self._render_flow_template(question, functions, visible_nodes, callers)
        elif intent in {"HOW_BEHAVIOR", "WHERE_LOCATION", "WHAT_SUMMARY"}:
            sections = self._render_function_template(question, functions, visible_nodes, callers)
        elif intent == "FILE":
            sections = self._render_file_template(question, functions, visible_nodes)
        elif intent == "WHY":
            sections = self._render_why(primary, functions, visible_nodes, callers)
        elif intent == "CONCEPT":
            sections = [
                self._section(
                    "Overview",
                    "This looks like a concept-level question. Only minimal examples that appear in the current retrieved context are safe to describe.",
                ),
                self._section(
                    "Quick Summary",
                    "No concrete concept implementation was fully available in the current context.",
                ),
            ]
        elif intent == "UNKNOWN":
            sections = [
                self._section("Overview", "Please specify a function, file, or concept to analyze."),
                self._section("Quick Summary", "Not available in current context."),
            ]
        elif intent == "EXPLAIN_CALLERS":
            sections = self._render_callers(context, callers)
        elif intent == "EXPLAIN_DEAD":
            sections = self._render_dead(context, dead_nodes)
        elif intent == "EXPLAIN_ENTRY":
            sections = self._render_entry(functions, callers)
        else:
            sections = self._render_analysis_template(question, functions, visible_nodes, callers)

        observations = self._build_observations(functions, metadata_insights, dead_nodes)
        if observations:
            sections.append(self._section("Key Insights", self._numbered(observations)))

        body = "\n\n".join(section for section in sections if section.strip())
        return f"{body}\n\n---\n{self._footer(context)}"

    def _render_flow(self, visible_nodes: list[FunctionNode], functions: list[FunctionNode]) -> list[str]:
        """Render a flow-focused explanation."""
        flow_line = " -> ".join(f"`{node.name}`" for node in visible_nodes) if visible_nodes else "Insufficient context"
        happening = (
            "The retrieved context forms a call path that can be followed step by step."
            if visible_nodes
            else "I could not confirm a reliable call path from the retrieved functions."
        )
        insights = []
        if visible_nodes:
            insights.append(
                f"The flow starts in `{visible_nodes[0].file}` and currently ends in `{visible_nodes[-1].file}`."
            )
        relationships = self._relationship_summary(visible_nodes or functions)
        if relationships:
            insights.append("Key relationships: " + relationships)
        return [
            self._section("Short Answer", f"The clearest confirmed flow is {flow_line}."),
            self._section("What Is Happening", happening),
            self._section("Step-by-Step Flow", self._build_flow_narrative([node.node_id for node in visible_nodes], {node.node_id: node for node in visible_nodes})),
            self._section("Key Insights", self._numbered(insights) if insights else "1. No additional structural insight was confirmed."),
        ]

    def _render_how(self, primary: FunctionNode | None, visible_nodes: list[FunctionNode]) -> list[str]:
        """Render a behavior-focused explanation."""
        if primary is None:
            return self._render_general([], visible_nodes, "")
        facts = self._extract_facts(primary)
        symbol_kind = str(facts["symbol_kind"])
        parameter_list = ", ".join(f"`{self._format_parameter(parameter)}`" for parameter in primary.parameters) or "no explicit parameters"
        short_answer = (
            f"`{primary.name}` is the main {symbol_kind} to inspect. It appears in `{primary.file}`."
        )
        happening_lines = [
            f"It is represented as `{facts['signature']}`.",
        ]
        if self._is_explicit_function(primary):
            happening_lines.append(f"It accepts {len(primary.parameters)} parameter(s): {parameter_list}.")
            happening_lines.append(f"It spans about {facts['line_count']} line(s) in `{primary.file}`.")
        if facts["docstring"]:
            first_sentence = self._docstring_summary(str(facts["docstring"]))
            if first_sentence:
                happening_lines.append(f'Its docstring says: "{first_sentence}."')
        if facts["calls_list"]:
            happening_lines.append("It delegates work to " + ", ".join(f"`{call}`" for call in facts["calls_list"][:5]) + ".")
        if facts["return_hint"]:
            happening_lines.append(f"It returns {self._friendly_return_summary(str(facts['return_hint']), sentence=False)}.")
        if len(visible_nodes) > 1:
            happening_lines.append(
                "The confirmed path continues through "
                + " -> ".join(f"`{node.name}`" for node in visible_nodes[:4])
                + "."
            )
        insights = [
            f"`{primary.name}` has {len(primary.calls)} visible downstream reference(s) in the retrieved context."
        ]
        if len(visible_nodes) <= 1 and facts["calls_list"]:
            insights.append("Only a narrow slice of the path was retrieved, so downstream details may be incomplete.")
        elif len(visible_nodes) > 1:
            insights.append(f"{len(visible_nodes)} connected step(s) were confirmed in the current path.")
        return [
            self._section("Short Answer", short_answer),
            self._section("What Is Happening", self._numbered(happening_lines)),
            self._section("Step-by-Step Flow", self._build_flow_narrative([node.node_id for node in visible_nodes], {node.node_id: node for node in visible_nodes or [primary]})),
            self._section("Key Insights", self._numbered(insights)),
        ]

    def _render_analysis_template(
        self,
        question: str,
        functions: list[FunctionNode],
        visible_nodes: list[FunctionNode],
        callers: dict[str, list[str]],
    ) -> list[str]:
        """Render a structured analysis view for Ask responses."""
        selected = visible_nodes or list(islice(functions, 5))
        if not selected:
            return [
                self._section("Overview", "Not available in current context."),
                self._section("Quick Summary", "- One-line explanation: Not available in current context."),
            ]

        primary = selected[0]
        overview = (
            f"`{primary.name}` is the clearest starting point in the retrieved context. "
            f"It appears to coordinate the visible flow through {len(selected)} confirmed step(s). "
            f"Its broader role beyond those linked steps is unknown from the current context."
        )

        component_rows = [
            "| Step | Component / Function | Defined In | Calls Next | Responsibility |",
            "|------|----------------------|------------|------------|----------------|",
        ]
        for index, node in enumerate(selected, start=1):
            next_call = (
                ", ".join(f"`{call}`" for call in node.calls[:2])
                if node.calls
                else "Not available in current context"
            )
            component_rows.append(
                f"| {index} | `{self._extract_facts(node)['signature']}` | `{node.file}` | {next_call} | {self._derive_purpose(node)} |"
            )

        detail_blocks = []
        for index, node in enumerate(selected, start=1):
            facts = self._extract_facts(node)
            inputs = ", ".join(f"`{self._format_parameter(parameter)}`" for parameter in node.parameters) or "Not available in current context"
            internal_behavior = []
            if node.calls:
                internal_behavior.append("Calls " + ", ".join(f"`{call}`" for call in node.calls[:4]) + ".")
                side_effects = self._side_effect_calls(node.calls)
                if side_effects:
                    internal_behavior.append("Side effects may occur through " + ", ".join(f"`{call}`" for call in side_effects) + ".")
            if facts["return_hint"]:
                internal_behavior.append("Returns " + self._friendly_return_summary(str(facts["return_hint"]), sentence=False) + ".")
            if not internal_behavior:
                internal_behavior.append("Internal behavior not available in current context.")
            next_call = ", ".join(f"`{call}`" for call in node.calls[:2]) if node.calls else "Next step not available in current context"
            detail_blocks.append(
                "\n".join(
                    [
                        f"{index}. `{facts['signature']}`",
                        f"   - Location: `{node.file}`",
                        f"   - Input received: {inputs}",
                        f"   - Internal behavior: {' '.join(internal_behavior)}",
                        f"   - Next call: {next_call}",
                        f"   - Return value: {self._friendly_return_summary(str(facts['return_hint']), sentence=False) if facts['return_hint'] else 'Not available in current context'}",
                    ]
                )
            )

        flow_lines: list[str] = []
        for node in selected:
            flow_lines.append(f"`{node.name}`")
            if len(node.calls) > 1:
                for call in node.calls[:4]:
                    flow_lines.append(f"├─> `{call}`")
            elif node.calls:
                flow_lines.append(f"↓ `{node.calls[0]}`")
        flow_diagram = "\n".join(flow_lines)
        if len(visible_nodes) < len(functions):
            flow_diagram += "\n(flow truncated due to limited context)"

        object_rows = [
            "| Object | Created At | Modified At | Returned At | Purpose |",
            "|--------|------------|-------------|-------------|---------|",
        ]
        seen_objects = False
        for node in selected:
            for line in node.code.splitlines():
                stripped = line.strip()
                match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", stripped)
                if match:
                    seen_objects = True
                    object_name = match.group(1)
                    object_rows.append(
                        f"| `{object_name}` | `{node.name}` | `{node.name}` | {'`' + node.name + '`' if object_name in (self._return_hint(node.code) or '') else 'Not available in current context'} | Local value used in `{node.name}` |"
                    )
        if not seen_objects:
            object_rows.append("| Not available in current context | - | - | - | - |")

        design_insights = [
            f"`{selected[0].name}` is the first confirmed step in the retrieved path.",
            f"{len(selected)} confirmed step(s) are linked in the current call graph view.",
        ]
        if callers.get(selected[0].node_id):
            design_insights.append(
                f"`{selected[0].name}` has visible caller(s) in the retrieved context."
            )

        snippet_blocks = []
        for node in selected[:2]:
            code_lines = [
                line.rstrip()
                for line in node.code.splitlines()
                if line.strip() and not line.strip().startswith(('"""', "'''"))
            ][:4]
            if not code_lines:
                continue
            annotated = []
            for line in code_lines:
                comment = "# passes work forward" if "(" in line and ")" in line else "# local step"
                annotated.append(f"{line}  {comment}")
            snippet_blocks.append(f"`{node.name}` in `{node.file}`:\n```python\n" + "\n".join(annotated) + "\n```")

        quick_summary = [
            f"- One-line explanation: `{primary.name}` is the main confirmed entry in the retrieved flow.",
            f"- Key facts: {len(selected)} confirmed step(s); defined in `{primary.file}`; next calls are limited to retrieved context.",
            f"- When to use: when tracing how the visible path moves through `{primary.name}` and its linked steps.",
            "- When NOT to use: when you need deeper downstream behavior that is not present in the current context.",
        ]

        return [
            self._section("Overview", overview),
            self._section("Component Table", "\n".join(component_rows)),
            self._section("Detailed Steps", "\n\n".join(detail_blocks)),
            self._section("Flow Diagram", flow_diagram),
            self._section("Object Lifecycle", "\n".join(object_rows)),
            self._section("Key Insights", self._numbered(design_insights)),
            self._section("Annotated Code Snippets", "\n\n".join(snippet_blocks) if snippet_blocks else "Not available in current context."),
            self._section("Quick Summary", "\n".join(quick_summary)),
        ]

    def _render_function_template(
        self,
        question: str,
        functions: list[FunctionNode],
        visible_nodes: list[FunctionNode],
        callers: dict[str, list[str]],
    ) -> list[str]:
        """Render a dedicated function-focused Ask response."""
        selected = visible_nodes or list(islice(functions, 5))
        if not selected:
            return [
                self._section("Overview", f"No grounded function match was confirmed for `{question}`."),
                self._section("Quick Summary", "Not available in current context."),
            ]

        primary = selected[0]
        facts = self._extract_facts(primary)
        direct_callees = [call for call in primary.calls[:5]]
        parameter_list = ", ".join(f"`{self._format_parameter(parameter)}`" for parameter in primary.parameters) or "No explicit parameters"
        detail_lines = [
            f"1. Signature: `{facts['signature']}`",
            f"2. Location: `{primary.file}`",
            f"3. Input received: {parameter_list}",
            f"4. Internal behavior: {self._derive_purpose(primary)}.",
            f"5. Called by: {', '.join(f'`{self._display_name(node_id)}`' for node_id in callers.get(primary.node_id, [])) if callers.get(primary.node_id, []) else 'No callers found in the current retrieved context'}",
            f"6. Direct callees: {', '.join(f'`{call}`' for call in direct_callees) if direct_callees else 'Not available in current context'}",
            f"7. Return value: {self._friendly_return_summary(str(facts['return_hint']), sentence=False) if facts['return_hint'] else 'Not available in current context'}",
        ]
        if len(selected) > 1:
            detail_lines.append(
                "8. Confirmed continuation: " + " -> ".join(f"`{node.name}`" for node in selected[:5])
            )

        component_rows = [
            "|:---:|:---|",
            "| Field | Value |",
            f"| Name | `{primary.name}` |",
            f"| Signature | `{facts['signature']}` |",
            f"| Defined In | `{primary.file}` |",
            f"| Called By | {', '.join(f'`{self._display_name(node_id)}`' for node_id in callers.get(primary.node_id, [])) if callers.get(primary.node_id, []) else 'No callers found in the current retrieved context'} |",
            f"| Direct Callees | {', '.join(f'`{call}`' for call in direct_callees) if direct_callees else 'Not available in current context'} |",
        ]
        return [
            self._section(
                "Overview",
                f"`{primary.name}` is the strongest grounded match for this question. It appears in `{primary.file}` and the sections below describe only the confirmed structure around it.",
            ),
            self._section("Component Table", "\n".join(component_rows)),
            self._section("Detailed Steps", "\n".join(detail_lines)),
            self._section(
                "Flow Diagram",
                self._build_flow_narrative([node.node_id for node in selected], {node.node_id: node for node in selected}),
            ),
            self._section(
                "Key Insights",
                self._numbered(
                    [
                        f"`{primary.name}` exposes {len(primary.parameters)} explicit parameter(s).",
                        f"{len(direct_callees)} direct callee(s) were confirmed in the retrieved context." if direct_callees else "No direct callees were confirmed in the retrieved context.",
                    ]
                ),
            ),
            self._section(
                "Quick Summary",
                "\n".join(
                    [
                        f"- One-line explanation: `{primary.name}` is the main grounded function for this query.",
                        f"- Key facts: defined in `{primary.file}`; signature `{facts['signature']}`; explicit flow limited to retrieved nodes.",
                        "- When to use: when you want the role of one concrete function or method.",
                        "- When NOT to use: when you need file-wide structure or a broader execution chain.",
                    ]
                ),
            ),
        ]

    def _render_flow_template(
        self,
        question: str,
        functions: list[FunctionNode],
        visible_nodes: list[FunctionNode],
        callers: dict[str, list[str]],
    ) -> list[str]:
        """Render a dedicated flow-focused Ask response."""
        selected = visible_nodes or list(islice(functions, 5))
        if not selected:
            return [
                self._section("Overview", f"No grounded execution path was confirmed for `{question}`."),
                self._section("Quick Summary", "Not available in current context."),
            ]

        component_rows = [
            "|:---:|:---|:---|:---|:---|:---|",
            "| Step | Component / Logic | File | Called By | Linkage (Next Call) | Responsibility (Derived Purpose) |",
        ]
        for index, node in enumerate(selected, start=1):
            next_call = ", ".join(f"`{call}`" for call in node.calls[:3]) if node.calls else "Not available in current context"
            caller_text = ", ".join(f"`{self._display_name(node_id)}`" for node_id in callers.get(node.node_id, [])) if callers.get(node.node_id, []) else "No callers found in the current retrieved context"
            component_rows.append(
                f"| {index} | `{self._extract_facts(node)['signature']}` | `{node.file}` | {caller_text} | {next_call} | {self._derive_purpose(node)} |"
            )

        detail_lines = []
        for index, node in enumerate(selected, start=1):
            facts = self._extract_facts(node)
            next_call = ", ".join(f"`{call}`" for call in node.calls[:3]) if node.calls else "Next step not available in current context"
            caller_text = ", ".join(f"`{self._display_name(node_id)}`" for node_id in callers.get(node.node_id, [])) if callers.get(node.node_id, []) else "No callers found in the current retrieved context"
            line = f"{index}. `{facts['signature']}` in `{node.file}` {self._derive_purpose(node)}. Called by: {caller_text}. Next call: {next_call}."
            side_effects = self._side_effect_calls(node.calls)
            if side_effects:
                line += " External action: " + ", ".join(f"`{call}`" for call in side_effects) + "."
            detail_lines.append(line)

        flow_diagram = []
        for node in selected:
            flow_diagram.append(f"`{node.name}`")
            if len(node.calls) > 1:
                for call in node.calls[:4]:
                    flow_diagram.append(f"├─> `{call}`")
            elif node.calls:
                flow_diagram.append(f"↓ `{node.calls[0]}`")
        if len(selected) < len(functions):
            flow_diagram.append("(flow truncated due to limited context)")

        return [
            self._section(
                "Overview",
                f"This architectural flow shows only explicitly connected steps retrieved for `{question}`. The path starts at `{selected[0].name}` and any missing downstream behavior is left unfilled on purpose.",
            ),
            self._section("Component Table", "\n".join(component_rows)),
            self._section("Detailed Steps", "\n".join(detail_lines)),
            self._section("Flow Diagram", "\n".join(flow_diagram)),
            self._section("Key Insights", self._numbered([
                f"{len(selected)} reachable step(s) were confirmed in the current call graph slice.",
                "Branching is shown only where a function makes multiple explicit calls." if any(len(node.calls) > 1 for node in selected) else "No confirmed branching was visible in the retrieved path.",
            ])),
            self._section(
                "Quick Summary",
                "\n".join(
                    [
                        f"- One-line explanation: the visible flow starts from `{selected[0].name}`.",
                        f"- Key facts: {len(selected)} confirmed step(s); files involved: {len({node.file for node in selected})}; only explicit calls are shown.",
                        "- When to use: when you want the observed execution path.",
                        "- When NOT to use: when you need runtime behavior beyond the retrieved edges.",
                    ]
                ),
            ),
        ]

    def _render_file_template(
        self,
        question: str,
        functions: list[FunctionNode],
        visible_nodes: list[FunctionNode],
    ) -> list[str]:
        """Render a dedicated file-focused Ask response."""
        selected = visible_nodes or list(islice(functions, 8))
        if not selected:
            return [
                self._section("Overview", f"No file-level match was confirmed for `{question}`."),
                self._section("Quick Summary", "Not available in current context."),
            ]

        file_path = selected[0].file
        file_nodes = [node for node in selected if node.file == file_path] or selected
        component_rows = [
            "| Symbol | Kind | Defined In | Calls Next | Responsibility |",
            "|--------|------|------------|------------|----------------|",
        ]
        for node in file_nodes[:8]:
            facts = self._extract_facts(node)
            next_call = ", ".join(f"`{call}`" for call in node.calls[:2]) if node.calls else "Not available in current context"
            component_rows.append(
                f"| `{node.name}` | {facts['symbol_kind']} | `{node.file}` | {next_call} | {self._derive_purpose(node)} |"
            )

        detail_lines = []
        for index, node in enumerate(file_nodes[:8], start=1):
            facts = self._extract_facts(node)
            detail_lines.append(
                f"{index}. `{facts['signature']}` is defined in `{node.file}` and {self._derive_purpose(node)}."
            )

        connected = [
            f"`{node.name}` -> {', '.join(f'`{call}`' for call in node.calls[:3])}"
            for node in file_nodes
            if node.calls
        ]

        return [
            self._section(
                "Overview",
                f"`{file_path}` is the matched file in the current context. Only symbols defined in that file and their direct connections are listed here.",
            ),
            self._section("Component Table", "\n".join(component_rows)),
            self._section("Detailed Steps", "\n".join(detail_lines) if detail_lines else "Not available in current context."),
            self._section(
                "Flow Diagram",
                "\n".join(connected[:8]) if connected else "No direct execution relationships found between retrieved functions.",
            ),
            self._section(
                "Key Insights",
                self._numbered(
                    [
                        f"{len(file_nodes)} symbol(s) were confirmed in `{file_path}`.",
                        "Cross-file links are shown only when they are explicit calls." if connected else "No explicit cross-symbol calls were confirmed in the current file context.",
                    ]
                ),
            ),
            self._section(
                "Quick Summary",
                "\n".join(
                    [
                        f"- One-line explanation: `{file_path}` defines the file-local symbols shown above.",
                        f"- Key facts: {len(file_nodes)} symbol(s) listed; matched file `{file_path}`; only explicit direct connections are shown.",
                        "- When to use: when you want a grounded file or module summary.",
                        "- When NOT to use: when you need a repository-wide flow that is not linked to this file.",
                    ]
                ),
            ),
        ]

    def _render_why(
        self,
        primary: FunctionNode | None,
        functions: list[FunctionNode],
        visible_nodes: list[FunctionNode],
        callers: dict[str, list[str]],
    ) -> list[str]:
        """Render a structural why-analysis without runtime speculation."""
        if primary is None:
            return [self._section("Overview", "No specific function was identified for this question. Please include a function name to get a diagnostic answer.")]

        facts = self._extract_facts(primary)
        caller_list = callers.get(primary.node_id, [])
        known_callees = facts["calls_list"][:5]
        return_hint = str(facts["return_hint"]) if facts["return_hint"] else ""
        first_sentence = str(facts["docstring"]).split(".")[0].strip() if facts["docstring"] else ""
        intended_action = "perform its visible operation"
        if return_hint:
            intended_action = f"compute `{return_hint}`"
        elif known_callees:
            intended_action = "delegate work to " + ", ".join(f"`{call}`" for call in known_callees[:2])
        elif first_sentence:
            intended_action = first_sentence
        short_answer = f"`{primary.name}` is intended to {intended_action}."

        happening = [
            f"- Function signature: `{facts['signature']}`",
            f"- Known callers: {', '.join(f'`{self._display_name(node_id)}`' for node_id in caller_list) if caller_list else 'none visible in the current context'}",
            f"- Known callees: {', '.join(f'`{call}`' for call in known_callees) if known_callees else 'none visible in the current context'}",
        ]
        if return_hint:
            happening.append(f"- Return behavior: `{return_hint}`")
        else:
            happening.append("- Return behavior: no explicit return was visible")

        red_flags = self._generate_problem_list(primary, functions, callers)
        if not red_flags:
            red_flags.append("No immediate structural red flag is visible in the retrieved context.")

        investigation = [
            f"Check if `{primary.name}` is registered or called from the expected path.",
            f"Verify execution flow reaches `{primary.name}` before the failing behavior.",
        ]
        if caller_list:
            investigation.append("Inspect upstream caller behavior for the functions listed above.")
        if any(token in primary.code for token in ["current_app", "request", "session", "g.", "global "]):
            investigation.append(f"Confirm required state is set before `{primary.name}` runs.")

        return [
            self._section("Short Answer", short_answer),
            self._section("What Is Happening", "\n".join(happening)),
            self._section("Possible Causes", self._numbered(red_flags)),
            self._section("Investigation Steps", "\n".join(f"- {line}" for line in self._dedupe_lines(investigation))),
        ]

    def _render_where(self, functions: list[FunctionNode], visible_nodes: list[FunctionNode]) -> list[str]:
        """Render a location-focused explanation."""
        selected = visible_nodes or list(islice(functions, 5))
        if not selected:
            return self._render_general([], [], "")
        short_answer = f"Start in `{selected[0].file}` with `{selected[0].name}`."
        happening = self._numbered(
            [f"`{node.name}` is defined in `{node.file}`." for node in selected[:5]]
        )
        flow = self._numbered(
            [f"Open `{node.file}` and inspect `{self._extract_facts(node)['signature']}`." for node in selected[:5]]
        )
        insights = self._numbered(
            [f"{len({node.file for node in selected})} relevant file(s) appear in the retrieved context."]
        )
        return [
            self._section("Short Answer", short_answer),
            self._section("What Is Happening", happening),
            self._section("Step-by-Step Flow", flow),
            self._section("Key Insights", insights),
        ]

    def _render_what(self, functions: list[FunctionNode], visible_nodes: list[FunctionNode], question: str) -> list[str]:
        """Render a summary-focused explanation."""
        selected = visible_nodes or list(islice(functions, 5))
        summaries = [
            f"`{self._extract_facts(node)['signature']}` in `{node.file}`: {self._derive_purpose(node)}."
            for node in selected[:5]
        ]
        short_answer = (
            f"The question is best answered by `{selected[0].name}`."
            if selected
            else f"I could not find a strong match for \"{question}\" in the retrieved context."
        )
        relationships = self._relationship_summary(selected)
        insights = [relationships] if relationships else ["No repeated caller-to-callee pattern stood out in the current set."]
        return [
            self._section("Short Answer", short_answer),
            self._section("What Is Happening", self._numbered(summaries) if summaries else "1. Insufficient context."),
            self._section("Step-by-Step Flow", self._build_flow_narrative([node.node_id for node in selected], {node.node_id: node for node in selected})),
            self._section("Key Insights", self._numbered(insights)),
        ]

    def _render_callers(self, context: dict[str, FunctionNode], callers: dict[str, list[str]]) -> list[str]:
        """Render a caller-focused explanation."""
        target_id = next(iter(context.keys()), "")
        target_name = self._display_name(target_id) if target_id else "target"
        caller_lines = []
        for caller_id in callers.get(target_id, []):
            caller_node = context.get(caller_id)
            if caller_node:
                caller_lines.append(f"`{caller_node.name}` in `{caller_node.file}` calls `{target_name}`.")
            else:
                caller_lines.append(f"`{self._display_name(caller_id)}` calls `{target_name}`.")
        if not caller_lines:
            caller_lines = [f"No callers were found in the current retrieved context for `{target_name}`."]
        return [
            self._section("Short Answer", f"These are the confirmed callers for `{target_name}`."),
            self._section("What Is Happening", self._numbered(caller_lines)),
            self._section("Step-by-Step Flow", self._numbered(["Start from each caller above and follow its use of the target function."])),
            self._section("Key Insights", self._numbered([f"`{target_name}` may be an entry point or unused if no callers were found."])),
        ]

    def _render_dead(self, context: dict[str, FunctionNode], dead_nodes: list[str]) -> list[str]:
        """Render a dead-code explanation."""
        selected = [context[node_id] for node_id in dead_nodes if node_id in context]
        lines = [
            f"`{node.name}` in `{node.file}` lines {node.start_line}-{node.end_line} has no inbound callers."
            for node in selected
        ]
        if not lines:
            lines = ["No unreachable functions were confirmed in the current selection."]
        return [
            self._section("Short Answer", f"I found {len(selected)} unreachable function(s) in the current context."),
            self._section("What Is Happening", self._numbered(lines)),
            self._section("Step-by-Step Flow", self._numbered(["Review each function above and decide whether it should be deleted, wired in, or kept as an intentional entry point."])),
            self._section("Key Insights", self._numbered(["Dead code is defined, but nothing else in the indexed graph currently calls it."])),
        ]

    def _render_entry(self, functions: list[FunctionNode], callers: dict[str, list[str]]) -> list[str]:
        """Render an entry-point explanation."""
        entry_nodes = [node for node in functions if not callers.get(node.node_id)]
        top_entry = self._pick_top_entry(entry_nodes or functions)
        lines = [f"`{node.name}` in `{node.file}` has no visible callers." for node in entry_nodes[:5]]
        if not lines:
            lines = ["No uncalled entry points were confirmed in the current selection."]
        return [
            self._section("Short Answer", f"`{top_entry.name}` looks like the strongest entry point in the retrieved set."),
            self._section("What Is Happening", self._numbered(lines)),
            self._section("Step-by-Step Flow", self._numbered([f"Start with `{top_entry.name}` and follow its callees to understand the outward flow."])),
            self._section("Key Insights", self._numbered([f"`{top_entry.name}` was chosen because {self._entry_reason(top_entry)}."])),
        ]

    def _render_general(self, functions: list[FunctionNode], visible_nodes: list[FunctionNode], question: str) -> list[str]:
        """Render a general explanation when intent is broad or unclear."""
        selected = visible_nodes or list(islice(functions, 5))
        short_answer = (
            f"The strongest match for this question is `{selected[0].name}`."
            if selected
            else f"I could not confidently answer \"{question}\" from the retrieved context alone."
        )
        happening = self._numbered(
            [f"`{self._extract_facts(node)['signature']}` in `{node.file}`: {self._derive_purpose(node)}." for node in selected[:5]]
        ) if selected else "1. Insufficient context."
        insights = self._numbered(
            ["The answer is based only on the currently retrieved functions and call relationships."]
        )
        return [
            self._section("Short Answer", short_answer),
            self._section("What Is Happening", happening),
            self._section("Step-by-Step Flow", self._build_flow_narrative([node.node_id for node in selected], {node.node_id: node for node in selected})),
            self._section("Key Insights", insights),
        ]

    def _build_observations(
        self,
        functions: list[FunctionNode],
        metadata_insights: dict[str, object],
        dead_nodes: list[str],
        ) -> list[str]:
        """Build grounded observations from the retrieved subgraph."""
        observations: list[str] = []
        if metadata_insights.get("retrieved_function_count", 0) <= 2 or metadata_insights.get("subgraph_node_count", 0) <= 2:
            observations.append("This may be due to partial context retrieval.")
        return self._dedupe_lines(observations)

    def _generate_problem_list(
        self,
        node: FunctionNode,
        functions: list[FunctionNode],
        callers: dict[str, list[str]] | None = None,
    ) -> list[str]:
        """Generate grounded structural red flags for diagnostics."""
        caller_map = callers or {}
        problems: list[str] = []
        if not caller_map.get(node.node_id, []):
            problems.append(f"`{node.name}` has no visible callers.")
        if "except:" in node.code or "except Exception:" in node.code:
            if re.search(r"except(?:\s+Exception)?\s*:\s*pass", node.code):
                problems.append(f"`{node.name}` suppresses exceptions with `pass`.")
        branch_count = len(re.findall(r"\bif\b|\belif\b", node.code))
        if branch_count > 5:
            problems.append(f"`{node.name}` has {branch_count} conditional branches, which raises structural complexity.")
        if re.search(r"\bglobal\b", node.code):
            problems.append(f"`{node.name}` uses global state.")
        return_hint = self._return_hint(node.code)
        if not return_hint and re.search(r"[\+\-\*/=]", node.code):
            problems.append(f"`{node.name}` performs work but has no explicit return.")
        missing_callees = [call for call in node.calls[:8] if call not in {fn.name for fn in functions}]
        if missing_callees:
            problems.append("It calls function(s) not present in the current context: " + ", ".join(f"`{call}`" for call in missing_callees) + ".")
        if re.search(r"\bif\b", node.code) and not re.search(r"\belse\b", node.code):
            problems.append(f"`{node.name}` has a conditional path without a visible fallback branch.")
        if any(token in node.code for token in ["current_app", "request", "session", "g.", "global "]):
            problems.append(f"`{node.name}` depends on external application or request state.")
        return self._dedupe_lines(problems)

    def _visible_nodes(self, call_chain: list[str], context: dict[str, FunctionNode]) -> list[FunctionNode]:
        """Return unique visible nodes in call-chain order."""
        seen: set[str] = set()
        visible: list[FunctionNode] = []
        for node_id in call_chain:
            if node_id in context and node_id not in seen:
                visible.append(context[node_id])
                seen.add(node_id)
        return visible

    def _section(self, title: str, body: str) -> str:
        """Build a consistently formatted section."""
        return f"## {title}\n{body.strip()}"

    def _numbered(self, lines: list[str]) -> str:
        """Render numbered lines with consistent spacing and deduplication."""
        cleaned = self._dedupe_lines([line.strip() for line in lines if line and line.strip()])
        if not cleaned:
            return "1. Insufficient context."
        return "\n".join(f"{index}. {line}" for index, line in enumerate(cleaned, start=1))

    def _dedupe_lines(self, lines: list[str]) -> list[str]:
        """Deduplicate lines while preserving order."""
        seen: set[str] = set()
        deduped: list[str] = []
        for line in lines:
            normalized = " ".join(line.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(line)
        return deduped

    def _format_parameter(self, parameter: Parameter) -> str:
        """Format a parameter for display."""
        if parameter.type_hint:
            return f"{parameter.name}: {parameter.type_hint}"
        return parameter.name

    def _is_explicit_function(self, node: FunctionNode) -> bool:
        """Return whether the symbol is explicitly defined like a function."""
        stripped = node.code.lstrip()
        return bool(node.parameters) or stripped.startswith("def ") or stripped.startswith("async def ")

    def _symbol_kind(self, node: FunctionNode) -> str:
        """Classify the symbol conservatively."""
        if self._is_explicit_function(node):
            return "function"
        return "object / proxy"

    def _return_hint(self, code: str) -> str:
        """Extract the first return expression from code."""
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("return "):
                return stripped[len("return ") :].strip()[:60]
        return ""

    def _display_name(self, node_id: str) -> str:
        """Get the short function name from a node id."""
        return node_id.split("::")[-1]

    def _pick_top_entry(self, entries: list[FunctionNode]) -> FunctionNode:
        """Choose the most likely entry point."""
        return sorted(
            entries,
            key=lambda node: (
                0 if node.name.lower() in {"main", "run", "start"} else 1,
                -len(node.calls),
                node.name,
            ),
        )[0]

    def _entry_reason(self, node: FunctionNode) -> str:
        """Explain the likely entry-point choice."""
        if node.name.lower() in {"main", "run", "start"}:
            return "its name suggests startup behavior"
        if node.calls:
            return "it has the most callees among visible entry functions"
        return "it is an uncalled top-level function"

    def _relationship_summary(self, functions: list[FunctionNode]) -> str:
        """Summarize top caller-to-callee relationships."""
        pairs = []
        for function in functions:
            for callee in function.calls[:2]:
                if callee == function.name:
                    continue
                pairs.append(f"`{function.name}` calls `{callee}`")
        return ", ".join(islice(self._dedupe_lines(pairs), 3))

    def _footer(self, context: dict[str, FunctionNode]) -> str:
        """Build the standard analysis footer."""
        file_count = len({node.file for node in context.values()})
        return f"_Analysis based on static code structure. {len(context)} function(s) retrieved from {file_count} file(s)._"

    def _normalize_sentence(self, text: str) -> str:
        """Normalize text into a cleaner sentence."""
        text = " ".join(text.split())
        return text.rstrip(".")

    def _docstring_summary(self, docstring: str) -> str:
        """Extract the first concise sentence from a docstring."""
        cleaned = " ".join(
            line.strip()
            for line in docstring.splitlines()
            if line.strip() and not line.strip().startswith(">>>")
        )
        if not cleaned:
            return ""
        summary = re.split(r"\.(?:\s|$)|:param\b|Args:|Parameters\b|Returns\b|Example[s]?\b", cleaned, maxsplit=1)[0]
        return self._normalize_sentence(summary.strip())

    def _friendly_return_summary(self, return_hint: str, sentence: bool = True) -> str:
        """Turn a return expression into a friendlier summary."""
        cleaned = return_hint.strip()
        if ".upper()" in cleaned:
            summary = "the helper result converted to uppercase"
        elif ".lower()" in cleaned:
            summary = "the helper result converted to lowercase"
        elif "ensure_sync" in cleaned:
            summary = "the selected view wrapped so it can run safely in the current request flow"
        elif "make_response" in cleaned:
            summary = "a response object built from the view result"
        elif 'f"' in cleaned or "f'" in cleaned:
            summary = "a formatted string"
        elif cleaned in {"None", ""}:
            summary = "no meaningful value"
        else:
            summary = cleaned
        if sentence:
            return f"returns {summary}"
        return summary

    def _side_effect_calls(self, calls: list[str]) -> list[str]:
        """Return calls that likely represent external side effects."""
        markers = ("request", "post", "get", "put", "delete", "connect", "execute", "open", "write", "save", "send")
        side_effects: list[str] = []
        for call in calls:
            lowered = call.lower()
            if "." in call or any(marker in lowered for marker in markers):
                if call not in side_effects:
                    side_effects.append(call)
        return side_effects[:4]


class Explainer:
    """Explain search results using free providers with a local fallback."""

    def __init__(self, llm_client: FreeLLMClient) -> None:
        self.llm_client = llm_client
        self._local = LocalExplainer()

    def _classify_query_type(self, question: str) -> str:
        """Classify a query before retrieval routing."""
        lowered = question.lower()
        if any(token in lowered for token in ["fix", "bug", "issue", "debug"]):
            return "DEBUG"
        if any(token in lowered for token in ["who calls", "called by"]):
            return "CALLERS"
        if any(token in lowered for token in ["how", "flow", "works", "trace"]):
            return "FLOW"
        if any(token in lowered for token in ["what is", "define"]):
            return "DEFINITION"
        return "GENERAL"

    def _detect_target_global(self, question: str, all_functions: list[FunctionNode]) -> str:
        """Detect the best global target match from all indexed functions."""
        lowered_question = question.lower()
        normalized_question = re.sub(r"[\s_]+", "", lowered_question)
        dotted_parts = [
            part
            for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_\.]*", lowered_question)
            if "." in token
            for part in token.split(".")
            if part
        ]
        dotted_tokens = set(dotted_parts)
        tail_token = dotted_parts[-1] if dotted_parts else ""
        candidates: list[tuple[int, int, int, int, str]] = []
        for function in all_functions:
            normalized_name = re.sub(r"[\s_]+", "", function.name.lower())
            if not normalized_name:
                continue
            if len(normalized_name) == 1 and function.name.lower() not in dotted_tokens and function.name.lower() != tail_token:
                continue
            exact = 1 if re.search(rf"\b{re.escape(function.name.lower())}\b", lowered_question) else 0
            dotted = 1 if function.name.lower() in dotted_tokens else 0
            tail = 1 if tail_token and function.name.lower() == tail_token else 0
            partial = 1 if normalized_name in normalized_question else 0
            if exact or partial:
                candidates.append((tail, dotted, exact, len(normalized_name), function.node_id))
        if not candidates:
            return ""
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        return candidates[0][4]

    def stream_explain(
        self,
        question: str,
        context: list[dict[str, str | int]],
        call_chain: list[str],
        callers: dict[str, list[str]] | None = None,
        dead_nodes: list[str] | None = None,
        metadata_insights: dict[str, object] | None = None,
    ) -> Iterator[str]:
        """Stream an explanation for the supplied context."""
        node_context = {
            str(item["node_id"]): FunctionNode(
                node_id=str(item["node_id"]),
                name=str(item["name"]),
                file=str(item["file"]),
                language="python",
                start_line=int(item.get("start_line", 0)),
                end_line=int(item.get("start_line", 0)) + max(len(str(item.get("code", "")).splitlines()) - 1, 0),
                docstring=str(item.get("docstring", "")) or None,
                parameters=[
                    Parameter(
                        name=str(parameter.get("name", "")),
                        type_hint=str(parameter["type_hint"]) if parameter.get("type_hint") else None,
                    )
                    for parameter in item.get("params", [])
                    if isinstance(parameter, dict) and parameter.get("name")
                ],
                calls=[
                    str(call)
                    for call in item.get("calls", _extract_calls_from_code(str(item.get("code", ""))))
                    if str(call)
                ],
                code=str(item.get("code", "")),
            )
            for item in context
        }
        _diagnostic_tokens = {"why", "failing", "fails", "not working", "not closing", "error"}
        _is_diagnostic = any(token in question.lower() for token in _diagnostic_tokens)

        system_prompt = """You are a coordinated system of expert agents working as a Principal Software Architect and Static Code Analysis Expert.

Follow this workflow internally:
1. PLANNER: identify entry points, key functions, call paths, unknowns, and what must be verified.
2. ANALYZER: reason only from the retrieved call graph and code. Trace execution, data transformations, cross-file interactions, state changes, external systems, sensitive logic, dead code, fan-in/fan-out, and hotspots when supported.
3. VERIFIER: remove any unsupported claim. If something is missing, say "Insufficient context".

You MUST use only the provided context. Do not hallucinate missing logic.

Final response style:
- Start with a direct concise answer
- Then explain naturally like a senior developer
- Keep it conversational but technical
- Use short paragraphs
- Use backticks for function names
- Use these sections when helpful: Short Answer, What's Happening Overall, Step-by-Step Flow, Important Technical Insights, Issues / Observations
- Avoid sounding like a report or prompt dump
"""
        formatted_context_blocks = "\n\n".join(
            (
                f"File path: {item['file']}\n"
                f"Function: {item['name']}\n"
                f"Callers: {', '.join((callers or {}).get(str(item['node_id']), [])) or 'Unknown from current context'}\n"
                f"Callees: {', '.join(_extract_calls_from_code(str(item.get('code', '')))) or 'None visible'}\n"
                f"Raw code:\n{item['code']}"
            )
            for item in context
        )
        user_prompt = (
            "### SYSTEM CONTEXT\n"
            "- Project Name: CodeLens\n"
            "- Analysis Type: Function-level call graph (directed)\n"
            "- Retrieval Method: Hybrid Semantic Search + Graph Expansion\n"
            "- Node Type: AST-extracted function nodes\n\n"
            f"### USER QUERY\n{question}\n\n"
            "### RETRIEVED CONTEXTUAL SUBGRAPH (SOURCE OF TRUTH)\n"
            f"{formatted_context_blocks}\n\n"
            "### ARCHITECTURAL METADATA / INSIGHTS\n"
            f"{metadata_insights or {}}\n"
        )
        if not _is_diagnostic:
            try:
                produced_output = False
                buffered_chunks: list[str] = []
                for chunk in self.llm_client.stream(user_prompt, system=system_prompt):
                    buffered_chunks.append(chunk)
                combined = "".join(buffered_chunks).strip()
                if combined and not combined.startswith("Local fallback response for prompt:"):
                    produced_output = True
                    for chunk in buffered_chunks:
                        yield chunk
                if produced_output:
                    return
            except Exception:
                pass
        yield from self._local.explain(
            question,
            node_context,
            call_chain,
            callers=callers,
            dead_nodes=dead_nodes,
            metadata_insights=metadata_insights,
        )


def _extract_calls_from_code(code: str) -> list[str]:
    """Extract rough call names from source code."""
    pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_\.]*)\(")
    calls: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(("def ", "async def ", "class ")):
            continue
        for match in pattern.findall(stripped):
            if match not in calls and match != "return":
                calls.append(match)
    return calls

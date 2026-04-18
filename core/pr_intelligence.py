from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .models import FunctionNode


@dataclass(slots=True)
class PRAnalysisResult:
    impacted_files: list[str]
    affected_functions: list[dict[str, object]]
    risk_level: str
    summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "impacted_files": self.impacted_files,
            "affected_functions": self.affected_functions,
            "risk_level": self.risk_level,
            "summary": self.summary,
        }


class PRIssueIntelligence:
    """Analyze GitHub issues and diffs against the indexed codebase."""

    def __init__(self, functions: Iterable[FunctionNode]) -> None:
        self.functions = list(functions)
        self.functions_by_file: dict[str, list[FunctionNode]] = {}
        for function in self.functions:
            self.functions_by_file.setdefault(function.file, []).append(function)

    def analyze(self, issue_text: str = "", pr_diff: str = "") -> PRAnalysisResult:
        """Return impacted files, affected functions, risk, and a summary."""
        issue_text, pr_diff = self._normalize_inputs(issue_text, pr_diff)
        impacted_files = self._extract_impacted_files(pr_diff)
        changed_ranges = self._extract_changed_ranges(pr_diff)
        changed_lines = self._extract_changed_lines(pr_diff)
        issue_tokens = self._issue_tokens(issue_text)
        affected = self._affected_functions(impacted_files, changed_ranges, changed_lines, issue_tokens)
        risk_level = self._risk_level(impacted_files, affected, pr_diff, issue_text)
        summary = self._summary(issue_text, impacted_files, affected, risk_level)
        return PRAnalysisResult(
            impacted_files=impacted_files,
            affected_functions=affected,
            risk_level=risk_level,
            summary=summary,
        )

    def _normalize_inputs(self, issue_text: str, pr_diff: str) -> tuple[str, str]:
        """Detect and correct swapped issue/diff inputs from the UI."""
        issue_text = issue_text.strip()
        pr_diff = pr_diff.strip()
        issue_looks_like_diff = self._looks_like_diff(issue_text)
        diff_looks_like_diff = self._looks_like_diff(pr_diff)
        if issue_looks_like_diff and not diff_looks_like_diff:
            return pr_diff, issue_text
        return issue_text, pr_diff

    def _looks_like_diff(self, text: str) -> bool:
        """Heuristically detect unified diff content."""
        lowered = text.lower()
        markers = ("diff --git ", "@@ ", "+++ ", "--- ", "index ")
        return any(marker in lowered for marker in markers)

    def _extract_impacted_files(self, pr_diff: str) -> list[str]:
        files: list[str] = []
        for line in pr_diff.splitlines():
            if line.startswith("diff --git "):
                parts = line.split()
                if len(parts) >= 4:
                    candidate = parts[3]
                    if candidate.startswith("b/"):
                        candidate = candidate[2:]
                    if candidate not in files:
                        files.append(candidate)
            elif line.startswith("+++ b/"):
                candidate = line[6:]
                if candidate not in files:
                    files.append(candidate)
        return files

    def _extract_changed_ranges(self, pr_diff: str) -> dict[str, list[tuple[int, int]]]:
        file_ranges: dict[str, list[tuple[int, int]]] = {}
        current_file: str | None = None
        hunk_pattern = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
        for line in pr_diff.splitlines():
            if line.startswith("diff --git "):
                parts = line.split()
                current_file = parts[3][2:] if len(parts) >= 4 and parts[3].startswith("b/") else None
            elif line.startswith("+++ b/"):
                current_file = line[6:]
            elif current_file and line.startswith("@@"):
                match = hunk_pattern.search(line)
                if not match:
                    continue
                start = int(match.group(1))
                length = int(match.group(2) or "1")
                file_ranges.setdefault(current_file, []).append((start, start + max(length - 1, 0)))
        return file_ranges

    def _extract_changed_lines(self, pr_diff: str) -> dict[str, list[str]]:
        """Collect added/removed line content per file for lightweight semantic matching."""
        changed_lines: dict[str, list[str]] = {}
        current_file: str | None = None
        for raw_line in pr_diff.splitlines():
            line = raw_line.rstrip("\n")
            if line.startswith("diff --git "):
                parts = line.split()
                current_file = parts[3][2:] if len(parts) >= 4 and parts[3].startswith("b/") else None
            elif line.startswith("+++ b/"):
                current_file = line[6:]
            elif current_file and (line.startswith("+") or line.startswith("-")):
                if line.startswith("+++") or line.startswith("---"):
                    continue
                stripped = line[1:].strip()
                if stripped:
                    changed_lines.setdefault(current_file, []).append(stripped)
        return changed_lines

    def _issue_tokens(self, issue_text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", issue_text.lower())
            if token not in {"the", "and", "for", "with", "that", "this", "from"}
        }

    def _affected_functions(
        self,
        impacted_files: list[str],
        changed_ranges: dict[str, list[tuple[int, int]]],
        changed_lines: dict[str, list[str]],
        issue_tokens: set[str],
    ) -> list[dict[str, object]]:
        scored: list[tuple[int, dict[str, object]]] = []

        for file_path in impacted_files:
            for function in self.functions_by_file.get(file_path, []):
                score, reasons = self._score_function(
                    function,
                    changed_ranges.get(file_path, []),
                    changed_lines.get(file_path, []),
                    issue_tokens,
                )
                if score > 0:
                    scored.append((score, self._function_payload(function, reasons, score)))

        if not scored and issue_tokens:
            for function in self.functions:
                score, reasons = self._score_function(function, [], [], issue_tokens)
                if score > 0:
                    scored.append((score, self._function_payload(function, reasons, score)))
                if len(scored) >= 8:
                    break

        scored.sort(key=lambda item: (item[0], -item[1]["start_line"]), reverse=True)
        unique: list[dict[str, object]] = []
        seen: set[str] = set()
        for _, payload in scored:
            if payload["node_id"] in seen:
                continue
            seen.add(payload["node_id"])
            unique.append(payload)
        return unique[:12]

    def _score_function(
        self,
        function: FunctionNode,
        ranges: list[tuple[int, int]],
        changed_lines: list[str],
        issue_tokens: set[str],
    ) -> tuple[int, list[str]]:
        """Score how strongly a function matches the issue and diff."""
        score = 0
        reasons: list[str] = []

        if self._intersects(function, ranges):
            score += 5
            reasons.append("Changed lines overlap this function")

        lowered_changed = " ".join(changed_lines).lower()
        if function.name.lower() in lowered_changed:
            score += 3
            reasons.append("Function name appears directly in the diff")

        if changed_lines and any(line.strip() and line.strip() in function.code for line in changed_lines[:8]):
            score += 2
            reasons.append("Changed code text matches this function body")

        token_hits = self._issue_overlap(function, issue_tokens)
        if token_hits:
            score += min(token_hits, 3)
            reasons.append("Issue wording overlaps this function")

        return score, reasons

    def _intersects(self, function: FunctionNode, ranges: list[tuple[int, int]]) -> bool:
        for start, end in ranges:
            if function.start_line <= end and function.end_line >= start:
                return True
        return False

    def _matches_issue(self, function: FunctionNode, issue_tokens: set[str]) -> bool:
        return self._issue_overlap(function, issue_tokens) > 0

    def _issue_overlap(self, function: FunctionNode, issue_tokens: set[str]) -> int:
        """Count how many issue tokens overlap a function's local context."""
        haystack = " ".join(
            [
                function.name.lower(),
                function.file.lower(),
                (function.docstring or "").lower(),
                " ".join(call.lower() for call in function.calls),
            ]
        )
        return sum(1 for token in issue_tokens if token in haystack)

    def _function_payload(self, function: FunctionNode, reasons: list[str], score: int) -> dict[str, object]:
        return {
            "node_id": function.node_id,
            "name": function.name,
            "file": function.file,
            "start_line": function.start_line,
            "end_line": function.end_line,
            "reason": "; ".join(reasons) if reasons else self._reason(function),
            "match_score": score,
        }

    def _reason(self, function: FunctionNode) -> str:
        if function.docstring:
            return function.docstring.split(". ")[0].strip().rstrip(".")
        if function.calls:
            return f"Calls {', '.join(function.calls[:2])}"
        return "Touches local logic in this file"

    def _risk_level(
        self,
        impacted_files: list[str],
        affected_functions: list[dict[str, object]],
        pr_diff: str,
        issue_text: str,
    ) -> str:
        score = 0
        score += len(impacted_files)
        score += min(len(affected_functions), 6)
        if any("/tests/" in file_path.replace("\\", "/") or file_path.startswith("tests/") for file_path in impacted_files):
            score -= 1
        if any(keyword in pr_diff.lower() for keyword in ["auth", "session", "security", "permission"]):
            score += 3
        if any(keyword in issue_text.lower() for keyword in ["crash", "production", "security", "data loss"]):
            score += 3
        if score >= 10:
            return "HIGH"
        if score >= 5:
            return "MEDIUM"
        return "LOW"

    def _summary(
        self,
        issue_text: str,
        impacted_files: list[str],
        affected_functions: list[dict[str, object]],
        risk_level: str,
    ) -> str:
        if not impacted_files and not affected_functions:
            return (
                "## Short Answer\n\n"
                "I could not map the issue or diff to concrete files or functions yet.\n\n"
                "## Issues / Observations\n\n"
                "Try pasting a fuller issue summary or a larger diff so I have more structure to match against the indexed codebase."
            )

        issue_line = issue_text.strip().splitlines()[0] if issue_text.strip() else "No issue summary was provided."
        file_text = ", ".join(f"`{file_path}`" for file_path in impacted_files[:3]) or "the indexed codebase"
        function_text = ", ".join(f"`{item['name']}`" for item in affected_functions[:4]) or "file-level changes"
        top_reason = affected_functions[0]["reason"] if affected_functions else "No function-level explanation was available."

        return (
            "## Short Answer\n\n"
            "Here is the likely impact for this GitHub change.\n\n"
            "## What Changed\n\n"
            f"Issue focus: {issue_line}\n\n"
            "## Important Technical Insights\n\n"
            f"The change appears to touch {file_text}. The most relevant function-level hits are {function_text}.\n\n"
            f"Top match reason: {top_reason}.\n\n"
            "## Issues / Observations\n\n"
            f"Risk looks {risk_level.lower()} based on how much code is affected and how close the change is to runtime behavior."
        )

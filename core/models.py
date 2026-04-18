from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Parameter:
    name: str
    type_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FunctionNode:
    node_id: str
    name: str
    file: str
    language: str
    start_line: int
    end_line: int
    docstring: str | None = None
    parameters: list[Parameter] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    code: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["params"] = [parameter.to_dict() for parameter in self.parameters]
        payload.pop("parameters", None)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FunctionNode":
        return cls(
            node_id=payload["node_id"],
            name=payload["name"],
            file=payload["file"],
            language=payload["language"],
            start_line=payload["start_line"],
            end_line=payload["end_line"],
            docstring=payload.get("docstring"),
            parameters=[
                Parameter(**parameter)
                for parameter in payload.get("params", payload.get("parameters", []))
            ],
            calls=list(payload.get("calls", [])),
            code=payload.get("code", ""),
        )


@dataclass(slots=True)
class RepoFile:
    path: str
    language: str
    content: str
    hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".java": "java",
}

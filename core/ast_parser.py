from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable

from tree_sitter import Node
from tree_sitter_languages import get_parser

from .models import FunctionNode, Parameter, RepoFile


FUNCTION_TYPES = {
    "python": {"function_definition"},
    "javascript": {"function_declaration", "method_definition"},
    "typescript": {"function_declaration", "method_definition"},
    "go": {"function_declaration", "method_declaration"},
    "java": {"method_declaration", "constructor_declaration"},
}

CALL_TYPES = {
    "python": {"call"},
    "javascript": {"call_expression"},
    "typescript": {"call_expression"},
    "go": {"call_expression"},
    "java": {"method_invocation"},
}

STRING_TYPES = {
    "python": {"string", "string_literal"},
    "javascript": {"string", "template_string"},
    "typescript": {"string", "template_string"},
    "go": {"interpreted_string_literal", "raw_string_literal"},
    "java": {"string_literal"},
}


@dataclass(slots=True)
class _ParseContext:
    lines: list[str]
    file: RepoFile


def parse_functions(files: Iterable[RepoFile]) -> list[FunctionNode]:
    functions: list[FunctionNode] = []
    for file in files:
        if file.language == "python":
            functions.extend(_parse_python_functions(file))
            continue
        try:
            parser = get_parser(file.language)
        except Exception:
            continue
        tree = parser.parse(file.content.encode("utf-8"))
        context = _ParseContext(lines=file.content.splitlines(), file=file)
        functions.extend(_collect_functions(tree.root_node, file.language, context))
    return functions


def _parse_python_functions(file: RepoFile) -> list[FunctionNode]:
    module = ast.parse(file.content)
    lines = file.content.splitlines()
    functions: list[FunctionNode] = []

    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", node.lineno)
        code = "\n".join(lines[start_line - 1 : end_line])
        parameters: list[Parameter] = []
        for arg in node.args.args:
            type_hint = ast.unparse(arg.annotation) if arg.annotation else None
            parameters.append(Parameter(name=arg.arg, type_hint=type_hint))

        functions.append(
            FunctionNode(
                node_id=f"{file.path}::{node.name}",
                name=node.name,
                file=file.path,
                language=file.language,
                start_line=start_line,
                end_line=end_line,
                docstring=ast.get_docstring(node),
                parameters=parameters,
                calls=sorted(_collect_python_calls(node)),
                code=code,
            )
        )

    return functions


def _collect_python_calls(node: ast.AST) -> set[str]:
    calls: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            calls.add(_python_call_name(child.func))
    return {call for call in calls if call}


def _python_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _python_call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _collect_functions(node: Node, language: str, context: _ParseContext) -> list[FunctionNode]:
    items: list[FunctionNode] = []
    if node.type in FUNCTION_TYPES.get(language, set()):
        function = _build_function(node, language, context)
        if function:
            items.append(function)
    for child in node.children:
        items.extend(_collect_functions(child, language, context))
    return items


def _build_function(node: Node, language: str, context: _ParseContext) -> FunctionNode | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None

    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    code = "\n".join(context.lines[start_line - 1 : end_line])
    name = _node_text(name_node, context.file.content)

    return FunctionNode(
        node_id=f"{context.file.path}::{name}",
        name=name,
        file=context.file.path,
        language=language,
        start_line=start_line,
        end_line=end_line,
        docstring=_extract_docstring(node, language, context.file.content),
        parameters=_extract_parameters(node, language, context.file.content),
        calls=sorted(_extract_calls(node, language, context.file.content)),
        code=code,
    )


def _extract_parameters(node: Node, language: str, source: str) -> list[Parameter]:
    parameters_node = node.child_by_field_name("parameters")
    if parameters_node is None:
        return []
    parameters: list[Parameter] = []
    for child in parameters_node.children:
        if not child.is_named:
            continue
        name_node = child.child_by_field_name("name") or child
        type_node = child.child_by_field_name("type")
        if language == "python" and child.type == "identifier":
            name_node = child
        name = _node_text(name_node, source).strip()
        if not name or name in {",", "(", ")"}:
            continue
        type_hint = _node_text(type_node, source).strip() if type_node else None
        parameters.append(Parameter(name=name, type_hint=type_hint or None))
    return parameters


def _extract_docstring(node: Node, language: str, source: str) -> str | None:
    body = node.child_by_field_name("body")
    if body is None:
        return None
    for child in body.children:
        if child.type in {"block", "statement_block"}:
            for nested in child.children:
                if nested.type in STRING_TYPES.get(language, set()):
                    return _clean_string(_node_text(nested, source))
                if nested.type == "expression_statement":
                    for expr_child in nested.children:
                        if expr_child.type in STRING_TYPES.get(language, set()):
                            return _clean_string(_node_text(expr_child, source))
        if child.type in STRING_TYPES.get(language, set()):
            return _clean_string(_node_text(child, source))
    return None


def _extract_calls(node: Node, language: str, source: str) -> set[str]:
    calls: set[str] = set()
    for child in node.children:
        if child.type in CALL_TYPES.get(language, set()):
            target = child.child_by_field_name("function") or child.child_by_field_name("name")
            if target is not None:
                calls.add(_node_text(target, source).strip())
        calls.update(_extract_calls(child, language, source))
    return {call for call in calls if call}


def _clean_string(value: str) -> str:
    return value.strip().strip('`"\'')


def _node_text(node: Node | None, source: str) -> str:
    if node is None:
        return ""
    return source[node.start_byte : node.end_byte]

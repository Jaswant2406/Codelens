from pathlib import Path

from core.ast_parser import parse_functions
from core.models import RepoFile


def test_parse_functions_extracts_python_details() -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample_repo" / "app.py"
    content = fixture.read_text(encoding="utf-8")
    functions = parse_functions(
        [
            RepoFile(
                path="app.py",
                language="python",
                content=content,
                hash="fixture",
            )
        ]
    )

    by_name = {function.name: function for function in functions}
    assert {"helper", "greet", "unused"} <= set(by_name)
    assert by_name["helper"].docstring == "Create a greeting."
    assert by_name["greet"].calls == ["helper", "message.upper"]
    assert by_name["helper"].parameters[0].name == "name"

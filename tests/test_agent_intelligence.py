from pathlib import Path

from core.agent import CodeLensAgent
from core.ast_parser import parse_functions
from core.graph_builder import CallGraphBuilder
from core.models import RepoFile
from core.pr_intelligence import PRIssueIntelligence


def _fixture_functions():
    fixture = Path(__file__).parent / "fixtures" / "sample_repo" / "app.py"
    content = fixture.read_text(encoding="utf-8")
    return parse_functions(
        [
            RepoFile(
                path="app.py",
                language="python",
                content=content,
                hash="fixture",
            )
        ]
    )


def test_pr_intelligence_detects_impacted_file_and_function() -> None:
    functions = _fixture_functions()
    intelligence = PRIssueIntelligence(functions)
    diff = """diff --git a/app.py b/app.py
++ b/app.py
@@ -5,3 +5,4 @@
 def greet(user: str) -> str:
     message = helper(user)
     return message.upper()
"""
    result = intelligence.analyze(issue_text="Greeting flow bug", pr_diff=diff)
    assert "app.py" in result.impacted_files
    assert any(item["name"] == "greet" for item in result.affected_functions)


def test_pr_intelligence_sets_reasonable_risk_level() -> None:
    functions = _fixture_functions()
    intelligence = PRIssueIntelligence(functions)
    diff = """diff --git a/app.py b/app.py
++ b/app.py
@@ -1,3 +1,4 @@
 def helper(name: str) -> str:
     return f"Hello {name}"
"""
    result = intelligence.analyze(issue_text="Production auth bug", pr_diff=diff)
    assert result.risk_level in {"LOW", "MEDIUM", "HIGH"}


class _StubService:
    def __init__(self):
        self.functions = _fixture_functions()
        self.graph_builder = CallGraphBuilder()
        self.graph_builder.build(self.functions)

        class _QueryEngine:
            def __init__(self, functions):
                self.functions = functions

            def search(self, query, top_k=6):
                return self.functions[:top_k]

        self.query_engine = _QueryEngine(self.functions)

    def load_state(self):
        return self.functions, self.graph_builder.graph

    def node_details(self, node_id: str):
        for function in self.functions:
            if function.node_id == node_id:
                return function.to_dict()
        return None


def test_agent_runs_tools_and_returns_answer() -> None:
    agent = CodeLensAgent(_StubService())
    result = agent.run("where is greeting flow and how to fix bug?")
    assert "answer" in result.to_dict()
    assert result.steps
    assert "search_functions" in result.tool_results

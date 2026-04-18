from pathlib import Path

from core.ast_parser import parse_functions
from core.config import CodeLensConfig
from core.embedder import EmbeddingEngine
from core.fusion import mmr_rerank, reciprocal_rank_fusion
from core.graph_builder import CallGraphBuilder
from core.keyword_retriever import KeywordRetriever
from core.models import RepoFile
from core.query_engine import MultiRAGQueryEngine
from core.query_rewriter import QueryRewriter


class MockLLMClient:
    def __init__(self, responses: list[str] | None = None, should_fail: bool = False) -> None:
        self.responses = responses or []
        self.should_fail = should_fail

    def complete(self, prompt: str, system: str = "") -> str:
        if self.should_fail:
            raise RuntimeError("boom")
        return self.responses.pop(0)


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


def test_query_rewriter_expand_with_mock() -> None:
    rewriter = QueryRewriter(
        MockLLMClient(
            responses=[
                "def answer():\n    return greet('Ada')",
                "1. What does greet do?\n2. Which function calls helper?",
            ]
        )
    )
    expanded = rewriter.expand("How does greet work?")
    assert "How does greet work?" in expanded
    assert any("def answer" in item for item in expanded)
    assert any("What does greet do?" == item for item in expanded)
    assert any("Which function calls helper?" == item for item in expanded)


def test_query_rewriter_expand_fallback() -> None:
    rewriter = QueryRewriter(MockLLMClient(should_fail=True))
    assert rewriter.expand("How does greet work?") == ["How does greet work?"]


def test_keyword_retriever_build_and_search() -> None:
    functions = _fixture_functions()
    retriever = KeywordRetriever()
    retriever.build(functions)
    results = retriever.search("greet", top_k=3)
    assert any(function.name == "greet" for function, _ in results)


def test_rrf_basic() -> None:
    fused = reciprocal_rank_fusion(
        [
            [("shared", 0.9), ("only_a", 0.8)],
            [("shared", 0.7), ("only_b", 0.6)],
        ]
    )
    assert fused[0][0] == "shared"


def test_mmr_rerank_no_embeddings() -> None:
    candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    assert mmr_rerank(candidates, {}, top_k=2) == ["a", "b"]


def test_multirag_search_integration() -> None:
    functions = _fixture_functions()
    graph = CallGraphBuilder().build(functions)
    retriever = KeywordRetriever()
    retriever.build(functions)
    config = CodeLensConfig()
    config.data_dir = Path(".codelens-test")
    config.chroma_path = config.data_dir / "chroma"
    config.graph_path = config.data_dir / "graph.json"
    config.functions_path = config.data_dir / "functions.json"
    config.cache_path = config.data_dir / "embed_cache.json"
    config.ensure_paths()
    embedder = EmbeddingEngine(config)
    embedder.index_functions(functions, {"app.py": "fixture"})
    query_engine = MultiRAGQueryEngine(
        embedder=embedder,
        graph=graph,
        keyword_retriever=retriever,
        query_rewriter=QueryRewriter(
            MockLLMClient(
                responses=[
                    "def greet_flow():\n    return greet('Ada')",
                    "What does greet do?\nWhich function calls helper?",
                ]
            )
        ),
    )
    results = query_engine.search("what does greet do")
    assert any(function.name == "greet" for function in results)

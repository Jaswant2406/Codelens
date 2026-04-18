from core.graph_builder import CallGraphBuilder
from core.models import FunctionNode, Parameter


def test_graph_builder_links_callers_and_callees() -> None:
    helper = FunctionNode(
        node_id="app.py::helper",
        name="helper",
        file="app.py",
        language="python",
        start_line=1,
        end_line=2,
        parameters=[Parameter(name="name")],
        calls=[],
        code="",
    )
    greet = FunctionNode(
        node_id="app.py::greet",
        name="greet",
        file="app.py",
        language="python",
        start_line=4,
        end_line=6,
        parameters=[Parameter(name="user")],
        calls=["helper"],
        code="",
    )
    graph = CallGraphBuilder()
    graph.build([helper, greet])

    assert graph.get_callees("app.py::greet") == ["app.py::helper"]
    assert graph.get_callers("app.py::helper") == ["app.py::greet"]
    assert graph.get_chain("app.py::greet", depth=1) == ["app.py::greet", "app.py::helper"]

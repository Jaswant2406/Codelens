from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from core.service import CodeLensError, CodeLensService

app = FastAPI(title="CodeLens API")
service = CodeLensService()
ui_dist = Path(__file__).resolve().parents[1] / "ui" / "dist"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexRequest(BaseModel):
    repo_url: str


class AskRequest(BaseModel):
    question: str
    depth: int | None = None


class IntelligenceRequest(BaseModel):
    issue_text: str = ""
    pr_diff: str = ""


class AgentRequest(BaseModel):
    query: str


class AIRequest(BaseModel):
    question: str
    api_key: str
    file_path: str = ""


@app.post("/index")
def index_repo(payload: IndexRequest) -> dict[str, object]:
    try:
        return service.index(payload.repo_url).to_dict()
    except CodeLensError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask_question(payload: AskRequest) -> StreamingResponse:
    try:
        matches, subgraph, stream = service.ask(payload.question, payload.depth)
    except CodeLensError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def event_stream():
        call_chain = list(subgraph.nodes) if hasattr(subgraph, "nodes") else []
        edges = (
            [
                {"source": source, "target": target, **data}
                for source, target, data in subgraph.edges(data=True)
            ]
            if hasattr(subgraph, "edges")
            else []
        )
        yield json.dumps(
            {
                "type": "context",
                "matches": [match.to_dict() for match in matches],
                "call_chain": call_chain,
                "edges": edges,
            }
        ) + "\n"
        for chunk in stream:
            yield json.dumps({"type": "token", "token": chunk}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/ai/query")
def run_ai_query(payload: AIRequest) -> dict[str, object]:
    try:
        return service.ai_query(payload.question, payload.api_key, payload.file_path)
    except CodeLensError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/impact/{function_name}")
def impact(function_name: str) -> list[dict[str, object]]:
    try:
        return service.impact(function_name)
    except CodeLensError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/deadcode")
def deadcode() -> list[dict[str, object]]:
    try:
        return service.deadcode()
    except CodeLensError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/node/{node_id:path}")
def get_node(node_id: str) -> dict[str, object]:
    try:
        details = service.node_details(node_id)
    except CodeLensError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not details:
        raise HTTPException(status_code=404, detail="Function not found")
    return details


@app.get("/nodes/by-name/{function_name:path}")
def get_nodes_by_name(function_name: str) -> list[dict[str, object]]:
    try:
        return service.nodes_by_name(function_name)
    except CodeLensError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/graph/{node_id:path}")
def get_graph_context(node_id: str) -> dict[str, object]:
    try:
        return service.graph_context(node_id)
    except CodeLensError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug/retrieval")
async def debug_retrieval(q: str) -> dict[str, object]:
    try:
        return service.query_engine.get_retriever_debug(q)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/intelligence/github")
def analyze_github_intelligence(payload: IntelligenceRequest) -> dict[str, object]:
    try:
        return service.analyze_pr_or_issue(issue_text=payload.issue_text, pr_diff=payload.pr_diff)
    except CodeLensError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/agent/query")
def run_agent(payload: AgentRequest) -> dict[str, object]:
    try:
        return service.run_agent(payload.query)
    except CodeLensError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/{path:path}")
def frontend(path: str) -> FileResponse:
    asset = ui_dist / path
    if path and asset.exists() and asset.is_file():
        return FileResponse(asset)
    index_file = ui_dist / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Frontend not built")

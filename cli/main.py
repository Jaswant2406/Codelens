from __future__ import annotations

import json

import typer

from core.service import CodeLensService

app = typer.Typer(help="CodeLens codebase analysis tool")


@app.command()
def index(repo_url_or_path: str) -> None:
    service = CodeLensService()
    stats = service.index(repo_url_or_path)
    typer.echo(json.dumps(stats.to_dict(), indent=2))


@app.command()
def ask(question: str) -> None:
    service = CodeLensService()
    matches, subgraph, stream = service.ask(question)
    typer.echo("Matches:")
    for match in matches:
        typer.echo(f"- {match.node_id}")
    typer.echo("\nCall chain:")
    for node_id in subgraph.nodes:
        typer.echo(f"- {node_id}")
    typer.echo("\nExplanation:")
    for chunk in stream:
        typer.echo(chunk, nl=False)
    typer.echo()


@app.command()
def impact(function_name: str) -> None:
    service = CodeLensService()
    typer.echo(json.dumps(service.impact(function_name), indent=2))


@app.command()
def deadcode() -> None:
    service = CodeLensService()
    typer.echo(json.dumps(service.deadcode(), indent=2))


if __name__ == "__main__":
    app()

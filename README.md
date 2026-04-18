---
title: CodeLens
emoji: "🔎"
colorFrom: orange
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# CodeLens

CodeLens is a repository analysis workspace for understanding code structure faster. It indexes a local or GitHub repository, extracts functions, builds a call graph, stores retrieval data, and then lets you explore the project through grounded explanations, search, impact inspection, dead-code review, and file-focused AI help.

## Live links

- GitHub: [https://github.com/Jaswant2406/Codelens](https://github.com/Jaswant2406/Codelens)
- Hugging Face Space: [https://huggingface.co/spaces/jaswant2406/codelens](https://huggingface.co/spaces/jaswant2406/codelens)

## What CodeLens does

After indexing a repository, CodeLens can:

- explain what a project, file, function, or visible flow does
- search files, functions, and retrieved symbols
- show caller impact before you change a function
- surface functions with no inbound callers
- preview retrieved code directly from the UI
- run Gemini against a selected file or indexed context for focused analysis

The goal is not just to summarize code, but to answer repository questions using visible code structure and retrieved evidence.

## Main features

### Ask

Ask is the architecture and explanation tab. Use it for:

- project overview questions
- file explanations
- function and flow explanations
- debugging-style questions

Example prompts:

- `What does this project do?`
- `Explain api/main.py`
- `Explain the repository structure`
- `How does indexing work?`
- `Why is area_of_rectangle failing?`

### AI

AI is a focused Gemini workflow. It supports:

- repository-wide semantic retrieval
- file-specific analysis for a single selected file

The most reliable path is file-focused use:

1. index the repository
2. open the AI tab
3. select a file
4. ask a focused question such as `What does this file do?`

### Search

Search helps you quickly find:

- matching file paths
- functions and symbols
- retrieval hits from the indexed repository

### Inspect

Inspect is the caller-impact view. Enter a function name to see which callers depend on it before you make a change.

### Dead code

Dead code lists functions with no inbound callers in the current graph and lets you jump directly into Code preview.

### Code preview

Code preview is the shared inspection panel used by Ask, AI, Search, Inspect, and Dead code. It lets you:

- open a retrieved function
- inspect the code body
- copy the code
- jump back
- run Inspect directly on the selected function

## How it works

At a high level, CodeLens works like this:

1. load a repository from a GitHub URL or local path
2. parse supported source files into function-level nodes
3. build a call graph from explicit call relationships
4. index embeddings and keyword retrieval data
5. answer questions against the indexed repository

The service layer is handled by `core/service.py`, the API is exposed by `api/main.py`, and the frontend is a React app in `ui/`.

## Supported languages

Repository indexing currently supports source files with these extensions:

- `.py`
- `.js`
- `.ts`
- `.go`
- `.java`

If a repository has no supported source files, indexing will fail with a clear error.

## Repository layout

```text
api/     FastAPI routes and HTTP entrypoints
cli/     Typer-based CLI commands
core/    Indexing, parsing, retrieval, graph, AI, and service logic
ui/      React frontend
tests/   Test suite
```

Important files:

- `api/main.py` — FastAPI application and routes
- `cli/main.py` — CLI commands
- `core/service.py` — main orchestration layer
- `codelens.yaml` — local configuration
- `Dockerfile` — container build for deployment
- `render.yaml` — Render blueprint

## Local development

### Prerequisites

- Python `3.11`
- Node.js `20+`
- npm
- git

### Backend setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev]
python -m uvicorn api.main:app --reload
```

### Frontend build

```bash
cd ui
npm install
npm run build
```

The backend serves the built frontend from `ui/dist`.

## CLI usage

CodeLens also includes a small CLI:

```bash
codelens index <repo-url-or-path>
codelens ask "How does indexing work?"
codelens impact <function_name>
codelens deadcode
```

## API routes

Main API routes exposed by `api/main.py`:

- `POST /index`
- `POST /ask`
- `POST /ai/query`
- `GET /impact/{function_name}`
- `GET /deadcode`
- `GET /node/{node_id}`
- `GET /nodes/by-name/{function_name}`
- `GET /graph/{node_id}`
- `GET /debug/retrieval`
- `GET /health`

## Configuration

Local configuration lives in `codelens.yaml`.

Important settings include:

- data directory paths
- embedding model name
- retrieval depth and MultiRAG settings
- LLM provider configuration

Current default LLM provider configuration includes:

- Groq
- OpenRouter
- Ollama

## Environment variables

These may be useful depending on which paths you use:

- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`

For the AI tab, the Google AI key is entered in the UI per request rather than stored by the server.

## Example repositories to test

- [https://github.com/joaks1/python-debugging](https://github.com/joaks1/python-debugging)
- [https://github.com/faif/python-patterns](https://github.com/faif/python-patterns)
- [https://github.com/psf/requests](https://github.com/psf/requests)
- [https://github.com/Deeptanshu-sankhwar/semantic-code-navigator](https://github.com/Deeptanshu-sankhwar/semantic-code-navigator)
- local repo path: `C:\Users\jaswa\OneDrive\Documents\New project\codelens`

## Example questions

### Project structure

- `What this project does`
- `Explain the repository structure`
- `Explain api/main.py`

### Flow

- `How requests.get works internally`
- `How does indexing work?`
- `Explain the request flow`

### File mode

- `Explain behavioral/command.py`
- `What are the functions in patterns/__init__.py`

### Debugging

- `Why area_of_rectangle is failing`
- `Why rectangle_area.py is failing`

### Search and inspect

- `area_of_rectangle`
- `request`
- `calculate_rectangle_area`

## Testing checklist

After deployment or local setup, a quick smoke test is:

1. index a small repository
2. verify Ask returns an answer
3. verify Search returns file and function matches
4. verify Inspect returns callers or a clean empty state
5. verify Dead code loads
6. verify Code preview opens and copy works
7. verify AI works best with a selected file

## Deployment

### Hugging Face Spaces

This repository is configured for Docker-based Hugging Face Spaces and serves the app on port `7860`.

High-level flow:

1. create a Docker Space
2. push this repository to the Space repo
3. let Hugging Face build the `Dockerfile`

### Render

This repository also includes a Docker-based Render setup through `render.yaml`.

High-level flow:

1. push the repository to GitHub
2. create a new Render Blueprint or Web Service
3. connect the repository
4. deploy the Docker service

## Troubleshooting

### Indexing fails

Check:

- the repo path or GitHub URL is valid
- the repository contains supported source files
- the repository contains parseable functions

### AI tab fails

Check:

- the Google AI key is valid
- quota is available
- file-focused AI mode is selected for smaller, cheaper requests

### No callers found

This can mean:

- the function is unused
- the function is an entry-style function
- the current graph does not show inbound relationships for that node

### No code preview result

Code preview opens from a selected function node, not a file name alone. Use Ask, Search, Inspect, AI, or Dead code to open a node.

## Notes

- normal analysis ignores test files unless explicitly requested
- AI works best when you select a single file for focused questions
- large repositories may take longer to index depending on size and model availability
- retrieval and explanations are only as strong as the indexed repository state

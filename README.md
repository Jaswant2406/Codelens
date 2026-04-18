# CodeLens

CodeLens is a production-oriented codebase analysis tool with a Python backend, FastAPI API, and React UI.

## Free deployment on Render

This repository includes a Docker-based Render setup.

1. Push the repository to GitHub.
2. Create a new Web Service on Render.
3. Connect this repository.
4. Render will detect `render.yaml` and deploy the Docker service.

The app starts with:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

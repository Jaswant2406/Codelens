from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from git import Repo

from .models import LANGUAGE_BY_EXTENSION, RepoFile

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".codelens",
}


def clone_or_resolve_repo(repo_url_or_path: str) -> Path:
    normalized = repo_url_or_path.strip()
    target = Path(normalized)
    if target.exists():
        return target.resolve()

    temp_dir = Path(tempfile.mkdtemp(prefix="codelens_repo_"))
    Repo.clone_from(normalized, temp_dir)
    return temp_dir


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_repository(repo_url_or_path: str) -> tuple[Path, list[RepoFile]]:
    root = clone_or_resolve_repo(repo_url_or_path)
    files: list[RepoFile] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        language = LANGUAGE_BY_EXTENSION.get(path.suffix.lower())
        if not language:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="ignore")
        files.append(
            RepoFile(
                path=str(path.relative_to(root)).replace("\\", "/"),
                language=language,
                content=content,
                hash=_file_hash(content),
            )
        )
    return root, files

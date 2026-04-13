from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

from ..backend_data import resolve_vendored_backend_root


def python_executable() -> str:
    return sys.executable


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def append_pythonpath(existing: str | None, *paths: str | Path) -> str:
    ordered: list[str] = []
    if existing:
        ordered.extend(part for part in existing.split(os.pathsep) if part)
    for path in paths:
        resolved = str(Path(path))
        if resolved not in ordered:
            ordered.insert(0, resolved)
    return os.pathsep.join(ordered)


def build_subprocess_env(
    *,
    backend: str,
    repo_root: str | Path | None = None,
    extra_env: Mapping[str, str | int | float] | None = None,
) -> dict[str, str]:
    resolved_root = resolve_vendored_backend_root(backend, repo_root)
    src_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = append_pythonpath(env.get("PYTHONPATH"), resolved_root, src_root)
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    return env

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_local_path(path: str | Path, *, root: str | Path | None = None) -> Path:
    """Resolve paths embedded in manifests after moving between Windows and WSL."""

    candidate = Path(path)
    if candidate.exists():
        return candidate

    base = Path(root) if root is not None else project_root()
    raw = str(path)
    normalised = raw.replace("\\", "/")
    marker = "Seg-MoE-v2/"
    if marker in normalised:
        relocated = base / normalised.split(marker, 1)[1]
        if relocated.exists():
            return relocated
        return relocated

    windows_parts = PureWindowsPath(raw).parts
    if "Seg-MoE-v2" in windows_parts:
        idx = windows_parts.index("Seg-MoE-v2")
        relocated = base.joinpath(*windows_parts[idx + 1 :])
        if relocated.exists():
            return relocated
        return relocated
    windows_path = PureWindowsPath(raw)
    if windows_path.drive:
        drive = windows_path.drive.rstrip(":").lower()
        relocated = Path("/mnt") / drive
        relocated = relocated.joinpath(*windows_path.parts[1:])
        if relocated.exists():
            return relocated
        return relocated
    if normalised != raw:
        relocated = base / normalised
        if relocated.exists():
            return relocated
        return relocated
    return candidate


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return Path(path)


def save_json(payload: Any, path: str | Path) -> Path:
    path = ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return Path(path)


def save_pickle(payload: Any, path: str | Path) -> Path:
    path = ensure_parent(path)
    with Path(path).open("wb") as handle:
        pickle.dump(payload, handle)
    return Path(path)


def save_csv_rows(
    rows: Iterable[dict[str, Any]],
    path: str | Path,
    *,
    fieldnames: list[str] | tuple[str, ...] | None = None,
) -> Path:
    materialized = list(rows)
    path = ensure_parent(path)
    if fieldnames is None:
        ordered: list[str] = []
        for row in materialized:
            for key in row.keys():
                if key not in ordered:
                    ordered.append(str(key))
        fieldnames = ordered
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return Path(path)


def stable_hash(payload: Any) -> str:
    dumped = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(dumped.encode("utf-8")).hexdigest()

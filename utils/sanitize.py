"""Path sanitization and filesystem safety for sandboxed output/."""

from __future__ import annotations

import re
from pathlib import Path

# Windows reserved device names
_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)

MAX_FILENAME_LEN = 255


def sanitize_filename(name: str) -> str:
    """
    Reduce filename to a safe basename: no path separators, no traversal.
    Strips dangerous characters; limits length.
    """
    if not name or not name.strip():
        return "untitled.txt"

    # Take basename only
    base = Path(name.replace("\\", "/")).name
    # Remove null bytes and control chars
    base = "".join(c for c in base if ord(c) >= 32 and c not in '<>:"|?*')
    base = base.strip(" .")
    if not base:
        base = "untitled.txt"

    stem = Path(base).stem
    suffix = Path(base).suffix
    if stem.upper() in _RESERVED:
        stem = f"file_{stem}"

    safe = f"{stem}{suffix}" if suffix else stem
    if len(safe) > MAX_FILENAME_LEN:
        safe = safe[: MAX_FILENAME_LEN - 4] + (suffix[:4] if suffix else ".txt")

    return safe or "untitled.txt"


def safe_path(filename: str, base_dir: str | Path = "output") -> Path:
    """
    Resolve a path under base_dir only. Rejects path traversal.

    Uses pathlib.resolve() and relative_to check (Python 3.9+).
    """
    base = Path(base_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)

    clean_name = sanitize_filename(filename)
    target = (base / clean_name).resolve()

    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError("Path traversal detected: path must stay under output directory") from exc

    return target


def ensure_under_output(path: Path, base_dir: str | Path = "output") -> Path:
    """Verify an already-resolved path stays under base."""
    base = Path(base_dir).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError("Path must remain inside the output directory") from exc
    return resolved


def resolve_output_subdir(folder: str | None, root: Path) -> Path:
    """
    Build a nested directory strictly under root. Rejects '..' segments.
    Creates parent directories as needed.
    """
    root = root.resolve()
    if not folder or not str(folder).strip():
        return root

    parts = [p for p in str(folder).replace("\\", "/").split("/") if p]
    cur = root
    for p in parts:
        if p in (".", ".."):
            raise ValueError("Invalid path segment")
        seg = sanitize_filename(p)
        cur = (cur / seg).resolve()
        try:
            cur.relative_to(root)
        except ValueError as exc:
            raise ValueError("Path traversal detected") from exc
    cur.mkdir(parents=True, exist_ok=True)
    return cur

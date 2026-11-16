"""Central configuration for SteelPCR-RAG.

Every credential and every machine-specific path is read from an environment
variable.

A local ``.env`` file at the repository root is loaded automatically when it
exists; ``.env.example`` lists every recognised variable with a safe default.
Environment variables that are already set take precedence over ``.env``.

Typical use from a script in a sub-package::

    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT))
    from steelpcr_config import env, env_required, repo_path
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "env",
    "env_bool",
    "env_float",
    "env_int",
    "env_required",
    "repo_path",
]

#: Absolute path of the repository root (the directory holding this file).
REPO_ROOT: Path = Path(__file__).resolve().parent

_ENV_FILE: Path = REPO_ROOT / ".env"
_dotenv_loaded: bool = False


def _load_dotenv(path: Path = _ENV_FILE) -> None:
    """Populate ``os.environ`` from ``path`` once, without overriding real variables."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def env(name: str, default: str | None = None) -> str | None:
    """Return the value of ``name``, or ``default`` when unset or blank."""
    _load_dotenv()
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def env_required(name: str, purpose: str = "") -> str:
    """Return the value of ``name``; raise a clear error when it is missing."""
    value = env(name)
    if value is None:
        hint = f" ({purpose})" if purpose else ""
        raise RuntimeError(
            f"Environment variable {name} is not set{hint}. "
            f"Copy .env.example to .env and fill it in, or export {name}."
        )
    return value


def env_int(name: str, default: int) -> int:
    """Return ``name`` parsed as ``int``, falling back to ``default``."""
    raw = env(name)
    try:
        return int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    """Return ``name`` parsed as ``float``, falling back to ``default``."""
    raw = env(name)
    try:
        return float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool) -> bool:
    """Return ``name`` parsed as a boolean (``1/true/yes/on`` are truthy)."""
    value = env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def repo_path(name: str, default_relative: str) -> str:
    """Resolve a path from ``name``, defaulting to ``default_relative`` under the repo root.

    An absolute value in the environment variable is used verbatim, so the
    repository can live anywhere while large artefacts (indexes, caches) may
    stay outside it.
    """
    value = env(name)
    if value is None:
        return str(REPO_ROOT / default_relative)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate)

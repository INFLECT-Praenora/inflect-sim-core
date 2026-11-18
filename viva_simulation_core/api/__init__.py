"""Optional public digital-twin API (FastAPI).

Import-guarded: this package is never imported by ``viva_simulation_core`` or
the CLI, so the base install stays lean. Install with ``pip install '.[api]'``.
"""
from __future__ import annotations

try:  # pragma: no cover - exercised via tests that simulate absence
    import fastapi  # noqa: F401

    HAS_API = True
except ImportError:
    HAS_API = False


def require_api() -> None:
    """Raise a clear ImportError if the optional API dependencies are absent."""
    if not HAS_API:
        raise ImportError(
            "The simulation-core API requires optional dependencies. "
            "Install them with: pip install '.[api]'"
        )


__all__ = ["HAS_API", "require_api"]

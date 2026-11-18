"""``python -m viva_simulation_core.api`` -- single-worker uvicorn launcher."""
from __future__ import annotations

import argparse
import sys


def check_workers(workers: int) -> None:
    """The twin registry is in-process, so more than one worker would split state."""
    if workers > 1:
        raise ValueError(
            "the digital-twin API keeps its registry in-process and must run with a "
            f"single worker (got --workers {workers})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="viva_simulation_core.api")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    try:
        check_workers(args.workers)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    import uvicorn

    from .app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()

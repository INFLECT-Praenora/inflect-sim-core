"""Fit a baseline surrogate and score it honestly (the viva-surrogate framework).

Repointed at ``viva_simulation_core.surrogate``: the framework's validate harness
produces a PASS/FAIL scorecard so a silently-degenerate surrogate cannot ship.
This demo fits the persistence and linear baselines on a toy decaying system and
prints each scorecard.

Run: ``python -m examples.surrogate_demo``.
"""
from __future__ import annotations

from viva_simulation_core.surrogate.fit_and_score import _demo


def scorecards() -> dict:
    return {name: _demo(name) for name in ("persistence", "linear")}


def main() -> None:
    for name, sc in scorecards().items():
        print(f"{name:12} ->", sc)


if __name__ == "__main__":
    main()

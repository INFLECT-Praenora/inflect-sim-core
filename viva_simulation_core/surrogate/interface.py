"""Core protocols + scorecard for the pluggable surrogate framework.

A surrogate is fitted by a ``SurrogateBackend`` and exposes a ``FittedSurrogate``
(it can ``rollout`` from an initial condition). Any fitted surrogate is scored by
the validate harness (see ``validate.py``), which fills a ``Scorecard`` whose
``verdict`` is PASS only if every honesty gate holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FittedSurrogate(Protocol):
    """A fitted surrogate: roll it out from an initial state."""

    observable_names: list[str]

    def rollout(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """Return an array of shape ``(n_steps, len(observable_names))``."""
        ...


@runtime_checkable
class SurrogateBackend(Protocol):
    """A method that fits trajectories into a ``FittedSurrogate``."""

    name: str

    def fit(self, trajectories: list[np.ndarray], observable_names: list[str],
            dt: float) -> FittedSurrogate:
        ...


@dataclass
class Scorecard:
    """The honest surrogate scorecard. ``verdict`` is PASS iff all gates hold —
    speed never buys a pass."""

    backend: str
    rollout_nrmse: float
    physicality: float
    responsiveness: float
    speedup: float | None
    gates: dict[str, bool] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return "PASS" if self.gates and all(self.gates.values()) else "FAIL"

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "rollout_nrmse": self.rollout_nrmse,
            "physicality": self.physicality,
            "responsiveness": self.responsiveness,
            "speedup": self.speedup,
            "gates": dict(self.gates),
            "verdict": self.verdict,
        }

    def __str__(self) -> str:
        sp = "n/a" if self.speedup is None else f"{self.speedup:,.0f}x"
        gates = "  ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in self.gates.items())
        return (
            f"Surrogate scorecard — {self.backend}: {self.verdict}  "
            f"(PASS = responsive + physical; rollout_nrmse is reported, not gated)\n"
            f"  rollout_nrmse  {self.rollout_nrmse:.4f}\n"
            f"  physicality    {self.physicality:.3f}\n"
            f"  responsiveness {self.responsiveness:.4f}\n"
            f"  speedup        {sp}\n"
            f"  gates          {gates}"
        )

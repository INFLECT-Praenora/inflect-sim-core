"""viva-surrogate — a pluggable surrogate/emulator framework.

v1 prototype home inside simulation-core; extracts to a standalone ``viva-surrogate``
package once the interface is proven. The centerpiece is the validate harness, which
produces an honest PASS/FAIL scorecard so a silently-degenerate surrogate cannot ship.
"""
from __future__ import annotations

from viva_simulation_core.surrogate.interface import (
    FittedSurrogate,
    Scorecard,
    SurrogateBackend,
)

__all__ = ["FittedSurrogate", "SurrogateBackend", "Scorecard"]

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
from viva_simulation_core.surrogate.validate import validate_surrogate
from viva_simulation_core.surrogate.fit_and_score import fit_and_score
from viva_simulation_core.surrogate.backends.baseline import (
    PersistenceBackend,
    LinearBackend,
)
from viva_simulation_core.surrogate.backends.neural import NeuralBackend

__all__ = [
    "FittedSurrogate",
    "SurrogateBackend",
    "Scorecard",
    "validate_surrogate",
    "fit_and_score",
    "PersistenceBackend",
    "LinearBackend",
    "NeuralBackend",
]

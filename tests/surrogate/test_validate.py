"""Task 2 — the validate harness. These tests ARE the framework's value:
a fast-but-degenerate or nonphysical surrogate must score FAIL."""
import numpy as np
import pytest

from viva_simulation_core.surrogate.validate import validate_surrogate

NAMES = ["a", "b"]
BOUNDS = {"a": (0.0, 10.0), "b": (0.0, 10.0)}
DT = 1.0

# three truth trajectories with distinct initial conditions (exponential decay)
def _truth(ic, T=6):
    xs = [np.array(ic, float)]
    for _ in range(T - 1):
        xs.append(xs[-1] * 0.8)
    return np.array(xs)

TRUTHS = [_truth([8.0, 2.0]), _truth([4.0, 6.0]), _truth([2.0, 9.0])]
DRIVER_ICS = [t[0] for t in TRUTHS]


class ConstantFitted:
    """Degenerate: same endpoint regardless of x0 (what the sepsis net does)."""
    observable_names = NAMES
    def rollout(self, x0, n_steps):
        return np.tile(np.array([5.0, 5.0]), (n_steps, 1))


class OutOfRangeFitted:
    """Responsive to x0 but blows out of physical bounds."""
    observable_names = NAMES
    def rollout(self, x0, n_steps):
        out = np.tile(np.asarray(x0, float), (n_steps, 1))
        out[-1] = [-50.0, 99.0]  # out of bounds
        return out


class TruthEchoFitted:
    """Good: reproduces the matched truth trajectory for each IC."""
    observable_names = NAMES
    def __init__(self, truths):
        self._by_ic = {tuple(np.round(t[0], 6)): t for t in truths}
    def rollout(self, x0, n_steps):
        t = self._by_ic[tuple(np.round(x0, 6))]
        return t[:n_steps]


def test_degenerate_surrogate_fails_responsiveness():
    sc = validate_surrogate(ConstantFitted(), TRUTHS, DRIVER_ICS, BOUNDS, DT)
    assert sc.responsiveness < 0.02
    assert sc.gates["responsive"] is False
    assert sc.verdict == "FAIL"


def test_nonphysical_surrogate_fails_physicality():
    sc = validate_surrogate(OutOfRangeFitted(), TRUTHS, DRIVER_ICS, BOUNDS, DT)
    assert sc.physicality < 0.95
    assert sc.gates["physical"] is False
    assert sc.verdict == "FAIL"


def test_good_surrogate_passes():
    sc = validate_surrogate(TruthEchoFitted(TRUTHS), TRUTHS, DRIVER_ICS, BOUNDS, DT,
                            speedup=1234.0)
    assert sc.verdict == "PASS"
    assert sc.rollout_nrmse < 1e-6
    assert sc.physicality == 1.0
    assert sc.responsiveness > 0.02


def test_empty_or_mismatched_inputs_raise():
    with pytest.raises(ValueError):
        validate_surrogate(ConstantFitted(), [], DRIVER_ICS, BOUNDS, DT)
    with pytest.raises(ValueError):
        validate_surrogate(ConstantFitted(), TRUTHS, [], BOUNDS, DT)

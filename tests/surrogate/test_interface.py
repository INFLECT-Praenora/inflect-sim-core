"""Task 1 — surrogate framework protocols + Scorecard."""
import numpy as np

from viva_simulation_core.surrogate.interface import (
    Scorecard,
    FittedSurrogate,
    SurrogateBackend,
)


def test_scorecard_verdict_pass_when_all_gates_true():
    sc = Scorecard(backend="x", rollout_nrmse=0.05, physicality=1.0,
                   responsiveness=0.5, speedup=1000.0,
                   gates={"responsive": True, "physical": True})
    assert sc.verdict == "PASS"


def test_scorecard_verdict_fail_when_any_gate_false():
    sc = Scorecard(backend="x", rollout_nrmse=0.05, physicality=0.3,
                   responsiveness=0.5, speedup=1000.0,
                   gates={"responsive": True, "physical": False})
    assert sc.verdict == "FAIL"


def test_scorecard_as_dict_and_str():
    sc = Scorecard(backend="neural", rollout_nrmse=0.1, physicality=0.9,
                   responsiveness=0.0, speedup=None,
                   gates={"responsive": False, "physical": False})
    d = sc.as_dict()
    assert d["backend"] == "neural" and d["verdict"] == "FAIL"
    s = str(sc)
    assert "neural" in s and "FAIL" in s


def test_protocols_are_structural():
    # a minimal object satisfies FittedSurrogate structurally
    class F:
        observable_names = ["a"]
        def rollout(self, x0, n_steps):
            return np.zeros((n_steps, 1))
    f = F()
    assert isinstance(f, FittedSurrogate)

    class B:
        name = "b"
        def fit(self, trajectories, observable_names, dt):
            return F()
    assert isinstance(B(), SurrogateBackend)

"""Task 5 — fit_and_score orchestration + public API."""
import numpy as np

from viva_simulation_core.surrogate import (
    fit_and_score,
    PersistenceBackend,
    LinearBackend,
    Scorecard,
)

NAMES = ["a", "b"]
DT = 1.0


def _decay_trajs(rate=0.6, T=8):
    trajs = []
    for ic in ([9.0, 3.0], [2.0, 8.0], [5.0, 5.0], [7.0, 1.0]):
        xs = [np.array(ic, float)]
        for _ in range(T - 1):
            xs.append(xs[-1] * rate)
        trajs.append(np.array(xs))
    return trajs


def test_fit_and_score_returns_scorecard():
    trajs = _decay_trajs()
    ics = [t[0] for t in trajs]
    bounds = {k: (-1.0, 100.0) for k in NAMES}
    sc = fit_and_score(LinearBackend(), trajs, NAMES, DT, bounds, ics, trajs)
    assert isinstance(sc, Scorecard)
    assert sc.backend == "linear"


def test_linear_beats_persistence_on_dynamics():
    trajs = _decay_trajs()
    ics = [t[0] for t in trajs]
    bounds = {k: (-1.0, 100.0) for k in NAMES}
    lin = fit_and_score(LinearBackend(), trajs, NAMES, DT, bounds, ics, trajs)
    per = fit_and_score(PersistenceBackend(), trajs, NAMES, DT, bounds, ics, trajs)
    assert lin.rollout_nrmse < per.rollout_nrmse   # linear captures the decay; persistence is the floor
    assert lin.verdict == "PASS"


def test_public_api_exports():
    import viva_simulation_core.surrogate as S
    for sym in ("fit_and_score", "validate_surrogate", "Scorecard",
                "PersistenceBackend", "LinearBackend", "NeuralBackend"):
        assert hasattr(S, sym), sym

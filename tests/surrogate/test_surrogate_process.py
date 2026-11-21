"""Task 6 — SurrogateProcess: a fitted surrogate as an interchangeable box."""
import numpy as np

from viva_simulation_core.core import build_core
from viva_simulation_core.surrogate.backends.baseline import LinearBackend
from viva_simulation_core.surrogate.process import SurrogateProcess, register


NAMES = ["a", "b"]


def _fit():
    trajs = []
    for ic in ([9.0, 3.0], [2.0, 8.0], [5.0, 5.0]):
        xs = [np.array(ic, float)]
        for _ in range(6):
            xs.append(xs[-1] * 0.6)
        trajs.append(np.array(xs))
    return LinearBackend().fit(trajs, NAMES, 1.0)


def test_ports_match_the_observables():
    core = build_core()
    sp = SurrogateProcess({"fitted": _fit(), "observables": NAMES}, core)
    assert set(sp.inputs()) == set(NAMES)
    assert set(sp.outputs()) == set(NAMES)


def test_update_advances_one_step_finite():
    core = build_core()
    sp = SurrogateProcess({"fitted": _fit(), "observables": NAMES}, core)
    out = sp.update({"a": 8.0, "b": 8.0}, 1.0)
    assert set(out) == set(NAMES)
    assert all(np.isfinite(v) for v in out.values())
    # linear 0.6-decay: one step from 8.0 -> ~4.8
    assert abs(out["a"] - 4.8) < 0.5


def test_register_adds_it_to_the_core():
    core = build_core()
    register(core)
    assert core.link_registry.get("SurrogateProcess") is SurrogateProcess

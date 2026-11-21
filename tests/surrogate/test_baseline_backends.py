"""Task 3 — persistence + linear baseline backends."""
import numpy as np

from viva_simulation_core.surrogate.interface import FittedSurrogate
from viva_simulation_core.surrogate.backends.baseline import (
    PersistenceBackend,
    LinearBackend,
)

NAMES = ["a", "b"]
DT = 1.0


def _linear_trajs(rate=0.5, T=8):
    trajs = []
    for ic in ([10.0, 4.0], [3.0, 9.0], [6.0, 6.0]):
        xs = [np.array(ic, float)]
        for _ in range(T - 1):
            xs.append(xs[-1] * rate)
        trajs.append(np.array(xs))
    return trajs


def test_persistence_holds_the_initial_condition():
    b = PersistenceBackend()
    f = b.fit(_linear_trajs(), NAMES, DT)
    assert isinstance(f, FittedSurrogate)
    r = f.rollout(np.array([7.0, 2.0]), 5)
    assert r.shape == (5, 2)
    assert np.allclose(r, np.array([7.0, 2.0]))  # constant = x0


def test_linear_fits_and_rolls_out_a_linear_system():
    trajs = _linear_trajs(rate=0.5)
    f = LinearBackend().fit(trajs, NAMES, DT)
    assert isinstance(f, FittedSurrogate)
    ic = np.array([8.0, 8.0])
    r = f.rollout(ic, 5)
    # rollout convention: row 0 is x0, then one applied step per row
    expected = np.array([ic * 0.5 ** k for k in range(5)])
    assert np.allclose(r, expected, atol=1e-6)


def test_backends_satisfy_backend_protocol():
    from viva_simulation_core.surrogate.interface import SurrogateBackend
    assert isinstance(PersistenceBackend(), SurrogateBackend)
    assert isinstance(LinearBackend(), SurrogateBackend)

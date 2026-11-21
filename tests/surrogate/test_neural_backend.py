"""Task 4 — neural backend wrapping viva-torch (engine-agnostic)."""

import numpy as np
import pytest

from viva_simulation_core.surrogate.backends.neural import (
    NeuralBackend,
    _overrides_from_x0,
)


def test_overrides_from_x0_nests_values_at_feature_paths():
    paths = [("stores", "a"), ("stores", "b"), ("p", "c")]
    ov = _overrides_from_x0(paths, np.array([1.0, 2.0, 3.0]))
    assert ov == {"stores": {"a": 1.0, "b": 2.0}, "p": {"c": 3.0}}


def test_neural_fitted_rollout_honors_x0_and_shape_with_stub():
    # inject a stub rollout_fn so we test wiring without training a net
    paths = [("stores", "a"), ("stores", "b")]

    def stub_rollout_fn(spec, ckpt, core, overrides, n, dt):
        # echo the seeded x0 forward (so we can check x0 seeding + shape)
        x0 = np.array([overrides["stores"]["a"], overrides["stores"]["b"]])
        return np.tile(x0, (n, 1))

    fitted = NeuralBackend._fitted_for_test(
        feature_paths=paths, target_names=["a", "b"], timestep=1.0,
        ckpt_path="unused", core_provider=lambda: None, rollout_fn=stub_rollout_fn,
    )
    r = fitted.rollout(np.array([5.0, 7.0]), 4)
    assert r.shape == (4, 2)
    assert np.allclose(r[0], [5.0, 7.0])         # row 0 is x0 (framework convention)
    assert np.allclose(r[1:], [5.0, 7.0])        # stub echoes the seeded x0
    assert fitted.observable_names == ["a", "b"]

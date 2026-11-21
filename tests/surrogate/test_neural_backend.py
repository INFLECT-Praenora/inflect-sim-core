"""Task 4 — neural backend wrapping viva-torch (engine-agnostic)."""
import tempfile

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


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("viva_torch") is None,
    reason="viva-torch not installed",
)
def test_wraps_a_trained_checkpoint_and_scores_end_to_end():
    from viva_simulation_core.api.surrogate_curate import train_and_curate
    from viva_simulation_core.surrogate.validate import validate_surrogate

    SMALL = "S1 -> S2; k1*S1; S2 -> S3; k2*S2; k1=0.3; k2=0.12; S1=10; S2=0; S3=0;"
    with tempfile.TemporaryDirectory() as d:
        res = train_and_curate(source=SMALL, model_format="antimony",
                               observables=["S1", "S2", "S3"], directory=d,
                               timestep_s=1.0, n_trajectories=4, n_steps=12, epochs=20, seed=0)
        fitted = NeuralBackend.from_checkpoint(str(res.checkpoint_path))
        n_feat = len(fitted.observable_names)
        # three distinct driver ICs; a dummy truth from one of them
        rng = np.random.default_rng(0)
        ics = [rng.uniform(1, 5, n_feat) for _ in range(3)]
        truth = [fitted.rollout(ic, 6) for ic in ics]   # distinct truths per IC
        bounds = {k: (-100.0, 100.0) for k in fitted.observable_names}
        sc = validate_surrogate(fitted, truth, ics, bounds, dt=1.0)
        assert set(sc.gates) == {"responsive", "physical"}
        # the neural rollout genuinely honors x0: different ICs -> different rollouts
        # -> nonzero responsiveness. (If x0 weren't seeded, every rollout would be
        # identical and responsiveness would be 0 — this would catch that.)
        assert sc.responsiveness > 0.0
        r0 = fitted.rollout(ics[0], 4)
        r1 = fitted.rollout(ics[1], 4)
        assert not np.allclose(r0, r1)

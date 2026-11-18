"""Torch-free checks of checkpoint path containment (``_resolve`` never loads a net)."""
import pytest

from viva_simulation_core.api import surrogate_torch as st
from viva_simulation_core.api.surrogate_backend import SurrogateUnavailable, seed_in_bounds, SurrogateEntry


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "HAS_TORCH_BACKEND", True)  # constructor guard only; no torch used
    d = tmp_path / "ck"
    d.mkdir()
    (d / "ok.pt").write_bytes(b"x")
    return st.TorchCheckpointBackend(d), d, tmp_path


def test_resolve_accepts_inside(backend):
    b, d, _ = backend
    assert b._resolve("ok.pt") == (d / "ok.pt").resolve()


def test_resolve_rejects_dotdot_absolute_and_symlink(backend):
    b, d, root = backend
    (root / "secret.pt").write_bytes(b"x")
    (d / "link.pt").symlink_to(root / "secret.pt")
    for ref in ("../secret.pt", str(root / "secret.pt"), "link.pt", "missing.pt"):
        with pytest.raises(SurrogateUnavailable):
            b._resolve(ref)


def test_seed_in_bounds():
    e = SurrogateEntry("c", ["A"], 1.0, target_bounds={"A": (0.0, 10.0)})
    assert seed_in_bounds(e, {"A": 10.0}) and not seed_in_bounds(e, {"A": 10.5})
    assert seed_in_bounds(SurrogateEntry("c", ["A"], 1.0), {"A": 1e9})

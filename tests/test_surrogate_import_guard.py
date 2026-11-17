import importlib

import pytest


def test_package_imports_without_torch():
    import viva_simulation_core  # noqa: F401
    surrogate = importlib.import_module("viva_simulation_core.surrogate")
    assert isinstance(surrogate.HAS_VIVA_TORCH, bool)


def test_calling_surrogate_without_torch_errors_clearly(monkeypatch):
    from viva_simulation_core import surrogate

    monkeypatch.setattr(surrogate, "HAS_VIVA_TORCH", False)
    with pytest.raises(ImportError, match=r"\[surrogate\]"):
        surrogate.require_viva_torch()
    with pytest.raises(ImportError, match=r"\[surrogate\]"):
        surrogate.get("train_surrogate")

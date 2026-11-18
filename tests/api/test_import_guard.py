import importlib
import sys

import pytest


def _reload_api(monkeypatch, fastapi_present: bool):
    for name in [m for m in sys.modules if m == "viva_simulation_core.api"]:
        monkeypatch.delitem(sys.modules, name)
    if not fastapi_present:
        monkeypatch.setitem(sys.modules, "fastapi", None)  # makes import raise ImportError
    return importlib.import_module("viva_simulation_core.api")


def test_require_api_raises_when_fastapi_missing(monkeypatch):
    mod = _reload_api(monkeypatch, fastapi_present=False)
    assert mod.HAS_API is False
    with pytest.raises(ImportError, match=r"\[api\]"):
        mod.require_api()


def test_has_api_true_when_fastapi_present(monkeypatch):
    pytest.importorskip("fastapi")
    mod = _reload_api(monkeypatch, fastapi_present=True)
    assert mod.HAS_API is True
    mod.require_api()


def test_base_package_does_not_import_fastapi():
    import subprocess
    code = "import sys, viva_simulation_core; sys.exit('fastapi' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0

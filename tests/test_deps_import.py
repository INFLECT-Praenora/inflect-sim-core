import importlib
import os

import pytest

LOCAL = ["viva_core", "viva_tellurium", "viva_biogears"]  # not on PyPI
PYPI = ["viva_emitters"]  # published on PyPI; local checkout is stale (pbg_emitters)


@pytest.mark.parametrize("mod", ["process_bigraph"] + PYPI + LOCAL)
def test_core_deps_import(mod):
    importlib.import_module(mod)


@pytest.mark.parametrize("mod", LOCAL)
def test_deps_are_local_checkouts(mod):
    m = importlib.import_module(mod)
    root = os.path.realpath(os.path.expanduser("~/code")) + os.sep
    assert os.path.realpath(m.__file__).startswith(root), m.__file__
    assert "site-packages" not in m.__file__, m.__file__

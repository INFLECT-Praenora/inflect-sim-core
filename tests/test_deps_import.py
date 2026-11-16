import importlib

import pytest


@pytest.mark.parametrize(
    "mod",
    ["process_bigraph", "viva_core", "viva_tellurium", "viva_biogears", "viva_emitters"],
)
def test_core_deps_import(mod):
    importlib.import_module(mod)

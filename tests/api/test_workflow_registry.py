"""Task 1: viva-torch is a base dep and NeuralProcess registers in build_core()."""
import importlib


def test_neural_process_registered_in_core():
    from viva_simulation_core.core import build_core

    assert "NeuralProcess" in build_core().list_processes()


def test_viva_torch_is_importable_as_base_dep():
    # viva-torch moved from the [surrogate] extra into base deps, so it is always
    # installed; the [surrogate] extra remains as an empty back-compat alias.
    assert importlib.util.find_spec("viva_torch") is not None

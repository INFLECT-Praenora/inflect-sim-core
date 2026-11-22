"""Optional simulation-engine registry + lazy, guarded loaders.

Every simulation ENGINE wrapper is an **optional** install (see the
``[project.optional-dependencies]`` extras in ``pyproject.toml``):

    engine key   extra                 wraps
    ----------   -------------------   --------------------------------------------
    tellurium    [tellurium]           Tellurium / roadrunner (exact SBML, CVODE)
    copasi       [copasi]              COPASI (exact SBML, via basico)
    biogears     [biogears]            BioGears 8.2.0 (C++ whole-body physiology)
    surrogate    [surrogate]           PyTorch learned surrogate (viva-torch)
    rust         [rust]                viva-core (Rust execution backend)
    julia        [julia]               Julia DifferentialEquations.jl (exact ODE)

A bare ``pip install simulation-core`` pulls in **no** engine — just the
framework, the workbench, and the CLI. Install the ones you want:

    pip install "simulation-core[tellurium]"          # one engine
    pip install "simulation-core[tellurium,copasi]"   # several
    pip install "simulation-core[sbml]"               # all exact-SBML engines
    pip install "simulation-core[all]"                # every engine

This module is the single place the package touches an engine import, so the
rest of the code never imports one at module top-level: ``import
viva_simulation_core`` and rendering studies/reports work with zero engines
installed. Asking for an engine that isn't installed raises
:class:`EngineNotInstalled`, which names the exact extra to install — never a
bare ``ModuleNotFoundError``.
"""
from __future__ import annotations

import importlib
import importlib.util

# engine key -> (top-level import module, extra name, human description)
_ENGINES: dict[str, tuple[str, str, str]] = {
    "tellurium": ("viva_tellurium", "tellurium", "Tellurium / roadrunner (exact SBML, CVODE)"),
    "copasi": ("viva_copasi", "copasi", "COPASI (exact SBML, via basico)"),
    "biogears": ("viva_biogears", "biogears", "BioGears 8.2.0 (C++ whole-body physiology)"),
    "surrogate": ("viva_torch", "surrogate", "PyTorch learned surrogate"),
    "rust": ("viva_core", "rust", "viva-core (Rust execution backend)"),
    "julia": ("viva_julia_diffeqs", "julia", "Julia DifferentialEquations.jl (exact ODE)"),
}


class EngineNotInstalled(ImportError):
    """A simulation engine was requested but its optional extra isn't installed."""


def known_engines() -> list[str]:
    """Every engine this package knows how to wrap (installed or not)."""
    return list(_ENGINES)


def is_available(engine: str) -> bool:
    """True if ``engine``'s wrapper package can be imported (the extra is installed)."""
    spec = _ENGINES.get(engine)
    if spec is None:
        return False
    return importlib.util.find_spec(spec[0]) is not None


def available_engines() -> list[str]:
    """The engines whose extras are installed in this environment."""
    return [e for e in _ENGINES if is_available(e)]


def describe(engine: str) -> str:
    """Human description of an engine key."""
    spec = _ENGINES.get(engine)
    return spec[2] if spec else engine


def require(engine: str):
    """Import and return an engine's top-level module, or raise EngineNotInstalled.

    The error message names the exact extra to install, so a user who runs a
    composite without the engine gets an actionable instruction rather than an
    opaque import error.
    """
    spec = _ENGINES.get(engine)
    if spec is None:
        raise EngineNotInstalled(
            f"unknown engine {engine!r}; known engines: {known_engines()}"
        )
    module, extra, _desc = spec
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise EngineNotInstalled(
            f"the {engine!r} simulation engine is not installed. Install it with:\n"
            f"    pip install 'simulation-core[{extra}]'\n"
            f"    # or, with uv:  uv sync --extra {extra}\n"
            f"Installed engines: {available_engines() or 'none'}."
        ) from exc

"""Bridge to the curated model database (``inflect-models-db``).

Every model in ``inflect-models-db`` ships a ``card.yaml`` that is, in effect, a
complete run spec — the Antimony, named scenarios (initial conditions), the
observable species, and the time window. This module lets the simulation core
**discover that whole database and run any model in it by slug**:

    from viva_simulation_core import simulate
    out = simulate("reynolds-2006-acute-inflammation", scenario="aseptic-death")

and registers every curated model as a composite in the catalog, so it shows up
in the workbench alongside the workspace's own composites.

The dependency is **optional and lazy**, exactly like the engine wrappers: a bare
``import viva_simulation_core`` works with ``inflect-models-db`` absent; the
helpers here return "not available" / empty instead of raising. Install it with
``pip install "simulation-core[models-db]"`` (an editable/dev install exposes the
curated ``models/`` directory; or point ``INFLECT_MODELS_DB_ROOT`` at a checkout).
"""
from __future__ import annotations

import importlib
from typing import Any


def available() -> bool:
    """Whether ``inflect-models-db`` is importable."""
    return importlib.util.find_spec("inflect_models_db") is not None


def _bigraph():
    """The ``inflect_models_db.bigraph`` module, or ``None`` if not installed.
    Importing it is engine-free (it only touches process-bigraph + the cards)."""
    try:
        return importlib.import_module("inflect_models_db.bigraph")
    except Exception:  # noqa: BLE001 — optional dep: never raise on absence
        return None


def _registry():
    try:
        return importlib.import_module("inflect_models_db.registry")
    except Exception:  # noqa: BLE001
        return None


def list_models() -> list[str]:
    """Slugs of every curated model, or ``[]`` when the database isn't installed."""
    reg = _registry()
    if reg is None:
        return []
    try:
        return list(reg.list_models())
    except Exception:  # noqa: BLE001
        return []


def is_slug(name: Any) -> bool:
    """Whether ``name`` is the slug of a known curated model."""
    return isinstance(name, str) and name in set(list_models())


def register_all() -> list[str]:
    """Register every curated model as a composite generator in process-bigraph's
    registry, so the catalog/workbench lists them. No-op (``[]``) when the database
    isn't installed. Engine-free; safe to call on package import."""
    bg = _bigraph()
    if bg is None:
        return []
    try:
        return list(bg.register_all())
    except Exception:  # noqa: BLE001 — discovery must never break a build
        return []


def resolve(slug: str, scenario: str | None = None) -> dict:
    """The run spec for one curated model + scenario.

    Returns ``{slug, scenario, antimony, initials, observables, duration,
    interval, start}`` (see ``inflect_models_db.bigraph.scenario_request``).
    Raises ``LookupError`` if the database isn't installed or the slug is unknown.
    """
    bg = _bigraph()
    reg = _registry()
    if bg is None or reg is None:
        raise LookupError(
            "inflect-models-db is not installed — "
            'pip install "simulation-core[models-db]" '
            "(or set INFLECT_MODELS_DB_ROOT to a checkout)"
        )
    try:
        card = reg.get_model(slug)
    except Exception as e:  # noqa: BLE001
        raise LookupError(f"no curated model {slug!r}: {e}") from e
    return bg.scenario_request(card, scenario)


def apply_to_request_kwargs(slug: str, scenario: str | None, kwargs: dict) -> tuple[str, dict]:
    """Resolve a models-db ``slug`` into ``(antimony_text, request_kwargs)`` for
    :func:`viva_simulation_core.simulate`.

    Fills the request fields the card determines, **without overriding anything the
    caller set explicitly**: ``model_format='antimony'``, ``observables`` (the
    card's species), ``duration``/``interval`` (the card's time block), and merges
    the scenario's initial conditions into ``parameters`` (caller params win).
    """
    spec = resolve(slug, scenario)
    out = dict(kwargs)
    out.setdefault("model_format", "antimony")
    out.setdefault("observables", list(spec["observables"]))
    out.setdefault("duration", float(spec["duration"]))
    out.setdefault("interval", float(spec["interval"]))
    # Scenario initial conditions as parameter overrides; caller's own params win.
    params = {**spec["initials"], **(out.get("parameters") or {})}
    out["parameters"] = params
    return spec["antimony"], out

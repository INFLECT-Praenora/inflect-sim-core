"""Enforce that every engine wrapper the workspace can install (a pyproject extra)
is declared in workspace.yaml `imports`, so the catalog (Repositories tab) always
shows every engine the workspace imports.

Without this, an engine extra added to pyproject silently fails to appear in the
catalog unless it also happens to be in the viva ecosystem registry (viva-biogears
and viva-core are not) — exactly the gap this test closes.
"""
import re
import pathlib

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _engine_wrapper_packages() -> set[str]:
    """viva-* wrapper distributions pulled in by pyproject's engine extras."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = (data.get("project", {}).get("optional-dependencies", {}) or {})
    pkgs: set[str] = set()
    for deps in extras.values():
        for dep in deps:
            # "viva-tellurium @ git+https://..."  (skip bundle extras that just name other extras)
            m = re.match(r"\s*(viva-[a-z0-9-]+)\s*@", dep)
            if m:
                pkgs.add(m.group(1))
    return pkgs


def _declared_import_names() -> set[str]:
    ws = yaml.safe_load((ROOT / "workspace.yaml").read_text(encoding="utf-8")) or {}
    imports = ws.get("imports") or {}
    names: set[str] = set()
    for key, spec in imports.items():
        spec = spec if isinstance(spec, dict) else {}
        for v in (key, spec.get("package")):
            if v:
                names.add(str(v).lower().replace("_", "-"))
    return names


def test_every_engine_extra_is_declared_in_workspace_imports():
    wrappers = _engine_wrapper_packages()
    assert wrappers, "no engine wrapper extras found in pyproject.toml"
    declared = _declared_import_names()
    missing = sorted(w for w in wrappers if w.replace("_", "-") not in declared)
    assert not missing, (
        "engine wrapper extras not declared in workspace.yaml `imports` "
        f"(so they won't show in the catalog): {missing}. Add them under `imports:`."
    )


def test_declared_engine_imports_are_reference_mode():
    """Engine imports must be mode: reference so build_core() does not eagerly import
    them (preserving the lazy-engine / works-with-zero-engines guarantee)."""
    ws = yaml.safe_load((ROOT / "workspace.yaml").read_text(encoding="utf-8")) or {}
    imports = ws.get("imports") or {}
    wrappers = {w.replace("_", "-") for w in _engine_wrapper_packages()}
    for key, spec in imports.items():
        spec = spec if isinstance(spec, dict) else {}
        name = str(spec.get("package") or key).lower().replace("_", "-")
        if name in wrappers:
            assert str(spec.get("mode", "")).lower() == "reference", (
                f"engine import {key!r} must be `mode: reference` so build_core() stays lazy"
            )

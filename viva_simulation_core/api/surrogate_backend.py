"""Tiered-engine layer: surrogate protocol, operator-curated checkpoint sidecars,
and the envelope-gated engine selector.

No torch here by design. The real viva-torch adapter (which loads the checkpoint
named by ``SurrogateEntry.checkpoint_ref``) is a TODO that plugs in behind
``SurrogateBackend``.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


class SurrogateUnavailable(Exception):
    """Surrogate demanded/needed but no usable checkpoint or runtime (HTTP 409)."""


@dataclass
class SurrogateEntry:
    checkpoint_ref: str
    targets: list[str]
    horizon_s: float
    param_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    state_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    val_error: dict = field(default_factory=dict)


class SurrogateBackend(Protocol):
    def lookup(self, model_hash: str) -> SurrogateEntry | None: ...

    def predict(self, entry: SurrogateEntry, request: Any) -> dict: ...


class NullBackend:
    def lookup(self, model_hash: str) -> SurrogateEntry | None:
        return None

    def predict(self, entry: SurrogateEntry, request: Any) -> dict:
        raise SurrogateUnavailable("no surrogate backend configured")


_HASH_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _ranges(d: dict) -> dict[str, tuple[float, float]]:
    return {k: (float(v[0]), float(v[1])) for k, v in (d or {}).items()}


class CheckpointStore:
    """Reads ``<model_hash>.json`` envelope sidecars from an operator-curated dir."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def lookup(self, model_hash: str) -> SurrogateEntry | None:
        if not _HASH_RE.match(model_hash) or model_hash.startswith("."):
            return None
        path = self.directory / f"{model_hash}.json"
        if not path.is_file():
            return None
        try:
            d = json.loads(path.read_text())
            entry = SurrogateEntry(
                checkpoint_ref=str(d["checkpoint_ref"]),
                targets=[str(t) for t in d["targets"]],
                horizon_s=float(d["horizon_s"]),
                param_ranges=_ranges(d.get("param_ranges")),
                state_ranges=_ranges(d.get("state_ranges")),
                val_error=dict(d.get("val_error") or {}),
            )
            bounds = [b for r in (*entry.param_ranges.values(), *entry.state_ranges.values()) for b in r]
            if not all(math.isfinite(x) for x in (entry.horizon_s, *bounds)):
                return None
            return entry
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return None

    def predict(self, entry: SurrogateEntry, request: Any) -> dict:
        # TODO: real viva-torch adapter loads entry.checkpoint_ref and runs inference.
        raise SurrogateUnavailable("sidecar known, runtime adapter not installed")


class FakeBackend:
    """Test double returning a canned prediction."""

    def __init__(self, entry: SurrogateEntry | None, predict_fn: Callable[[SurrogateEntry, Any], dict]):
        self.entry = entry
        self.predict_fn = predict_fn

    def lookup(self, model_hash: str) -> SurrogateEntry | None:
        return self.entry

    def predict(self, entry: SurrogateEntry, request: Any) -> dict:
        return self.predict_fn(entry, request)


def _within(values: dict | None, ranges: dict) -> bool:
    for k, v in (values or {}).items():
        if k in ranges:
            lo, hi = ranges[k]
            try:
                if not (lo <= v <= hi):
                    return False
            except TypeError:
                return False
        else:
            return False  # unknown input: outside the validated envelope
    return True


def _eligible(request: Any, entry: SurrogateEntry | None) -> bool:
    if entry is None:
        return False
    return (
        set(request.observables) <= set(entry.targets)
        and request.horizon_s <= entry.horizon_s
        and _within(getattr(request, "parameters", None), entry.param_ranges)
        and _within(getattr(request, "initial_state", None), entry.state_ranges)
    )


def select_engine(request: Any, entry: SurrogateEntry | None, requested_engine: str) -> tuple[str, bool]:
    if requested_engine == "exact":
        return "exact", True
    if requested_engine == "surrogate":
        if entry is None:
            raise SurrogateUnavailable("no surrogate checkpoint for this model")
        return "surrogate", _eligible(request, entry)
    # auto
    if _eligible(request, entry):
        return "surrogate", True
    return "exact", True

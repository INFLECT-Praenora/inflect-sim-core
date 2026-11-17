"""Import-guarded access to viva-torch (optional ``[surrogate]`` extra).

Importing this module never requires torch. The thin viva-torch API is
re-exported only when available; otherwise use ``require_viva_torch()`` /
``get(name)``, which raise a clear ImportError naming the extra.
"""
from __future__ import annotations

from typing import Any

_API = (
    "SurrogateSpec",
    "TrajectorySampler",
    "SamplingPlan",
    "train_surrogate",
    "neural_process_node",
    "evaluate_surrogate",
)

try:
    import torch  # noqa: F401
    import viva_torch as _vt

    HAS_VIVA_TORCH: bool = True
except ImportError:
    _vt = None
    HAS_VIVA_TORCH = False

if HAS_VIVA_TORCH:
    for _name in _API:
        if hasattr(_vt, _name):
            globals()[_name] = getattr(_vt, _name)

_MSG = (
    "viva-torch / torch is not installed. Install the optional extra: "
    "pip install 'simulation-core[surrogate]'"
)


def require_viva_torch() -> None:
    """Raise ImportError (naming the [surrogate] extra) if viva-torch is absent."""
    if not HAS_VIVA_TORCH:
        raise ImportError(_MSG)


def get(name: str) -> Any:
    """Guarded accessor for a viva-torch API name."""
    require_viva_torch()
    return getattr(_vt, name)


__all__ = ["HAS_VIVA_TORCH", "require_viva_torch", "get", *_API]

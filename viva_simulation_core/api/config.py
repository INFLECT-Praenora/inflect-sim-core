"""Env-driven configuration for the API (prefix ``INFLECT_API_``)."""
from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, fields

ENV_PREFIX = "INFLECT_API_"


@dataclass(frozen=True)
class ApiConfig:
    max_model_bytes: int = 1024 * 1024
    max_species_reactions: int = 2000
    max_horizon_s: float = 24 * 3600.0
    max_steps_per_call: int = 10000
    max_ensemble_n: int = 256
    max_observables_per_call: int = 64
    request_timeout_s: float = 30.0
    register_compile_timeout_s: float = 60.0
    max_twins: int = 64
    twin_ttl_s: float = 3600.0
    db_path: str = ""  # SQLite file for twin persistence; "" = no persistence
    api_keys: str = ""  # comma-separated bearer keys (INFLECT_API_API_KEYS); empty = all rejected
    rate_capacity: float = 100.0  # token-bucket burst size per key
    rate_refill_per_s: float = 10.0  # tokens refilled per second per key
    cost_step: float = 1.0
    cost_register: float = 10.0
    cost_ensemble: float = 20.0

    @property
    def key_set(self) -> tuple[str, ...]:
        return tuple(k.strip() for k in self.api_keys.split(",") if k.strip())

    @property
    def costs(self) -> dict[str, float]:
        return {"step": self.cost_step, "register": self.cost_register,
                "ensemble": self.cost_ensemble}

    @classmethod
    def from_env(cls) -> "ApiConfig":
        kwargs = {}
        for f in fields(cls):
            raw = os.environ.get(ENV_PREFIX + f.name.upper())
            if raw is None or raw == "":
                continue
            kind = type(f.default) if f.default is not MISSING else float
            kwargs[f.name] = kind(raw)
        return cls(**kwargs)

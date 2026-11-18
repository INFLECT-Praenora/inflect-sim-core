"""Pydantic request/response schemas for the digital-twin API (shape only)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RegisterRequest(BaseModel):
    model: str
    model_format: Literal["antimony", "sbml"]
    metadata: dict = Field(default_factory=dict)
    observables: list[str] | dict[str, str] | None = None
    defaults: dict | None = None

    model_config = ConfigDict(protected_namespaces=())


class RegisterResponse(BaseModel):
    twin_id: str
    observables: list[str]
    parameters: list[str]
    engines_available: list[str]


class EnsembleSpec(BaseModel):
    members: list[dict] | None = None
    draws: dict[str, dict] | None = None
    n: int | None = None
    stochastic: bool = False
    seed: int | None = None
    quantiles: list[float] | None = None
    per_member: bool = False


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    observables: list[str]
    horizon_s: float
    interval_s: float
    initial_state: dict[str, float] | None = None
    parameters: dict[str, float] | None = None
    ensemble: EnsembleSpec | None = None
    engine: Literal["auto", "exact", "surrogate"] = "auto"
    return_: Literal["trajectory", "endpoint", "stats"] = Field(
        default="trajectory", alias="return"
    )


class StepRequest(BaseModel):
    dt_s: float | None = None
    parameters: dict[str, float] | None = None
    species: dict[str, float] | None = None
    observables: list[str] | None = None
    seq: int | None = None


class Provenance(BaseModel):
    engine_used: Literal["exact", "surrogate"]
    seed: int | None
    integrator: str
    wall_ms: float
    n_steps: int
    lib_versions: dict[str, str] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    engine_used: str
    observables: dict
    times: list[float] | None = None
    ensemble_stats: dict | None = None
    in_distribution: bool | None = None
    provenance: Provenance


class StepResponse(BaseModel):
    t: float
    observables: dict
    provenance: Provenance


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None

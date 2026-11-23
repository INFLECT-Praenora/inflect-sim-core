"""Closed-loop sepsis co-simulation composite (physiology engine <-> controller).

The CO-SIMULATION pillar (study 06). Two processes in one composite exchange state
on every tick:

  * ``physiology`` — the real BioGears-sepsis core run by the Tellurium engine
    (``sepsis_sbml``). Each step it writes all global parameters (MAP, SVR, the
    pathogen PB/PT, tissue integrity TI ...) to the shared ``parameters`` store and
    reads a ``parameters_in`` writeback before its next step.
  * ``controller`` — the :class:`AntibioticControllerProcess`: a closed-loop
    pharmacology process that reads ``parameters`` (senses bacteremia), integrates
    an antibiotic PK, and writes the drug-reduced pathogen back via
    ``parameters_in``.

The feedback edge ``controller.parameters_in -> physiology.parameters_in`` is what
makes this a co-simulation rather than two independent runs: the treatment the
controller computes at tick *n* changes the physiology the engine integrates at
tick *n+1*, whose new state the controller reads at tick *n+2*.

``treat=False`` disables the controller (inert), so the identical composite
produces the matched untreated control — an apples-to-apples closed-loop vs.
no-treatment comparison from one spec.

PT/PB are rate-rule state variables in the SBML (not assignment-rule outputs), so
the engine accepts the controller's mid-run writeback with no model surgery.
"""
from __future__ import annotations

from pathlib import Path

from process_bigraph.composite_generator import composite_generator

from viva_simulation_core.core import build_core
from viva_simulation_core.engines import require as require_engine
from viva_simulation_core.composites.sepsis_sbml import SBML_PATH, SEPSIS_OBSERVABLES


@composite_generator(
    name="sepsis_cosim",
    description="Closed-loop sepsis CO-SIMULATION: the Tellurium sepsis physiology "
                "engine and an antibiotic-PK controller process exchange state every "
                "tick. The controller senses bacteremia (PB), infuses an antibiotic "
                "with a one-compartment PK, and writes the drug-killed pathogen (PT/PB) "
                "back into the engine via its parameters_in port. treat=False disables "
                "the controller for the matched untreated control.",
    parameters={
        "severity": {"type": "integer", "default": 3,
                     "description": "0 none / 1 mild / 2 moderate / 3 severe"},
        "treat": {"type": "boolean", "default": True,
                  "description": "True = closed-loop antibiotic; False = matched untreated control"},
        "infection_start_s": {"type": "float", "default": 1.0,
                              "description": "Infection start time (s)"},
        "interval_s": {"type": "float", "default": 60.0, "description": "Seconds per co-sim tick"},
        "trigger_pb": {"type": "float", "default": 1.0,
                       "description": "Bacteremia (PB) that triggers treatment onset"},
        "start_time_s": {"type": "float", "default": 0.0,
                         "description": "Earliest treatment onset (0 = early; larger = delayed)"},
    },
    core_extensions=[build_core],
    default_n_steps=360,
    fidelity="reproduction",
)
def make_sepsis_cosim_composite(core=None, *, severity: int = 3, treat: bool = True,
                                infection_start_s: float = 1.0, interval_s: float = 60.0,
                                trigger_pb: float = 1.0, start_time_s: float = 0.0,
                                horizon_s: float = 86400.0) -> dict:
    """Build the coupled physiology<->controller composite document."""
    make_tellurium_document = require_engine("tellurium").make_tellurium_document

    overrides = {"severity": float(severity)}
    if severity > 0:
        overrides["infectionStart"] = float(infection_start_s)

    doc = make_tellurium_document(
        model_file=str(SBML_PATH), model_format="sbml",
        parameter_overrides=overrides, interval=float(interval_s))

    # Rename the engine node for clarity and wire its closed-loop input port.
    phys = doc.pop("tellurium")
    phys["inputs"] = {"parameters_in": ["stores", "parameters_in"]}
    doc["physiology"] = phys

    # The antibiotic-PK controller: reads the engine's parameters, writes the
    # drug-reduced pathogen back to parameters_in, and reports drug/treating.
    doc["controller"] = {
        "_type": "process",
        "address": "local:AntibioticControllerProcess",
        "config": {
            "enabled": bool(treat),
            "trigger_pb": float(trigger_pb),
            "start_time_s": float(start_time_s),
            "pathogen_ids": ["PT", "PB"],
            "sense_id": "PB",
        },
        "interval": float(interval_s),
        "inputs": {"parameters": ["stores", "parameters"]},
        "outputs": {
            "parameters_in": ["stores", "parameters_in"],
            "drug": ["stores", "drug"],
            "treating": ["stores", "treating"],
        },
    }

    # The sepsis SBML has zero floating species, so drop the species/rates ports
    # from the emitter (an empty array[(0,)] store clashes with the emitter's
    # map[float] schema). Emit only parameters + time + the controller telemetry.
    emit = doc["emitter"]["config"]["emit"]
    for k in ("species", "rates"):
        emit.pop(k, None)
        doc["emitter"]["inputs"].pop(k, None)
    emit.update({"drug": "float", "treating": "float"})
    doc["emitter"]["inputs"].update({
        "drug": ["stores", "drug"],
        "treating": ["stores", "treating"],
    })
    return doc


def extract_cosim(rows) -> dict[str, list[float]]:
    """Emitter rows -> {observable: [...]} including drug + treating telemetry."""
    out = {k: [] for k in SEPSIS_OBSERVABLES}
    out["time"] = []
    out["drug"] = []
    out["treating"] = []
    for r in rows:
        p = r.get("parameters")
        if not p:
            continue
        for k, v in SEPSIS_OBSERVABLES.items():
            out[k].append(float(p[v]))
        out["time"].append(float(r.get("global_time", r.get("time", 0.0))))
        out["drug"].append(float(r.get("drug", 0.0)))
        out["treating"].append(float(r.get("treating", 0.0)))
    return out if out["SBP"] else {}

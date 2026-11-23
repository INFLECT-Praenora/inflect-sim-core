"""AntibioticControllerProcess — a closed-loop treatment process for co-simulation.

This is the SECOND engine in the co-simulation study (06): a pharmacology process
that runs *coupled to* the sepsis physiology engine, exchanging state every tick.

The loop each interval:

  1. READ the physiology engine's current global parameters (via the shared
     ``parameters`` store) — specifically the blood pathogen ``PB`` (bacteremia).
  2. SENSE sepsis: once ``PB`` crosses ``trigger_pb`` the controller starts an
     antibiotic infusion (closed-loop — treatment is triggered by the patient's
     own state, not a fixed schedule).
  3. INTEGRATE a one-compartment antibiotic PK over the interval (first-order
     elimination ``ke``, infusion ``dose_rate`` into volume ``vd``), carrying the
     drug concentration across ticks in its own state.
  4. ACTUATE: compute a concentration-dependent bactericidal kill rate
     ``kkill = kill_max * C/(ec50 + C)`` (an Emax PD model) and write the reduced
     pathogen state ``PT``/``PB`` back into the physiology engine via its
     ``parameters_in`` port (applied before the engine's next step).

So the physiology engine and the pharmacology controller are two processes in one
composite that hand state back and forth on every tick — a genuine co-simulation,
not two independent runs compared afterwards. ``enabled=False`` makes the
controller inert (writes nothing), so the *same* composite yields the matched
untreated control.

PT/PB are rate-rule state variables in the BioGears-sepsis SBML (not
assignment-rule outputs), so roadrunner accepts a mid-run ``setValue`` on them;
that is what makes the real flagship model actuatable without any model surgery.
"""
from __future__ import annotations

import math

from process_bigraph import Process


class AntibioticControllerProcess(Process):
    """Closed-loop antibiotic controller coupled to the sepsis physiology engine."""

    config_schema = {
        # Pharmacodynamics (Emax bactericidal kill on the pathogen)
        "kill_max": {"_type": "float", "_default": 3.0e-3},   # max kill rate (1/s)
        "ec50": {"_type": "float", "_default": 1.0},          # drug conc at half-max kill
        # Pharmacokinetics (one-compartment, first-order elimination)
        "ke": {"_type": "float", "_default": 3.0e-4},         # elimination rate (1/s)
        "vd": {"_type": "float", "_default": 1.0},            # volume of distribution
        "dose_rate": {"_type": "float", "_default": 2.0e-3},  # infusion rate while treating (mass/s)
        # Closed-loop sensing
        "trigger_pb": {"_type": "float", "_default": 1.0},    # start treatment when PB >= this
        "pathogen_ids": {"_type": "list[string]", "_default": ["PT", "PB"]},
        "sense_id": {"_type": "string", "_default": "PB"},    # which observable triggers treatment
        "enabled": {"_type": "boolean", "_default": True},    # False -> inert (matched untreated arm)
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self._C = 0.0          # drug concentration, carried across ticks
        self._active = False   # has treatment been triggered?

    def inputs(self):
        # Reads the physiology engine's full global-parameter map (holds PB/PT/...).
        return {"parameters": "map[float]"}

    def outputs(self):
        # Writes the reduced pathogen back to the engine (overwrite: replaces, not
        # adds) plus drug/activity telemetry for the figure.
        return {
            "parameters_in": "overwrite[map[float]]",
            "drug": "overwrite[float]",
            "treating": "overwrite[float]",
        }

    def update(self, state, interval):
        params = state.get("parameters") or {}
        if not self.config.get("enabled", True):
            # Inert controller: the matched untreated control. Still report telemetry.
            return {"parameters_in": {}, "drug": 0.0, "treating": 0.0}

        sense = float(params.get(self.config["sense_id"], 0.0))
        if not self._active and sense >= float(self.config["trigger_pb"]):
            self._active = True

        ke = float(self.config["ke"])
        vd = float(self.config["vd"])
        dose_rate = float(self.config["dose_rate"]) if self._active else 0.0
        dt = float(interval)

        # One-compartment PK, analytic over a constant-infusion interval:
        #   C(t+dt) = C e^{-ke dt} + (dose_rate / (vd ke)) (1 - e^{-ke dt})
        if ke > 0:
            decay = math.exp(-ke * dt)
            css = dose_rate / (vd * ke)
            self._C = self._C * decay + css * (1.0 - decay)
        else:
            self._C += dose_rate / vd * dt

        # Emax bactericidal kill; apply as first-order decay on the pathogen state.
        kkill = float(self.config["kill_max"]) * self._C / (float(self.config["ec50"]) + self._C)
        kill_decay = math.exp(-kkill * dt)

        writeback = {}
        for pid in self.config["pathogen_ids"]:
            if pid in params:
                writeback[pid] = float(params[pid]) * kill_decay

        return {
            "parameters_in": writeback,
            "drug": float(self._C),
            "treating": 1.0 if self._active else 0.0,
        }

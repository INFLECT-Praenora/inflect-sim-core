"""BioGears-engine sepsis baseline composite (real BioGears via viva-biogears).

Wraps ``viva_biogears.composites.sepsis.biogears_sepsis`` (StandardMale, McDaniel
infection scenario) and exposes the SAME 9 observables as the SBML side
(``SEPSIS_OBSERVABLES``), mapped to BioGears readings below. Requires the native
build (pybiogears + ``BIOGEARS_RUNTIME``).

Severity uses the SBML side's int convention (0 none, 1 mild, 2 moderate, 3 severe);
it is translated to the engine's string levels internally.

Limitation: BioGears starts the infection at t=0 of the run (the severity is read
once from the setpoint), so ``infection_start_s`` must be None or 0.
"""
from __future__ import annotations

from viva_biogears.composites.sepsis import biogears_sepsis
from viva_biogears.core import build_core

from viva_simulation_core.composites.sepsis_sbml import SEPSIS_OBSERVABLES

SEVERITY_NAMES = {0: 'none', 1: 'mild', 2: 'moderate', 3: 'severe'}

# shared observable -> path under row['readings'] (viva-biogears >= 8d02019; was row['circuits']) (all 9 have an engine reading)
ENGINE_READINGS = {
    'SBP': ('cardiovascular', 'hemodynamics', 'systolic_pressure_mmHg'),            # mmHg
    'DBP': ('cardiovascular', 'hemodynamics', 'diastolic_pressure_mmHg'),           # mmHg
    'MAP': ('cardiovascular', 'hemodynamics', 'mean_arterial_pressure_mmHg'),       # mmHg
    'HR': ('cardiovascular', 'hemodynamics', 'heart_rate_bpm'),                     # 1/min
    'CO': ('cardiovascular', 'hemodynamics', 'cardiac_output_L_per_min'),           # L/min
    # mmHg s/mL, same unit as the SBML SVR = (MAP-CVP)/(CO*1000/60)
    'SVR': ('infection', 'clinical', 'systemic_vascular_resistance_mmHg_s_per_mL'),
    'blood_volume': ('cardiovascular', 'hemodynamics', 'blood_volume_mL'),         # mL
    'tissue_integrity': ('infection', 'inflammation', 'tissue_integrity'),          # 0..1 (InflammatoryResponse)
    'blood_pathogen': ('infection', 'inflammation', 'blood_pathogen'),              # InflammatoryResponse
}
assert set(ENGINE_READINGS) == set(SEPSIS_OBSERVABLES)


def make_sepsis_engine_composite(severity: int = 0,
                                 infection_start_s: float | None = None,
                                 horizon_s: float = 3600.0,
                                 interval_s: float = 60.0,
                                 core=None) -> dict:
    """Build the BioGears composite document (healthy at severity=0).

    ``horizon_s`` is advisory (run the Composite for that long). Run with the
    core from ``viva_biogears.core.build_core()``.
    """
    if int(severity) not in SEVERITY_NAMES:
        raise ValueError(f'severity must be one of {list(SEVERITY_NAMES)}, got {severity!r}')
    if infection_start_s not in (None, 0, 0.0):
        raise ValueError('the BioGears engine starts the infection at t=0; '
                         'infection_start_s must be None or 0')
    return biogears_sepsis(core or build_core(), severity=SEVERITY_NAMES[int(severity)],
                           interval=float(interval_s))


def extract_observables(rows) -> dict[str, list[float]]:
    """Emitter rows -> {observable: [values over time]}, same keys as the SBML side."""
    out = {k: [] for k in SEPSIS_OBSERVABLES}
    for r in rows:
        c = r.get('readings')
        if not c:
            continue
        for k, path in ENGINE_READINGS.items():
            v = c
            for p in path:
                v = v[p]
            out[k].append(float(v))
    return out if out['SBP'] else {}

"""Sepsis milestone extraction from an aligned time-series of observables.

Computes the McDaniel-2019 / extraction-report validation milestones (event
timings and ratios) so the mild/severe arms can report crossing times and the
engine-vs-SBML divergence table. Input is the same sample shape align.py uses:
``[{'time': seconds, 'SBP': ..., ...}, ...]``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Mapping, Optional, Sequence

_H = 3600.0


@dataclass(frozen=True)
class SepsisMilestones:
    sbp_below_100_h: Optional[float]
    map_below_65_h: Optional[float]
    hr_above_90_h: Optional[float]
    svr_ratio_min: Optional[float]
    svr_ratio_min_h: Optional[float]
    tissue_integrity_min: Optional[float]
    tissue_integrity_min_h: Optional[float]
    peak_blood_pathogen: Optional[float]
    peak_blood_pathogen_h: Optional[float]
    resolved: Optional[bool]

    def as_dict(self) -> dict:
        return asdict(self)


def _series(samples: Sequence[Mapping], key: str):
    out = [(float(s['time']), float(s[key])) for s in samples
           if key in s and s[key] is not None]
    return sorted(out)


def _first_sustained(pts, cond: Callable[[float], bool], sustain_s: float):
    """Onset time (h) of the first run of consecutive samples satisfying cond
    whose span (first to last sample of the run) is >= sustain_s. The run may
    still be ongoing at the end of the record. None if no such run."""
    start = None
    for t, v in pts:
        if cond(v):
            if start is None:
                start = t
            if t - start >= sustain_s:
                return start / _H
        else:
            start = None
    return None


def _extreme(pts, fn):
    if not pts:
        return None, None
    t, v = fn(pts, key=lambda p: p[1])
    return v, t / _H


def extract_milestones(samples: Sequence[Mapping], *, sustain_minutes: float = 30.0,
                       sbp_threshold: float = 100.0, map_threshold: float = 65.0,
                       hr_threshold: float = 90.0, resolved_frac: float = 0.01,
                       resolved_abs: float = 1e-6) -> SepsisMilestones:
    """Extract milestones. Threshold events report the ONSET of the first run
    that holds for >= ``sustain_minutes`` (0 => any single sample counts).
    ``resolved``: final blood_pathogen <= max(resolved_abs, resolved_frac*peak)."""
    sustain_s = sustain_minutes * 60.0
    sbp = _series(samples, 'SBP')
    mp = _series(samples, 'MAP')
    hr = _series(samples, 'HR')
    svr = _series(samples, 'SVR')
    tis = _series(samples, 'tissue_integrity')
    pat = _series(samples, 'blood_pathogen')

    svr_min = svr_min_h = None
    if svr and svr[0][1] != 0:
        base = svr[0][1]
        svr_min, svr_min_h = _extreme([(t, v / base) for t, v in svr], min)
    tis_min, tis_min_h = _extreme(tis, min)
    pk, pk_h = _extreme(pat, max)
    resolved = None if not pat else pat[-1][1] <= max(resolved_abs, resolved_frac * pk)

    return SepsisMilestones(
        sbp_below_100_h=_first_sustained(sbp, lambda v: v < sbp_threshold, sustain_s),
        map_below_65_h=_first_sustained(mp, lambda v: v < map_threshold, sustain_s),
        hr_above_90_h=_first_sustained(hr, lambda v: v > hr_threshold, sustain_s),
        svr_ratio_min=svr_min, svr_ratio_min_h=svr_min_h,
        tissue_integrity_min=tis_min, tissue_integrity_min_h=tis_min_h,
        peak_blood_pathogen=pk, peak_blood_pathogen_h=pk_h, resolved=resolved)

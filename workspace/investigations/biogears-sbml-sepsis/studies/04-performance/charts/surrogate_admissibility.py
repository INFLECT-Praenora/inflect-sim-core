#!/usr/bin/env python3
"""Study 04: surrogate admissibility on the severe arm — where acceleration fails.

The surrogate is ~40,000x faster (main finding), but speed is only useful if the
predictions are admissible. This checks the committed surrogate severe-arm trajectory
(data/real_trajectories.json, run at the surrogate's trained 300 s timestep) against
simple physical bounds, and against the exact engine's range. A prediction that leaves
physical bounds is not even screening-grade.

Writes charts/surrogate_admissibility.json and charts/04_surrogate_admissibility.svg|png.
Run from the investigation dir:
    PYTHONPATH=<worktree> python studies/04-performance/charts/surrogate_admissibility.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INV = HERE.parents[2]
TRAJ = INV / "data" / "real_trajectories.json"

# Physical admissibility bounds per observable.
BOUNDS = {
    "SBP": (0, 300), "DBP": (0, 300), "MAP": (0, 300), "HR": (0, 400),
    "CO": (0, 50), "SVR": (0, 50), "blood_volume": (0, 1e5),
    "tissue_integrity": (0.0, 1.0), "blood_pathogen": (0, None),
}

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})


def frac_out(vals, lo, hi):
    v = np.asarray(vals, float)
    bad = np.zeros(len(v), bool)
    if lo is not None:
        bad |= v < lo
    if hi is not None:
        bad |= v > hi
    return float(bad.mean()), float(v.min()), float(v.max())


def main() -> None:
    d = json.loads(TRAJ.read_text())
    su = d["surrogate_sev3"]["obs"]
    ex = d["exact_sev3"]["obs"]

    rows = []
    for obs, (lo, hi) in BOUNDS.items():
        f, vmin, vmax = frac_out(su[obs], lo, hi)
        ev = np.asarray(ex[obs], float)
        rows.append({"observable": obs, "bound_lo": lo, "bound_hi": hi,
                     "surr_min": vmin, "surr_max": vmax,
                     "exact_min": float(ev.min()), "exact_max": float(ev.max()),
                     "frac_out_of_bounds": f,
                     "admissible": f == 0.0})

    n_bad = sum(not r["admissible"] for r in rows)
    verdict = ("inadmissible" if n_bad else "admissible")
    out = {"meta": {"arm": "severe (severity 3)", "timestep_s": d["surrogate_sev3"].get("dt_s"),
                    "source": "data/real_trajectories.json"},
           "verdict": verdict, "n_observables": len(rows),
           "n_inadmissible": n_bad, "rows": rows}
    (HERE / "surrogate_admissibility.json").write_text(json.dumps(out, indent=1))

    # Figure: fraction of the trajectory out of physical bounds, per observable.
    labels = [r["observable"] for r in rows]
    fracs = [100 * r["frac_out_of_bounds"] for r in rows]
    colors = ["#c0392b" if r["frac_out_of_bounds"] > 0 else "#2e8b57" for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    y = np.arange(len(labels))
    ax.barh(y, fracs, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("% of the 48 h severe trajectory outside physical bounds")
    ax.set_title("Surrogate admissibility on severe sepsis — where acceleration fails",
                 fontsize=12, fontweight="bold", loc="left")
    for yi, r in zip(y, rows):
        if r["frac_out_of_bounds"] > 0:
            ax.text(fracs[yi] + 1, yi, f"min {r['surr_min']:.3g}", va="center", fontsize=8, color="#c0392b")
    ax.set_xlim(0, max(fracs) * 1.25 + 5)
    fig.tight_layout()
    stem = HERE / "04_surrogate_admissibility"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"))
    bad_names = [r["observable"] for r in rows if not r["admissible"]]
    caption = (
        "Surrogate predictions on the severe arm (severity 3, 48 h, at the surrogate's "
        "trained 300 s timestep) checked against physical bounds. "
        f"{n_bad} of {len(rows)} observables leave admissible ranges: {', '.join(bad_names)} — "
        "the surrogate reaches MAP ≈ -158 mmHg, tissue integrity ≈ -6.1, and negative blood "
        "pathogen, all physically impossible. So on severe sepsis the surrogate is not "
        "admissible, let alone mechanistically accurate: speed does not help here, and the "
        "exact engine is required. A fair held-out-severity admissibility protocol at the "
        "trained timestep is the pending validator (study 07, SURR-01).")
    stem.with_name(stem.name + ".meta.json").write_text(json.dumps({"caption": caption}, indent=1))
    print(f"verdict={verdict}  inadmissible={n_bad}/{len(rows)}  ->", ", ".join(bad_names))


if __name__ == "__main__":
    main()

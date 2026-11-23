#!/usr/bin/env python3
"""Study 04 figure: Tellurium vs COPASI on the same SBML. Reads copasi_vs_tellurium.json."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TE = "#2f6fb3"   # Tellurium (main SBML engine)
CP = "#d4952a"   # COPASI (second SBML engine)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 10.5,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white"})


def main():
    d = json.loads((HERE / "copasi_vs_tellurium.json").read_text())
    tr = d["trajectories"]; t = tr["time_h"]
    meta = d["meta"]; wall = d["wall"]; agr = d["agreement"]
    panels = [("MAP", "MAP (mmHg)"), ("SVR", "SVR"),
              ("blood_pathogen", "Blood pathogen"), ("tissue_integrity", "Tissue integrity")]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0))
    for ax, (k, lab) in zip(axes.ravel(), panels):
        ax.plot(t, tr[f"tellurium_{k}"], color=TE, lw=2.6, label="Tellurium (roadrunner)")
        ax.plot(t, tr[f"copasi_{k}"], color=CP, lw=1.6, ls="--", label="COPASI (LSODA)")
        ax.set_ylabel(lab); ax.set_xlabel("Time (h)")
        ax.set_title(f"{lab.split(' (')[0]} — rel diff {agr[k]['max_rel_diff']:.1e}")
    axes[0, 0].legend(loc="best", fontsize=8.5)
    worst = max(agr.values(), key=lambda v: v["max_rel_diff"])["max_rel_diff"]
    fig.suptitle(f"Same sepsis SBML, two exact engines: identical trajectory "
                 f"(max rel diff {worst:.1e}), {wall['copasi_over_tellurium']:.1f}× the cost",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    stem = HERE / "03_copasi_vs_tellurium"
    fig.savefig(stem.with_suffix(".svg")); fig.savefig(stem.with_suffix(".png"))
    cap = (f"The identical BioGears-sepsis SBML (severe, {meta['horizon_h']:g} h) under the main "
           f"engine Tellurium (roadrunner/CVODE) and a second independent exact-SBML integrator, "
           f"COPASI (LSODA, driven through libCOPASI directly). The two agree to within "
           f"{worst:.1e} across all nine shared observables — the exact-SBML result is "
           f"engine-independent, not a roadrunner artifact. The agreement costs "
           f"{wall['copasi_over_tellurium']:.1f}× more wall-clock in COPASI "
           f"({wall['tellurium_s']:.0f} s vs {wall['copasi_s']:.0f} s): the model is event-dense "
           "(the per-beat event), and COPASI evaluates each event through an interpreted math layer "
           "while roadrunner runs LLVM-compiled code.")
    stem.with_name(stem.name + ".meta.json").write_text(json.dumps({"caption": cap}, indent=1))
    print("wrote 03_copasi_vs_tellurium.svg | max rel diff %.2e | copasi/tellurium %.1fx"
          % (worst, wall["copasi_over_tellurium"]))


if __name__ == "__main__":
    main()

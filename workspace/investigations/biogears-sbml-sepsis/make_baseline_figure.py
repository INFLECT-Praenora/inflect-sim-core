#!/usr/bin/env python3
"""Study 01 (baseline parity) figure. Pure read of out/baseline.json.

    .venv/bin/python workspace/investigations/biogears-sbml-sepsis/make_baseline_figure.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STUDY = HERE / "studies" / "01-baseline-parity"
D = json.load(open(STUDY / "out" / "baseline.json"))
ENGINE = "#b3462f"; SBML = "#2f6fb3"; TARGET = "#7a7f87"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

obs = D["observables"]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
for ax, o in zip(axes, obs):
    e, s, t = D["engine"][o], D["sbml"][o], D["target"][o]
    bars = ax.bar([0, 1, 2], [e, s, t], color=[ENGINE, SBML, TARGET], width=0.72)
    for x, v in zip([0, 1, 2], [e, s, t]):
        ax.text(x, v, f"{v:g}", ha="center", va="bottom", fontsize=9)
    lo = min(e, s, t); hi = max(e, s, t); span = hi - lo
    ax.set_ylim(lo - max(span * 3, 0.06 * hi), hi + max(span * 3, 0.1 * hi))  # zoom so the tiny gap is visible
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["engine", "SBML", "target"], fontsize=9)
    ax.set_title(D["labels"][o])
    dmax = max(abs(e - s), abs(e - t), abs(s - t))
    ax.text(0.5, 0.04, f"max spread {dmax:.2g}", transform=ax.transAxes, ha="center", fontsize=8.5, color="0.4")
fig.suptitle("Baseline parity at rest (severity 0): BioGears engine vs SBML extraction vs StandardMale target",
             fontsize=13, fontweight="bold", y=1.04)
cap = ("At-rest baseline (no infection): the real BioGears 8.2.0 engine and the SBML extraction settle to the "
       "same StandardMale haemodynamics — SBP 114.9 vs 114.6 mmHg, DBP 74.5 vs 73.5, HR 72.2 vs 71.5 /min, "
       "cardiac output 5.81 vs 5.9 L/min — all within ~1 mmHg / ~0.7 bpm / ~0.1 L/min of each other and of "
       "the StandardMale target (y-axes zoomed to the target band so the small differences are visible). This "
       "is the parity gate: because both sides start from the same state, the divergence measured under "
       "infection in studies 02-03 is attributable to the model, not a mismatched starting point. Values are "
       "the committed severity-0 run results (engine 5 min, SBML 30 s).")
d = STUDY / "charts"; d.mkdir(parents=True, exist_ok=True)
fig.savefig(d / "01_baseline_parity.svg", bbox_inches="tight")
fig.savefig(d / "01_baseline_parity.png", bbox_inches="tight")
json.dump({"title": "01_baseline_parity", "interpretation": cap, "caption": cap},
          open(d / "01_baseline_parity.meta.json", "w"), indent=1)
plt.close(fig); print("wrote", d / "01_baseline_parity.svg")

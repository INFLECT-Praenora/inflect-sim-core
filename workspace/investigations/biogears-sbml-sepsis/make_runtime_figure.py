#!/usr/bin/env python3
"""Model-vs-model 48 h wall-clock: real BioGears engine vs SBML extraction.
Pure read of each arm's committed run meta (out/<arm>.json -> meta.wall_s).

    .venv/bin/python workspace/investigations/biogears-sbml-sepsis/make_runtime_figure.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
STUDIES = HERE / "studies"
ENGINE = "#b3462f"; SBML = "#2f6fb3"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

def wall_h(study, arm):
    m = json.load(open(STUDIES / study / "out" / f"{arm}.json"))["meta"]
    return m["wall_s"] / 3600.0

arms = {
    "Mild (severity 1)":   ("02-mild-infection", "engine-sev1", "sbml-sev1"),
    "Severe (severity 3)": ("03-severe-infection", "engine-sev3", "sbml-sev3"),
}
labels = list(arms); x = np.arange(len(labels)); w = 0.36
eng = [wall_h(arms[l][0], arms[l][1]) for l in labels]
sbm = [wall_h(arms[l][0], arms[l][2]) for l in labels]

fig, ax = plt.subplots(figsize=(8.4, 5.0))
b1 = ax.bar(x - w/2, eng, w, color=ENGINE, label="real BioGears engine (C++)")
b2 = ax.bar(x + w/2, sbm, w, color=SBML, label="SBML extraction (Tellurium / roadrunner)")
for xs, vals in ((x - w/2, eng), (x + w/2, sbm)):
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 0.04, f"{v:.2f} h", ha="center", va="bottom", fontsize=9.5)
# speed ratio annotations
for i, l in enumerate(labels):
    r = eng[i] / sbm[i]
    ax.annotate(f"SBML {r:.2f}× faster", (x[i], max(eng[i], sbm[i]) + 0.32),
                ha="center", fontsize=9.5, color="0.25",
                fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("wall-clock for a 48 h simulation (hours)")
ax.set_ylim(0, max(eng + sbm) + 0.7)
ax.set_title("Model-vs-model runtime: a 48 h sepsis simulation, engine vs SBML extraction")
ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.17))
cap = ("Wall-clock to run one matched 48 h sepsis simulation, real BioGears 8.2.0 engine vs the SBML/Tellurium "
       "extraction (committed run metadata, meta.wall_s). The extraction is modestly faster — 2.15 h vs 3.35 h "
       "(1.56×) at mild infection and 2.81 h vs 3.38 h (1.20×) at severe — but not orders of magnitude, because "
       "integrating the stiff 91-state ODE in roadrunner is itself heavy (see study 06). This is the "
       "model-vs-model comparison relevant to large virtual-patient sweeps; it is a different axis from the "
       "composition-engine swap (process-bigraph vs viva-core), which is bit-identical and overhead-only.")
d = STUDIES / "04-engine-runtime" / "charts"; d.mkdir(parents=True, exist_ok=True)
fig.savefig(d / "02_model_runtime_48h.svg", bbox_inches="tight")
fig.savefig(d / "02_model_runtime_48h.png", bbox_inches="tight")
json.dump({"title": "02_model_runtime_48h", "interpretation": cap, "caption": cap},
          open(d / "02_model_runtime_48h.meta.json", "w"), indent=1)
plt.close(fig); print("wrote", d / "02_model_runtime_48h.svg")
print(f"mild: engine {eng[0]:.2f}h sbml {sbm[0]:.2f}h ({eng[0]/sbm[0]:.2f}x) | severe: engine {eng[1]:.2f}h sbml {sbm[1]:.2f}h ({eng[1]/sbm[1]:.2f}x)")

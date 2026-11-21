#!/usr/bin/env python3
"""Figure for study 06 (ODE-integrator runtime). Pure read of integrator_bench.json.

    .venv/bin/python workspace/investigations/biogears-sbml-sepsis/make_integrator_figure.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
STUDY = HERE / "studies" / "06-ode-integrator-runtime"
D = json.load(open(STUDY / "out" / "integrator_bench.json"))
STIFF = "#2e7d4f"    # cvode (stiff, compiled C) — the right tool
EXPL  = "#b3462f"    # explicit rk4 (= viva-hybrid)
AMBER = "#c77f1a"    # rk45
GREY  = "#9aa0a6"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

def human(n):
    for u, d in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= d: return f"{n/d:.0f} {u}"
    return f"{n:.0f}"

fig = plt.figure(figsize=(16.5, 5.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0], wspace=0.28)

# ---- Panel A: RK4 accuracy vs step size ----
axA = fig.add_subplot(gs[0, 0])
dts = np.array([r["dt_s"] for r in D["rk4"]])
rel = np.array([r["max_relerr"] for r in D["rk4"]])
fin = np.array([r["finite"] for r in D["rk4"]])
CAP = 1e4
relc = np.where(np.isfinite(rel), rel, CAP)
# finite (converged-but-wrong) points
m = fin & np.isfinite(rel)
axA.plot(dts[m], relc[m], "o-", color=EXPL, lw=1.8, ms=7, label="RK4 (fixed-step = viva-hybrid)")
# non-finite (blow-up) points
mb = ~np.isfinite(rel)
axA.scatter(dts[mb], relc[mb], marker="x", s=80, color=EXPL, lw=2.4, zorder=5)
for dt in dts[mb]:
    pass
axA.axhspan(CAP*0.6, CAP*1.6, color=EXPL, alpha=0.06)
axA.text(dts[mb].max(), CAP*1.15, "blows up\n(non-finite)", color=EXPL, fontsize=9, ha="right", va="center")
axA.axhline(1e-2, ls="--", color="0.45", lw=1); axA.text(dts.min(), 1.3e-2, "1% accuracy target", fontsize=8.5, color="0.35")
axA.axhline(1.0, ls=":", color="0.55", lw=1); axA.text(dts.min(), 1.3, "100% error", fontsize=8.5, color="0.45")
axA.axhline(D["cvode"]["max_relerr"], ls="-", color=STIFF, lw=1.6)
axA.text(dts.max(), D["cvode"]["max_relerr"]*0.45, "CVODE (stiff, compiled C): accurate", color=STIFF, fontsize=9, ha="right")
# annotate 48h step counts on a couple of points
for r in D["rk4"]:
    if r["dt_s"] in (0.1, 0.001):
        y = r["max_relerr"] if np.isfinite(r["max_relerr"]) else CAP
        axA.annotate(human(r["steps_48h"]) + " steps / 48 h", (r["dt_s"], y),
                     textcoords="offset points", xytext=(0, -16), fontsize=8.2, color="0.3", ha="center")
axA.set_xscale("log"); axA.set_yscale("log"); axA.invert_xaxis()
axA.set_xlabel("RK4 fixed step size  dt (s)   →  finer"); axA.set_ylabel("max relative error vs stiff reference")
axA.set_title("Explicit RK4 never becomes usable")
axA.legend(loc="lower left", fontsize=8.5)

# ---- Panel B: per-state error at dt=0.001 ----
axB = fig.add_subplot(gs[0, 1])
ps = D["per_state_dt001"]
st = sorted(ps["all_states"], key=lambda x: x["relerr"], reverse=True)
vals = np.array([max(s["relerr"], 1e-6) for s in st])
cols = [EXPL if s["relerr"] > 0.01 else GREY for s in st]
x = np.arange(len(st))
axB.bar(x, vals, color=cols, width=0.9)
axB.set_yscale("log")
axB.axhline(1e-2, ls="--", color="0.45", lw=1); axB.axhline(1.0, ls=":", color="0.55", lw=1)
axB.text(len(st)-1, 1.3e-2, "1%", fontsize=8.5, color="0.35", ha="right")
axB.text(len(st)-1, 1.3, "100%", fontsize=8.5, color="0.45", ha="right")
for i, s in enumerate(st[:5]):
    axB.annotate(s["id"], (i, s["relerr"]), textcoords="offset points", xytext=(2, 3), fontsize=8.3, rotation=35, color=EXPL)
axB.set_xlabel("91 state variables (sorted by error)"); axB.set_ylabel("relative error vs stiff reference")
axB.set_title("At 173 M steps / 48 h, the fast\ncardiovascular states are still wrong")
axB.legend(handles=[Patch(color=EXPL, label=f">1% error: {ps['n_over_1pct']}/91  (cardiovascular)"),
                    Patch(color=GREY, label=f"within 1%: {91-ps['n_over_1pct']}/91  (inflammation, metabolic)")],
           loc="upper right", fontsize=8.5)
axB.text(0.5, 0.02, f"median error over non-trivial states: {ps['median_relerr_nonneg']*100:.1f}%   •   {ps['n_over_100pct']} states off by >100×",
         transform=axB.transAxes, fontsize=8.6, color="0.3", ha="center")

# ---- Panel C: verdict ----
axC = fig.add_subplot(gs[0, 2]); axC.axis("off")
axC.set_title("Reimplementing the integrator\nis not the speed lever", color="0.15")
rk45_txt = "did not finish a 60 s window in 45 s" if not D["rk45"]["completed"] else f"{D['rk45']['wall_s']:.1f} s"
rows = [
    (STIFF, "✓", "CVODE  (stiff BDF, compiled C)", "libroadrunner's default — accurate, seconds.\nAlready native C; a Rust rewrite can't beat it much."),
    (AMBER, "✗", "rk45  (adaptive explicit)", f"stability-limited: {rk45_txt}."),
    (EXPL,  "✗", "rk4  (fixed-step = viva-hybrid)", "unstable; finite only at dt ≤ 0.02 s\n(8.6 M steps / 48 h) and still wrong."),
]
y = 0.86
for col, mark, head, body in rows:
    axC.text(0.02, y, mark, color=col, fontsize=15, fontweight="bold", va="top")
    axC.text(0.11, y, head, color=col, fontsize=10.5, fontweight="bold", va="top")
    axC.text(0.11, y-0.055, body, color="0.25", fontsize=9.2, va="top")
    y -= 0.205
axC.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 0.205, transform=axC.transAxes, color="#f2f4f3", ec="none"))
axC.text(0.03, 0.175,
         "Independent check (viva-hybrid repo): on a NON-stiff benchmark — where explicit\n"
         "integration is valid — a from-scratch Rust engine beat compiled numba by only 2.0×.\n"
         "Real ODE-speed levers: a better / warm-started stiff solver, GPU-parallel patients,\n"
         "or a reduced-order surrogate — not a faster integrator in another language.",
         transform=axC.transAxes, fontsize=8.7, color="0.2", va="top")

fig.suptitle("Can a Rust / compiled integrator speed up the SBML sepsis ODE?  The model is stiff — so no.",
             fontsize=13.5, fontweight="bold", y=1.02)
cap = ("Integrator comparison on the real 91-state BioGears sepsis extraction (60 s window; step counts "
       "extrapolated to a 48 h run). The sepsis ODE is stiff. An explicit fixed-step RK4 — the algorithm "
       "viva-hybrid's Rust engine uses — blows up for dt ≥ 0.05 s and, at steps fine enough to stay finite "
       "(down to dt = 0.001 s = 173 M steps per 48 h run), still gets the fast cardiovascular states wrong by "
       ">100× (vena-cava pressure 406 vs the true 1.6; 30/91 states off by >1%, 5 by >100%) while the slow "
       "inflammation and metabolic states track (median 0.3%). The adaptive explicit method (rk45) is "
       "stability-limited and could not finish the window. The correct solver — a stiff BDF method (CVODE) — "
       "is already compiled C inside libroadrunner and integrates the window in seconds. An independent "
       "trial (viva-hybrid) shows that even on a non-stiff model, where explicit integration is valid, a "
       "from-scratch Rust integrator beats compiled numba by only ~2×. Reimplementing the integrator is "
       "therefore not an order-of-magnitude speed lever for this model; the levers are a better/warm-started "
       "stiff solver, GPU parallelism across virtual patients, or a reduced-order surrogate.")
d = STUDY / "charts"; d.mkdir(parents=True, exist_ok=True)
fig.savefig(d / "01_integrator_stiffness.svg", bbox_inches="tight")
fig.savefig(d / "01_integrator_stiffness.png", bbox_inches="tight")
json.dump({"title": "01_integrator_stiffness", "interpretation": cap, "caption": cap},
          open(d / "01_integrator_stiffness.meta.json", "w"), indent=1)
plt.close(fig); print("wrote", d / "01_integrator_stiffness.svg")

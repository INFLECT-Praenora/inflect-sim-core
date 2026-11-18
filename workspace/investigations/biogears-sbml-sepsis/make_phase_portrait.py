#!/usr/bin/env python3
"""Severe-sepsis CO–SVR phase portrait: engine vs SBML extraction.

Pure read of the committed per-arm severe run data (studies/03-.../out/*.json).
Plots the shock trajectory in cardiac-output x vascular-resistance space, with
iso-MAP hyperbolas (MAP = k*CO*SVR, k=17.34 fit from the data, CV<0.5% in both
arms) so that "same hypotension endpoint, opposite mechanism" reads as geometry:
the two start at the same point and land on the same low-MAP contour by
orthogonal paths — the engine slides LEFT (cardiac-output failure, SVR flat),
the extraction slides DOWN (vasodilation, CO held).

    .venv/bin/python workspace/investigations/biogears-sbml-sepsis/make_phase_portrait.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STUDIES = HERE / "studies"
ENGINE = "#b3462f"   # BioGears engine (brown-red, matches sibling charts)
SBML = "#2f6fb3"     # SBML extraction (steel blue)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "legend.frameon": False,
})

def load(name):
    s = json.load(open(STUDIES / "03-severe-infection" / "out" / name))["samples"]
    t = np.array([x["time"] for x in s]) / 3600.0
    return t, {o: np.array([x[o] for x in s]) for o in ["CO", "SVR", "MAP"]}

te, E = load("engine-sev3.json")
ts, S = load("sbml-sev3.json")

# iso-MAP constant k: MAP = k * CO * SVR. Fit from the pooled data (CV<0.5%).
k = float(np.median(np.concatenate([E["MAP"]/(E["CO"]*E["SVR"]),
                                    S["MAP"]/(S["CO"]*S["SVR"])])))

fig, ax = plt.subplots(figsize=(8.6, 7.2))

# --- iso-MAP hyperbolas: SVR = MAP / (k*CO) ---
co_lo, co_hi = 4.2, 6.35
co_grid = np.linspace(co_lo, co_hi, 200)
for MAP in (65, 75, 85, 95, 105):
    svr_line = MAP / (k * co_grid)
    ax.plot(co_grid, svr_line, color="0.78", lw=1.0, ls="--", zorder=1)
    # label near the right edge, inside the plotted SVR window
    yv = MAP / (k * co_hi)
    if 0.60 <= yv <= 1.18:
        ax.text(co_hi - 0.02, yv, f"MAP {MAP}", color="0.45", fontsize=8.5,
                ha="right", va="bottom", zorder=1)
ax.text(0.015, 0.015, "grey dashes: iso-MAP contours (MAP = 17.34 · CO · SVR)",
        transform=ax.transAxes, color="0.5", fontsize=8.5, style="italic")

# --- trajectories, colored start→end by time via marker ticks ---
def draw(t, d, color, label):
    ax.plot(d["CO"], d["SVR"], color=color, lw=2.4, zorder=3, label=label, solid_capstyle="round")
    # hour ticks every 6 h
    for hr in range(6, 49, 6):
        i = int(np.argmin(np.abs(t - hr)))
        ax.plot(d["CO"][i], d["SVR"][i], "o", color=color, ms=4.5, zorder=4,
                mec="white", mew=0.6)
    # direction arrow at ~2/3 along the path
    i2 = int(len(t) * 0.66); i1 = i2 - 30
    ax.annotate("", xy=(d["CO"][i2], d["SVR"][i2]), xytext=(d["CO"][i1], d["SVR"][i1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2), zorder=5)

draw(te, E, ENGINE, "BioGears engine (C++)")
draw(ts, S, SBML, "SBML extraction (Tellurium)")

# shared start (both ~5.8-5.9 / ~0.95) and the two endpoints
ax.plot(E["CO"][0], E["SVR"][0], "s", color="0.2", ms=11, zorder=6, mec="white", mew=1.2)
ax.annotate("t = 0  (shared healthy\nbaseline, MAP ≈ 96)",
            (E["CO"][0], E["SVR"][0]), textcoords="offset points", xytext=(-6, 14),
            ha="right", fontsize=9.5, color="0.2", fontweight="bold")
for t, d, color, txt, off in [
    (te, E, ENGINE, "engine 48 h\nCO 4.5 · SVR 0.93\nMAP 72 — LOW-OUTPUT (≥65)", (10, -26)),
    (ts, S, SBML, "extraction 48 h\nCO 6.0 · SVR 0.67\nMAP 70 — VASODILATORY (≥65)", (-8, -4)),
]:
    ax.plot(d["CO"][-1], d["SVR"][-1], "*", color=color, ms=18, zorder=6, mec="white", mew=1.0)
    ha = "left" if off[0] > 0 else "right"
    ax.annotate(txt, (d["CO"][-1], d["SVR"][-1]), textcoords="offset points",
                xytext=off, ha=ha, fontsize=9, color=color, fontweight="bold")

ax.set_xlabel("Cardiac output  (L/min)")
ax.set_ylabel("Systemic vascular resistance  (SVR, mmHg·s/mL)")
ax.set_xlim(co_lo, co_hi)
ax.set_ylim(0.60, 1.18)
ax.set_title("Severe sepsis as a trajectory in CO–SVR space:\nsame hypotension endpoint, opposite mechanism",
             fontsize=13.5, loc="left")
ax.legend(loc="upper left", fontsize=10.5, bbox_to_anchor=(0.012, 0.995))
ax.margins(0)
fig.tight_layout()

cap = ("Severe sepsis (severity 3) plotted as a path in cardiac-output x vascular-resistance "
       "space rather than as two separate time series. Dots mark 6-hour intervals; the square is "
       "the shared healthy baseline (t=0), the stars are the 48 h endpoints, and grey dashes are "
       "iso-MAP contours (MAP = 17.34 x CO x SVR — a relation that holds to <0.5% in BOTH arms with "
       "the same constant). Both arms begin at the same point and finish on nearly the same low-MAP "
       "contour (~72 vs ~70 mmHg — both still above the MAP<65 shock threshold and still falling at "
       "48 h), but by orthogonal routes: the real BioGears engine slides LEFT — "
       "cardiac output fails 5.8->4.5 L/min while SVR stays flat (cold, low-output decline) — whereas "
       "the SBML extraction slides DOWN — it holds/rebuilds output and lets SVR swing up then fall "
       "to 0.67 (warm, vasodilatory decline). The orthogonality is the finding: the extraction's single "
       "AIR->sympathetic arc overshoots the vasomotor axis that the engine's full cardiovascular "
       "control keeps muted, so the two approach the same low MAP by physiologically different routes. "
       "This is the geometric view of studies 03/05.")

d = STUDIES / "03-severe-infection" / "charts"; d.mkdir(parents=True, exist_ok=True)
base = "05_severe_phase_portrait_co_svr"
fig.savefig(d / f"{base}.svg", bbox_inches="tight")
fig.savefig(d / f"{base}.png", bbox_inches="tight")
json.dump({"title": base, "interpretation": cap, "caption": cap},
          open(d / f"{base}.meta.json", "w"), indent=1)
plt.close(fig)
print("wrote", d / f"{base}.svg", f"(k={k:.4f})")

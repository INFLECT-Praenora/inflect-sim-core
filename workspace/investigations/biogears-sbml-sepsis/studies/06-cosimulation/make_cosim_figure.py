#!/usr/bin/env python3
"""Co-simulation figure (study 06): the closed loop in one picture.

Reads out/cosim_arms.json (written by run_cosim.py) and renders charts/
01_cosim_closed_loop.svg|png (+ .meta.json caption):

  * top    — blood pathogen (bacteremia): untreated climbs, closed-loop treated is
             cleared. This is the controlled variable the two processes act on.
  * bottom — the controller's live state: antibiotic concentration (its own PK,
             integrated by the second process) and the treat-on flag it sets by
             sensing the pathogen.

Run from the investigation dir (after run_cosim.py):
    PYTHONPATH=<worktree> python studies/06-cosimulation/make_cosim_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
UNTREATED = "#b3462f"   # untreated (brown-red, matches the engine color elsewhere)
TREATED = "#2e8b57"     # closed-loop treated (sea green)
DRUG = "#2f6fb3"        # antibiotic concentration (steel blue)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


def main() -> None:
    data = json.loads((HERE / "out" / "cosim_arms.json").read_text())
    u, t = data["untreated"], data["treated"]
    meta = data["meta"]
    tu, tt = u["time_h"], t["time_h"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    # Top: bacteremia — the coupled variable.
    ax1.plot(tu, u["blood_pathogen"], color=UNTREATED, lw=2.2, label="untreated (controller inert)")
    ax1.plot(tt, t["blood_pathogen"], color=TREATED, lw=2.2, label="closed-loop antibiotic")
    ax1.set_ylabel("Blood pathogen (bacteremia)")
    ax1.set_title("Closed-loop co-simulation: bacteremia cleared tick-by-tick")
    ax1.legend(loc="upper left")
    pb_u = u["blood_pathogen"][-1] or 0.0
    pb_t = t["blood_pathogen"][-1] or 0.0
    ratio = (pb_u / pb_t) if pb_t > 1e-9 else float("inf")
    rtxt = "∞" if ratio == float("inf") else f"{ratio:.0f}×"
    ax1.annotate(f"final bacteremia\nuntreated / treated = {rtxt}",
                 xy=(tu[-1], pb_u), xytext=(0.62, 0.55), textcoords="axes fraction",
                 fontsize=9.5, color=UNTREATED,
                 arrowprops=dict(arrowstyle="->", color=UNTREATED, lw=1.2))

    # Bottom: the controller's own simulated state (the second engine is live).
    ax2.plot(tt, t["drug"], color=DRUG, lw=2.0, label="antibiotic concentration (controller PK)")
    ax2.fill_between(tt, 0, [m * max(t["drug"] + [1e-9]) for m in t["treating"]],
                     color=TREATED, alpha=0.12, step="pre", label="treatment on (sensed)")
    ax2.set_ylabel("Drug conc.")
    ax2.set_xlabel("Time (h)")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    out = HERE / "charts" / "01_cosim_closed_loop"
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".png"))
    caption = (
        f"Closed-loop sepsis co-simulation at severity {meta['severity']} over "
        f"{meta['horizon_h']:g} h ({meta['interval_s']:g} s/tick). ONE process-bigraph "
        "composite couples the Tellurium sepsis physiology engine to an antibiotic-PK "
        "controller process; they exchange state every tick (controller reads the "
        "pathogen, integrates a one-compartment PK, writes the drug-killed pathogen "
        "back via the engine's parameters_in port). Untreated (controller inert) is the "
        "matched control. The loop clears bacteremia — the driver of septic collapse — "
        "while the untreated arm's pathogen climbs. The downstream clinical endpoint "
        "(MAP < 65 mmHg at ~54 h, tissue integrity 0.24 at 72 h; studies 03/05) is a "
        "2–3 day process not reached at this horizon; this figure shows the controlled "
        "variable and the controller's live PK, i.e. the loop removing the cause. The "
        "treated arm's sawtooth is the per-tick coupling made visible: the pathogen "
        "regrows within each physiology step and the controller knocks it back down "
        "at every tick. (At this horizon the untreated arm's tissue integrity has only "
        "just begun to fall, ~0.983 vs 1.000 treated — the early edge of the collapse.)"
    )
    (out.with_name(out.name + ".meta.json")).write_text(json.dumps({"caption": caption}, indent=1))
    print("wrote", out.with_suffix(".svg"), "| untreated/treated bacteremia =", rtxt)


if __name__ == "__main__":
    main()

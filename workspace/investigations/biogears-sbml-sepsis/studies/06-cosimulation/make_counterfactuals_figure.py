#!/usr/bin/env python3
"""Study 06 figure: treatment counterfactuals + coupling-step convergence.

Reads out/cosim_counterfactuals.json and out/cosim_convergence.json and renders
charts/02_cosim_counterfactuals.svg|png (+ .meta.json):

  * left/top    — blood pathogen for untreated / early / delayed treatment.
  * left/bottom — tissue integrity for the same three arms.
  * right       — coupling-step convergence: final bacteremia vs step size.

Run after run_counterfactuals.py, from the investigation dir.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
C = {"untreated": "#b3462f", "early": "#2e8b57", "delayed": "#d4952a"}
LBL = {"untreated": "untreated", "early": "early treatment", "delayed": "delayed treatment"}

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
                     "axes.titlesize": 11.5, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def main() -> None:
    cf = json.loads((HERE / "out" / "cosim_counterfactuals.json").read_text())
    conv = json.loads((HERE / "out" / "cosim_convergence.json").read_text())
    meta = cf["meta"]

    fig = plt.figure(figsize=(10.4, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1.25], hspace=0.38, wspace=0.30)
    ax_pb = fig.add_subplot(gs[0, 0])
    ax_ti = fig.add_subplot(gs[1, 0], sharex=ax_pb)
    ax_cv = fig.add_subplot(gs[:, 1])

    for arm in ("untreated", "early", "delayed"):
        a = cf[arm]
        ax_pb.plot(a["time_h"], a["blood_pathogen"], color=C[arm], lw=2.0, label=LBL[arm])
        ax_ti.plot(a["time_h"], a["tissue_integrity"], color=C[arm], lw=2.0, label=LBL[arm])
    ax_pb.set_ylabel("Blood pathogen")
    ax_pb.set_title(f"Treatment counterfactuals (severity {meta['severity']}, "
                    f"delay {meta['delay_h']:g} h)")
    ax_pb.legend(loc="upper left", fontsize=9)
    ax_ti.set_ylabel("Tissue integrity")
    ax_ti.set_xlabel("Time (h)")

    # Convergence: final bacteremia vs coupling step size.
    runs = sorted(conv["runs"], key=lambda r: r["interval_s"])
    ivs = [r["interval_s"] for r in runs]
    pbs = [r["blood_pathogen_end"] for r in runs]
    ax_cv.plot(ivs, pbs, "o-", color="#2f6fb3", lw=2.0)
    ax_cv.set_xlabel("Coupling step (s/tick)")
    ax_cv.set_ylabel("Final blood pathogen (early arm, 1 h)")
    ax_cv.set_title("Coupling-step convergence")
    spread = max(pbs) - min(pbs)
    ax_cv.annotate(f"spread {spread:.3g}\nover {min(ivs):.0f}–{max(ivs):.0f} s",
                   xy=(0.5, 0.1), xycoords="axes fraction", fontsize=9, color="#555")

    fig.tight_layout()
    stem = HERE / "charts" / "02_cosim_counterfactuals"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"))

    pb = {k: cf[k]["blood_pathogen"][-1] for k in ("untreated", "early", "delayed")}
    ti = {k: cf[k]["tissue_integrity"][-1] for k in ("untreated", "early", "delayed")}
    caption = (
        f"Treatment counterfactuals from the identical sepsis_cosim composite (severity "
        f"{meta['severity']}, {meta['horizon_h']:g} h, delay {meta['delay_h']:g} h). Untreated, "
        "early, and delayed treatment differ only in when the controller starts. Early "
        f"treatment clears bacteremia (final {pb['early']:.2g}); delayed lets it accumulate "
        f"first ({pb['delayed']:.2g}) then clears it; untreated climbs ({pb['untreated']:.2g}). "
        f"Tissue integrity ends {ti['early']:.3f} early vs {ti['untreated']:.3f} untreated. "
        "The convergence panel runs the early arm at 240/120/60 s per tick: as the coupling "
        f"step shrinks the final bacteremia changes by {spread:.3g}, so the co-simulation is "
        "numerically consistent. This separates execution success (the coupling runs and "
        "converges) from pharmacological credibility (the PK is illustrative, not calibrated).")
    stem.with_name(stem.name + ".meta.json").write_text(json.dumps({"caption": caption}, indent=1))
    print("wrote", stem.with_suffix(".svg"))
    print("final PB:", {k: round(v, 3) for k, v in pb.items()}, "| TI:", {k: round(v, 4) for k, v in ti.items()})
    print("convergence PB_end:", [round(p, 4) for p in pbs], "spread", round(spread, 4))


if __name__ == "__main__":
    main()

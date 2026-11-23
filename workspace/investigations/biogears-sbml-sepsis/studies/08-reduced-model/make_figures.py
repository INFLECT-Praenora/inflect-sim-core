#!/usr/bin/env python3
"""Study 08 figures: fidelity overlay + speed comparison. Reads out/comparison.json."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FULL = "#b3462f"   # event-driven full model
EF = "#2e8b57"     # event-free reduced model
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 10.5,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white"})


def fidelity_fig(d):
    sev = d["arms"]["severe"]; t = sev["t_h"]
    panels = [("MAP", "MAP (mmHg)"), ("SVR", "SVR"),
              ("PB", "Blood pathogen"), ("TI", "Tissue integrity")]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0))
    for ax, (k, lab) in zip(axes.ravel(), panels):
        ax.plot(t, sev[f"ref_{k}"], color=FULL, lw=2.2, label="full model (event-driven)")
        ax.plot(t, sev[k], color=EF, lw=2.2, ls="--", label="event-free (reduced)")
        ax.set_ylabel(lab); ax.set_xlabel("Time (h)")
        f = sev["fidelity"][k]; ax.set_title(f"{lab.split(' (')[0]} — RMSE {f['pct_of_max']:.1f}%")
    axes[0, 0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Event-free reduced model vs full pulsatile model (severe sepsis, 48 h)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    stem = HERE / "charts" / "01_fidelity_overlay"
    fig.savefig(stem.with_suffix(".svg")); fig.savefig(stem.with_suffix(".png"))
    cap = ("Event-free (mean-hemodynamics) reduced model vs the exact pulsatile BioGears "
           "extraction, severe arm, 48 h. The reduced model drops the per-beat event and "
           "tracks blood pathogen and tissue integrity to within ~10% and the mean-MAP "
           "decline to ~4% RMSE. It under-reaches the deepest hemodynamic point (SVR ~0.78 "
           "vs 0.665; MAP ~80 vs 69.5 mmHg) — it captures the direction and most of the "
           "shock, trading hemodynamic depth and pulsatility (no SBP/DBP) for ~10^7x speed.")
    stem.with_name(stem.name + ".meta.json").write_text(json.dumps({"caption": cap}, indent=1))


def speed_fig(d):
    s = d["speed"]
    labels = ["full\n(roadrunner)", "full\n(COPASI)", "event-free\n(roadrunner)", "event-free\n(COPASI)"]
    vals = [s["full_roadrunner_s"], s["full_copasi_s"], s["eventfree_roadrunner_s"], s["eventfree_copasi_s"]]
    colors = [FULL, "#d4952a", EF, "#2f6fb3"]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors)
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis(); ax.set_xlabel("48 h run wall-clock (s, log scale)")
    ax.set_title("Removing the beat event collapses the cost — and the engine gap",
                 fontsize=11.5, fontweight="bold", loc="left")
    for yi, v in zip(y, vals):
        txt = f"{v:.0e} s" if v < 1 else (f"{v/3600:.1f} h" if v > 3600 else f"{v:.0f} s")
        ax.text(v * 1.3, yi, txt, va="center", fontsize=9)
    ax.set_xlim(1e-5, 1e5)
    fig.tight_layout()
    stem = HERE / "charts" / "02_speed_comparison"
    fig.savefig(stem.with_suffix(".svg")); fig.savefig(stem.with_suffix(".png"))
    cap = (f"48 h wall-clock, log scale. The full pulsatile model (~205k beat events) takes "
           f"~2.81 h under roadrunner and ~3.7 h under COPASI. The event-free model runs in "
           f"~{s['eventfree_roadrunner_s']*1e3:.2g} ms (roadrunner) and ~{s['eventfree_copasi_s']*1e3:.2g} ms "
           f"(COPASI) — ~{s['speedup_roadrunner']:.0g}x / ~{s['speedup_copasi']:.0g}x faster. With the "
           "events gone, COPASI is as fast as roadrunner (no per-event interpreter overhead), so the "
           "~3x engine gap vanishes too.")
    stem.with_name(stem.name + ".meta.json").write_text(json.dumps({"caption": cap}, indent=1))


def main():
    d = json.loads((HERE / "out" / "comparison.json").read_text())
    fidelity_fig(d); speed_fig(d)
    print("wrote charts/01_fidelity_overlay.svg and charts/02_speed_comparison.svg")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Study figure: the contract — results out (left) and verification out (right).
Reads out/contract_runs.json."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COL = {True: "#2e8b57", False: "#c0392b"}
OBSC = {"MAP": "#2f6fb3", "SVR": "#d4952a", "PB": "#b3462f", "TI": "#2e8b57"}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 10.5,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white"})


def main():
    d = json.loads((HERE / "out" / "contract_runs.json").read_text())
    runs = d["runs"]
    fig = plt.figure(figsize=(11.2, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.3], wspace=0.22)

    # LEFT: results out — one run's trajectory
    axr = fig.add_subplot(gs[0, 0])
    tr = next((r["trajectory"] for r in runs if r.get("trajectory")), None)
    if tr:
        t = tr["time_h"]
        for k in ("MAP", "SVR", "PB", "TI"):
            if k in tr:
                axr.plot(t, tr[k], color=OBSC.get(k, "#555"), lw=2.0, label=k)
        axr.set_xlabel("Time (h)"); axr.set_ylabel("observable value")
        axr.set_title("Results out — observables over time")
        axr.legend(loc="best", fontsize=9)

    # RIGHT: verification out — status panel per run
    axv = fig.add_subplot(gs[0, 1]); axv.axis("off")
    axv.set_title("Verification out — every run self-verified", loc="left")
    n = len(runs)
    for i, r in enumerate(runs):
        y = n - 1 - i
        v = r["verification"]; ok = v["passed"]; col = COL[ok]
        axv.add_patch(plt.Rectangle((0, y + 0.1), 2.0, 0.8, color=col, alpha=0.12, lw=0))
        axv.add_patch(plt.Rectangle((0, y + 0.1), 0.05, 0.8, color=col, lw=0))
        axv.text(0.09, y + 0.60, r["label"], fontsize=9.5, fontweight="bold")
        adm = v.get("admissible") or {}
        nadm = sum(1 for x in adm.values() if x)
        bits = [f"engine={r['engine']}", f"admissible {nadm}/{len(adm)}"]
        if v.get("cross_engine"):
            bits.append(f"cross-engine {v['cross_engine']['engine']} Δ={v['cross_engine']['max_rel_diff']:.1e}")
        axv.text(0.09, y + 0.26, "  ·  ".join(bits), fontsize=8.4, color="#444")
        axv.text(1.98, y + 0.5, "PASSED" if ok else "FAILED", fontsize=11, fontweight="bold",
                 color=col, ha="right")
    axv.set_xlim(0, 2.0); axv.set_ylim(0, n)

    fig.suptitle("The simulation-core contract: simulate(model, request) → results + verification",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem = HERE / "charts" / "01_contract"
    fig.savefig(stem.with_suffix(".svg")); fig.savefig(stem.with_suffix(".png"))
    npass = sum(1 for r in runs if r["verification"]["passed"])
    cap = ("The simulation-core contract, demonstrated. One interface — "
           "simulate(model, request) — takes a model (an SBML file, the same model with a "
           "cross-engine check, a different engine, or a registered composite) and returns "
           "results (left) plus a structured verification (right): ran, finite, physical "
           "admissibility per observable, and optional cross-engine agreement. "
           f"{npass}/{len(runs)} runs verified; the two exact-SBML engines (Tellurium, COPASI) "
           "agree to ~1e-4. Every run carries its own evidence.")
    stem.with_name(stem.name + ".meta.json").write_text(json.dumps({"caption": cap}, indent=1))
    print("wrote 01_contract.svg |", npass, "/", len(runs), "verified")


if __name__ == "__main__":
    main()

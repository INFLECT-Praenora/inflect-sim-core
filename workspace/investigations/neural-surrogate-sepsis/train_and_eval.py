#!/usr/bin/env python3
"""Train the sepsis SBML surrogate on the sampled dataset, then evaluate it on
HELD-OUT infection timings: rollout NRMSE, a real-vs-surrogate demonstration
figure, and a wall-clock speed comparison.

The honest test is the rollout, not one-step error: from a held-out initial
condition the surrogate is stepped autonomously for the full horizon and
compared to the real SBML/roadrunner trajectory it never saw at that timing.

Run (needs the [surrogate] extra):
    .venv/bin/python workspace/investigations/neural-surrogate-sepsis/train_and_eval.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viva_torch import TransitionDataset, train_surrogate
from viva_torch.evaluate import evaluate_surrogate, _rollout_target, _rollout_surrogate
from viva_simulation_core.composites.sepsis_sbml import SEPSIS_OBSERVABLES
from viva_simulation_core.composites.sepsis_sbml_state import build_sepsis_state_composite

HERE = pathlib.Path(__file__).resolve().parent
STUDY = HERE / "studies" / "01-sample-and-train"
OUT = STUDY / "out"
CHARTS = STUDY / "charts"
ENGINE = "#b3462f"  # real SBML engine (roadrunner)
SUR = "#2f6fb3"     # neural surrogate

# clinical observables to show in the demonstration figure
FIG_OBS = ["SBP", "MAP", "HR", "SVR", "tissue_integrity", "blood_pathogen"]
LABELS = {"SBP": "Systolic BP (mmHg)", "MAP": "MAP (mmHg)", "HR": "Heart rate (/min)",
          "SVR": "SVR (mmHg·s/mL)", "tissue_integrity": "Tissue integrity",
          "blood_pathogen": "Blood pathogen"}


def _core(sev, jit, interval):
    return build_sepsis_state_composite(int(sev), float(jit), interval)[1]


def col_for(spec, sbml_var):
    """Index of the target column whose path ends in `sbml_var`, or None."""
    for i, p in enumerate(spec.target_paths_t):
        if p[-1] == sbml_var:
            return i
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=lambda s: tuple(int(x) for x in s.split(",")),
                    default=(128, 128))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    meta = json.loads((OUT / "dataset_meta.json").read_text())
    ds = TransitionDataset.load(str(OUT / "dataset.npz"))
    spec = ds.spec
    interval = float(meta["timestep_s"])
    n_steps = int(meta["n_steps_per_trajectory"])
    print(f"dataset: {ds.n_transitions} transitions, {spec.n_features} features, "
          f"{spec.n_targets} targets, dt={interval}s, {n_steps} steps/traj "
          f"({n_steps * interval / 3600:.1f} h)")
    print(f"train ICs: {meta['train_ics']}")
    print(f"eval  ICs (held out): {meta['eval_ics']}")

    # ----------------------------------------------------------------- train
    t0 = time.perf_counter()
    net, hist = train_surrogate(ds, hidden=a.hidden, epochs=a.epochs, lr=a.lr, seed=a.seed)
    train_s = time.perf_counter() - t0
    ckpt = str(OUT / "surrogate.pt")
    net.save(ckpt)
    val = hist.get("val_loss", [float("nan")])[-1]
    print(f"trained {a.hidden} for {a.epochs} epochs in {train_s:.1f}s; final val loss {val:.4g}")

    # ------------------------------------------ held-out rollout NRMSE per IC
    eval_ics = [tuple(c) for c in meta["eval_ics"]]  # (severity, infection_start_s)
    per_ic = {}
    for sev, jit in eval_ics:
        doc, _ = build_sepsis_state_composite(int(sev), float(jit), interval)
        m = evaluate_surrogate({"state": doc}, spec, ckpt,
                               lambda s=sev, j=jit: _core(s, j, interval),
                               [{}], n_steps, interval)
        key = f"sev{int(sev)}_start{jit / 3600:.0f}h"
        per_ic[key] = m
        print(f"  held-out sev={int(sev)} start={jit / 3600:.1f}h : "
              f"rollout NRMSE {m['overall_nrmse']:.4f}")
    mean_nrmse = float(np.mean([m["overall_nrmse"] for m in per_ic.values()]))
    print(f"mean held-out rollout NRMSE: {mean_nrmse:.4f}")

    # ------------------------------- demonstration figure + speed (most severe IC)
    sev, jit = max(eval_ics, key=lambda c: c[0])
    hrs = np.arange(1, n_steps + 1) * interval / 3600.0

    tr = time.perf_counter()
    tgt = _rollout_target({"state": build_sepsis_state_composite(int(sev), float(jit), interval)[0]},
                          spec, _core(sev, jit, interval), {}, n_steps, interval)
    t_real = time.perf_counter() - tr

    ts = time.perf_counter()
    sur = _rollout_surrogate(spec, ckpt, _core(sev, jit, interval), {}, n_steps, interval)
    t_sur = time.perf_counter() - ts
    speedup = t_real / t_sur if t_sur else float("nan")
    print(f"rollout wall-clock ({n_steps} steps, sev={int(sev)} start={jit/3600:.0f}h): "
          f"real SBML {t_real*1e3:.0f} ms vs surrogate {t_sur*1e3:.0f} ms -> {speedup:.1f}x")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, obs in zip(axes.flat, FIG_OBS):
        c = col_for(spec, SEPSIS_OBSERVABLES[obs])
        if c is None:
            ax.set_visible(False)
            continue
        ax.plot(hrs, tgt[:, c], color=ENGINE, lw=1.8, label="real SBML (roadrunner)")
        ax.plot(hrs, sur[:, c], color=SUR, lw=1.8, ls="--", label="neural surrogate")
        ax.set_title(LABELS[obs]); ax.set_xlabel("Hours after infection"); ax.margins(x=0)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle(f"Surrogate rollout vs real SBML on a HELD-OUT infection timing "
                 f"(severity {int(sev)}, start {jit/3600:.0f} h) — mean held-out NRMSE "
                 f"{mean_nrmse:.3f}, {speedup:.0f}x faster",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    CHARTS.mkdir(parents=True, exist_ok=True)
    base = CHARTS / "01_surrogate_vs_real_heldout"
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.png", bbox_inches="tight")
    cap = (f"Neural surrogate (dashed) rolled out autonomously for {n_steps*interval/3600:.0f} h "
           f"from a held-out infection timing (severity {int(sev)}, start {jit/3600:.0f} h — not in "
           f"the training set) vs the real SBML/roadrunner trajectory (solid). Mean rollout NRMSE "
           f"across all held-out timings {mean_nrmse:.3f}; the surrogate runs {speedup:.0f}x faster.")
    json.dump({"title": base.name, "interpretation": cap, "caption": cap},
              open(f"{base}.meta.json", "w"), indent=1)
    print("wrote", base)

    # ---------------------------------------------------------------- metrics
    metrics = {
        "dataset": {"n_transitions": int(ds.n_transitions), "n_features": spec.n_features,
                    "n_targets": spec.n_targets, "timestep_s": interval,
                    "horizon_h": n_steps * interval / 3600.0,
                    "train_ics": meta["train_ics"], "eval_ics": meta["eval_ics"]},
        "training": {"hidden": list(a.hidden), "epochs": a.epochs, "lr": a.lr,
                     "train_seconds": round(train_s, 1), "final_val_loss": float(val)},
        "heldout_rollout_nrmse": {k: round(v["overall_nrmse"], 4) for k, v in per_ic.items()},
        "mean_heldout_nrmse": round(mean_nrmse, 4),
        "speed": {"real_sbml_ms": round(t_real * 1e3, 1), "surrogate_ms": round(t_sur * 1e3, 1),
                  "speedup": round(speedup, 1), "n_steps": n_steps,
                  "benchmark_ic": [int(sev), jit / 3600.0]},
    }
    (OUT / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    print("wrote", OUT / "eval_metrics.json")
    return metrics


if __name__ == "__main__":
    main()

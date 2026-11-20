#!/usr/bin/env python3
"""Parallel, checkpoint-reusing evaluation of the trained sepsis surrogate.

`train_and_eval.py` trains AND evaluates in one serial pass; its held-out
rollouts call the real stiff SBML (roadrunner ~10 s per 300 s macro-step), so a
full 48 h / 576-step eval over the 3 held-out infection timings runs ~7 h
serially. The real-SBML rollouts are independent, so this driver runs one
subprocess per held-out IC (bounded by a timeout, mirroring the sampling
driver's per-worker robustness) against the EXISTING checkpoint — no retraining,
identical numbers, ~1.6 h wall-clock on a multicore box.

  # driver (spawns one worker per held-out IC, then reduces):
  .venv/bin/python workspace/investigations/neural-surrogate-sepsis/eval_parallel.py
  # single worker (used internally):
  .venv/bin/python .../eval_parallel.py --worker <severity> <infection_start_s> <key>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viva_torch import TransitionDataset
from viva_torch.evaluate import _rollout_target, _rollout_surrogate
from viva_simulation_core.composites.sepsis_sbml import SEPSIS_OBSERVABLES
from viva_simulation_core.composites.sepsis_sbml_state import build_sepsis_state_composite

HERE = pathlib.Path(__file__).resolve().parent
STUDY = HERE / "studies" / "01-sample-and-train"
OUT = STUDY / "out"
CHARTS = STUDY / "charts"
ENGINE = "#b3462f"  # real SBML engine (roadrunner)
SUR = "#2f6fb3"     # neural surrogate

FIG_OBS = ["SBP", "MAP", "HR", "SVR", "tissue_integrity", "blood_pathogen"]
LABELS = {"SBP": "Systolic BP (mmHg)", "MAP": "MAP (mmHg)", "HR": "Heart rate (/min)",
          "SVR": "SVR (mmHg·s/mL)", "tissue_integrity": "Tissue integrity",
          "blood_pathogen": "Blood pathogen"}

PER_IC_TIMEOUT_S = 3 * 3600  # hard safety net per held-out rollout


def _meta():
    return json.loads((OUT / "dataset_meta.json").read_text())


def _spec():
    return TransitionDataset.load(str(OUT / "dataset.npz")).spec


def key_for(sev, jit):
    return f"sev{int(sev)}_start{jit / 3600:.0f}h"


def col_for(spec, sbml_var):
    for i, p in enumerate(spec.target_paths_t):
        if p[-1] == sbml_var:
            return i
    return None


# --------------------------------------------------------------------- worker
def run_worker(sev, jit, key):
    meta = _meta()
    spec = _spec()
    interval = float(meta["timestep_s"])
    n_steps = int(meta["n_steps_per_trajectory"])
    ckpt = str(OUT / "surrogate.pt")

    doc = build_sepsis_state_composite(int(sev), float(jit), interval)[0]
    t0 = time.perf_counter()
    tgt = _rollout_target({"state": doc}, spec,
                          build_sepsis_state_composite(int(sev), float(jit), interval)[1],
                          {}, n_steps, interval)
    t_real = time.perf_counter() - t0

    t1 = time.perf_counter()
    sur = _rollout_surrogate(spec, ckpt,
                             build_sepsis_state_composite(int(sev), float(jit), interval)[1],
                             {}, n_steps, interval)
    t_sur = time.perf_counter() - t1

    np.save(OUT / f"eval_tgt_{key}.npy", tgt)
    np.save(OUT / f"eval_sur_{key}.npy", sur)
    (OUT / f"eval_timing_{key}.json").write_text(json.dumps(
        {"key": key, "severity": int(sev), "infection_start_s": float(jit),
         "real_sbml_s": t_real, "surrogate_s": t_sur,
         "n_steps": n_steps, "timestep_s": interval}))
    print(f"[worker {key}] real SBML {t_real:.0f}s, surrogate {t_sur:.1f}s -> saved", flush=True)


# ------------------------------------------------------------------- reducer
def _nrmse(tgt, sur):
    """Per-target NRMSE pooled over the rollout, matching evaluate_surrogate;
    computed over finite (target, surrogate) pairs. Returns (overall, diverged_frac)."""
    nrmses = []
    nonfinite = 0
    total = 0
    for i in range(tgt.shape[1]):
        t = tgt[:, i]
        s = sur[:, i]
        m = np.isfinite(t) & np.isfinite(s)
        nonfinite += int((~np.isfinite(s)).sum())
        total += s.size
        if m.sum() < 2:
            continue
        rmse = float(np.sqrt(np.mean((t[m] - s[m]) ** 2)))
        scale = float(np.std(t[m])) or 1.0
        nrmses.append(rmse / scale)
    overall = float(np.mean(nrmses)) if nrmses else float("nan")
    return overall, (nonfinite / total if total else 0.0)


def reduce_and_figure():
    meta = _meta()
    spec = _spec()
    interval = float(meta["timestep_s"])
    n_steps = int(meta["n_steps_per_trajectory"])
    eval_ics = [tuple(c) for c in meta["eval_ics"]]

    per_ic = {}
    diverged = {}
    timing = {}
    for sev, jit in eval_ics:
        key = key_for(sev, jit)
        tgt = np.load(OUT / f"eval_tgt_{key}.npy")
        sur = np.load(OUT / f"eval_sur_{key}.npy")
        overall, divf = _nrmse(tgt, sur)
        per_ic[key] = overall
        diverged[key] = divf
        timing[key] = json.loads((OUT / f"eval_timing_{key}.json").read_text())
        print(f"  held-out sev={int(sev)} start={jit/3600:.1f}h : "
              f"rollout NRMSE {overall:.4f}  (surrogate non-finite frac {divf:.3f})")
    finite_nrmse = [v for v in per_ic.values() if np.isfinite(v)]
    mean_nrmse = float(np.mean(finite_nrmse)) if finite_nrmse else float("nan")
    print(f"mean held-out rollout NRMSE: {mean_nrmse:.4f}")

    # demonstration figure + speed on the most-severe held-out IC
    sev, jit = max(eval_ics, key=lambda c: c[0])
    key = key_for(sev, jit)
    tgt = np.load(OUT / f"eval_tgt_{key}.npy")
    sur = np.load(OUT / f"eval_sur_{key}.npy")
    hrs = np.arange(1, n_steps + 1) * interval / 3600.0
    t_real = timing[key]["real_sbml_s"]
    t_sur = timing[key]["surrogate_s"]
    speedup = t_real / t_sur if t_sur else float("nan")
    print(f"rollout wall-clock (sev={int(sev)} start={jit/3600:.0f}h): real SBML "
          f"{t_real:.0f}s vs surrogate {t_sur:.2f}s -> {speedup:.0f}x")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, obs in zip(axes.flat, FIG_OBS):
        c = col_for(spec, SEPSIS_OBSERVABLES[obs])
        if c is None:
            ax.set_visible(False)
            continue
        st = sur[:, c]
        finite = np.isfinite(st)
        ax.plot(hrs, tgt[:, c], color=ENGINE, lw=1.8, label="real SBML (roadrunner)")
        ax.plot(hrs[finite], st[finite], color=SUR, lw=1.8, ls="--", label="neural surrogate")
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

    metrics = {
        "dataset": {"n_transitions": int(meta["n_transitions"]), "n_features": spec.n_features,
                    "n_targets": spec.n_targets, "timestep_s": interval,
                    "horizon_h": n_steps * interval / 3600.0,
                    "train_ics": meta["train_ics"], "eval_ics": meta["eval_ics"]},
        "heldout_rollout_nrmse": {k: (round(v, 4) if np.isfinite(v) else None)
                                  for k, v in per_ic.items()},
        "surrogate_nonfinite_frac": {k: round(v, 4) for k, v in diverged.items()},
        "mean_heldout_nrmse": (round(mean_nrmse, 4) if np.isfinite(mean_nrmse) else None),
        "speed": {"real_sbml_s": round(t_real, 1), "surrogate_s": round(t_sur, 2),
                  "speedup": round(speedup, 1), "n_steps": n_steps,
                  "benchmark_ic": [int(sev), jit / 3600.0]},
    }
    (OUT / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    print("wrote", OUT / "eval_metrics.json")
    return metrics


# --------------------------------------------------------------------- driver
def run_driver():
    meta = _meta()
    eval_ics = [tuple(c) for c in meta["eval_ics"]]
    if not (OUT / "surrogate.pt").exists():
        sys.exit("surrogate.pt missing — train first (train_and_eval.py)")
    print(f"eval ICs (held out): {eval_ics}; checkpoint {OUT/'surrogate.pt'}")

    procs = {}
    for sev, jit in eval_ics:
        key = key_for(sev, jit)
        log = open(OUT / f"eval_worker_{key}.log", "w")
        p = subprocess.Popen(
            [sys.executable, str(HERE / "eval_parallel.py"),
             "--worker", str(int(sev)), str(float(jit)), key],
            stdout=log, stderr=subprocess.STDOUT)
        procs[key] = (p, log, time.time())
        print(f"spawned worker {key} (pid {p.pid})")

    failed = []
    for key, (p, log, t0) in procs.items():
        try:
            rc = p.wait(timeout=PER_IC_TIMEOUT_S - (time.time() - t0))
        except subprocess.TimeoutExpired:
            p.kill()
            rc = -1
        log.close()
        print(f"worker {key} finished rc={rc} in {time.time()-t0:.0f}s")
        if rc != 0:
            failed.append(key)
    if failed:
        sys.exit(f"workers failed/timed out: {failed} — see eval_worker_*.log")
    return reduce_and_figure()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=3, metavar=("SEV", "JIT", "KEY"))
    a = ap.parse_args(argv)
    if a.worker:
        run_worker(float(a.worker[0]), float(a.worker[1]), a.worker[2])
    else:
        run_driver()


if __name__ == "__main__":
    main()

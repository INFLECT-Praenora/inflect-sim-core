#!/usr/bin/env python3
"""Rollout-stable training for the sepsis surrogate.

The baseline (train_and_eval.py) trains the delta-MLP one-step/teacher-forced.
It is accurate one-step but, fed its own predictions autonomously for 576 steps,
compounding error explodes (~40 h in) — held-out rollout NRMSE ~1e31. This
script fixes that with UNROLLED (multi-step) training: the SAME architecture is
optimized on K-step free-running rollouts of its own output, so it learns to be
contractive rather than merely one-step accurate. Output is a drop-in
`surrogate.pt` the existing NeuralProcess/eval loads unchanged.

Key facts it relies on (verified against viva_torch):
  * spec.parameterization == "delta", driver_paths == []  -> pure 98-dim
    autoregression  x_{t+1} = x_t + net.forward(x_t)  (net.forward = raw delta).
  * eval reloads base SurrogateNet, so stabilization must be in the WEIGHTS;
    the per-step clamp here is a TRAINING-ONLY finiteness guard, never baked in.

Fast loop: train (minutes) -> `--smoke` checks the surrogate-only 576-step
rollout stays bounded (seconds, no roadrunner) -> only then the ~2 h real-SBML
held-out eval via eval_parallel.py.

Run:
  .venv/bin/python .../train_stable.py --warm .../out/surrogate.pt --out .../out/surrogate.pt
  .venv/bin/python .../train_stable.py --smoke .../out/surrogate.pt   # stability check only
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from torch import nn

from viva_torch import TransitionDataset
from viva_torch.model import SurrogateNet
from viva_torch.spec import Normalizer

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "studies" / "01-sample-and-train" / "out"


# --------------------------------------------------------- trajectory assembly
def build_trajectories(ds: TransitionDataset):
    """Return (traj_states, order_ok): list of (L+1, n_targets) absolute-state
    sequences, one per trajectory, reconstructed from consecutive transitions.
    X[t] is the state at t; Y[t] the state at t+1. Rows within a trajectory are
    assumed appended in time order; we verify Y[i] == X[i+1]."""
    n_t = ds.spec.n_targets
    seqs = []
    order_ok = True
    for tid in np.unique(ds.traj_id):
        idx = np.where(ds.traj_id == tid)[0]  # preserves insertion order
        Xs = ds.X[idx, :n_t]
        Ys = ds.Y[idx, :n_t]
        if len(idx) > 1:
            mism = np.abs(Ys[:-1] - Xs[1:])
            scale = np.abs(Xs[1:]) + 1e-6
            if np.nanmax(mism / scale) > 1e-3:
                order_ok = False
        states = np.vstack([Xs, Ys[-1:]])  # (L+1, n_t)
        seqs.append(states)
    return seqs, order_ok


# ------------------------------------------------------------------ unrolling
def unroll(net: SurrogateNet, x0: torch.Tensor, K: int,
           fmean: torch.Tensor, fstd: torch.Tensor,
           clamp_sigma: float = 15.0, tbptt: int = 48):
    """Autoregressive K-step rollout from x0 (B, n). Returns (B, K, n) predicted
    states x_1..x_K. Each step is clamped in NORMALIZED space to +/-clamp_sigma
    (keeps the graph finite WITHOUT masking near-divergence from the loss, unlike
    a wide raw clamp), and the graph is detached every `tbptt` steps (truncated
    BPTT) so full-horizon K stays cheap and gradients don't explode through time."""
    preds = []
    x = x0
    for k in range(K):
        x = x + net.forward(x)                       # raw delta step
        z = torch.clamp((x - fmean) / fstd, -clamp_sigma, clamp_sigma)
        x = z * fstd + fmean
        preds.append(x)
        if tbptt and (k + 1) % tbptt == 0:
            x = x.detach()
    return torch.stack(preds, dim=1)


def surrogate_rollout_numpy(net: SurrogateNet, x0: np.ndarray, n_steps: int):
    """Surrogate-only rollout (no roadrunner), matching NeuralProcess.predict_next."""
    x = np.asarray(x0, dtype=np.float64).reshape(1, -1)
    traj = np.empty((n_steps, x.shape[1]))
    for i in range(n_steps):
        x = net.predict_next(x)  # delta: x + forward(x)
        traj[i] = x[0]
    return traj


# ------------------------------------------------------------------ smoke test
def smoke(checkpoint: str, n_steps: int = 576):
    ds = TransitionDataset.load(str(OUT / "dataset.npz"))
    net = SurrogateNet.load(checkpoint)
    seqs, _ = build_trajectories(ds)
    worst = 0.0
    nonfinite = 0
    for s in seqs:
        traj = surrogate_rollout_numpy(net, s[0], n_steps)
        fin = np.isfinite(traj)
        nonfinite += int((~fin).sum())
        if fin.any():
            worst = max(worst, float(np.abs(traj[fin]).max()))
    # reference: physiological magnitude from the data
    datamax = float(np.abs(ds.X).max())
    print(f"smoke: {len(seqs)} ICs x {n_steps} steps | max|state|={worst:.3e} "
          f"(data max {datamax:.3e}) | non-finite={nonfinite} "
          f"| {'STABLE' if (nonfinite == 0 and worst < 100 * datamax) else 'DIVERGES'}")
    return nonfinite == 0 and worst < 100 * datamax


# -------------------------------------------------------------------- training
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", default=None, help="checkpoint to warm-start from")
    ap.add_argument("--out", default=str(OUT / "surrogate.pt"))
    ap.add_argument("--smoke", default=None, help="just run the stability smoke test on this ckpt")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=lambda s: tuple(int(x) for x in s.split(",")), default=(128, 128))
    ap.add_argument("--barrier_b", type=float, default=6.0, help="penalize states beyond this many sigma")
    ap.add_argument("--barrier_w", type=float, default=0.5, help="weight on the out-of-distribution barrier")
    ap.add_argument("--clamp_sigma", type=float, default=15.0)
    # curriculum: "K:steps,K:steps,..." (steps = gradient steps, one minibatch each)
    ap.add_argument("--curriculum", default="16:300,64:400,192:400,576:500")
    a = ap.parse_args(argv)

    if a.smoke:
        ok = smoke(a.smoke)
        raise SystemExit(0 if ok else 1)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ds = TransitionDataset.load(str(OUT / "dataset.npz"))
    spec = ds.spec
    seqs, order_ok = build_trajectories(ds)
    if not order_ok:
        print("WARNING: within-trajectory ordering check failed — unroll targets may be misaligned")
    L = min(len(s) for s in seqs)
    traj = np.stack([s[:L] for s in seqs]).astype(np.float32)  # (n_traj, L, n)
    print(f"{traj.shape[0]} trajectories x {L} states x {spec.n_targets} dims; order_ok={order_ok}")

    # net: warm-start (keeps one-step accuracy + normalization) or fresh
    if a.warm and pathlib.Path(a.warm).exists():
        net = SurrogateNet.load(a.warm)
        print(f"warm-started from {a.warm}")
    else:
        feat_norm = Normalizer.from_data(ds.X)
        from viva_torch.trainer import _training_signal
        targ_norm = Normalizer.from_data(_training_signal(ds))
        net = SurrogateNet(spec, feat_norm, targ_norm, hidden=a.hidden)
        print("fresh net")

    fmean, fstd = net.feat_mean, net.feat_std  # state-space normalizer (even-weighted loss)

    def nstate(x):  # normalize states for the loss
        return (x - fmean) / fstd

    traj_t = torch.tensor(traj)
    n_traj = traj_t.shape[0]
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    phases = [(int(k), int(e)) for k, e in (p.split(":") for p in a.curriculum.split(","))]

    for K, epochs in phases:
        maxstart = L - 1 - K
        if maxstart < 0:
            print(f"skip K={K} (trajectory too short)"); continue
        for ep in range(epochs):
            net.train()
            # sample B windows (traj, start)
            tj = rng.integers(0, n_traj, size=a.batch)
            st = rng.integers(0, maxstart + 1, size=a.batch)
            x0 = traj_t[tj, st]                                   # (B, n)
            real = torch.stack([traj_t[tj[b], st[b] + 1: st[b] + 1 + K] for b in range(a.batch)])  # (B,K,n)
            pred = unroll(net, x0, K, fmean, fstd, a.clamp_sigma)  # (B,K,n)
            znp = nstate(pred)
            # per-step weight: emphasize later (stability) steps mildly
            w = torch.linspace(1.0, 2.0, K).view(1, K, 1)
            loss_roll = (w * (znp - nstate(real)) ** 2).mean()
            # barrier: penalize predicted states leaving the data distribution (|z| > b)
            barrier = (torch.relu(znp.abs() - a.barrier_b) ** 2).mean()
            # one-step anchor over a fresh batch of all transitions
            tj2 = rng.integers(0, n_traj, size=a.batch)
            st2 = rng.integers(0, L - 1, size=a.batch)
            xa = traj_t[tj2, st2]
            xb = traj_t[tj2, st2 + 1]
            one = ((nstate(xa + net.forward(xa)) - nstate(xb)) ** 2).mean()
            loss = loss_roll + a.barrier_w * barrier + one
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if ep % max(1, epochs // 4) == 0 or ep == epochs - 1:
                print(f"  K={K:3d} step {ep:3d}/{epochs}: roll {loss_roll.item():.4g}  "
                      f"barrier {barrier.item():.4g}  one {one.item():.4g}")
        # stability checkpoint + smoke after each phase
        net.save(a.out)
        smoke(a.out)

    net.save(a.out)
    print("saved", a.out)
    ok = smoke(a.out)
    print("FINAL:", "STABLE" if ok else "STILL DIVERGES")


if __name__ == "__main__":
    main()

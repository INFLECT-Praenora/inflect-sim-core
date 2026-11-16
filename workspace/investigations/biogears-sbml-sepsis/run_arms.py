#!/usr/bin/env python
"""Sepsis arms runner: one arm (engine|sbml x severity) per invocation, plus `analyze`.

  python run_arms.py --engine {engine,sbml} --severity {1,3} --horizon-hours H
                     [--interval-s 60] [--out DIR] [--log-every-h 1] [--no-resume]
  python run_arms.py analyze --severity {1,3} [--out DIR]

Default --out: studies/02-mild-infection/out (severity 1), studies/03-severe-infection/out (3).
Infection is seeded at t=0 on both sides. Outputs in <out>:
  <engine>-sev<S>.json              {meta, samples:[{time,<9 obs>}...]}  (checkpointed at each progress line)
  <engine>-sev<S>.milestones.json   extract_milestones() of the arm
  compare-sev<S>.json               analyze output (align + milestone divergence)
Resume: if the arm json exists with the same meta and a shorter horizon, a rerun
restarts from scratch (composite state is not serialisable); a complete run
(sim_time >= horizon) is skipped unless --no-resume.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_BY_SEV = {1: HERE / 'studies/02-mild-infection/out', 3: HERE / 'studies/03-severe-infection/out'}


def default_out(sev):
    return OUT_BY_SEV.get(sev, HERE / 'out')


def arm_paths(out: Path, engine: str, sev: int):
    return out / f'{engine}-sev{sev}.json', out / f'{engine}-sev{sev}.milestones.json'


def rows_to_samples(rows, extract, observables):
    """Emitter rows -> [{'time': s, obs: v}]; drops rows without readings (as extract does)."""
    obs = extract(rows)
    if not obs:
        return []
    keep = [r for r in rows if (r.get('parameters') if 'parameters' in r else r.get('circuits'))]
    times = [float(r.get('time', r.get('global_time', 0.0))) for r in keep]
    n = len(obs['SBP'])
    assert len(times) == n, (len(times), n)
    return [{'time': times[i], **{k: obs[k][i] for k in observables}} for i in range(n)]


def build(engine, sev, interval_s, horizon_s):
    from process_bigraph import Composite
    if engine == 'sbml':
        from process_bigraph import allocate_core
        from process_bigraph.emitter import RAMEmitter
        from viva_tellurium.processes import TelluriumProcess
        from viva_simulation_core.composites.sepsis_sbml import (
            SEPSIS_OBSERVABLES, extract_observables, make_sepsis_sbml_composite)
        core = allocate_core()
        core.register_link('TelluriumProcess', TelluriumProcess)
        core.register_link('ram-emitter', RAMEmitter)
        doc = make_sepsis_sbml_composite(severity=sev, infection_start_s=0, horizon_s=horizon_s,
                                         interval_s=interval_s)
    else:
        from viva_biogears.core import build_core
        from viva_simulation_core.composites.sepsis_engine import (
            SEPSIS_OBSERVABLES, extract_observables, make_sepsis_engine_composite)
        core = build_core()
        doc = make_sepsis_engine_composite(severity=sev, infection_start_s=0, horizon_s=horizon_s,
                                           interval_s=interval_s, core=core)
    return Composite({'state': doc}, core=core), extract_observables, list(SEPSIS_OBSERVABLES)


def write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def run_arm(engine, sev, horizon_h, interval_s, out: Path, log_every_h=1.0, resume=True):
    from process_bigraph import gather_emitter_results
    from viva_simulation_core.analysis.milestones import extract_milestones
    out.mkdir(parents=True, exist_ok=True)
    series_p, ms_p = arm_paths(out, engine, sev)
    horizon_s = horizon_h * 3600.0
    meta = dict(engine=engine, severity=sev, interval_s=interval_s, horizon_s=horizon_s,
                infection_start_s=0, patient='StandardMale' if engine == 'engine' else 'sbml-default')
    if resume and series_p.exists():
        old = json.loads(series_p.read_text())
        om = old.get('meta', {})
        if (om.get('complete') and all(om.get(k) == meta[k] for k in
                                       ('engine', 'severity', 'interval_s', 'horizon_s'))):
            print(f'[{engine} sev{sev}] already complete, skipping', flush=True)
            return
    sim, extract, observables = build(engine, sev, interval_s, horizon_s)
    t0 = time.perf_counter()
    chunk = max(interval_s, log_every_h * 3600.0)
    done = 0.0
    def snapshot(complete):
        rows = gather_emitter_results(sim)[('emitter',)]
        samples = rows_to_samples(rows, extract, observables)
        write_json(series_p, {'meta': {**meta, 'complete': complete, 'sim_time_s': done,
                                       'wall_s': time.perf_counter() - t0}, 'samples': samples})
        return samples
    while done < horizon_s - 1e-9:
        step = min(chunk, horizon_s - done)
        sim.run(step)
        done += step
        samples = snapshot(done >= horizon_s - 1e-9)
        last = samples[-1] if samples else {}
        print(f'[{engine} sev{sev}] sim {done/3600:.2f}h/{horizon_h:g}h wall {time.perf_counter()-t0:.0f}s '
              f'SBP={last.get("SBP", float("nan")):.1f} HR={last.get("HR", float("nan")):.1f} '
              f'pathogen={last.get("blood_pathogen", float("nan")):.3g}', flush=True)
    ms = extract_milestones(samples).as_dict()
    write_json(ms_p, ms)
    print(f'[{engine} sev{sev}] done; milestones {ms}', flush=True)


def analyze(sev, out: Path):
    from viva_simulation_core.analysis.align import align_observables
    from viva_simulation_core.analysis.milestones import extract_milestones
    arms = {}
    for e in ('engine', 'sbml'):
        p, _ = arm_paths(out, e, sev)
        if not p.exists():
            sys.exit(f'missing arm file {p}')
        arms[e] = json.loads(p.read_text())
    s_e, s_s = arms['engine']['samples'], arms['sbml']['samples']
    cmp_ = align_observables(s_e, s_s, require_all=True)
    m_e, m_s = (extract_milestones(s).as_dict() for s in (s_e, s_s))
    table = {}
    for k in m_e:
        a, b = m_e[k], m_s[k]
        num = isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool)
        table[k] = dict(engine=a, sbml=b, diff=(b - a) if num else None,
                        agree=(a == b) if (a is None or b is None or isinstance(a, bool)) else None)
    res = dict(
        severity=sev,
        horizon_s=dict(engine=arms['engine']['meta']['horizon_s'], sbml=arms['sbml']['meta']['horizon_s']),
        complete=dict(engine=arms['engine']['meta'].get('complete'), sbml=arms['sbml']['meta'].get('complete')),
        aligned=dict(n_points=int(len(cmp_.time)), passed=cmp_.passed,
                     only_engine=cmp_.only_engine, only_sbml=cmp_.only_sbml,
                     missing_both=cmp_.missing_both, no_overlap=cmp_.no_overlap,
                     per_observable={k: dict(max_abs=d.max_abs, max_rel=d.max_rel, passed=d.passed)
                                     for k, d in cmp_.per_observable.items()}),
        milestones=table)
    p = out / f'compare-sev{sev}.json'
    write_json(p, res)
    print(f'wrote {p}; aligned.passed={cmp_.passed} points={len(cmp_.time)}')
    for k, d in cmp_.per_observable.items():
        print(f'  {k:18s} max_abs={d.max_abs:.4g} max_rel={d.max_rel:.3g} pass={d.passed}')
    return res


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'analyze':
        ap = argparse.ArgumentParser(prog='run_arms.py analyze')
        ap.add_argument('--severity', type=int, required=True, choices=[1, 3])
        ap.add_argument('--out', type=Path)
        a = ap.parse_args(argv[1:])
        analyze(a.severity, a.out or default_out(a.severity))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True, choices=['engine', 'sbml'])
    ap.add_argument('--severity', type=int, required=True, choices=[1, 3])
    ap.add_argument('--horizon-hours', type=float, required=True)
    ap.add_argument('--interval-s', type=float, default=60.0)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--log-every-h', type=float, default=1.0)
    ap.add_argument('--no-resume', action='store_true')
    a = ap.parse_args(argv)
    run_arm(a.engine, a.severity, a.horizon_hours, a.interval_s, a.out or default_out(a.severity),
            a.log_every_h, not a.no_resume)


if __name__ == '__main__':
    main()

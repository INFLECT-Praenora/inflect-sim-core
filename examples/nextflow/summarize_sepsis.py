#!/usr/bin/env python3
"""Nextflow SUMMARIZE stage: gather per-severity sepsis task outputs into one
clinical table. Reads the `sev_<N>.txt` files emitted by `viva-sim run sepsis_sbml`
(lines like `parameters/MAP = 72.3`) and scores each run for septic shock.

Usage:  python summarize_sepsis.py sev_0.txt sev_1.txt sev_3.txt > sepsis_summary.tsv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FIELD = re.compile(r"\b(MAP|TI|PB)\s*=\s*([-\d.eE+]+)")


def parse(path: Path) -> dict[str, float]:
    vals: dict[str, float] = {}
    for line in path.read_text().splitlines():
        for m in FIELD.finditer(line):
            try:
                vals[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return vals


def severity_of(path: Path) -> str:
    m = re.search(r"sev_(\d+)", path.name)
    return m.group(1) if m else "?"


def main(argv: list[str]) -> int:
    rows = []
    for p in sorted(argv, key=lambda s: Path(s).name):
        path = Path(p)
        v = parse(path)
        sev = severity_of(path)
        mapv = v.get("MAP", float("nan"))
        shock = "SHOCK" if mapv < 65 else "stable"
        rows.append((sev, mapv, v.get("TI", float("nan")), v.get("PB", float("nan")), shock))

    print("severity\tMAP_mmHg\ttissue_integrity\tblood_pathogen\tstatus")
    for sev, mapv, ti, pb, shock in sorted(rows):
        print(f"{sev}\t{mapv:.2f}\t{ti:.4f}\t{pb:.4g}\t{shock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

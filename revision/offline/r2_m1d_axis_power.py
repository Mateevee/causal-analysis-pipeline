"""
R2-m1d / R3-1d -- Power curve of the time-axis rules (criterion C5 of
revision/PROTOCOL_time_axis_rule.md, frozen; SHA-256 in the authors' lab notes). The
decision only; no pipeline, no F1.

Design (fixed before running, 2026-11-09)
  Rules: R0 (original), R1 ((A)+(B)) and R2 (blockwise Hosking, the one adopted),
    m = 5, q = 0.05, exactly as in r2_m1c_axis_rules.py (same functions),
    on the Stage 2 series of the pipeline.
  P1 weak cross-dependence, without autocorrelation of their own: the repository's VAR
    generator (Data/synthetic_var), ar_coef_range = (0, 0), coef_strength in
    {(0.05; 0.15), (0.15; 0.30)} (the generator's default is (0.3; 0.6)), T in
    {50, 100, 300}, configurations of r3_1 (5 entities, prob 0.35, up to 2 parents;
    12 entities, prob 0.3, up to 3 parents; lag up to 3), seeds 5101-5120.
  P2 trend without cross-dependence: 12 white series; to 4 of them (30%) the
    linear trend beta * (t/T - 1/2) is added, beta in {0.5; 1; 2} standard deviations; T in {50,
    100, 300}; seeds 5101-5120.
  P3 seasonality without cross-dependence: 12 white series; in 4 of them, A *
    sin(2 pi t / 12 + phase), A in {0.3; 0.6; 1.0}; T in {50, 100, 300}.
  In all of them, the right answer by the definition is "with axis".
  Report: fraction of "with axis" per cell and per rule (power curve). No
    pass criterion: power should fall with a weak effect and a small T, and the
    report shows where.

Usage:  python revision/offline/r2_m1d_axis_power.py [--verificar]
Output: Results_Revisao/r2_m1d_poder_eixo/ (file and column names kept in Portuguese; see revision/README.md, Notes)
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_m1c_axis_rules as cmp  # noqa: E402  (sets sys.path)
import common as c  # noqa: E402
import common_exec as ce  # noqa: E402

NAME = "r2_m1d_poder_eixo"
SEEDS = list(range(5101, 5121))
TS = (50, 100, 300)
CONFIGS = [("n5", 5, 0.35, 2, 3), ("n12", 12, 0.3, 3, 3)]
STRENGTHS = [(0.05, 0.15), (0.15, 0.30)]
K = "m5_q0.05"


def p1(seeds):
    for strength in STRENGTHS:
        for T in TS:
            for tag, n, p, mp, ml in CONFIGS:
                with contextlib.redirect_stdout(io.StringIO()):
                    cs = ce.generate_var(seeds, T, n, p, mp, ml, min_edges=2,
                                         name_prefix=f"poder_{tag}_T{T}_c{strength[0]}", ar_coef_range=(0.0, 0.0),
                                         coef_strength=strength)
                for scen in cs:
                    yield "P1 cruzada fraca", f"coef {strength[0]}-{strength[1]}, {tag}", T, scen["name"], scen["df"]


def p23(seeds):
    for name, levels in (("P2 tendência", (0.5, 1.0, 2.0)), ("P3 sazonalidade", (0.3, 0.6, 1.0))):
        for level in levels:
            for T in TS:
                for s in seeds:
                    rng = np.random.default_rng(s * 100 + T + int(level * 10))
                    X = rng.standard_normal((T, 12))
                    t = np.arange(T)
                    for j in range(4):
                        if name.startswith("P2"):
                            X[:, j] += level * (t / T - 0.5)
                        else:
                            X[:, j] += level * np.sin(2 * np.pi * t / 12 + rng.uniform(0, 2 * np.pi))
                    X = X[:, rng.permutation(12)]
                    label = f"beta {level}" if name.startswith("P2") else f"A {level}"
                    yield name, label, T, s, cmp.long_format(X)


def main():
    check = "--verificar" in sys.argv
    seeds = SEEDS[:2] if check else SEEDS
    rel = c.Report(NAME + ("_verificacao" if check else ""),
                   "R2-m1d — Power curve of the time-axis rules (fraction of 'with axis'; right = with axis)")
    rows = []
    for i, (arm, level, T, s, df) in enumerate(list(p1(seeds)) + list(p23(seeds)), 1):
        if i % 50 == 0:
            print(f"  {i}")
        dec = cmp.decisions(cmp.stages_0_2(df))
        rows.append({"braco": arm, "nivel": level, "T": T, "semente": s, "n_datas": df["Date"].nunique(),
                     "com_eixo_R0": not dec["R0"], "com_eixo_R1": not dec[f"R1_{K}"],
                     "com_eixo_R2": not dec[f"R2_{K}"], "n": dec["n"], "d": dec["d"],
                     "series_R2": dec.get("diag_sig_R2"), "pares_R2": dec.get("pares_sig_R2")})
    D = pd.DataFrame(rows)
    rel.csv("por_cenario.csv", D)
    cut = D[D["n"] != D["n_datas"]]
    rel.p(f"Check: scenarios in which Stage 2 does not have every time point of the input: {len(cut)}.")
    if len(cut):
        raise RuntimeError("series cut at the input; invalid result")
    body = []
    for (b, nv, T), g in D.groupby(["braco", "nivel", "T"], sort=True):
        body.append([b, nv, T, len(g), c.fmt(g["com_eixo_R0"].mean(), 2), c.fmt(g["com_eixo_R1"].mean(), 2),
                     c.fmt(g["com_eixo_R2"].mean(), 2)])
    rel.table(["Arm", "Level", "T", "Scenarios", "R0 original", "R1 (A)+(B)", "R2 blockwise Hosking"], body)
    rel.save()


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        main()

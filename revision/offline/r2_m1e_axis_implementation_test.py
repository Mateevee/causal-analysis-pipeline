"""
Criterion C1 of revision/PROTOCOL_time_axis_rule.md -- tests of the implementation of the
new time-axis rule in the pipeline.

Design (fixed before running, 2026-11-09; protocol frozen, SHA-256 in the authors' lab notes)
  (i)   Pipeline.discovery._detect_time_axis with the default (rule='median_acf1')
        decides as the rule R0 stored by r2_m1c_axis_rules.py, on the 90 scenarios
        of the paper and the 400 simulated ones.
  (ii)  With rule='block_portmanteau', it decides as R2 (m = 5, q = 0.05)
        stored by r2_m1c_axis_rules.py, on the same scenarios.
  (iii) Synthetic i.i.d. case (8 white series, T = 500, seed 1) -> no axis.
  (iv)  Y_t = 0.6 X_{t-1} + noise, X white (T = 500, seed 1) -> with axis
        under the new rule (and no axis under the old one: it is the defect corrected).
  (v)   drop_lagged_without_time_axis passes the rule on (case iv: with the old rule
        it discards the links; with the new one, it does not).
  Success: 100% agreement in (i) and (ii) and the four cases (iii)-(v)
  right. Any disagreement is listed.

Usage:  python revision/offline/r2_m1e_axis_implementation_test.py
Output: Results_Revisao/r2_m1e_teste_implementacao_eixo/relatorio.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_m1c_axis_rules as cmp  # noqa: E402  (sets sys.path)
import common as c  # noqa: E402
from Pipeline import contemporaneous  # noqa: E402
from Pipeline.discovery import _detect_time_axis  # noqa: E402

NAME = "r2_m1e_teste_implementacao_eixo"
K = "R2_m5_q0.05"


def main():
    rel = c.Report(NAME, "C1 — Tests of the implementation of the new time-axis rule")
    ref = os.path.join(c.OUTPUT_ROOT, "r2_m1c_regras_eixo")
    A = pd.read_csv(os.path.join(ref, "artigo.csv")).set_index("cenario")
    S = pd.read_csv(os.path.join(ref, "simulado.csv"))
    S["chave"] = S["braco"] + "|" + S["d"].astype(str) + "|" + S["T"].astype(str) + "|" + S["semente"].astype(str)
    S = S.set_index("chave")

    disc_i, disc_ii, tot = [], [], 0
    for s in A.index:
        proc = cmp.stages_0_2(pd.read_csv(os.path.join(c.folder(s), "source.csv")))
        old = not _detect_time_axis(proc, verbose=False)
        new = not _detect_time_axis(proc, verbose=False, rule="block_portmanteau")
        tot += 1
        if old != bool(A.loc[s, "R0"]):
            disc_i.append(s)
        if new != bool(A.loc[s, K]):
            disc_ii.append(s)
    for arm, d, T, seed, df in cmp.simulated_scenarios(False):
        key = f"{arm}|{d}|{T}|{seed}"
        proc = cmp.stages_0_2(df)
        old = not _detect_time_axis(proc, verbose=False)
        new = not _detect_time_axis(proc, verbose=False, rule="block_portmanteau")
        tot += 1
        if old != bool(S.loc[key, "R0"]):
            disc_i.append(key)
        if new != bool(S.loc[key, K]):
            disc_ii.append(key)
    rel.p(f"(i) default = stored R0: {tot - len(disc_i)}/{tot} "
          f"({'ok' if not disc_i else 'DISAGREES: ' + ', '.join(disc_i)}).")
    rel.p(f"(ii) block_portmanteau = stored R2: {tot - len(disc_ii)}/{tot} "
          f"({'ok' if not disc_ii else 'DISAGREES: ' + ', '.join(disc_ii)}).")

    rng = np.random.default_rng(1)
    iid = {f"s{j}": rng.standard_normal(500) for j in range(8)}
    t3 = (_detect_time_axis(iid, verbose=False, rule="block_portmanteau") is False)
    rel.p(f"(iii) i.i.d. -> no axis under the new rule: {'ok' if t3 else 'FAILED'}.")

    rng = np.random.default_rng(1)
    x = rng.standard_normal(501)
    y = 0.6 * x[:-1] + rng.standard_normal(500)
    cr = {"x": x[1:], "y": y}
    new_rule = _detect_time_axis(cr, verbose=False, rule="block_portmanteau")
    old_rule = _detect_time_axis(cr, verbose=False)
    t4 = (new_rule is True) and (old_rule is False)
    rel.p(f"(iv) Y = 0.6 X(t-1) + noise: new rule with axis = {new_rule}, old rule with axis = {old_rule} "
          f"-> {'ok' if t4 else 'FAILED'}.")

    res = pd.DataFrame({"significant": [True, True], "lag": [1, 2]})
    _, n_old = contemporaneous.drop_lagged_without_time_axis(res.copy(), cr)
    _, n_new = contemporaneous.drop_lagged_without_time_axis(res.copy(), cr, rule="block_portmanteau")
    t5 = (n_old == 2) and (n_new is None)
    rel.p(f"(v) Stage 8: the old rule discards {n_old} links, the new rule {n_new} -> {'ok' if t5 else 'FAILED'}.")

    ok = not disc_i and not disc_ii and t3 and t4 and t5
    rel.p("")
    rel.p(f"**Result: {'all tests passed' if ok else 'THERE ARE FAILURES'}.**")
    rel.data = {"ok": ok, "discordancias_i": disc_i, "discordancias_ii": disc_ii}
    rel.save()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        main()

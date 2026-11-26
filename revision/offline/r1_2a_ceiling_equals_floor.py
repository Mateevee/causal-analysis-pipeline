"""
R1-2a / D6 — on how many scenarios the F1 of the per-family ceiling (A2) and that of the "one scenario left out"
condition are exactly the trivial floor (declare all pairs: F1 = 2p/(1 + p), p = n_causal / n_universe, as in
generators/paper_numbers.py). A descriptive count, first done in the terminal on 2026-11-21 and versioned here because
it goes into the text (V-A, table of the tuned baselines).

Input: Results_Revisao/_outra_maquina/rodada2_grade_ampliada_2026-11-10/r1_2f_ajuste_por_familia/a2_por_cenario.csv
(checked on two machines by D1). Floor: common.rows(). Equality: |F1 − floor| < 1e-6.
Usage (from the repository root):  python revision/offline/r1_2a_ceiling_equals_floor.py
Output: Results_Revisao/r1_2a_teto_igual_piso/
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "execution")]
import common as c  # noqa: E402

NAME = "r1_2a_teto_igual_piso"
INPUT = os.path.join(c.OUTPUT_ROOT, "_outra_maquina", "rodada2_grade_ampliada_2026-11-10", "r1_2f_ajuste_por_familia",
                     "a2_por_cenario.csv")


def main():
    rel = c.Report(NAME, "D6 — per-family ceiling equal to the trivial floor")
    A = pd.read_csv(INPUT)
    rows = c.rows()

    def floor_of(cen):
        d = rows[c.scenario_key(cen)][c.PIPE]
        p = d["n_causal"] / d["n_universe"]
        return 2 * p / (1 + p)

    A["piso"] = A.cenario.map(floor_of)
    A["teto_igual_piso"] = (A.f1_teto - A.piso).abs() < 1e-6
    A["fora_igual_piso"] = (A.f1_um_cenario_de_fora - A.piso).abs() < 1e-6
    rel.csv("por_cenario.csv", A)
    table_rows = []
    for part, msk in (("held-out", A.heldout == True), ("development", A.heldout == False)):  # noqa: E712
        for m, g in A[msk].groupby("metodo"):
            with_ceiling = g.f1_teto.notna()
            with_left_out = g.f1_um_cenario_de_fora.notna()
            table_rows.append([part, m, f"{int(g.teto_igual_piso.sum())} of {int(with_ceiling.sum())}",
                               f"{int(g.fora_igual_piso.sum())} of {int(with_left_out.sum())}",
                               c.fmt(g.f1_teto.mean()), c.fmt(g.piso[with_ceiling].mean())])
    rel.table(["Part", "Method", "Ceiling = floor", "One scenario left out = floor", "Mean ceiling", "Mean floor"],
              table_rows)
    rel.save()
    print(open(os.path.join(rel.dir, "relatorio.md"), encoding="utf-8").read())


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        main()

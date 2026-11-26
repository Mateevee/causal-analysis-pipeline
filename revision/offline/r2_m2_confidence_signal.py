"""
R2-m2 — stability as a confidence signal without the truth (E1 of 2026-11-13). POST HOC, labelled as such.

Versions the computation done in the terminal on 2026-11-13 (entry "Exploratory analyses of round 5" of NOTAS.md),
so that it can support the recommendation to the user in the response to R2-m2. The numbers of 13/11 were seen before.
This script redoes them with the same definition; if they diverge, this script prevails and the divergence is noted.

Definition (that of 13/11):
  Data: the pipeline's R2-m2 (round 5, B = 20 contiguous 80% subsamples per scenario, 90 scenarios) and the
    full-sample graph (paper run, patch_tau10).
  (a) Each edge of the full graph, by the fraction of subsamples in which it appears: < 50%, 50–80%, ≥ 80%. Precision of
      each band against the truth, aggregated (all, development, held-out).
  (b) Spearman, per scenario, between the Jaccard between subsamples and the full-sample F1 (all, dev, held-out).
  (c) Hard filter: F1 with only the ≥ 80% edges, against the full F1 (mean; better / equal / worse).

Usage (from the repository root):  python revision/offline/r2_m2_confidence_signal.py
Output: Results_Revisao/r2_m2_sinal_confianca/
"""
from __future__ import annotations

import itertools
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "execution")]
import common as c  # noqa: E402
import r2_m2_jaccard as rj  # noqa: E402

NAME = "r2_m2_sinal_confianca"
RUN = os.path.join(c.OUTPUT_ROOT, "_outra_maquina", "rodada5_R2", "r2_m2_jaccard", "run")


def main():
    rel = c.Report(NAME, "R2-m2 — stability as a confidence signal (POST HOC; redoes E1 of 13/11)")
    groups = {}
    for s in os.listdir(RUN):
        m = re.match(r"(.+)__sub\d+$", s)
        if m:
            groups.setdefault(m.group(1), []).append(s)
    edges, scen_rows = [], []
    for base, subs in sorted(groups.items()):
        k = c.scenario_key(base)
        if not k:
            continue
        T, _ = c.truth(k)
        full = rj.graph(os.path.join(c.folder(k), c.PIPE_FOLDER, "results.csv"))
        G = [g for g in (rj.graph(os.path.join(RUN, s, "patch_tau10", "results.csv")) for s in sorted(subs))
             if g is not None]
        if full is None or not G:
            continue
        ho = c.is_heldout(k)
        freq = {e: np.mean([e in g for g in G]) for e in full}
        for e, f in freq.items():
            edges.append({"cenario": k, "heldout": ho, "freq": f, "verdadeira": e in T})
        stable = {e for e, f in freq.items() if f >= 0.8}
        scen_rows.append({"cenario": k, "heldout": ho, "B": len(G),
                          "jaccard_entre": np.mean([rj.jacc(a, b) for a, b in itertools.combinations(G, 2)]),
                          "f1_completo": c.f1_set(full, T), "f1_so_estaveis": c.f1_set(stable, T)})
    A, C = pd.DataFrame(edges), pd.DataFrame(scen_rows)
    rel.csv("arestas.csv", A)
    rel.csv("por_cenario.csv", C)
    A["faixa"] = pd.cut(A.freq, [-0.01, 0.4999999, 0.7999999, 1.0], labels=["< 50%", "50–80%", "≥ 80%"])
    rel.h("## (a) Precision per edge-stability band")
    rows = []
    for part, msk in (("all", A.heldout | ~A.heldout), ("dev", ~A.heldout), ("held-out", A.heldout)):
        g = A[msk].groupby("faixa", observed=False).verdadeira.agg(["size", "mean"])
        rows.append([part] + [f"{int(g.loc[f, 'size'])} edges, {g.loc[f, 'mean']:.3f}" for f in g.index])
    rel.table(["Part", "< 50%", "50–80%", "≥ 80%"], rows)
    rel.h("## (b) Spearman per scenario: Jaccard between subsamples × full-sample F1")
    rows = []
    for part, msk in (("all", C.heldout | ~C.heldout), ("dev", ~C.heldout), ("held-out", C.heldout)):
        r = spearmanr(C[msk].jaccard_entre, C[msk].f1_completo)
        rows.append([part, int(msk.sum()), c.fmt(r.statistic), c.fmt(r.pvalue)])
    rel.table(["Part", "Scenarios", "Spearman", "p"], rows)
    rel.h("## (c) Hard filter: only the ≥ 80% edges")
    rows = []
    for part, msk in (("dev", ~C.heldout), ("held-out", C.heldout)):
        g = C[msk]
        w, e, l = c.wins(g.f1_so_estaveis, g.f1_completo)
        rows.append([part, len(g), c.fmt(g.f1_completo.mean()), c.fmt(g.f1_so_estaveis.mean()), f"{w} / {e} / {l}"])
    rel.table(["Part", "Scenarios", "Full F1", "F1 stable only", "Better / equal / worse"], rows)
    rel.save()
    print(open(os.path.join(rel.dir, "relatorio.md"), encoding="utf-8").read())


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        main()

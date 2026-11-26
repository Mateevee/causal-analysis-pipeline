"""
R2-m2 / A8 — complement with the pipeline on the SAME subsamples as the baselines (pending item of 2026-11-20).

Why: in the delivered A8 (execucao_2026-11-19), the pipeline enters with B = 20 (round 5) and the baselines with B = 10.
The subsamples of r2_m2_jaccard.subsamples are nested (the first ones of a larger B are those of a smaller B), so
sub00..sub09 of the pipeline are the same windows as those of the baselines.

Design (fixed before running, 2026-11-21; the frozen `avaliar` of A8 does not change):
  Main: pipeline only on sub00..sub09 against the baselines as delivered; same metrics (_metrics of A8),
    same rules (means on the scenarios where the 6 methods have a result; paired Wilcoxon and Holm over the 5
    comparisons; all, development and held-out; with and without the scenarios with an empty graph in all subsamples).
  Secondary (paired): for each scenario x baseline, the pipeline restricted exactly to the subsample indices that the
    baseline has (some baselines have fewer than 10 in a scenario), only for the Jaccard between subsamples.
  Reading: descriptive, alongside the delivered A8; whatever changes relative to it is reported.

Usage (from the repository root):  python revision/offline/r2_m2_a8_sub10_complement.py
Output: Results_Revisao/r2_m2_a8_complemento_sub10/
"""
from __future__ import annotations

import itertools
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "execution")]
import common as c  # noqa: E402
import r2_m2_jaccard as rj  # noqa: E402
import r2_m2_jaccard_baselines as ab  # noqa: E402

NAME = "r2_m2_a8_complemento_sub10"
RUN_P = os.path.join(c.OUTPUT_ROOT, "_outra_maquina", "rodada5_R2", "r2_m2_jaccard", "run")
RUN_B = os.path.join(c.OUTPUT_ROOT, "_outra_maquina", "execucao_2026-11-19", "r2_m2_jaccard_baselines", "run")
SUBS = 10


def groups(run):
    g = {}
    for s in os.listdir(run):
        m = re.match(r"(.+)__sub(\d+)$", s)
        if m:
            g.setdefault(m.group(1), {})[int(m.group(2))] = s
    return g


def main():
    rel = c.Report(NAME, "A8 — complement: pipeline on the same subsamples as the baselines (sub00..sub09)")
    rel.p(f"Pipeline: `{RUN_P}` (round 5). Baselines: `{RUN_B}` (run of 19/11, B = 10, partial).")
    gp, gb = groups(RUN_P), groups(RUN_B)
    rows_m, idx_b = [], {}
    for base in sorted(set(gp) | set(gb)):
        k = c.scenario_key(base)
        if not k:
            continue
        T, _ = c.truth(k)
        if base in gp:
            full = rj.graph(os.path.join(c.folder(k), c.PIPE_FOLDER, "results.csv"))
            subs = [gp[base][i] for i in sorted(gp[base]) if i < SUBS]
            G = [g for g in (rj.graph(os.path.join(RUN_P, s, "patch_tau10", "results.csv")) for s in subs)
                 if g is not None]
            if G and full is not None:
                rows_m.append({"cenario": k, "familia": c.family(k), "heldout": c.is_heldout(k),
                               "metodo": "pipeline", **ab._metrics(G, full, T, c)})
        if base in gb:
            for b in ab.BASELINES:
                idx_graphs = [(i, rj.graph(os.path.join(RUN_B, gb[base][i], f"{b}_tau10", "results.csv")))
                              for i in sorted(gb[base])]
                idx_graphs = [(i, g) for i, g in idx_graphs if g is not None]
                if not idx_graphs:
                    continue
                idx_b[(k, b)] = [i for i, _ in idx_graphs]
                full = rj.graph(os.path.join(c.folder(k, b), f"{b}_tau10", "results.csv")) if b in c.rows()[k] else None
                rows_m.append({"cenario": k, "familia": c.family(k), "heldout": c.is_heldout(k), "metodo": b,
                               **ab._metrics([g for _, g in idx_graphs], full, T, c)})
    A = pd.DataFrame(rows_m)
    rel.csv("por_cenario_metodo.csv", A)
    p = A[A.metodo == "pipeline"]
    rel.p(f"Pipeline: {len(p)} scenarios; subsamples per scenario from {int(p.B.min())} to {int(p.B.max())}.")
    for b in ab.BASELINES:
        g = A[A.metodo == b]
        rel.p(f"- {b}: {len(g)} scenarios; B from {int(g.B.min()) if len(g) else 0} to {int(g.B.max()) if len(g) else 0}.")
    methods = ["pipeline"] + ab.BASELINES
    for metric, title in (("jaccard_entre_subamostras", "Jaccard between subsamples (main)"),
                          ("jaccard_vs_completa", "Jaccard against the full sample (secondary)"),
                          ("arestas_estaveis_frac", "Fraction of stable edges (≥ 80%)"),
                          ("f1_subamostras", "Mean F1 of the subsamples")):
        rel.h(f"## {title}")
        X = A.pivot_table(index="cenario", columns="metodo", values=metric)
        X = X[[m for m in methods if m in X.columns]]
        Xc = X.dropna()
        ho = A.drop_duplicates("cenario").set_index("cenario").heldout.reindex(Xc.index).astype(bool)
        rel.p(f"Means on the {len(Xc)} scenarios where the {Xc.shape[1]} methods have a result.")
        rel.table(["Part", "Scenarios"] + list(Xc.columns),
                  [[pt, int(msk.sum())] + [c.fmt(Xc.loc[msk, m].mean()) for m in Xc.columns]
                   for pt, msk in (("all", ho | ~ho), ("dev", ~ho), ("held-out", ho))])
        hoX = A.drop_duplicates("cenario").set_index("cenario").heldout
        for part in ("todos", "dev", "held-out"):
            for no_empty in (False, True):
                pv, line_by = {}, {}
                for b in [m for m in ab.BASELINES if m in X.columns]:
                    ss = X[["pipeline", b]].dropna()
                    if part != "todos":
                        ss = ss[hoX.reindex(ss.index).astype(bool) == (part == "held-out")]
                    if no_empty:
                        vz = set(A[(A.vazio_em_todas == True) & A.metodo.isin(["pipeline", b])].cenario)  # noqa: E712
                        ss = ss[~ss.index.isin(vz)]
                    if len(ss) < 2:
                        continue
                    w, e, l = c.wins(ss["pipeline"], ss[b])
                    pv[b] = c.wilcoxon_p(ss["pipeline"], ss[b])
                    line_by[b] = [f"pipeline × {b}", len(ss), c.fmt((ss["pipeline"] - ss[b]).mean()), f"{w} / {e} / {l}"]
                if line_by:
                    ph = c.holm(pv)
                    rel.p(f"{part}, {'without the scenarios with an empty graph in all subsamples' if no_empty else 'all scenarios'}:")
                    rel.table(["Comparison", "Scenarios", "Mean difference", "Better / equal / worse", "Wilcoxon p",
                               "Holm p (5)"], [line_by[b] + [c.fmt(pv[b]), c.fmt(ph[b])] for b in line_by])
            if metric != "jaccard_entre_subamostras":
                break
    # ---- secondary: paired by subsample index
    rel.h("## Secondary: Jaccard between subsamples, pipeline restricted to the same indices of each baseline")
    pv, line_by, rows = {}, {}, []
    for b in ab.BASELINES:
        diffs = []
        for (k, bb), idx in idx_b.items():
            if bb != b or len(idx) < 2:
                continue
            base = next((x for x in gp if c.scenario_key(x) == k), None)
            if base is None:
                continue
            Gp = [rj.graph(os.path.join(RUN_P, gp[base][i], "patch_tau10", "results.csv")) for i in idx if i in gp[base]]
            Gb = [rj.graph(os.path.join(RUN_B, next(x for x in gb if c.scenario_key(x) == k) and
                                        gb[next(x for x in gb if c.scenario_key(x) == k)][i], f"{b}_tau10", "results.csv"))
                  for i in idx]
            if len(Gp) != len(idx) or any(g is None for g in Gp + Gb):
                continue
            jp = np.mean([rj.jacc(x, y) for x, y in itertools.combinations(Gp, 2)])
            jb = np.mean([rj.jacc(x, y) for x, y in itertools.combinations(Gb, 2)])
            diffs.append((k, jp, jb))
            rows.append({"cenario": k, "baseline": b, "n_subamostras": len(idx), "jaccard_pipeline": jp,
                         "jaccard_baseline": jb})
        if len(diffs) >= 2:
            sp, sb = pd.Series([d[1] for d in diffs]), pd.Series([d[2] for d in diffs])
            w, e, l = c.wins(sp, sb)
            pv[b] = c.wilcoxon_p(sp, sb)
            line_by[b] = [f"pipeline × {b}", len(diffs), c.fmt((sp - sb).mean()), f"{w} / {e} / {l}"]
    ph = c.holm(pv)
    rel.table(["Comparison", "Scenarios", "Mean difference", "Better / equal / worse", "Wilcoxon p", "Holm p (5)"],
              [line_by[b] + [c.fmt(pv[b]), c.fmt(ph[b])] for b in line_by])
    rel.csv("pareada_por_indice.csv", pd.DataFrame(rows))
    rel.save()
    print(open(os.path.join(rel.dir, "relatorio.md"), encoding="utf-8").read())


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        main()

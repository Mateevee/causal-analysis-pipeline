"""
R2-m2 / A8 -- Stability of the graph under subsampling (Jaccard) for the 5 baselines, to compare with the pipeline.

Request
  R2: "the sensitivity of the resulting edge set to noise perturbations or sub-sampling is not reported.
       Stability metrics, such as Jaccard similarity across bootstrap runs, would increase confidence".
  R2-m2 (round 5, the runner) measured the pipeline. Without the baselines, the number has no reference (A8, approved
  by the author on 2026-11-19 in the decision "kill all of those that take less than 20 hours").

Design (fixed before running; 2026-11-19)
  Subsamples: the SAME ones as in R2-m2. The function r2_m2_jaccard.subsamples is imported unchanged (contiguous
    block with 80% of the dates; seed 7000 + CRC32 of the scenario name; the first subsamples of a larger B are the
    same as those of a smaller B).
  Methods: the 5 baselines of the paper at tau_max = 10 (pcmci_tigramite, pcmciplus, dynotears, cuts_plus, varlingam),
    paper configuration, default threads (as in the paper runs), one scenario at a time.
  Stages: rodar --B 5, then --B 10, then --B 20 (modular strategy of R2-m2). Resumable.
  Stopping point (fixed beforehand): until done or until the author orders R1-3b to start. Whatever does not reach
    B = 20 on the 90 scenarios is labelled PARTIAL, with the B reached per method and per scenario.
  Metrics per scenario and method (those of R2-m2): mean Jaccard between pairs of subsamples; mean Jaccard between
    each subsample and the full-sample graph (paper run); fraction of the full-sample edges present in
    >= 80% of the subsamples; mean F1 of the subsamples.
  **Main metric of the comparison between methods: the Jaccard between subsamples.** All subsamples of all
    methods run on the same machine. The Jaccard against the full sample compares with the graph of the paper
    runs, and DYNOTEARS and CUTS+ depend on threads and on the machine (round 3: 41 of 360 runs). Therefore,
    in that metric the instability of the baselines would include the machine effect. It is reported as secondary,
    with that caveat.
  Empty graph: the Jaccard of two empty graphs is 1 (R2-m2 definition). The report counts, per method, the scenarios
    with an empty graph in all subsamples and repeats the comparison without them.
  Means per method: only on the scenarios where all methods have a result (as in R1-5d).
  Comparison: pipeline (the R2-m2 of round 5) against each baseline, on the scenarios where both have a result.
    Mean difference, better / equal / worse, paired Wilcoxon and Holm over the 5 comparisons; overall, on
    development and on held-out. Every result is reported, including those unfavourable to the pipeline.

Usage
  python revision/execution/r2_m2_jaccard_baselines.py rodar --B 5|10|20 [--filtro name ...]
  python revision/execution/r2_m2_jaccard_baselines.py avaliar
Output: Results_Revisao/r2_m2_jaccard_baselines/
"""
from __future__ import annotations

import itertools
import os
import re
import sys
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import r2_m2_jaccard as rj  # noqa: E402  (subsamples, jacc, graph, arg: unchanged)

NAME = "r2_m2_jaccard_baselines"
BASELINES = ["pcmci_tigramite", "pcmciplus", "dynotears", "cuts_plus", "varlingam"]
# The pipeline's R2-m2: on the runner, where it ran, it is in the local folder; on the main machine, in the round 5
# copy.
PIPE_JACCARD_CANDIDATES = (os.path.join("r2_m2_jaccard", "por_cenario.csv"),
                           os.path.join("_outra_maquina", "rodada5_R2", "r2_m2_jaccard", "por_cenario.csv"))
B_FINAL = 20


def run():
    import common_exec as ce
    B = int(rj.arg("--B", ["20"])[0])
    if B not in (5, 10, 20):
        raise SystemExit("--B must be 5, 10 or 20 (stages of the design)")
    name_filter = rj.arg("--filtro", None)
    folder = ce.output_dir(os.path.join(NAME, "run"))
    dfs = ce.paper_scenarios(name_filter)
    if not name_filter and len(dfs) != 90:
        raise SystemExit(f"expected 90 scenarios, got {len(dfs)}")
    models = ce.models([f"{b}_tau10" for b in BASELINES])
    for k, d in enumerate(sorted(dfs, key=lambda x: x["name"]), 1):
        seed = 7000 + zlib.crc32(d["name"].encode("utf-8"))
        print(f"[{k}/{len(dfs)}] {d['name']} (B = {B})", flush=True)
        ce.run_resuming(rj.subsamples(d, B, seed), models, folder)


def _metrics(G, full, T, c):
    freq = {e: np.mean([e in g for g in G]) for e in full} if full is not None else {}
    return {"B": len(G),
            "jaccard_entre_subamostras": np.mean([rj.jacc(a, b) for a, b in itertools.combinations(G, 2)])
            if len(G) > 1 else np.nan,
            "jaccard_vs_completa": np.mean([rj.jacc(g, full) for g in G]) if full is not None else np.nan,
            "arestas_estaveis_frac": np.mean([v >= 0.8 for v in freq.values()]) if freq else np.nan,
            "f1_subamostras": np.mean([c.f1_set(g, T) for g in G]),
            "vazio_em_todas": all(len(g) == 0 for g in G)}


def evaluate():
    import common as c
    rel = c.Report(NAME, "R2-m2 / A8 — Stability under subsampling (Jaccard): baselines against the pipeline")
    run = os.path.join(c.OUTPUT_ROOT, NAME, "run")
    groups = {}
    if os.path.isdir(run):
        for s in os.listdir(run):
            m = re.match(r"(.+)__sub(\d+)$", s)
            if m:
                groups.setdefault(m.group(1), []).append(s)
    rows = []
    for base, subs in sorted(groups.items()):
        k = c.scenario_key(base)
        if not k:
            continue
        T, _ = c.truth(k)
        for b in BASELINES:
            G = [g for g in (rj.graph(os.path.join(run, s, f"{b}_tau10", "results.csv")) for s in sorted(subs))
                 if g is not None]
            if not G:
                continue
            full = rj.graph(os.path.join(c.folder(k, b), f"{b}_tau10", "results.csv")) if b in c.rows()[k] else None
            rows.append({"cenario": k, "familia": c.family(k), "heldout": c.is_heldout(k), "metodo": b,
                         **_metrics(G, full, T, c)})
    D = pd.DataFrame(rows)
    source = next((p for p in PIPE_JACCARD_CANDIDATES if os.path.exists(os.path.join(c.OUTPUT_ROOT, p))), None)
    if source is None:
        raise SystemExit(f"Pipeline R2-m2 not found in {PIPE_JACCARD_CANDIDATES}")
    rel.p(f"Pipeline: R2-m2 read from Results_Revisao/{source} (round 5, B = 20 on the 90).")
    P = pd.read_csv(os.path.join(c.OUTPUT_ROOT, source))
    P["metodo"] = "pipeline"
    run_p = os.path.join(c.OUTPUT_ROOT, os.path.dirname(source), "run")
    empty_map = {}
    for base in P.cenario:
        subs = [s for s in os.listdir(run_p) if re.match(re.escape(base) + r"__sub\d+$", s)] if os.path.isdir(run_p) else []
        G = [rj.graph(os.path.join(run_p, s, "patch_tau10", "results.csv")) for s in subs]
        empty_map[base] = bool(G) and all(g is not None and len(g) == 0 for g in G)
    P["vazio_em_todas"] = P.cenario.map(empty_map)
    cols = ["cenario", "familia", "heldout", "metodo", "B", "jaccard_entre_subamostras", "jaccard_vs_completa",
            "arestas_estaveis_frac", "f1_subamostras", "vazio_em_todas"]
    A = pd.concat([P[cols], D[cols]] if len(D) else [P[cols]], ignore_index=True)
    rel.csv("por_cenario_metodo.csv", A)
    if D.empty:
        rel.p("No baseline run yet.")
        rel.save()
        return
    partial = []
    for b in BASELINES:
        g = D[D.metodo == b]
        rel.p(f"- {b}: {len(g)} scenarios; subsamples per scenario from {int(g.B.min()) if len(g) else 0} to "
              f"{int(g.B.max()) if len(g) else 0}; empty in all subsamples in {int(g.vazio_em_todas.sum())}.")
        if len(g) < 90 or (len(g) and int(g.B.min()) < B_FINAL):
            partial.append(b)
    if partial:
        rel.p(f"**PARTIAL** ({', '.join(partial)}): the design is B = {B_FINAL} on the 90 scenarios; the stopping point "
              "fixed beforehand was reached before that. The numbers below hold for what ran.")
    methods = ["pipeline"] + BASELINES
    for metric, title in (("jaccard_entre_subamostras", "Jaccard between subsamples (main)"),
                          ("jaccard_vs_completa", "Jaccard against the full sample (secondary; mixes machines "
                                                  "for the baselines)"),
                          ("arestas_estaveis_frac", "Fraction of stable edges (≥ 80%)"),
                          ("f1_subamostras", "Mean F1 of the subsamples")):
        rel.h(f"## {title}")
        X = A.pivot_table(index="cenario", columns="metodo", values=metric)
        X = X[[m for m in methods if m in X.columns]]
        Xc = X.dropna()  # means only on the scenarios where all methods have a result (as in R1-5d)
        ho = A.drop_duplicates("cenario").set_index("cenario").heldout.reindex(Xc.index).astype(bool)
        rel.p(f"Means on the {len(Xc)} scenarios where the {Xc.shape[1]} methods have a result.")
        rel.table(["Part", "Scenarios"] + list(Xc.columns),
                  [[p, int(msk.sum())] + [c.fmt(Xc.loc[msk, m].mean()) for m in Xc.columns]
                   for p, msk in (("all", ho | ~ho), ("dev", ~ho), ("held-out", ho))])
        for no_empty in (False, True):
            pv, line_by = {}, {}
            for b in [m for m in BASELINES if m in X.columns]:
                ss = X[["pipeline", b]].dropna()
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
                rel.p("Without the scenarios with an empty graph in all subsamples:" if no_empty else "All scenarios:")
                rel.table(["Comparison", "Scenarios", "Mean difference", "Better / equal / worse", "Wilcoxon p",
                           "Holm p (5)"], [line_by[b] + [c.fmt(pv[b]), c.fmt(ph[b])] for b in line_by])
    common = A.pivot_table(index="cenario", columns="metodo", values="jaccard_entre_subamostras").dropna().index
    F = A[A.cenario.isin(common)].groupby(["familia", "metodo"]).jaccard_entre_subamostras.mean().unstack()
    F = F[[m for m in methods if m in F.columns]]
    rel.h("## Jaccard between subsamples per family")
    rel.table(["Family"] + list(F.columns), [[f] + [c.fmt(F.loc[f, m]) for m in F.columns] for f in F.index])
    rel.save()
    print(open(os.path.join(rel.dir, "relatorio.md"), encoding="utf-8").read())


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        if len(sys.argv) > 1 and sys.argv[1] == "rodar":
            run()
        else:
            evaluate()

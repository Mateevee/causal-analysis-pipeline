"""
"Safe choice": accuracy regret of the six methods. POST HOC ANALYSIS, labelled as such.

Origin: proposal of 2026-11-20 in NOTAS.md ("[texto] Pergunta de pesquisa como contribuição...", section 4). The
expression "safe choice" may only go into the text if the computation of component 3 is done with the definition
fixed before computing. This docstring IS the definition; the SHA-256 of this file is recorded in NOTAS.md before the
first run (2026-11-21). Post hoc: the F1 values it starts from have already been seen (Table III, R1-2c).

Definition (fixed before computing)
  Methods: the six with the paper's configuration, τ_max = 10: pipeline (patch_tau10), PCMCI, PCMCI+, DYNOTEARS,
    CUTS+, VARLiNGAM.
  Sets, always kept separate: H = the paper's 39 held-out scenarios (F1 from the paper's runs, common.rows()); N = the 86
    of R1-2c (Results_Revisao/r1_2c_benchmark_novo/T1_por_cenario.csv, the official report rerun on the main machine).
  Component 1, coverage: scenarios with a result / total, per method and set.
  Component 2, stability: not recomputed here. It is the Jaccard between subsamples of A8 with the complement
    sub00..sub09 (r2_m2_a8_complemento_sub10), cited in the write-up.
  Component 3, regret: reg_m(s) = max_k F1_k(s) − F1_m(s), with the maximum over the six methods.
    Version A (main): a scenario without a result for a method counts as F1 = 0 for it (enters the maximum as 0).
    Version B: the method without a result is left out of that scenario (maximum over those with a result; the method
      receives no regret there).
    Per-method summaries: mean, median, 90th percentile, maximum, fraction with reg ≤ 0.05, fraction with reg = 0 (it
      is the best), and the worst case per family = largest mean regret among the families of the set.
    Test: regret of the pipeline against that of each baseline, two-sided paired Wilcoxon per scenario, Holm over
      the 5 comparisons, per set and per version.
  Reading, per set (version A decides; B reported alongside):
    - the pipeline has the lowest mean AND the lowest 90th percentile of the six → "lowest regret" in the set;
    - it does not, but no baseline has significantly lower regret (Holm ≥ 0.05) → "regret
      comparable to that of the best fixed method";
    - some baseline has significantly lower regret (Holm < 0.05) → "not the lowest-regret
      choice" in the set, naming the baseline.
  Every result is reported, including those unfavourable to the pipeline.

Usage (from the repository root):  python revision/offline/safe_choice.py
Output: Results_Revisao/escolha_segura/
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "execution")]
import common as c  # noqa: E402

NAME = "escolha_segura"
METHODS = ["pipeline", "pcmci_tigramite", "pcmciplus", "dynotears", "cuts_plus", "varlingam"]
LABELS = {"pipeline": "Pipeline", "pcmci_tigramite": "PCMCI", "pcmciplus": "PCMCI+", "dynotears": "DYNOTEARS",
          "cuts_plus": "CUTS+", "varlingam": "VARLiNGAM"}


def table_H() -> pd.DataFrame:
    rows = c.rows()
    out = []
    for s in c.scenarios("ho"):
        r = {"cenario": s, "familia": c.family(s)}
        for m in METHODS:
            k = c.PIPE if m == "pipeline" else m
            d = rows[s].get(k)
            r[m] = c.f1(d["tp"], d["fp"], d["fn"]) if d else np.nan
        out.append(r)
    return pd.DataFrame(out)


def table_N() -> pd.DataFrame:
    T = pd.read_csv(os.path.join(c.OUTPUT_ROOT, "r1_2c_benchmark_novo", "T1_por_cenario.csv"))
    T = T[T.status == "ok"]
    model_name = {"pipeline": "patch_tau10", **{m: f"{m}_tau10" for m in METHODS[1:]}}
    P = T.pivot_table(index="cenario", columns="model", values="f1", aggfunc="first")
    fam = T.drop_duplicates("cenario").set_index("cenario").familia
    D = pd.DataFrame({"cenario": P.index, "familia": fam.reindex(P.index).values})
    for m in METHODS:
        D[m] = P[model_name[m]].values if model_name[m] in P.columns else np.nan
    return D.reset_index(drop=True)


def regret(D: pd.DataFrame, version: str) -> pd.DataFrame:
    F = D[METHODS].copy()
    if version == "A":
        F = F.fillna(0.0)
    best = F.max(axis=1, skipna=True)
    R = F.rsub(best, axis=0)  # NaN stays NaN in version B
    return R


def summary(R: pd.DataFrame, D: pd.DataFrame) -> list:
    rows = []
    for m in METHODS:
        r = R[m].dropna()
        fam = pd.DataFrame({"familia": D.familia, "r": R[m]}).dropna().groupby("familia").r.mean()
        rows.append([LABELS[m], len(r), c.fmt(r.mean()), c.fmt(r.median()), c.fmt(r.quantile(0.9)), c.fmt(r.max()),
                     f"{(r <= 0.05).mean():.0%}", f"{(r <= 1e-12).mean():.0%}",
                     f"{c.fmt(fam.max())} ({fam.idxmax()})" if len(fam) else "—"])
    return rows


def main():
    rel = c.Report(NAME, "Safe choice — accuracy regret (POST HOC, definition fixed beforehand)")
    rel.p("**Post hoc analysis.** The definition is in the script's docstring; its SHA-256 was recorded in NOTAS.md "
          "before running.")
    readings = {}
    for cj, D in (("H (39 held-out from the paper)", table_H()), ("N (86 from R1-2c)", table_N())):
        rel.h(f"## Set {cj}")
        cov = [[LABELS[m], f"{int(D[m].notna().sum())} of {len(D)}"] for m in METHODS]
        rel.p("Component 1, coverage:")
        rel.table(["Method", "Scenarios with a result"], cov)
        for version in ("A", "B"):
            R = regret(D, version)
            rel.p(f"Component 3, regret, version {version} "
                  f"({'no result = F1 0' if version == 'A' else 'no result is left out'}):")
            rel.table(["Method", "Scenarios", "Mean", "Median", "P90", "Maximum", "≤ 0.05", "Is the best",
                       "Worst family (mean)"], summary(R, D))
            pv, line = {}, {}
            for b in METHODS[1:]:
                ss = R[["pipeline", b]].dropna()
                if len(ss) < 2:
                    continue
                w, e, l = c.wins(-ss["pipeline"], -ss[b])
                pv[b] = c.wilcoxon_p(ss["pipeline"], ss[b])
                line[b] = [f"pipeline × {LABELS[b]}", len(ss), c.fmt((ss["pipeline"] - ss[b]).mean()), f"{w} / {e} / {l}"]
            ph = c.holm(pv)
            rel.table(["Comparison (regret)", "Scenarios", "Mean difference (pipeline − baseline)",
                       "Pipeline lower / equal / higher", "Wilcoxon p", "Holm p (5)"],
                      [line[b] + [c.fmt(pv[b]), c.fmt(ph[b])] for b in line])
            mean_r = R.mean()
            p90 = R.quantile(0.9)
            if version == "A":
                better = [b for b in METHODS[1:] if b in ph and ph[b] < 0.05 and mean_r[b] < mean_r["pipeline"]]
                if mean_r["pipeline"] <= mean_r.min() + 1e-12 and p90["pipeline"] <= p90.min() + 1e-12:
                    readings[cj] = "lowest regret of the six (mean and P90)"
                elif not better:
                    readings[cj] = ("regret comparable to that of the best fixed method (no baseline "
                                    "significantly lower after Holm)")
                else:
                    readings[cj] = ("not the lowest-regret choice: " +
                                    ", ".join(f"{LABELS[b]} (Holm {c.fmt(ph[b])})" for b in better) +
                                    " has significantly lower regret")
        rel.csv(f"f1_{cj.split()[0]}.csv", D)
    rel.h("## Reading (fixed beforehand; version A)")
    for cj, t in readings.items():
        rel.p(f"- **{cj}:** {t}.")
    rel.p("Component 2 (stability): see `r2_m2_a8_complemento_sub10` (Jaccard between subsamples, sub00..sub09).")
    rel.data.update({"leituras": readings})
    rel.save()
    print(open(os.path.join(rel.dir, "relatorio.md"), encoding="utf-8").read())


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        main()

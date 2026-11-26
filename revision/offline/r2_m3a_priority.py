"""
R2-m3a -- Prioritization of the log for analyst review.

Request
  R2: "it is unclear how operators could filter or prioritize these audit
       traces without decision fatigue, and whether an automated risk-scoring
       hierarchy for analyst review is envisaged."

Design (fixed before running; links reported at tau_max = 10)
  Priority score, using only recorded fields (no ground truth):
    entry path           Transfer Entropy = 3, Granger = 2, Pearson = 1,
                         others (lag 0, predictive competition, guard) = 1;
    cascade confidence   high = 3, medium = 2, low = 1;
    tie-break            |t| of the DML.
  Lexicographic order (path, confidence, |t|). Bands for the analyst:
    A = Transfer Entropy and high confidence; B = the others with high or medium confidence;
    C = the rest.
  Reporting: precision per band (pooled over the scenarios), fraction of the links in
  each band; precision@k (k = 5, 10, 20) of the ordered links, mean per
  scenario with at least k links, against the precision of all links of the
  scenario (random order); development and held-out separately.

Usage:  python revision/offline/r2_m3a_priority.py [--verificar]
Output: Results_Revisao/r2_m3a_prioridade/
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c  # noqa: E402

NAME = "r2_m3a_prioridade"
KS = (5, 10, 20)


def entry_path(fm) -> int:
    fm = str(fm).lower()
    if "te_" in fm or "transfer" in fm:
        return 3
    if "granger" in fm:
        return 2
    return 1


def main():
    check = "--verificar" in sys.argv
    rel = c.Report(NAME + ("_verificacao" if check else ""), "R2-m3a — Prioritization of the log")
    scens = c.scenarios()
    if check:
        scens = c.scenarios("dev")[:6] + c.scenarios("ho")[:6]
    L = []
    for s in scens:
        T, _ = c.truth(s)
        d = c.original(s)
        if d.empty:
            continue
        d = d[d["significant"].astype(str) == "True"].copy()
        if d.empty:
            continue
        d["cam"] = d["filter_method"].map(entry_path)
        d["conf"] = d["cascade_confidence"].map({"high": 3, "medium": 2, "low": 1}).fillna(1)
        d["abs_t"] = pd.to_numeric(d["t_stat"], errors="coerce").abs().fillna(0)
        d["par"] = c.pairs(d)
        # one link per pair (the one with the highest priority across lags and properties)
        d = d.sort_values(["cam", "conf", "abs_t"], ascending=False).drop_duplicates("par")
        d["y"] = d["par"].map(lambda p: p in T).astype(int)
        d["faixa"] = np.where((d["cam"] == 3) & (d["conf"] == 3), "A",
                              np.where(d["conf"] >= 2, "B", "C"))
        d["cenario"], d["heldout"] = s, c.is_heldout(s)
        d["ordem"] = range(1, len(d) + 1)
        L.append(d[["cenario", "heldout", "par", "y", "cam", "conf", "abs_t", "faixa", "ordem"]])
    D = pd.concat(L, ignore_index=True)
    rel.csv("links.csv", D.astype({"par": str}))
    for part, msk in (("all", D["heldout"] | ~D["heldout"]), ("development", ~D["heldout"]),
                      ("held-out", D["heldout"])):
        X = D[msk]
        if X.empty:
            continue
        rel.p(f"**{part}** — {len(X)} links reported in {X['cenario'].nunique()} scenarios; "
              f"precision of all: {X['y'].mean():.3f}")
        rel.table(["Band", "Links", "Fraction of links", "Precision"],
                  [[f, int((X["faixa"] == f).sum()), c.fmt((X["faixa"] == f).mean()),
                    c.fmt(X.loc[X["faixa"] == f, "y"].mean())] for f in ("A", "B", "C")])
        body = []
        for k in KS:
            pk, base = [], []
            for s, g in X.groupby("cenario"):
                if len(g) >= k:
                    pk.append(g.nsmallest(k, "ordem")["y"].mean())
                    base.append(g["y"].mean())
            if pk:
                body.append([k, len(pk), c.fmt(np.mean(pk)), c.fmt(np.mean(base))])
        rel.table(["k", "Scenarios with ≥ k links", "Precision@k (priority order)",
                   "Precision of all links (random order)"], body)
        rel.p("")
    rel.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        main()

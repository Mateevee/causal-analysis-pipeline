"""
R1-m6a -- PC algorithm on the no-time-axis data (ALARM and Sachs).

Request
  R1: "The fully nonparametric CMIknn, and PC on no-time-axis data, were not
       evaluated (you say so); I would add them since the claim is
       domain-agnosticism."

Design (fixed before running)
  Data: the same source.csv as the paper's runs (ALARM, 500 samples;
    Sachs, 300 cells), one column per entity.
  PC-stable from causal-learn, alpha = 0.05, with the Fisher-z test on both
    datasets and, on ALARM, whose values are categories, also the chi-square test.
  Output: the CPDAG. An oriented edge counts in its direction; an unoriented one
    counts in both, the paper's rule for unoriented lag 0 (Section IV-C).
  Reporting: F1, precision and recall of PC next to the paper's six methods.

Requirement: pip install causal-learn
Usage:  python revision/execution/r1_m6a_pc.py
Output: Results_Revisao/r1_m6a_pc/
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as c  # noqa: E402

NAME = "r1_m6a_pc"
SEARCH = ("alarm", "sachs")   # current or old folder names (alarm_icu/alarm_uti, sachs_proteins/proteinas)
ALPHA = 0.05


def load_data(scen):
    d = pd.read_csv(os.path.join(c.folder(scen), "source.csv"))
    props = [k for k in d.columns if k not in ("Date", "Name")]
    w = d.pivot_table(index="Date", columns="Name", values=props[0], aggfunc="first").sort_index()
    w = w.loc[:, w.std() > 0].dropna()
    return w


def pc_edges(w, test):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import chisq, fisherz
    X = w.to_numpy(float)
    if test == "chisq":
        X = np.column_stack([pd.factorize(w[k])[0] for k in w.columns]).astype(float)
    g = pc(X, alpha=ALPHA, indep_test={"fisherz": fisherz, "chisq": chisq}[test], stable=True,
           show_progress=False).G.graph
    names = list(w.columns)
    S = set()
    n = len(names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if g[j, i] == 1 and g[i, j] == -1:          # i -> j
                S.add((names[i], names[j]))
            elif g[i, j] == -1 and g[j, i] == -1:       # i - j: both directions
                S.add((names[i], names[j]))
    return S


def main():
    rep = c.Report(NAME, "R1-m6a — PC on the no-time-axis data")
    try:
        import causallearn  # noqa: F401
    except ImportError:
        rep.p("causal-learn is not installed: pip install causal-learn")
        rep.save()
        return
    body = []
    for s in [s for s in c.scenarios() if any(b in s for b in SEARCH)]:
        T, _ = c.truth(s)
        w = load_data(s)
        for test in (["fisherz", "chisq"] if "alarm" in s else ["fisherz"]):
            S = pc_edges(w, test)
            tp, fp, fn = c.counts(S, T)
            body.append([s, f"PC ({test})", len(S), c.fmt(tp / (tp + fp) if tp + fp else 0.0),
                         c.fmt(tp / (tp + fn) if tp + fn else 0.0), c.fmt(c.f1(tp, fp, fn))])
        for m in [c.PIPE] + c.BASELINES:
            r = c.rows()[s].get(m)
            if r:
                body.append([s, c.LABEL[m], r["n_pred"], c.fmt(r["tp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else 0.0),
                             c.fmt(r["tp"] / (r["tp"] + r["fn"]) if r["tp"] + r["fn"] else 0.0),
                             c.fmt(c.f1(r["tp"], r["fp"], r["fn"]))])
    rep.table(["Scenario", "Method", "Reported pairs", "Precision", "Recall", "F1"], body)
    rep.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        main()

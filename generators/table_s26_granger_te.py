"""
Granger and Transfer Entropy alone (Table S26 of the supplement).

Both are ingredients of the pipeline itself (Stage 5 rescue and Stage 4). This
script reads the run Results_GrangerTE/Run_20261012_182259 (main_granger_te.py:
same scenarios, data and GT as the reported run; each test applied alone to
every ordered pair, without screening, lag penalization, DML or Stage 8,
decision alpha = 0.05 with FDR-BH over all pairs, like PCMCI) and compares
the pipeline with each one:

  1. The three methods (F1, precision, recall, F1 = 0, below the floor).
  2. Pipeline against each one, per scenario (sign, Wilcoxon, Holm over the two).
  3. Mean per family.
  4. Mean rank with the two added to the six methods of the paper (eight methods).
  5. Window 24.

Loading, groups and definitions are those of paper_numbers.py. Before using the run,
checks that n_causal and n_universe of each scenario are those of the reported run.
The RCAEval of this run uses the same window (first 500 rows) as the reported run.

Output: Table_S26_GrangerTE.md (tables for the supplement).
"""
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare, rankdata, studentized_range, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_numbers as na

import origins as O  # noqa: E402
GTE = O.require("GRANGER_TE_RUN")   # in the paper, Results_GrangerTE/Run_20261012_182259
OUT = O.out("Table_S26_GrangerTE.md")
P = na.PIPELINE_MODEL
EPS = na.EPS
ING = [("granger", "Granger alone"), ("transfer_entropy", "Transfer Entropy alone")]
LAB = {P: "**Pipeline**", **dict(ING), **dict(na.BASELINES)}
SIX = [P] + [m for m, _ in na.BASELINES]
EIGHT = SIX + [m for m, _ in ING]
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
STAR = chr(92) + "*"
SETS = [("All scenarios", lambda s: True),
        ("Development suite", lambda s: na.group_of(s) not in na.HELDOUT),
        ("Held-out datasets", lambda s: na.group_of(s) in na.HELDOUT)]
FAMILIES = [("Synthetic VAR", "Internal synthetic VAR"), ("tigramite SCM", "tigramite SCMs"),
            ("Logistic maps", "Coupled logistic maps"), ("Lorenz-96", "Lorenz-96"),
            ("CausalTime", "CausalTime"), ("NetSim", "NetSim"),
            ("Real-world", "TEP, ALARM, PeMS-BAY, and Sachs"), ("Walker", "Walker Circulation"),
            ("PetShop", "PetShop"), ("BEELINE", "BEELINE"), ("DREAM3", "DREAM3"), ("RCAEval", "RCAEval"),
            ("EasyVista", "EasyVista"), ("Causal Chambers", "Causal Chambers")]


def pf(p):
    s = f"{p:.2g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m} × 10{str(int(e)).translate(SUP)}"
    return s


def table(header, body):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(x) for x in r) + " |" for r in body] + [""]


def F(mm, m):
    return round(na.f1(mm[m]["tp"], mm[m]["fp"], mm[m]["fn"]), 4)


def wtl(d):
    c = Counter("W" if x > EPS else ("L" if x < -EPS else "T") for x in d)
    return c["W"], c["T"], c["L"]


def pw(d):
    return wilcoxon(d).pvalue if np.any(np.abs(d) > EPS) else 1.0


def floor(mm):
    p = mm[P]["n_causal"] / mm[P]["n_universe"]
    return 2 * p / (1 + p)


def load_window(window):
    """rows of the paper with Granger and TE alone added."""
    na.WINDOW = window
    rows, _ = na.load()
    rows = {s: mm for s, mm in rows.items() if P in mm}
    suf = "_tau10" if window == "10" else ""
    for s, mm in rows.items():
        v = pd.read_csv(os.path.join(GTE, s, "validation_summary.csv")).set_index("model")
        for m, _ in ING:
            r = v.loc[m + suf]
            if r["status"] != "ok":
                raise RuntimeError(f"{s}: {m + suf} with status {r['status']}")
            d = dict(tp=int(r["TP"]), fp=int(r["FP"]), fn=int(r["FN"]), n_causal=int(r["n_causal"]),
                     n_universe=int(r["n_universe"]), n_pred=int(r["n_predicted"]), run="gte:" + m + suf)
            if (d["n_causal"], d["n_universe"]) != (mm[P]["n_causal"], mm[P]["n_universe"]):
                raise RuntimeError(f"{s}: GT or universe different from the reported run's")
            mm[m] = d
    return rows


def t_methods(rows):
    """1. Pipeline, Granger and TE: means and counts."""
    body = []
    for name, keep in SETS:
        sc = [s for s in sorted(rows) if keep(s)]
        for m in [P] + [m for m, _ in ING]:
            c = [rows[s][m] for s in sc]
            f = np.array([F(rows[s], m) for s in sc])
            pr = np.mean([na.prf((x["tp"], x["fp"], x["fn"]))[0] for x in c])
            rc = np.mean([na.prf((x["tp"], x["fp"], x["fn"]))[1] for x in c])
            body.append([name, LAB[m], len(sc), f"{f.mean():.3f}", f"{pr:.3f}", f"{rc:.3f}",
                         int((f == 0).sum()), int(sum(F(rows[s], m) < floor(rows[s]) - EPS for s in sc)),
                         sum(x["tp"] for x in c), sum(x["fp"] for x in c)])
    return table(["Scenarios", "Method", "n", "Mean F1", "Mean precision", "Mean recall", "F1 = 0",
                  "Below the trivial floor", "True positives (total)", "False positives (total)"], body)


def t_one_to_one(rows):
    """2. Pipeline against each ingredient, per scenario; Holm over the two."""
    body = []
    for name, keep in SETS:
        sc = [s for s in sorted(rows) if keep(s)]
        d = {m: np.array([F(rows[s], P) - F(rows[s], m) for s in sc]) for m, _ in ING}
        pv = {m: pw(x) for m, x in d.items()}
        holm, acc = {}, 0.0
        for i, (m, p) in enumerate(sorted(pv.items(), key=lambda kv: kv[1])):
            acc = max(acc, min(1.0, p * (len(pv) - i)))
            holm[m] = acc
        for m, lab in ING:
            w, t, l = wtl(d[m])
            body.append([name, lab, len(sc), f"{w} / {t} / {l}", f"{d[m].mean():+.3f}",
                         pf(binomtest(w, w + l).pvalue), pf(pv[m]), pf(holm[m])])
    return table(["Scenarios", "Against", "n", "Pipeline higher / tied / lower", "Mean F1 difference",
                  "Sign test p", "Wilcoxon p (two-sided)", "Holm-adjusted p (two comparisons)"], body)


def t_family(rows):
    """3. Mean per family."""
    body = []
    for key, lab in FAMILIES:
        sc = [s for s in sorted(rows) if na.family_of(na.group_of(s)) == key]
        cols = [np.array([F(rows[s], m) for s in sc]) for m in [P] + [m for m, _ in ING]]
        top = max(x.mean() for x in cols)
        cells = [f"**{x.mean():.3f}**" if abs(x.mean() - top) < EPS else f"{x.mean():.3f}" for x in cols]
        body.append([lab, len(sc)] + cells +
                    [" / ".join(map(str, wtl(cols[0] - c))) for c in cols[1:]])
    return table(["Family", "Scenarios", "Pipeline, mean F1", "Granger alone, mean F1",
                  "Transfer Entropy alone, mean F1", "Pipeline higher / tied / lower against Granger",
                  "Pipeline higher / tied / lower against Transfer Entropy"], body)


def block(M):
    rk = np.array([rankdata(-r) for r in M]).mean(axis=0)
    k, n = M.shape[1], M.shape[0]
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    return rk, n, float(friedmanchisquare(*M.T).pvalue), float(cd)


def t_rank(rows):
    """4. Mean rank of the eight methods (the six of the paper, Granger and TE)."""
    res = []
    for name, keep in SETS:
        sc = [s for s in sorted(rows) if keep(s) and all(m in rows[s] for m in EIGHT)]
        res.append((name, block(np.array([[F(rows[s], m) for m in EIGHT] for s in sc]))))
    body = []
    for i, m in enumerate(EIGHT):
        row = [LAB[m]]
        for _, (rk, _, _, cd) in res:
            # bold on the lowest rank of the column; * = behind the pipeline by more than the CD
            x = f"{rk[i]:.2f}" + (STAR if m != P and rk[i] - rk[0] > cd else "")
            row.append(f"**{x}**" if rk[i] <= rk.min() + EPS else x)
        body.append(row)
    body.append(["Units (n)"] + [b[1] for _, b in res])
    body.append(["Friedman p"] + [pf(b[2]) for _, b in res])
    body.append(["Nemenyi critical difference (5%)"] + [f"{b[3]:.2f}" for _, b in res])
    return table(["Method"] + [name for name, _ in res], body)


PARTS = [("The pipeline and its two first detectors run alone", t_methods),
         ("Pipeline against each detector run alone, by scenario", t_one_to_one),
         ("Means by dataset family", t_family),
         ("Mean rank with the two detectors added to the six methods", t_rank)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = ["# Table S26 -- generated by generators/table_s26_granger_te.py (paste into the supplement)", ""]
    rows = load_window("10")
    print(f"window 10: {len(rows)} scenarios; GT and universe checked against the reported run")
    for i, (title, fn) in enumerate(PARTS, 1):
        out += [f"### S26.{i} — {title} (τ_max = 10)", ""] + fn(rows)
    rows = load_window("24")
    print(f"window 24: {len(rows)} scenarios; GT and universe checked against the reported run")
    out += [f"### S26.{len(PARTS) + 1} — The same comparisons at τ_max = 24", ""]
    for title, fn in PARTS:
        out += [f"**{title}**", ""] + fn(rows)
    # typographic minus sign, as in the paper's tables
    txt = re.sub(r"(?<![\w.])-(?=\d)", "−", "\n".join(out))
    open(OUT, "w", encoding="utf-8").write(O.canonical(txt))
    print(txt)


if __name__ == "__main__":
    main()

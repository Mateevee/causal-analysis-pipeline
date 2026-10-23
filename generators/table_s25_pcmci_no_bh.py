"""
PCMCI without the Benjamini-Hochberg correction (Table S22 of the supplement).

The paper's PCMCI runs with BH over the p-value matrix, an option of the method
that is not the library's default (Section 4.3, Table S6.0). This script reads the run
Results_PCMCI_NoBH/Run_20261011_202543 (main_pcmci_nobh.py: same scenarios,
data and GT as the reported run; the only change is pcmci_correction = 'none', that is,
raw MCI p-value < 0.05) and redoes, with PCMCI without the correction in place of
PCMCI with it, the comparisons of the main text in which PCMCI takes part:

  1. PCMCI itself with and without the correction (F1, precision, recall, F1 = 0).
  2. Pipeline against PCMCI, per scenario (sign, Wilcoxon, Holm).
  3. Mean rank of the six methods (Friedman, Nemenyi), as in section 3g of
     paper_numbers.
  4. Shortfall to the best of the six and pipeline against the best of the five
     baselines (sections 3e and 3f).
  5. Mean per family.
  6. Window 24.

Loading, groups and definitions are those of paper_numbers.py, so that the column
"with the correction" is the paper's. Before using the run, checks that n_causal and
n_universe of each scenario are those of the reported run.

Output: Table_S22_PCMCI_NoBH.md (tables for the supplement).
"""
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare, rankdata, studentized_range, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_numbers as na

import origins as O  # noqa: E402
NOBH = O.require("PCMCI_NOBH_RUN")   # in the paper, Results_PCMCI_NoBH/Run_20261011_202543
OUT = O.out("Table_S22_PCMCI_NoBH.md")
P, PC = na.PIPELINE_MODEL, "pcmci_tigramite"
EPS = na.EPS
SIX = [P] + [m for m, _ in na.BASELINES]
FIVE = [m for m in SIX if m != "varlingam"]
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
STAR = chr(92) + "*"
SETS = [("All scenarios", lambda s: True),
        ("Development suite", lambda s: na.group_of(s) not in na.HELDOUT),
        ("Held-out datasets", lambda s: na.group_of(s) in na.HELDOUT)]
SETTINGS = ["With the correction", "Without the correction"]
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


def load_window(window):
    """rows of the paper and the variant with PCMCI without the correction in its place."""
    na.WINDOW = window
    rows, _ = na.load()
    rows = {s: mm for s, mm in rows.items() if P in mm}
    name = "pcmci_tigramite_nobh" + ("_tau10" if window == "10" else "")
    alt = {}
    for s, mm in rows.items():
        r = pd.read_csv(os.path.join(NOBH, s, "validation_summary.csv")).set_index("model").loc[name]
        if r["status"] != "ok":
            raise RuntimeError(f"{s}: {name} with status {r['status']}")
        d = dict(tp=int(r["TP"]), fp=int(r["FP"]), fn=int(r["FN"]), n_causal=int(r["n_causal"]),
                 n_universe=int(r["n_universe"]), n_pred=int(r["n_predicted"]), run="nobh:" + name)
        if (d["n_causal"], d["n_universe"]) != (mm[PC]["n_causal"], mm[PC]["n_universe"]):
            raise RuntimeError(f"{s}: GT or universe different from the reported run's")
        alt[s] = {**mm, PC: d}
    return rows, alt


def floor(mm):
    p = mm[P]["n_causal"] / mm[P]["n_universe"]
    return 2 * p / (1 + p)


def t_pcmci(rows, alt):
    """1. PCMCI with and without the correction."""
    body, cmpb = [], []
    for name, keep in SETS:
        sc = [s for s in sorted(rows) if keep(s)]
        for lab, R in zip(SETTINGS, (rows, alt)):
            c = [R[s][PC] for s in sc]
            f = np.array([F(R[s], PC) for s in sc])
            pr = np.mean([na.prf((x["tp"], x["fp"], x["fn"]))[0] for x in c])
            rc = np.mean([na.prf((x["tp"], x["fp"], x["fn"]))[1] for x in c])
            body.append([name, lab, len(sc), f"{f.mean():.3f}", f"{pr:.3f}", f"{rc:.3f}", int((f == 0).sum()),
                         sum(1 for x in c if x["n_pred"] == 0),
                         int(sum(F(R[s], PC) < floor(R[s]) - EPS for s in sc)),
                         sum(x["tp"] for x in c), sum(x["fp"] for x in c)])
        d = np.array([F(alt[s], PC) - F(rows[s], PC) for s in sc])
        w, t, l = wtl(d)
        cmpb.append([name, len(sc), w, t, l, f"{d.mean():+.3f}", pf(pw(d))])
    out = table(["Scenarios", "PCMCI", "n", "Mean F1", "Mean precision", "Mean recall", "F1 = 0",
                 "No pair declared", "Below the trivial floor", "True positives (total)",
                 "False positives (total)"], body)
    out += ["*PCMCI without the correction against PCMCI with it, by scenario:*", ""]
    out += table(["Scenarios", "n", "F1 higher without", "F1 equal", "F1 lower without", "Mean F1 difference",
                  "Wilcoxon p (two-sided)"], cmpb)
    return out


def t_one_to_one(rows, alt):
    """2. Pipeline against PCMCI, per scenario; Holm over the five baselines."""
    body = []
    for name, keep in SETS:
        sc = [s for s in sorted(rows) if keep(s)]
        for lab, R in zip(SETTINGS, (rows, alt)):
            pv = {}
            for m, _ in na.BASELINES:
                pv[m] = pw(np.array([F(R[s], P) - F(R[s], m) for s in sc if m in R[s]]))
            holm, acc = {}, 0.0
            for i, (m, p) in enumerate(sorted(pv.items(), key=lambda kv: kv[1])):
                acc = max(acc, min(1.0, p * (len(pv) - i)))
                holm[m] = acc
            d = np.array([F(R[s], P) - F(R[s], PC) for s in sc])
            w, t, l = wtl(d)
            body.append([name, lab, len(sc), f"{w} / {t} / {l}", f"{d.mean():+.3f}",
                         pf(binomtest(w, w + l).pvalue), pf(pv[PC]), pf(holm[PC])])
    return table(["Scenarios", "PCMCI", "n", "Pipeline higher / tied / lower", "Mean F1 difference",
                  "Sign test p", "Wilcoxon p (two-sided)", "Holm-adjusted p (five baselines)"], body)


def block(M):
    rk = np.array([rankdata(-r) for r in M]).mean(axis=0)
    k, n = M.shape[1], M.shape[0]
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    return rk, n, float(friedmanchisquare(*M.T).pvalue), float(cd)


def rank_cols(cols):
    """cols: list of (label, methods, M). One column per block."""
    res = [(lab, ms, block(M)) for lab, ms, M in cols]
    body = []
    for m in SIX:
        row = ["**Pipeline**" if m == P else na.LABEL[m]]
        for _, ms, (rk, _, _, cd) in res:
            if m not in ms:
                row.append("—")
                continue
            x = rk[ms.index(m)]
            row.append(f"**{x:.2f}**" if m == P else f"{x:.2f}" + (STAR if x - rk[ms.index(P)] > cd else ""))
        body.append(row)
    body.append(["Units (n)"] + [b[1] for _, _, b in res])
    body.append(["Friedman p"] + [pf(b[2]) for _, _, b in res])
    body.append(["Nemenyi critical difference (5%)"] + [f"{b[3]:.2f}" for _, _, b in res])
    return table(["Method"] + [lab for lab, _, _ in res], body)


def t_rank(rows, alt):
    """3. Mean rank of the six methods with PCMCI in each setting."""
    cols = []
    for name, keep in SETS:
        sc = [s for s in sorted(rows) if keep(s) and all(m in rows[s] for m in SIX)]
        for lab, R in zip(("with", "without"), (rows, alt)):
            cols.append((f"{name}, {lab}", SIX, np.array([[F(R[s], m) for m in SIX] for s in sc])))
    out = rank_cols(cols)
    cols = []
    sc = sorted(rows)
    with6 = [s for s in sc if all(m in rows[s] for m in SIX)]
    for lab, R in zip(("with", "without"), (rows, alt)):
        cols.append((f"Scenarios, five methods, {lab}", FIVE, np.array([[F(R[s], m) for m in FIVE] for s in sc])))
    for lab, R in zip(("with", "without"), (rows, alt)):
        byg = defaultdict(list)
        for s in with6:
            byg[na.group_of(s)].append([F(R[s], m) for m in SIX])
        cols.append((f"Scenario groups, six methods, {lab}", SIX, np.array([np.mean(v, axis=0) for v in byg.values()])))
    out += ["*All scenarios, with the other two units of Table S8.2 (five methods on every scenario; "
            "scenario groups):*", ""]
    return out + rank_cols(cols)


def boot_best5(R):
    """Interval of the difference to the best of the five baselines, with the
    scenario group as the unit. Repeats section 3f of paper_numbers -- same
    seed and same order of draws (subsets x five baselines and the best of
    five) -- so that the interval "with the correction" is the paper's and
    "without the correction" uses the same draws."""
    rng = np.random.default_rng(0)
    out = {}
    for name, keep in SETS:
        for m in [b for b, _ in na.BASELINES] + ["best5"]:
            byg = defaultdict(list)
            for s, mm in sorted(R.items()):
                if not keep(s) or (m != "best5" and m not in mm):
                    continue
                ref = max(F(mm, b) for b, _ in na.BASELINES if b in mm) if m == "best5" else F(mm, m)
                byg[na.group_of(s)].append(F(mm, P) - ref)
            groups = list(byg)
            boots = [np.concatenate([byg[g] for g in rng.choice(groups, len(groups))]).mean() for _ in range(5000)]
            if m == "best5":
                out[name] = np.percentile(boots, [2.5, 97.5])
    return out


def t_shortfall(rows, alt):
    """4. Shortfall to the best of the six; pipeline against the best of the five baselines."""
    body, best = [], []
    ci = {lab: boot_best5(R) for lab, R in zip(SETTINGS, (rows, alt))}
    for name, keep in SETS:
        sc = [s for s in sorted(rows) if keep(s)]
        for lab, R in zip(SETTINGS, (rows, alt)):
            top = np.array([max(F(R[s], m) for m in SIX if m in R[s]) for s in sc])
            for m in (P, PC):
                g = top - np.array([F(R[s], m) for s in sc])
                body.append([name, lab, "**Pipeline**" if m == P else "PCMCI", len(sc), f"{g.mean():.3f}",
                             f"{g.max():.3f}", int((g <= 0.05 + EPS).sum()), int((g <= 0.10 + EPS).sum()),
                             int((g > 0.20 + EPS).sum()), int((g <= EPS).sum())])
            d = {s: F(R[s], P) - max(F(R[s], m) for m, _ in na.BASELINES if m in R[s]) for s in sc}
            lo, hi = ci[lab][name]
            w, t, l = wtl(d.values())
            best.append([name, lab, len(sc), f"{w} / {t} / {l}", f"{np.mean(list(d.values())):+.3f}",
                         f"[{lo:+.3f}, {hi:+.3f}]"])
    out = table(["Scenarios", "PCMCI", "Method", "n", "Mean shortfall", "Largest shortfall",
                 "Within 0.05 of the best", "Within 0.10 of the best", "More than 0.20 behind",
                 "Best or tied"], body)
    out += ["*Pipeline against the best of the five baselines in each scenario:*", ""]
    out += table(["Scenarios", "PCMCI", "n", "Pipeline higher / tied / lower", "Mean F1 difference",
                  "95% interval (bootstrap over scenario groups)"], best)
    return out


def t_family(rows, alt):
    """5. Mean per family."""
    body = []
    for key, lab in FAMILIES:
        sc = [s for s in sorted(rows) if na.family_of(na.group_of(s)) == key]
        a = np.array([F(rows[s], P) for s in sc])
        b = np.array([F(rows[s], PC) for s in sc])
        c = np.array([F(alt[s], PC) for s in sc])
        w, t, l = wtl(a - c)
        cells = [f"{x.mean():.3f}" for x in (a, b, c)]
        top = max(x.mean() for x in (a, b, c))
        cells = [f"**{v}**" if abs(x.mean() - top) < EPS else v for v, x in zip(cells, (a, b, c))]
        body.append([lab, len(sc)] + cells + [int((b == 0).sum()), int((c == 0).sum()), f"{w} / {t} / {l}"])
    return table(["Family", "Scenarios", "Pipeline, mean F1", "PCMCI with the correction, mean F1",
                  "PCMCI without the correction, mean F1", "PCMCI at F1 = 0, with", "PCMCI at F1 = 0, without",
                  "Pipeline higher / tied / lower against PCMCI without the correction"], body)


PARTS = [("PCMCI with and without the correction", t_pcmci),
         ("Pipeline against PCMCI, by scenario", t_one_to_one),
         ("Mean rank of the six methods", t_rank),
         ("Shortfall to the best of the six, and pipeline against the best baseline of each scenario", t_shortfall),
         ("Means by dataset family", t_family)]


def main():
    out = ["# Table S22 -- generated by generators/table_s25_pcmci_no_bh.py (paste into the supplement)", ""]
    rows, alt = load_window("10")
    print(f"window 10: {len(rows)} scenarios; GT and universe checked against the reported run")
    for i, (title, fn) in enumerate(PARTS, 1):
        out += [f"### S22.{i} — {title} (τ_max = 10)", ""] + fn(rows, alt)
    rows, alt = load_window("24")
    print(f"window 24: {len(rows)} scenarios; GT and universe checked against the reported run")
    out += [f"### S22.{len(PARTS) + 1} — The same comparisons at τ_max = 24", ""]
    for title, fn in PARTS:
        out += [f"**{title}**", ""] + fn(rows, alt)
    # typographic minus sign, as in the paper's tables
    txt = re.sub(r"(?<![\w.])-(?=\d)", "−", "\n".join(out))
    open(OUT, "w", encoding="utf-8").write(O.canonical(txt))
    print(txt)


if __name__ == "__main__":
    main()

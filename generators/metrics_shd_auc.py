"""
SHD, AUROC and AUPRC of the six methods, from the run METRICS_RUN in
origins.py (in the paper, Results/Run_20261011_132346; auroc_auprc_shd_per_scenario.csv, with
the definitions in auroc_auprc_shd_summary.txt). The run repeats the reported one
and adds the three metrics; this script checks both things before using the
numbers:

  0. Checks: TP/FP/FN of the run against the reported ones (paper_numbers.load,
     both windows); internal identities of the table; independent recomputation
     of AUROC, AUPRC and SHD from the run's files (window 10).
  1. Mean per method (all, development, held-out), window 10.
  2. Mean rank per metric (Friedman, Nemenyi), as in section 3g of
     paper_numbers: five methods on every scenario and six where all six run.
  3. Pipeline against each baseline, per scenario (Wilcoxon, Holm).
  4. SHD against the empty graph (the trivial reference for SHD).
  5. Mean per family.
  6. AUPRC of the pipeline per group (Section 5.3, Tables S23.3 and S16.3).
  7. Window 24.

Definitions (those of the run's summary; unit = ordered pair of entities,
like the paper's F1):
  score   pipeline: max |penalized_effect| of the pair in original.csv (0 if the
          pair was never scored); baselines: two tiers -- pair in the final graph
          (results.csv) above pair only in original.csv, above pair never
          scored -- and, within the tier, 1 - smallest p-value.
  *_two_tier  the two-tier score for every method, the pipeline included.
  shd         UNORDERED entity pairs whose link (none, one direction, the
              other, both) differs from the true one: one per pair.
  shd_double  ordered pairs that differ = FP + FN.
  shd_norm    shd_double / number of ordered pairs evaluated.

Output: Metrics_SHD_AUC.md (report) and Table_S7_Metrics.md (tables for the
supplement).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_numbers as na

import origins as O  # noqa: E402
RUN = O.require("METRICS_RUN")
CSV = os.path.join(RUN, "auroc_auprc_shd_per_scenario.csv")
OUT = os.path.join(na.OUT_DIR, "Metrics_SHD_AUC.md")
OUT_S = os.path.join(na.OUT_DIR, "Table_S7_Metrics.md")
EPS = na.EPS
CANON = [na.PIPELINE_MODEL] + [m for m, _ in na.BASELINES]
LABEL = dict([(na.PIPELINE_MODEL, "Pipeline")] + na.BASELINES)
NAME = {"10": {c: ("patch_tau10" if c == na.PIPELINE_MODEL else c + "_tau10") for c in CANON},
        "24": {c: c for c in CANON}}
# (column, report label, supplement label, higher is better, decimals)
METRICS = [("auprc", "AUPRC", "AUPRC", True, 3), ("auprc_two_tier", "AUPRC (two tiers)", "AUPRC (two-tier score)", True, 3),
           ("auroc", "AUROC", "AUROC", True, 3), ("auroc_two_tier", "AUROC (two tiers)", "AUROC (two-tier score)", True, 3),
           ("shd", "SHD", "SHD", False, 1), ("shd_double", "SHD (ordered pairs)", "SHD (ordered pairs)", False, 1)]
MEANS = [("f1", "F1", 3), ("auroc", "AUROC", 3), ("auprc", "AUPRC", 3), ("auroc_two_tier", "AUROC (two-tier)", 3),
         ("auprc_two_tier", "AUPRC (two-tier)", 3), ("auroc_pvalue_only", "AUROC (p-value only)", 3),
         ("auprc_pvalue_only", "AUPRC (p-value only)", 3), ("shd", "SHD", 1), ("shd_double", "SHD (ordered pairs)", 1),
         ("shd_norm", "Normalized SHD", 3)]
# The p-value-only ranking stays in the report and out of the supplement: it is not the
# pipeline's score (penalized effect), and the like-for-like comparison is the two-tier one.
MEANS_S = [m for m in MEANS if "pvalue_only" not in m[0]]
short = lambda s: s.replace("benchmark_", "").replace("real_", "")
held = lambda s: na.group_of(s) in na.HELDOUT


def table(header, body):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(x) for x in r) + " |" for r in body]


def recompute(scen, model):
    """AUROC, AUPRC (the method's score and the two-tier one) and both SHDs, from the run's files."""
    def rd(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    key = lambda df: df[["source_entity", "target_entity"]].astype(str).apply(tuple, axis=1)
    gt = pd.read_csv(os.path.join(RUN, scen, "ground_truth.csv"))
    gt["pos"] = gt["causal_gt"].astype(str).str.lower().isin(["true", "1"])
    truth = gt.groupby(["source_entity", "target_entity"])["pos"].any()
    truth.index = [tuple(map(str, k)) for k in truth.index]
    o = rd(os.path.join(RUN, scen, model, "original.csv"))
    r = rd(os.path.join(RUN, scen, model, "results.csv"))
    pred = set(key(r[r["significant"].astype(str) == "True"])) if len(r) else set()
    cand = set(key(o)) if len(o) else set()
    univ = list(truth.index) + sorted(pred - set(truth.index))
    y = np.array([bool(truth.get(k, False)) for k in univ]).astype(int)
    pv = pd.to_numeric(o["pvalue"], errors="coerce").groupby(key(o)).min() if len(o) else pd.Series(dtype=float)
    p_of = lambda k: 1.0 if pd.isna(pv.get(k, 1.0)) else pv.get(k, 1.0)
    tt = np.array([(2 if k in pred else 1) + (1 - p_of(k)) * 0.999 if (k in pred or k in cand) else 0.0 for k in univ])
    if model.startswith("patch"):
        pe = pd.to_numeric(o["penalized_effect"], errors="coerce").abs().groupby(key(o)).max() if len(o) else pd.Series(dtype=float)
        s = np.array([0.0 if pd.isna(pe.get(k, 0.0)) else pe.get(k, 0.0) for k in univ])
    else:
        s = tt
    T = {k for k in truth.index if truth[k]}
    link = lambda S, u: tuple((a, b) in S for a, b in ((u[0], u[-1]), (u[-1], u[0])))
    und = {tuple(sorted(set(k))) for k in (T | pred)}
    return dict(auprc=average_precision_score(y, s), auroc=roc_auc_score(y, s),
                auprc_two_tier=average_precision_score(y, tt), auroc_two_tier=roc_auc_score(y, tt),
                shd=sum(link(T, u) != link(pred, u) for u in und), shd_double=len(pred - T) + len(T - pred))


def main():
    D = pd.read_csv(CSV)
    out = ["# SHD, AUROC and AUPRC -- generated by generators/metrics_shd_auc.py", "",
           f"Run: `{os.path.basename(RUN)}`; {D.scenario.nunique()} scenarios, {D.model.nunique()} models, {len(D)} rows."]
    sup = ["# Table S7 -- generated by generators/metrics_shd_auc.py (paste into the supplement)", ""]

    # ------------------------------------------------------------------ 0
    out += ["", "## 0. Checks", ""]
    for win in ("10", "24"):
        na.WINDOW = win
        rows, _ = na.load()
        n, diff, outside = 0, [], set()
        for c in CANON:
            d = D[D.model == NAME[win][c]].set_index("scenario")
            for s, mm in rows.items():
                if c not in mm:
                    continue
                if s not in d.index:
                    outside.add(s)
                    continue
                n += 1
                a = (mm[c]["tp"], mm[c]["fp"], mm[c]["fn"])
                b = tuple(int(d.loc[s, k]) for k in ("TP", "FP", "FN"))
                if a != b:
                    diff.append(f"{short(s)} / {LABEL[c]}: reported {a}, run {b}")
        out.append(f"- Window {win}: {n} (scenario, method) pairs compared with the reported ones; TP/FP/FN differ on {len(diff)}"
                   + (": " + "; ".join(diff) if diff else "") + f". Scenarios of the paper not in the run ({len(outside)}): "
                   + (", ".join(short(s) for s in sorted(outside)) or "none") + ".")
    ident = [("shd_double = FP + FN", int((D.shd_double != D.FP + D.FN).sum())),
             ("shd <= shd_double", int((D.shd > D.shd_double).sum())),
             ("shd_norm = shd_double / n_universe", int((abs(D.shd_norm - D.shd_double / D.n_universe) > 6e-5).sum())),
             ("TP + FN = n_causal", int((D.TP + D.FN != D.n_causal).sum())),
             ("held-out scope = the paper's HELDOUT", int(((D.scope == "held_out") != D.scenario.map(held)).sum()))]
    out.append("- Internal identities (rows that violate them): " + "; ".join(f"{k}: {v}" for k, v in ident) + ".")
    d10 = D[D.model.isin(NAME["10"].values())]
    bad = {k: 0 for k in ("auprc", "auroc", "auprc_two_tier", "auroc_two_tier", "shd", "shd_double")}
    for r in d10.itertuples():
        v = recompute(r.scenario, r.model)
        for k in bad:
            bad[k] += abs(v[k] - getattr(r, k)) > (6e-5 if k.startswith("au") else 0)
    out.append(f"- Independent recomputation (window 10, {len(d10)} rows), rows that diverge from the table: "
               + "; ".join(f"{k}: {v}" for k, v in bad.items()) + ".")

    # wide tables per metric, window 10
    inv = {v: k for k, v in NAME["10"].items()}
    d10 = d10.assign(canon=d10.model.map(inv))
    W = {col: d10.pivot(index="scenario", columns="canon", values=col)[CANON]
         for col in set([m[0] for m in METRICS] + [m[0] for m in MEANS] + ["n_causal", "n_predicted", "prevalence"])}
    scens = list(W["f1"].index)
    SETS = [("All", "All scenarios", scens), ("Development", "Development suite", [s for s in scens if not held(s)]),
            ("Held-out", "Held-out datasets", [s for s in scens if held(s)])]
    five = [c for c in CANON if c != "varlingam"]

    # ------------------------------------------------------------------ 1
    out += ["", "## 1. Mean per method (window 10)"]
    sup += ["### S7.1 — Mean of each metric per method (τ_max = 10)", ""]
    for rep, sp, sc in SETS:
        body = []
        for c in CANON:
            n = int(W["f1"].loc[sc, c].notna().sum())
            body.append([LABEL[c], n] + [f"{W[col].loc[sc, c].mean():.{dec}f}" for col, _, dec in MEANS])
        head = ["Method", "Scenarios"] + [lab for _, lab, _ in MEANS]
        out += ["", f"### {rep} ({len(sc)} scenarios)", ""] + table(head, body)
        keep = [0, 1] + [2 + i for i, m in enumerate(MEANS) if m in MEANS_S]
        sup += [f"**{sp} ({len(sc)} scenarios)**", ""] + \
            table([head[i] for i in keep], [[r[i] for i in keep] for r in body]) + [""]

    # ------------------------------------------------------------------ 2
    def block(w, hib):
        rk = np.array([rankdata(-r if hib else r) for r in w.values]).mean(axis=0)
        k = w.shape[1]
        cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * len(w)))
        return rk, cd, friedmanchisquare(*w.values.T).pvalue

    out += ["", "## 2. Mean rank per metric (window 10)", "",
            "Rank 1 = best; * = rank worse than the pipeline's by more than the Nemenyi CD at 5%; "
            "† = rank better than the pipeline's by more than the CD."]
    sup += ["### S7.2 — Mean rank of the six methods under each metric (τ_max = 10)", ""]
    summary = {}
    for col, rep, sp, hib, _ in METRICS:
        cols, lines, pe = [], {c: [] for c in CANON}, []
        for srep, ssp, sc in SETS:
            for methods, tag in ((five, "five methods"), (CANON, "six methods")):
                w = W[col].loc[sc, methods].dropna()
                rk, cd, p = block(w, hib)
                cols.append(f"{ssp}, {tag} (n = {len(w)})")
                pe.append((cd, p))
                for c in CANON:
                    if c not in methods:
                        lines[c].append("—")
                        continue
                    x = rk[methods.index(c)]
                    diff = x - rk[0]
                    lines[c].append(f"{x:.2f}" + ("" if c == CANON[0] else " *" if diff > cd else " †" if -diff > cd else ""))
                summary[(col, ssp, tag)] = dict(rank=dict(zip(methods, rk)), cd=cd, n=len(w))
        body = [[LABEL[c]] + lines[c] for c in CANON]
        body.append(["Nemenyi critical difference (5%)"] + [f"{cd:.2f}" for cd, _ in pe])
        body.append(["Friedman p"] + [f"{p:.2g}" for _, p in pe])
        out += ["", f"### {rep}", ""] + table(["Method"] + cols, body)
        sup += [f"**{sp}** ({'higher' if hib else 'lower'} is better)", ""] + table(["Method"] + cols, body) + [""]
    sup += ["\\* mean rank worse than the pipeline's by more than the critical difference; † better than the pipeline's by more than it.", ""]

    # ------------------------------------------------------------------ 3
    out += ["", "## 3. Pipeline against each baseline, per scenario (window 10)", "",
            "better / equal / worse for the pipeline; two-sided Wilcoxon p with Holm over the five baselines."]
    body = []
    for col, rep, sp, hib, _ in METRICS:
        cell = {c: [] for c in CANON[1:]}
        for _, _, sc in SETS:
            pv, wtl = {}, {}
            for c in CANON[1:]:
                x = ((W[col].loc[sc, CANON[0]] - W[col].loc[sc, c]) * (1 if hib else -1)).dropna()
                wtl[c] = (int((x > EPS).sum()), int((abs(x) <= EPS).sum()), int((x < -EPS).sum()))
                pv[c] = wilcoxon(x).pvalue if (abs(x) > EPS).any() else 1.0
            acc = 0.0
            for i, (c, p) in enumerate(sorted(pv.items(), key=lambda kv: kv[1])):
                acc = max(acc, min(1.0, p * (len(pv) - i)))
                cell[c].append(f"{wtl[c][0]} / {wtl[c][1]} / {wtl[c][2]} ({acc:.2g})")
        for c in CANON[1:]:
            body.append([sp, LABEL[c]] + cell[c])
    head = ["Metric", "Against"] + [f"{ssp} — pipeline better / equal / worse (Holm p)" for _, ssp, _ in SETS]
    out += [""] + table(head, body)
    sup += ["### S7.3 — Pipeline against each baseline, by scenario (τ_max = 10)", "",
            "Two-sided paired Wilcoxon signed-rank test, Holm-adjusted over the five baselines within each metric and set.", ""] \
        + table(head, body) + [""]

    # ------------------------------------------------------------------ 4
    out += ["", "## 4. SHD against the empty graph (window 10)", "",
            "The empty graph has SHD (ordered pairs) = number of true pairs; a method falls below it "
            "if and only if FP < TP, that is, precision above 0.5.", ""]
    body = []
    for c in CANON:
        row = [LABEL[c]]
        for _, _, sc in SETS:
            m = W["shd_double"].loc[sc, c].notna()
            ss = [s for s in sc if m[s]]
            below = int((W["shd_double"].loc[ss, c] < W["n_causal"].loc[ss, c]).sum())
            row += [f"{below} / {len(ss)}", f"{W['n_predicted'].loc[ss, c].mean():.1f}"]
        body.append(row)
    row = ["True pairs (mean)"]
    for _, _, sc in SETS:
        row += ["—", f"{W['n_causal'].loc[sc, CANON[0]].mean():.1f}"]
    body.append(row)
    head = ["Method"] + [x for _, ssp, _ in SETS for x in (f"{ssp}: scenarios with SHD below the empty graph's",
                                                           f"{ssp}: mean number of pairs reported")]
    out += table(head, body)
    out += ["", "Mean normalized SHD of the empty graph (= mean prevalence): " +
            "; ".join(f"{srep} {W['prevalence'].loc[sc, CANON[0]].mean():.3f}" for srep, _, sc in SETS) + "."]
    sup += ["### S7.4 — SHD against the empty graph (τ_max = 10)", ""] + table(head, body) + \
        ["", "Mean normalized SHD of the empty graph (the mean prevalence): " +
         "; ".join(f"{ssp.lower()} {W['prevalence'].loc[sc, CANON[0]].mean():.3f}" for _, ssp, sc in SETS) + ".", ""]

    # ------------------------------------------------------------------ 5
    fam = d10.drop_duplicates("scenario").set_index("scenario")["family"]
    out += ["", "## 5. Mean per family (window 10)"]
    sup += ["### S7.5 — Means by dataset family (τ_max = 10)", ""]
    order = sorted(fam.unique(), key=lambda f: (held(fam[fam == f].index[0]), f))
    for col, lab, dec in (("auprc", "AUPRC", 3), ("auroc", "AUROC", 3), ("shd", "SHD", 1), ("shd_norm", "Normalized SHD", 3)):
        body = []
        for f in order:
            sc = list(fam[fam == f].index)
            body.append([f, "held-out" if held(sc[0]) else "development", len(sc), f"{W['prevalence'].loc[sc, CANON[0]].mean():.3f}"]
                        + [("—" if W[col].loc[sc, c].notna().sum() == 0 else f"{W[col].loc[sc, c].mean():.{dec}f}") for c in CANON])
        head = ["Family", "Set", "Scenarios", "Prevalence"] + [LABEL[c] for c in CANON]
        out += ["", f"### {lab}", ""] + table(head, body)
        sup += [f"**{lab}**", ""] + table(head, body) + [""]

    # ------------------------------------------------------------------ 6
    out += ["", "## 6. AUPRC of the pipeline per group (Section 5.3; Tables S23.3 and S16.3)", "",
            "Mean over the group's realizations; chance = prevalence. Score: max |penalized_effect| of the pair.", ""]
    g_of = D.drop_duplicates("scenario").set_index("scenario").index.to_series().map(na.group_of)
    body = []
    for g in [g for g in na.ALL_GROUPS if (g_of == g).any()]:
        sc = list(g_of[g_of == g].index)
        row = [g, len(sc), f"{D[D.scenario.isin(sc)].drop_duplicates('scenario').prevalence.mean():.3f}"]
        for win in ("10", "24"):
            x = D[(D.model == NAME[win][CANON[0]]) & D.scenario.isin(sc)]
            row += [f"{x.auprc.mean():.3f}", f"{x.auroc.mean():.3f}"]
        body.append(row)
    out += table(["Group", "Realizations", "Prevalence", "AUPRC (window 10)", "AUROC (window 10)", "AUPRC (window 24)",
                  "AUROC (window 24)"], body)

    # ------------------------------------------------------------------ 7
    d24 = D[D.model.isin(NAME["24"].values())]
    out += ["", "## 7. Mean per method (window 24)"]
    sup += ["### S7.6 — Mean of each metric per method at τ_max = 24", ""]
    for rep, sp, sc in SETS:
        body = []
        for c in CANON:
            x = d24[(d24.model == NAME["24"][c]) & d24.scenario.isin(sc)]
            body.append([LABEL[c], len(x)] + [f"{x[col].mean():.{dec}f}" for col, _, dec in MEANS])
        head = ["Method", "Scenarios"] + [lab for _, lab, _ in MEANS]
        out += ["", f"### {rep} ({len(sc)} scenarios)", ""] + table(head, body)
        keep = [0, 1] + [2 + i for i, m in enumerate(MEANS) if m in MEANS_S]
        sup += [f"**{sp} ({len(sc)} scenarios)**", ""] + \
            table([head[i] for i in keep], [[r[i] for i in keep] for r in body]) + [""]

    open(OUT, "w", encoding="utf-8").write(O.canonical("\n".join(out) + "\n"))
    open(OUT_S, "w", encoding="utf-8").write(O.canonical("\n".join(sup) + "\n"))
    print("saved to", OUT, "and", OUT_S)


if __name__ == "__main__":
    main()

"""
Stage 7 against a simple adaptive acceptance rule (rule 11).

Repeats the Stage 7 ablation of generators/paper_numbers.py::ablation() on
the original.csv files of MAIN_RUN, at both windows, and adds rule 11 of the
pre-registration (PREREGISTRATION.md):

    score of a pair = max |t_stat| over the pair's rows
    candidates      = pairs with score > 1.96 (n of them)
    n < 10          -> accept the candidates (the fixed cut-off alone)
    n >= 10         -> accept score > 45th percentile of the candidates

The Holm correction now covers 11 comparisons. Before any conclusion, the 10
existing rules are checked against section 8 of Paper_Numbers_tau{10,24}.md
(the report paper_numbers.py writes to OUT_DIR); the check is skipped, with a
message, when that report is not there. Only the stored output is rescored;
nothing is re-run.

Usage (from the repository root):
    python analysis/stage7_vs_adaptive_rule/adaptive_rule.py
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(REPO, "generators"))
import paper_numbers as NA  # noqa: E402

T_FIXED = 1.96
PERCENTILE = 45.0
MIN_CAND = 10

# Rule keys as paper_numbers.ablation() names them (the published report is
# matched on these), and their English labels for the output.
R11 = "11 |t| adaptive (P45)"
ORDER = ["1 Stage 7 (DS)", "2 No filter", "3 Cascade high/medium", "4 p_BH < 0.05", "5 |t| > 1.96",
         "6 p oracle", "7 |t| oracle", "8 |t| matched recall", "9 p_BH<0.05 and CI excludes 0",
         "10 p_BH<0.05 and asymmetry>0", R11, "Final (DS + Stage 8)"]
LABEL = {
    "1 Stage 7 (DS)": "1 Stage 7 (evidence fusion)",
    "2 No filter": "2 Accept every candidate reaching Stage 6",
    "3 Cascade high/medium": "3 Cascade confidence high or medium",
    "4 p_BH < 0.05": "4 DML p-value (BH) < 0.05",
    "5 |t| > 1.96": "5 |t| > 1.96",
    "6 p oracle": "6 DML p-value, oracle threshold",
    "7 |t| oracle": "7 |t|, oracle threshold",
    "8 |t| matched recall": "8 |t|, threshold matched to Stage 7's recall (oracle)",
    "9 p_BH<0.05 and CI excludes 0": "9 p (BH) < 0.05 and 95% CI excludes zero",
    "10 p_BH<0.05 and asymmetry>0": "10 p (BH) < 0.05 and directional asymmetry > 0",
    R11: "11 |t| above the 45th percentile of the dataset's candidates (adaptive)",
    "Final (DS + Stage 8)": "Full pipeline (Stage 7 + Stage 8)",
}
num = NA.num


def adaptive_rule(tmax):
    """tmax: max |t| per pair. Returns (accepted pairs, n candidates, effective cut-off)."""
    cand = tmax[tmax > T_FIXED]
    n = len(cand)
    if n < MIN_CAND:
        return set(cand.index), n, T_FIXED
    c = float(np.percentile(cand.values, PERCENTILE))
    return set(cand[cand > c].index), n, max(c, T_FIXED)


def evaluate(rows):
    """Copy of paper_numbers.ablation() (same metric and oracles) plus rule 11."""
    rules = {
        "1 Stage 7 (DS)": lambda d: num(d["causal_penalty_score"]) == 1,
        "2 No filter": lambda d: pd.Series(True, index=d.index),
        "3 Cascade high/medium": lambda d: d["cascade_confidence"].isin(["high", "medium"]),
        "4 p_BH < 0.05": lambda d: num(d["pvalue_bh"]) < 0.05,
        "5 |t| > 1.96": lambda d: num(d["t_stat"]).abs() > 1.96,
        "9 p_BH<0.05 and CI excludes 0": lambda d: (num(d["pvalue_bh"]) < 0.05) & (
            (num(d["ci_95_lower"]) > 0) | (num(d["ci_95_upper"]) < 0)),
        "10 p_BH<0.05 and asymmetry>0": lambda d: (num(d["pvalue_bh"]) < 0.05) & (num(d["causal_asymmetry"]) > 0),
        "Final (DS + Stage 8)": lambda d: d["significant"].astype(str) == "True",
    }
    per, npred, prec, rec = (defaultdict(dict) for _ in range(4))
    cutoff = {}
    for scen in sorted(rows):
        df, truth = NA.pair_table(NA.PIPELINE_RUN, scen, NA.PIPELINE_FOLDER[NA.WINDOW])
        if df is None or len(df) == 0:
            continue
        T = set(truth[truth].index)
        pairs = df[["source_entity", "target_entity"]].apply(tuple, axis=1)

        def put(k, S):
            c = (len(S & T), len(S - T), len(T - S))
            per[k][scen] = NA.f1(*c); npred[k][scen] = len(S)
            prec[k][scen], rec[k][scen] = NA.prf(c)[:2]
            return c

        for k, fn_ in rules.items():
            put(k, set(pairs[fn_(df).fillna(False)]))
        pmin = num(df["pvalue_bh"]).groupby(pairs).min()
        tmax = num(df["t_stat"]).abs().groupby(pairs).max()
        for k, sc, grid, better in [("6 p oracle", pmin, [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5], "lt"),
                                    ("7 |t| oracle", tmax, list(np.arange(0.5, 4.01, 0.25)), "gt")]:
            bestS, bestc = None, None
            for th in grid:
                S = set(sc[(sc < th) if better == "lt" else (sc > th)].index)
                c = (len(S & T), len(S - T), len(T - S))
                if bestc is None or NA.f1(*c) > NA.f1(*bestc) + NA.EPS:
                    bestS, bestc = S, c
            put(k, bestS)
        r_ds = rec["1 Stage 7 (DS)"][scen]
        bestS, bestc = None, None
        for th in sorted(tmax.dropna().unique(), reverse=True):
            S = set(tmax[tmax >= th].index)
            c = (len(S & T), len(S - T), len(T - S))
            if NA.prf(c)[1] >= r_ds - NA.EPS and (bestc is None or NA.prf(c)[0] > NA.prf(bestc)[0] + NA.EPS):
                bestS, bestc = S, c
        if bestS is not None:
            put("8 |t| matched recall", bestS)
        S, n, c = adaptive_rule(tmax)
        put(R11, S)
        cutoff[scen] = dict(n_pairs=len(tmax), n_candidates_above_1_96=n, cutoff=c, adaptive=n >= MIN_CAND)
    return per, npred, prec, rec, cutoff


def table(per, npred, prec, rec):
    ds = per["1 Stage 7 (DS)"]
    pv = {}
    for k in ORDER[1:]:
        common = [s for s in ds if s in per[k]]
        a = np.array([per[k][s] for s in common]); b = np.array([ds[s] for s in common])
        try:
            pv[k] = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
        except ValueError:
            pv[k] = 1.0
    it = sorted(pv.items(), key=lambda x: x[1]); m = len(it); run = 0; holm = {}
    for i, (k, p) in enumerate(it):
        run = max(run, min(1.0, p * (m - i))); holm[k] = run
    out = []
    for k in ORDER:
        common = [s for s in ds if s in per[k]]
        fam = defaultdict(list)
        for s in common:
            fam[NA.family_of(NA.group_of(s))].append(per[k][s])
        c = Counter(NA.cmp(per[k][s], ds[s]) for s in common)
        out.append(dict(
            key=k, rule=LABEL[k], n=len(common),
            precision=np.mean([prec[k][s] for s in common]), recall=np.mean([rec[k][s] for s in common]),
            f1=np.mean([per[k][s] for s in common]), median_f1=np.median([per[k][s] for s in common]),
            family_mean_f1=np.mean([np.mean(v) for v in fam.values()]),
            pairs_accepted=np.mean([npred[k][s] for s in common]),
            delta_f1=np.nan if k == ORDER[0] else np.mean([per[k][s] - ds[s] for s in common]),
            wins=c["W"], ties=c["T"], losses=c["L"],
            wilcoxon_p=pv.get(k, np.nan), holm_p_11=holm.get(k, np.nan)))
    return pd.DataFrame(out), m


def _key(name):
    return "".join(ch for ch in name if ch not in " |")


def published(window):
    """Rows of section 8 of Paper_Numbers_tau{w}.md (Holm over 10), or None."""
    path = os.path.join(NA.OUT_DIR, f"Paper_Numbers_tau{window}.md")
    if not os.path.exists(path):
        return None
    lines = open(path, encoding="utf-8").read().split("\n")
    i = next(j for j, l in enumerate(lines) if l.startswith("## 8. Stage 7 ablation"))
    pub = {}
    for l in lines[i + 1:]:
        if l.startswith("## ") or l.startswith("### "):
            break
        if l.startswith("| ") and not l.startswith("| Rule"):
            cel = [x.strip() for x in l.strip("|").split("|")]
            # names holding '|' (|t|) split into more cells; the name is what is left on the left
            pub[_key("|".join(cel[:-11]))] = cel[-11:]
    return pub


def check(tab, window):
    """None when the published report is absent; otherwise a list of mismatches."""
    pub = published(window)
    if pub is None:
        return None
    errors = []
    for _, r in tab.iterrows():
        if r.key == R11:
            continue
        if _key(r.key) not in pub:
            errors.append(f"{r.rule}: missing from the published report"); continue
        p = pub[_key(r.key)]
        mine = [str(int(r.n)), f"{r.precision:.3f}", f"{r.recall:.3f}", f"{r.f1:.3f}", f"{r.median_f1:.3f}",
                f"{r.family_mean_f1:.3f}", f"{r.pairs_accepted:.1f}"]
        if r.key != ORDER[0]:
            mine += [f"{r.delta_f1:+.3f}", f"{r.wins}/{r.ties}/{r.losses}", f"{r.wilcoxon_p:.2g}"]
        if mine != p[:len(mine)]:
            errors.append(f"{r.rule}: published {p[:len(mine)]}, recomputed {mine}")
    return errors


def fmt_p(p):
    return "—" if pd.isna(p) else ("< 0.0001" if p < 1e-4 else f"{p:.2g}")


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for w in ["10", "24"]:
        NA.WINDOW = w
        rows, _ = NA.load()
        per, npred, prec, rec, cutoff = evaluate(NA.core(rows))
        tab, m = table(per, npred, prec, rec)
        errors = check(tab, w)
        res[w] = (tab, m, errors, cutoff)
        tab.drop(columns="key").to_csv(os.path.join(OUT, f"rules_tau{w}.csv"), index=False)
        pd.DataFrame({LABEL[k]: per[k] for k in ORDER}).rename(index=NA.O.canonical).rename_axis("scenario").to_csv(
            os.path.join(OUT, f"f1_per_scenario_tau{w}.csv"))
        pd.DataFrame(cutoff).T.rename(index=NA.O.canonical).rename_axis("scenario").to_csv(os.path.join(OUT, f"adaptive_cutoff_tau{w}.csv"))
        status = "skipped (report not found)" if errors is None else ("OK" if not errors else "FAILED")
        print(f"tau={w}: {len(per[ORDER[0]])} scenarios, Holm over {m}, check {status}")
        for e in errors or []:
            print("   ", e)

    L = ["# Stage 7 against a simple adaptive rule — results", "",
         f"Written by `adaptive_rule.py` from `{os.path.basename(os.path.normpath(NA.PIPELINE_RUN))}` "
         "(rule and decision criterion in `PREREGISTRATION.md`).", ""]
    for w in ["10", "24"]:
        tab, m, errors, cutoff = res[w]
        if errors is None:
            chk = "not run (`Paper_Numbers_tau{w}.md` not found in OUT_DIR).".replace("{w}", w)
        elif not errors:
            chk = ("**reproduced** (n, P, R, F1, median, family mean, pairs accepted, ΔF1, W/T/L, "
                   "Wilcoxon p).")
        else:
            chk = "**NOT reproduced**:\n" + "\n".join(f"- {e}" for e in errors)
        L += [f"## τ_max = {w}", "",
              f"Check of the 10 existing rules against `Paper_Numbers_tau{w}.md` (section 8): {chk}", "",
              f"Holm correction over {m} comparisons.", "",
              "| Rule | n | P | R | F1 | Median F1 | Family-mean F1 | Pairs accepted | ΔF1 vs Stage 7 | W/T/L | Wilcoxon p | Holm p (11) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in tab.iterrows():
            L.append(f"| {r.rule.replace('|', chr(92) + '|')} | {r.n} | {r.precision:.3f} | {r.recall:.3f} | "
                     f"{r.f1:.3f} | {r.median_f1:.3f} | {r.family_mean_f1:.3f} | {r.pairs_accepted:.1f} | "
                     + ("— | — | — | —" if pd.isna(r.delta_f1) else
                        f"{r.delta_f1:+.3f} | {r.wins}/{r.ties}/{r.losses} | {fmt_p(r.wilcoxon_p)} | {fmt_p(r.holm_p_11)}")
                     + " |")
        c = pd.DataFrame(cutoff).T
        L += ["", f"Adaptive cut-off: applied on {int(c.adaptive.sum())} of {len(c)} scenarios (on the others, "
                  f"fewer than {MIN_CAND} candidates with |t| > 1.96); median cut-off "
                  f"{c[c.adaptive.astype(bool)].cutoff.median():.2f}.", ""]
        r11 = tab.set_index("key").loc[R11]
        if r11.holm_p_11 >= 0.05:
            v = "**Tie** (Holm p ≥ 0.05)"
        elif r11.delta_f1 < 0:
            v = "**Stage 7 wins** (higher F1 and Holm p < 0.05)"
        else:
            v = "**Rule 11 wins** (higher F1 and Holm p < 0.05)"
        L += [f"Pre-registered verdict at τ = {w}: {v}.", ""]
    open(os.path.join(OUT, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("ok ->", os.path.join(OUT, "RESULTS.md"))


if __name__ == "__main__":
    main()

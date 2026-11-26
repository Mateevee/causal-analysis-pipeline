"""
R1-m4 -- Bootstrap intervals per dataset.

Request
  R1: "Four of the 10 held-out datasets are single realizations, so held-out
       claims have very wide uncertainty; add bootstrap intervals for the
       per-dataset results."

Design (fixed before running)
  Resampling unit within a scenario: the pair of the universe (the ground-truth
  pairs plus any predicted pair outside it), with replacement; the graph
  of each method stays fixed. B = 2000, seed 2026.
  Per dataset (family of Table IV): mean F1 of the pipeline and of the best
  baseline of the dataset (the one with the highest mean F1 on it, fixed on the original sample),
  and the difference, with a 95% percentile interval.
  On the held-out aggregate: hierarchical bootstrap -- resamples the 10 datasets,
  then the pairs within each scenario -- for the mean difference between the
  pipeline and the best baseline of each scenario.
  Caveats to state: (1) the pairs of a scenario are not independent,
  so the pair-based interval is too narrow; (2) on a single-realization
  dataset it measures only the sampling uncertainty of the pairs, not that of
  another realization of the process; (3) on Walker (6 pairs) the interval is not
  informative (R1-m4b) and is reported only for completeness.

Usage:  python revision/offline/r1_m4_bootstrap.py [--verificar]
Output: Results_Revisao/r1_m4_bootstrap/
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c  # noqa: E402

NAME = "r1_m4_bootstrap"
B = 2000
SEED = 2026


def predicted(scen, model):
    mm = c.rows()[scen]
    if model not in mm:
        return None
    model_folder = c.PIPE_FOLDER if model == c.PIPE else f"{model}_tau10"
    p = os.path.join(c.folder(scen, model), model_folder, "results.csv")
    if not os.path.exists(p):
        return None
    r = pd.read_csv(p, skipinitialspace=True)
    if r.empty:
        return set()
    for col in ("source_entity", "target_entity"):
        r[col] = r[col].astype(str).str.strip()
    return set(c.pairs(r))


def matrix(scen, methods):
    """y[pairs] and P[method, pairs] over the scenario's universe."""
    T, U = c.truth(scen)
    S = {m: predicted(scen, m) for m in methods}
    pool = sorted(U | set().union(*[s for s in S.values() if s]))
    idx = {p: i for i, p in enumerate(pool)}
    y = np.zeros(len(pool), bool)
    for p in T:
        y[idx[p]] = True
    P = {}
    for m, s in S.items():
        if s is None:
            continue
        v = np.zeros(len(pool), bool)
        for p in s:
            v[idx[p]] = True
        P[m] = v
    return y, P


def f1_weights(w, y, v):
    tp = (w * (y & v)).sum(axis=-1)
    fp = (w * (~y & v)).sum(axis=-1)
    fn = (w * (y & ~v)).sum(axis=-1)
    den = 2 * tp + fp + fn
    return np.where(tp > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)


def main():
    check = "--verificar" in sys.argv
    rel = c.Report(NAME + ("_verificacao" if check else ""), "R1-m4 — Bootstrap intervals per dataset")
    rng = np.random.default_rng(SEED)
    scens = c.scenarios()
    if check:
        scens = [s for s in scens if c.family(s) in ("Walker Circulation", "Sachs", "DREAM3")]
    methods = [c.PIPE] + c.BASELINES
    by_fam = defaultdict(list)
    for s in scens:
        by_fam[c.family(s)].append(s)

    boot = {}       # scenario -> {method: vector of B F1s}
    point = {}      # scenario -> {method: F1}
    for s in scens:
        y, P = matrix(s, methods)
        W = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=B)   # resampling weights
        boot[s] = {m: f1_weights(W, y, v) for m, v in P.items()}
        point[s] = {m: float(f1_weights(np.ones((1, len(y))), y, v)[0]) for m, v in P.items()}

    body, out_rows = [], []
    for fam in c.FAMILIES_DEV + c.FAMILIES_HO:
        sc = by_fam.get(fam)
        if not sc:
            continue
        bls = [m for m in c.BASELINES if all(m in point[s] for s in sc)]
        best = max(bls, key=lambda m: np.mean([point[s][m] for s in sc])) if bls else None
        fp = np.mean([boot[s][c.PIPE] for s in sc], axis=0)
        lo, hi = np.percentile(fp, [2.5, 97.5])
        row = {"familia": fam, "cenarios": len(sc), "heldout": c.is_heldout(sc[0]),
               "f1_pipeline": np.mean([point[s][c.PIPE] for s in sc]), "ic_lo": lo, "ic_hi": hi}
        diff_txt = "—"
        if best:
            fb = np.mean([boot[s][best] for s in sc], axis=0)
            dd = fp - fb
            dlo, dhi = np.percentile(dd, [2.5, 97.5])
            diff = np.mean([point[s][c.PIPE] - point[s][best] for s in sc])
            row.update(melhor_baseline=best, dif=diff, dif_lo=dlo, dif_hi=dhi)
            diff_txt = f"{diff:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
        out_rows.append(row)
        body.append([fam, len(sc), "held-out" if row["heldout"] else "dev",
                     f"{row['f1_pipeline']:.3f} [{lo:.3f}, {hi:.3f}]", c.LABEL.get(best, best) or "—", diff_txt])
    rel.table(["Dataset", "Scenarios", "Part", "Pipeline F1 [95% CI]", "Best baseline of the dataset",
               "Pipeline − best baseline [95% CI]"], body)
    rel.csv("por_dataset.csv", pd.DataFrame(out_rows))

    # hierarchical on the held-out set: difference to the best baseline of each scenario
    ho_fams = [f for f in c.FAMILIES_HO if f in by_fam]
    if len(ho_fams) > 1:
        def scenario_diff(s, b):
            bl = [m for m in c.BASELINES if m in boot[s]]
            best_s = max(bl, key=lambda m: point[s][m])
            return boot[s][c.PIPE][b] - boot[s][best_s][b]
        est = []
        for b in range(B):
            fams = rng.choice(ho_fams, size=len(ho_fams), replace=True)
            vals = []
            for f in fams:
                sc = rng.choice(by_fam[f], size=len(by_fam[f]), replace=True)
                vals.append(np.mean([scenario_diff(s, b) for s in sc]))
            est.append(np.mean(vals))
        pt = np.mean([np.mean([point[s][c.PIPE] - max(point[s][m] for m in c.BASELINES if m in point[s])
                               for s in by_fam[f]]) for f in ho_fams])
        lo, hi = np.percentile(est, [2.5, 97.5])
        rel.p("")
        rel.p(f"Held-out, per-dataset mean of the difference to the best baseline of each scenario: {pt:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] (hierarchical bootstrap: datasets, scenarios and pairs; {len(ho_fams)} datasets).")
        rel.data["hierarquico_heldout"] = dict(pontual=pt, lo=lo, hi=hi)
    rel.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        main()

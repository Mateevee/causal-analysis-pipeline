"""
R1-5a, R1-5b, R1-5c -- Alternative scores: skeleton, without lag 0, and per lag.

Request
  R1: "Scoring is at ordered-pair level, ignoring lag; the contemporaneous step
       reports each adjacency in both directions (inflating false positives ...);
       lagged-only methods (PCMCI, CUTS+) cannot earn lag-0 credit. Please also
       report lag-aware scores where the ground truth allows it".

Design (fixed before running; tau_max = 10; graphs saved by the paper's runs)
  (A) Directed pair: the paper's score (reference; reproduces Table III).
  (B) Skeleton: unordered pairs {A, B}, for the six methods -- removes the
      effect of reporting the lag-0 adjacency in both directions (R1-5b).
  (C) Without lag 0: the directed pair, discarding lag-0 edges from ALL
      methods; no method gains or loses credit for lag 0 (R1-5c).
  (D) Per lag (triple source, target, lag), in the families whose ground truth
      records the lag: internal VAR (scenario_params.txt), tigramite SCMs
      (`links` dictionaries of Data/tigramite_scm.py) and logistic maps (lag 1 by
      construction). Lag-0 edges count as false (the ground truth does not have
      them). Unit: (source entity, target entity, lag) (R1-5a).
  Reporting: mean F1 per method for each score, per suite part (A-C) and per
  family (D); and the change in the pipeline's mean rank between (A) and (B), (C).

Usage:  python revision/offline/r1_5_scores.py [--verificar]
Output: Results_Revisao/r1_5_escores/
"""
from __future__ import annotations

import ast
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c  # noqa: E402

NAME = "r1_5_escores"


def results(scen, m):
    if m not in c.rows()[scen]:          # the method did not run on the scenario (e.g. VARLiNGAM on CausalTime)
        return None
    mfolder = c.PIPE_FOLDER if m == c.PIPE else f"{m}_tau10"
    p = os.path.join(c.folder(scen, m), mfolder, "results.csv")
    if not os.path.exists(p):
        return None
    r = pd.read_csv(p, skipinitialspace=True)
    for col in ("source_entity", "target_entity"):
        if col in r.columns:
            r[col] = r[col].astype(str).str.strip()
    r["lag"] = pd.to_numeric(r.get("lag", 0), errors="coerce").fillna(0).astype(int)
    return r


def true_triples(scen):
    """Set of (source, target, lag) of the ground truth, or None if the lag is not known."""
    fam = c.family(scen)
    folder = c.folder(scen)
    if fam == "VAR synthetic":
        txt = open(os.path.join(folder, "scenario_params.txt"), encoding="utf-8").read()
        edges = ast.literal_eval(re.search(r"causal_edges\s*:\s*(\[.*\])", txt).group(1))
        return {(e["source"], e["target"], int(e["lag"])) for e in edges}
    if fam == "Coupled logistic maps":
        T, _ = c.truth(scen)
        return {(a, b, 1) for a, b in T}
    if fam == "tigramite SCM":
        from Data.tigramite_scm import _BENCHMARK_SCENARIO_DEFINITIONS as DEF
        name = re.sub(r"^benchmark_tigramite_|_seed\d+$", "", scen)
        d = DEF[name]
        npr = d["n_properties"]
        T, _ = c.truth(scen)
        ents = sorted({e for p in T for e in p})
        prefix = re.match(r"^(.*?)(\d+)$", ents[0]).group(1) if ents else "ENT_"
        out = set()
        for target, lst in dict(d["links"]).items():
            for (source, tau), _coef, _f in lst:
                if source == target:
                    continue      # self-dependence, outside the ground truth
                out.add((f"{prefix}{source // npr}", f"{prefix}{target // npr}", -int(tau)))
        return out
    return None


def mean_ranks(F, methods, scens):
    M = np.array([[F[m][s] for m in methods] for s in scens if all(s in F[m] for m in methods)])
    return dict(zip(methods, np.array([rankdata(-r) for r in M]).mean(axis=0))) if len(M) else {}


def main():
    check = "--verificar" in sys.argv
    rel = c.Report(NAME + ("_verificacao" if check else ""), "R1-5 — Skeleton, without lag 0, and per lag")
    scens = c.scenarios()
    if check:
        scens = [s for s in scens if c.family(s) in ("VAR synthetic", "tigramite SCM", "NetSim", "Causal Chambers")]
    methods = [c.PIPE] + c.BASELINES
    F = {k: {m: {} for m in methods} for k in ("A", "B", "C", "D")}
    for s in scens:
        T, U = c.truth(s)
        Tu = {frozenset(p) for p in T}
        Tl = true_triples(s)
        for m in methods:
            r = results(s, m)
            if r is None:
                continue
            S = set(c.pairs(r)) if len(r) else set()
            F["A"][m][s] = c.f1_set(S, T)
            F["B"][m][s] = c.f1_set({frozenset(p) for p in S}, Tu)
            r1 = r[r["lag"] >= 1]
            F["C"][m][s] = c.f1_set(set(c.pairs(r1)) if len(r1) else set(), T)
            if Tl is not None:
                Sl = set(zip(r["source_entity"], r["target_entity"], r["lag"])) if len(r) else set()
                F["D"][m][s] = c.f1_set(Sl, Tl)
    names = {"A": "directed pair (paper)", "B": "skeleton", "C": "without lag 0"}
    for part, sc in (("all", scens), ("development", [s for s in scens if not c.is_heldout(s)]),
                     ("held-out", [s for s in scens if c.is_heldout(s)])):
        if not sc:
            continue
        body = []
        for k in ("A", "B", "C"):
            rk = mean_ranks(F[k], methods, sc)
            body.append([names[k]] + [f"{np.mean([F[k][m][s] for s in sc if s in F[k][m]]):.3f}"
                                      + (f" ({rk[m]:.2f})" if rk else "") for m in methods])
        rel.p(f"**{part}** — mean F1 (mean rank among the six, on the scenarios where all of them run)")
        rel.table(["Score"] + [c.LABEL[m] for m in methods], body)
        rel.p("")
    rel.h("## (D) Per lag, in the families with known lag")
    body = []
    for fam in ("VAR synthetic", "tigramite SCM", "Coupled logistic maps"):
        sc = [s for s in scens if c.family(s) == fam and s in F["D"][c.PIPE]]
        if sc:
            body.append([fam, len(sc)] + [c.fmt(np.mean([F["D"][m][s] for s in sc if s in F["D"][m]]))
                                          for m in methods])
    rel.table(["Family", "Scenarios"] + [c.LABEL[m] for m in methods], body)
    pd.DataFrame([{"escore": k, "metodo": m, "cenario": s, "f1": v} for k, d in F.items()
                  for m, dd in d.items() for s, v in dd.items()]).to_csv(os.path.join(rel.dir, "por_cenario.csv"),
                                                                       index=False)
    rel.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        main()

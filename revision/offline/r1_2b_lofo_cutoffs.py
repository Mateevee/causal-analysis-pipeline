"""
R1-2b -- Leave-one-family-out (partial) of the pipeline's cut-offs.

Request
  R1: "(b) a leave-one-family-out protocol, where cut-offs are frozen using
       other families only".

What can be done without running anything
  The development history cannot be undone (R1-2d), but the sensitivity
  run (Table S20) already has the pipeline output for 216 configurations of the
  nine cut-offs of Table I (default, 65 one at a time and 150 joint), on the 86
  scenarios (without Causal Chambers). For each family, the configuration
  that maximises the mean per-scenario F1 on the other families is chosen and
  measured on the left-out family, against the default.

Design (fixed before running)
  F1 per directional pair, seed 0 of each configuration (that of Table S20).
  Selection: highest mean F1 on the scenarios of the other families; tie -> default.
  Report per family: chosen configuration, F1 of the default and of the chosen one on
  the left-out family; in aggregate, the mean difference and how many families improve
  or worsen; and how many times the chosen one is the default itself.
  Caveat to declare: the paper's cut-offs were set by hand during
  development, not by this optimisation; the exercise shows how far the
  default is from what each family "would learn" without seeing itself; it does not
  redo the development.

Usage:  python revision/offline/r1_2b_lofo_cutoffs.py [--verificar]
Output: Results_Revisao/r1_2b_lofo_cutoffs/
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c  # noqa: E402

NAME = "r1_2b_lofo_cutoffs"


def load(run, scens=None):
    """F[scenario][config] = F1 of seed 0."""
    F = {}
    for path in sorted(glob.glob(os.path.join(run, "runs", "*.jsonl"))):
        s = os.path.basename(path)[:-6]
        s_canon = c.O.canonical(s) if hasattr(c.O, "canonical") else s
        if scens is not None and s_canon not in scens and s not in scens:
            continue
        gt = pd.read_csv(os.path.join(run, "gt", s + ".csv"))
        for col in ("source_entity", "target_entity"):
            gt[col] = gt[col].astype(str).str.strip()
        pos = gt["causal_gt"].astype(str).str.lower().isin(["true", "1"])
        T = set(map(tuple, gt.loc[pos, ["source_entity", "target_entity"]].values))
        U = set(map(tuple, gt[["source_entity", "target_entity"]].values))
        F[s_canon] = {}
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["status"] != "ok" or r["seed"] != 0:
                continue
            S = {(a, b) for a, b, _ in r["pairs"]} & U
            F[s_canon][r["config"]] = c.f1_set(S, T)
    return F


def main():
    check = "--verificar" in sys.argv
    rel = c.Report(NAME + ("_verificacao" if check else ""),
                   "R1-2b — Leave-one-family-out of the cut-offs (sensitivity run)")
    run = c.O.require("SENSITIVITY_RUN")
    scens = set(c.scenarios())
    if check:
        scens = {s for s in scens if c.family(s) in ("VAR synthetic", "Lorenz-96", "PetShop", "BEELINE")}
    F = load(run, scens)
    M = pd.DataFrame(F).T            # rows: scenarios; columns: configurations
    M = M.dropna(axis=1, how="any")  # only complete configurations
    fam = pd.Series({s: c.family(s) for s in M.index})
    rel.p(f"{len(M)} scenarios, {M.shape[1]} complete configurations, {fam.nunique()} families.")
    body, rows = [], []
    for f in sorted(fam.unique()):
        left_out = M[fam == f]
        inside = M[fam != f]
        mean = inside.mean(axis=0)
        best = mean.idxmax()
        if abs(mean[best] - mean["default"]) < 1e-12:
            best = "default"
        d0, d1 = float(left_out["default"].mean()), float(left_out[best].mean())
        rows.append({"familia": f, "cenarios": len(left_out), "heldout": c.is_heldout(left_out.index[0]),
                     "config": best, "f1_default": d0, "f1_lofo": d1})
        body.append([f, len(left_out), "held-out" if c.is_heldout(left_out.index[0]) else "dev", best,
                     c.fmt(d0), c.fmt(d1), f"{d1 - d0:+.3f}"])
    rel.table(["Left-out family", "Scenarios", "Part", "Configuration chosen on the others", "F1 default",
               "F1 chosen", "Difference"], body)
    L = pd.DataFrame(rows)
    rel.csv("por_familia.csv", L)
    dif = L["f1_lofo"] - L["f1_default"]
    rel.p("")
    rel.p(f"Chosen = default in {int((L['config'] == 'default').sum())} of {len(L)} families. "
          f"Mean difference per family {dif.mean():+.3f}; improves in {int((dif > 1e-9).sum())}, "
          f"worsens in {int((dif < -1e-9).sum())}.")
    for part, m in (("development", ~L["heldout"]), ("held-out", L["heldout"])):
        if m.any():
            rel.p(f"- {part}: mean difference {dif[m].mean():+.3f} over {int(m.sum())} families.")
    rel.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        main()

"""
R1-5d -- AUROC and AUPRC on the 90 scenarios, for the main text.

Request
  R1: "a metric that does not depend on a single operating point (AUPRC is in
       the supplement; bring the headline numbers into the main text)."

What was missing: Table S7 covers 86 of 90 scenarios; the 4 Causal Chambers ones
were not scored. They have already been run (Results_Extra/Run_20261009_causal_chamber,
the six methods), and Auxiliary/ranking_metrics.py scores an existing run without
running any model. To avoid writing inside the paper's run, the 4 scenarios
are copied to Results_Revisao/r1_5d_auprc_90/cc/ and scored there; the
result is joined with that of METRICS_RUN (the 86).

Design: the same definitions as Table S7 (penalized_effect score for the
pipeline; two_tier for the baselines; tau_max = 10). Reporting: mean AUROC and AUPRC
per method on development and held-out, mean rank and critical difference.

Usage:  python revision/offline/r1_5d_auprc_90.py
Output: Results_Revisao/r1_5d_auprc_90/
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, studentized_range

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c  # noqa: E402
from Auxiliary import ranking_metrics as rm  # noqa: E402

NAME = "r1_5d_auprc_90"
CC_RUN = os.path.join(c.ROOT, "Results_Extra", "Run_20261009_causal_chamber")


def main():
    rel = c.Report(NAME, "R1-5d — AUROC and AUPRC on the 90 scenarios")
    dst = os.path.join(rel.dir, "cc")
    if not os.path.isdir(dst):
        shutil.copytree(CC_RUN, dst)
    rm.run(dst)
    cc = pd.read_csv(os.path.join(dst, "auroc_auprc_shd_per_scenario.csv"))
    base = pd.read_csv(os.path.join(c.O.require("METRICS_RUN"), "auroc_auprc_shd_per_scenario.csv"))
    D = pd.concat([base, cc], ignore_index=True)
    D["scenario"] = D["scenario"].map(c.O.canonical)
    D = D[D["model"].astype(str).str.endswith("_tau10")]
    D["metodo"] = D["model"].str.replace("_tau10", "", regex=False).replace({"patch": c.PIPE})
    D["heldout"] = D["scenario"].map(c.is_heldout)
    rel.csv("por_cenario.csv", D)
    rel.p(f"Scenarios: {D['scenario'].nunique()} (86 from METRICS_RUN + {cc['scenario'].nunique()} from Causal Chambers).")
    methods = [c.PIPE] + c.BASELINES
    for met in ("auroc", "auprc"):
        body = []
        for part, msk in (("all", D["heldout"] | ~D["heldout"]), ("dev", ~D["heldout"]), ("held-out", D["heldout"])):
            X = D[msk].pivot_table(index="scenario", columns="metodo", values=met)
            X = X[[m for m in methods if m in X.columns]]
            comp = X.dropna()
            rk = np.array([rankdata(-r) for r in comp.to_numpy()]).mean(axis=0) if len(comp) else None
            k = comp.shape[1]
            cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * max(len(comp), 1)))
            body.append([part, len(X)] + [f"{X[m].mean():.3f}" + (f" ({rk[i]:.2f})" if rk is not None else "")
                                          for i, m in enumerate(X.columns)] + [f"{cd:.2f}"])
        rel.p(f"**{met.upper()}** — mean (mean rank on the scenarios where all six run)")
        rel.table(["Part", "Scenarios"] + [c.LABEL.get(m, m) for m in X.columns] + ["Critical difference"], body)
        rel.p("")
    rel.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        main()

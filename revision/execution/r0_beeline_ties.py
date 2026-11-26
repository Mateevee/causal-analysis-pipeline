"""
R0-BEELINE -- Sensitivity of the BEELINE results to the order of the cells with
tied pseudotime (a finding of the revision; no reviewer asked for it, transparency requires it).

Finding (NOTAS.md, R3-1b): Data/beeline_grn.py sorts ~2,000 cells by
pseudotime with an unstable sort and keeps the first 500; 92-98% of the
cells have a tied pseudotime, so the order among tied cells (and which ones enter the
cut) depends on the environment. The BEELINE F1 of the paper do not reproduce bit for bit
(Supplementary Material, Table S31.2, note a, and Text ST16).

Design (fixed before running, 2026-11-09; decision of the authors by delegation)
  Scenarios: the 6 of the paper (HSC and VSC, first 3 runs of each, 500 cells,
    as in main_heldout.py), loaded with
    - the default order (tie_seed=None, that of the paper in this environment), and
    - 10 random orders of the ties (tie_seed = 7101..7110): shuffles the
      cells and sorts them stably by pseudotime.
    66 scenarios in all.
  Models: the six of the paper at tau_max = 10 (patch_tau10 and the five baselines)
    and patch_tau10_final (new time-axis rule + guard G1).
  Report, whatever the result:
    - per scenario and method: F1 in the default order, mean, standard deviation, minimum and maximum over the
      10 random orders; and the F1 stored in the paper;
    - per order: mean of the 6 scenarios per method and the position of the pipeline among the
      six methods of the paper; fraction of the 10 orders in which the pipeline is above the
      best baseline in the family mean;
    - how much of the pipeline - best baseline difference on BEELINE fits within the variation
      due to the order alone.

Usage
  python revision/execution/r0_beeline_ties.py rodar [--paralelo N]   (N processes; sequential by default)
  python revision/execution/r0_beeline_ties.py avaliar
Output: Results_Revisao/r0_beeline_empates/ (file and column names kept in Portuguese; see revision/README.md, Notes)
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common_exec as ce  # noqa: E402

NAME = "r0_beeline_empates"
SEEDS = list(range(7101, 7111))
FINAL = ("patch_tau10", "patch_tau10_final",
         {"time_axis_rule": "block_portmanteau", "evidence_coherence_overlap": "coherent"})
SPECS = [(m, m, {}) for m in ce.PAPER_MODELS_TAU10] + [FINAL]
BASELINES = [m for m in ce.PAPER_MODELS_TAU10 if m != "patch_tau10"]


def scenarios():
    from Data.beeline_grn import generate_beeline_grn_benchmark_scenarios as g
    kw = dict(networks=["HSC", "VSC"], max_runs_per_network=3, max_cells=500,
              base_dir="./external_data/beeline_grn/BEELINE-data")
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        out += g(**kw, name_prefix="revisao_empates_padrao")
        for s in SEEDS:
            out += g(**kw, name_prefix=f"revisao_empates_s{s}", tie_seed=s)
    return out


def run():
    folder = ce.output_dir(os.path.join(NAME, "run"))
    dfs = scenarios()
    if "--paralelo" in sys.argv:
        ce.run_parallel(dfs, SPECS, folder, int(sys.argv[sys.argv.index("--paralelo") + 1]))
    else:
        ce.run_resuming(dfs, [ce.model(b, n, **o) for b, n, o in SPECS], folder)
    evaluate(folder)


def evaluate(folder=None):
    import common as c
    folder = folder or os.path.join(ce.OUTPUT_ROOT, NAME, "run")
    R = ce.summary(folder)
    R = R[R["status"] == "ok"].copy()
    R["f1"] = [ce.f1(a, b, d) for a, b, d in zip(R.TP, R.FP, R.FN)]
    R["ordem"] = R["cenario"].str.extract(r"revisao_empates_(padrao|s\d+)_")[0]
    R["base"] = R["cenario"].str.replace(r"revisao_empates_(padrao|s\d+)_", "", regex=True)
    R.to_csv(os.path.join(os.path.dirname(folder), "por_execucao.csv"), index=False)
    rnd = R[R["ordem"] != "padrao"]
    txt = ["# Sensitivity of BEELINE to the order of the pseudotime ties", "",
           f"6 scenarios x (default order + {len(SEEDS)} random orders); F1 under the scoring of the paper.", ""]
    # per scenario and method
    paper_key = {"patch_tau10": "patch", "pcmci_tigramite_tau10": "pcmci_tigramite", "pcmciplus_tau10": "pcmciplus",
                 "dynotears_tau10": "dynotears", "cuts_plus_tau10": "cuts_plus", "varlingam_tau10": "varlingam"}
    rows = []
    for (b, m), g in rnd.groupby(["base", "model"]):
        dflt = R[(R["ordem"] == "padrao") & (R["base"] == b) & (R["model"] == m)]["f1"]
        art = None
        k = c.scenario_key(f"benchmark_beeline_{b}")
        if k and paper_key.get(m) in c.rows().get(k, {}):
            art = c.f1_model(k, paper_key[m])
        rows.append([b, m, c.fmt(art), c.fmt(float(dflt.iloc[0]) if len(dflt) else None), c.fmt(g["f1"].mean()),
                     c.fmt(g["f1"].std()), c.fmt(g["f1"].min()), c.fmt(g["f1"].max()), len(g)])
    txt += ["## Per scenario and method", "", "| Scenario | Method | F1 paper | F1 default order | Mean of 10 orders | SD | "
            "Min | Max | n |", "|---|---|---|---|---|---|---|---|---|"]
    txt += ["| " + " | ".join(map(str, l)) + " |" for l in rows]
    # per order: family mean
    F = R.groupby(["ordem", "model"])["f1"].mean().unstack()
    txt += ["", "## Mean of the 6 scenarios, per order", "", F.round(3).to_markdown(), ""]
    if "patch_tau10" in F and all(b in F for b in BASELINES):
        Fa = F.drop(index="padrao", errors="ignore")
        best = Fa[BASELINES].max(axis=1)
        dif = Fa["patch_tau10"] - best
        pos = Fa[["patch_tau10"] + BASELINES].rank(axis=1, ascending=False)["patch_tau10"]
        txt += [f"- Pipeline (patch_tau10) above the best baseline in {int((dif > 0).sum())} of {len(Fa)} orders; "
                f"mean difference {dif.mean():+.3f} (min {dif.min():+.3f}, max {dif.max():+.3f}).",
                f"- Position of the pipeline among the 6 methods: min {int(pos.min())}, max {int(pos.max())}, "
                f"median {pos.median():.0f}.",
                f"- Standard deviation of the family mean across orders: pipeline {Fa['patch_tau10'].std():.3f}; baselines "
                + ", ".join(f"{b} {Fa[b].std():.3f}" for b in BASELINES) + "."]
    out = os.path.dirname(folder)
    open(os.path.join(out, "relatorio.md"), "w", encoding="utf-8").write("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        if len(sys.argv) > 1 and sys.argv[1] == "rodar":
            run()
        else:
            evaluate()

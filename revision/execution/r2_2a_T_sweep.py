"""
R2-2a -- Behaviour on short series: sweep of the length T.

Request
  R2: "Several benchmark datasets ... have very short horizons: 40 time points in
       CausalTime, 84 in DREAM3, and 200 in NetSim. At these sample sizes, the
       cross-fitting procedure and the Neyman orthogonality guarantees underlying
       double machine learning are likely to degrade."
  (Also covers limitation L3: patterns not tested on very short or very long series.)

Design (fixed before running)
  Internal VAR generator, two configurations of the paper (entries 2 and 3 of
    Config.execution.SYNTHETIC_VAR_SCENARIOS), T in {40, 84, 200, 300, 1000}
    (the first three are the T of CausalTime, DREAM3 and NetSim), seeds
    5101-5110: 100 scenarios.
  Models: the pipeline and the five baselines, tau_max = 10 (with the paper's
    rule for short series: the largest window that leaves 20 observations).
  Reporting per T: mean F1 of each method; for the pipeline, on the true links
    reported at the true lag, coverage of the 95% interval, correct sign and
    mean absolute error against the standardized coefficient.
  There is no asymptotic guarantee at T = 40 (R2-2d); the experiment shows what
  happens, it does not establish one.

Addendum (2026-11-12, before any complete run; NOTAS.md, entry "R2-2a — addendum of the final
method, before running"; authors' decision of 2026-11-11, by delegation, item 1 of the entry "today's round with at
most 5 h")
  The final method (patch_tau10_final = patch_tau10 + time_axis_rule='block_portmanteau' +
  evidence_coherence_overlap='coherent', the same definition as in r1_3d and r0_beeline_empates) runs alongside
  the six, on the same scenarios. Coverage, sign and error are reported for both pipeline versions. The scenarios,
  the T values, the seeds and the six methods do not change. --verificar now runs both pipeline versions.

Usage
  python revision/execution/r2_2a_T_sweep.py rodar [--verificar]   (several hours; resumes)
  python revision/execution/r2_2a_T_sweep.py avaliar
Output: Results_Revisao/r2_2a_varredura_T/
"""
from __future__ import annotations

import ast
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_exec as ce  # noqa: E402
from Config import execution as cfg_exec  # noqa: E402

NAME = "r2_2a_varredura_T"
TS = [40, 84, 200, 300, 1000]
SEEDS = list(range(5101, 5111))
FINAL = ("patch_tau10", "patch_tau10_final",
         {"time_axis_rule": "block_portmanteau", "evidence_coherence_overlap": "coherent"})
PIPELINES = ["patch_tau10", "patch_tau10_final"]


def models(check=False):
    """The six methods of the paper (with --verificar, only the pipeline) and the final method alongside."""
    base = ["patch_tau10"] if check else ce.PAPER_MODELS_TAU10
    return ce.models(base) + [ce.model(FINAL[0], FINAL[1], **FINAL[2])]


def scenarios(check=False):
    out = []
    confs = cfg_exec.SYNTHETIC_VAR_SCENARIOS[1:3]
    for (prefix, _s, _n, n_ent, prob, mp, ml, me) in (confs[:1] if check else confs):
        tag = re.sub(r"^benchmark_", "", prefix)
        for T in ([40] if check else TS):
            out += ce.generate_var(SEEDS[:1] if check else SEEDS, T, n_ent, prob, mp, ml, me,
                                   name_prefix=f"revisao_T{T}_{tag}")
    return out


def run():
    check = "--verificar" in sys.argv
    folder = ce.output_dir(os.path.join(NAME + ("_verificacao" if check else ""), "run"))
    ce.run_resuming(scenarios(check), models(check), folder)
    evaluate(folder)


def effects(scen_folder, model="patch_tau10"):
    """True links reported at the true lag: (theta, lo, hi, ref)."""
    txt = open(os.path.join(scen_folder, "scenario_params.txt"), encoding="utf-8").read()
    edges = ast.literal_eval(re.search(r"causal_edges\s*:\s*(\[.*\])", txt).group(1))
    src = pd.read_csv(os.path.join(scen_folder, "source.csv"))
    col = [k for k in src.columns if k not in ("Date", "Name")][0]
    sd = src.groupby("Name")[col].std()
    true_map = {(e["source"], e["target"]): (int(e["lag"]), e["coef"] * sd[e["source"]] / sd[e["target"]]) for e in edges}
    p = os.path.join(scen_folder, model, "results.csv")
    if not os.path.exists(p):
        return []
    d = pd.read_csv(p)
    out = []
    for r in d.itertuples():
        v = true_map.get((r.source_entity, r.target_entity))
        if v and int(r.lag) == v[0]:
            out.append((r.causal_effect, r.ci_95_lower, r.ci_95_upper, v[1]))
    return out


def evaluate(folder=None):
    folder = folder or os.path.join(ce.OUTPUT_ROOT, NAME, "run")
    R = ce.summary(folder)
    R["T"] = R["cenario"].str.extract(r"revisao_T(\d+)_").astype(int)
    R["f1"] = [ce.f1(a, b, c) if s == "ok" else np.nan for a, b, c, s in zip(R.TP, R.FP, R.FN, R.status)]
    tab = R.pivot_table(index="T", columns="model", values="f1", aggfunc="mean")
    n = R[R.model == "patch_tau10"].groupby("T").size()
    txt = ["# R2-2a — Sweep of T", "", "Mean F1 per method:", "", tab.round(3).to_markdown(), ""]
    for model in PIPELINES:
        eff = []
        for s in R["cenario"].unique():
            T = int(re.search(r"revisao_T(\d+)_", s).group(1))
            for th, lo, hi, ref in effects(os.path.join(folder, s), model):
                eff.append({"T": T, "cobre": lo <= ref <= hi, "sinal": np.sign(th) == np.sign(ref),
                            "erro": abs(th - ref)})
        E = pd.DataFrame(eff)
        if len(E):
            G = E.groupby("T").agg(links=("cobre", "size"), cobertura=("cobre", "mean"), sinal=("sinal", "mean"),
                                   erro_abs_medio=("erro", "mean"))
            G["cenarios"] = R[R.model == model].groupby("T").size()
            txt += [f"`{model}`, true links reported at the true lag:", "", G.round(3).to_markdown(), ""]
    out = os.path.dirname(folder)
    R.to_csv(os.path.join(out, "resumo.csv"), index=False)
    open(os.path.join(out, "relatorio.md"), "w", encoding="utf-8").write("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        if len(sys.argv) > 1 and sys.argv[1] == "rodar":
            run()
        else:
            evaluate()

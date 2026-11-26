"""
R3-1 (simulated part) -- Counterexample to the time-axis diagnostic.

Request
  R3: "The pipeline interprets low median lag-1 autocorrelation as the absence
       of meaningful temporal structure and removes all lagged edges. But
       individual series can have zero autocorrelation while exhibiting strong
       cross-series lagged dependence. Would be great to have revisions or
       targeted validation against such cases."
  (R2-m1 raises the same question.)

Design (fixed before running)
  Internal VAR generator without an autoregressive term (ar_coef_range = (0, 0)):
    roots are white noise and each child receives only the lagged influence of
    its parents, so that the ACF(1) of each series stays close to zero although
    there is lagged cross dependence.
  Control: the same graphs and seeds with the generator's default AR(1).
  Configurations: (5 entities, prob 0.35, up to 2 parents, lag up to 3) and
    (12 entities, prob 0.3, up to 3 parents, lag up to 3); T = 300; seeds
    3101-3110. 40 scenarios in total.
  Models: the pipeline and the five baselines, tau_max = 10, paper configuration.
  Reporting per condition (without AR / with AR): fraction of scenarios in which the rule
    declares "no time axis"; mean F1 of each method; mean number of lagged
    links the pipeline reports; and, per scenario, the rule statistic.
  Expectation to be tested, not assumed: without AR, the rule fires and the
    pipeline discards all lagged links.

Usage
  python revision/execution/r3_1_axis_counterexample.py rodar     (about 1-2 h)
  python revision/execution/r3_1_axis_counterexample.py avaliar
  python revision/execution/r3_1_axis_counterexample.py rodar --verificar   (2 scenarios, pipeline only)
Output: Results_Revisao/r3_1_contraexemplo_eixo/
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_exec as ce  # noqa: E402

NAME = "r3_1_contraexemplo_eixo"
CONFIGS = [("n5", 5, 0.35, 2, 3), ("n12", 12, 0.3, 3, 3)]
SEEDS = list(range(3101, 3111))
T = 300


def scenarios(check=False):
    out = []
    seeds = SEEDS[:1] if check else SEEDS
    configs = CONFIGS[:1] if check else CONFIGS
    for tag, n, p, mp, ml in configs:
        for cond, ar in (("semAR", (0.0, 0.0)), ("comAR", (0.2, 0.5))):
            out += ce.generate_var(seeds, T, n, p, mp, ml, min_edges=2,
                                   name_prefix=f"revisao_eixo_{cond}_{tag}", ar_coef_range=ar)
    return out


def run():
    check = "--verificar" in sys.argv
    folder = ce.output_dir(os.path.join(NAME + ("_verificacao" if check else ""), "run"))
    names = ["patch_tau10"] if check else ce.PAPER_MODELS_TAU10
    ce.run_resuming(scenarios(check), ce.models(names), folder)
    evaluate(folder)


def evaluate(folder=None):
    folder = folder or os.path.join(ce.OUTPUT_ROOT, NAME, "run")
    R = ce.summary(folder)
    rows = []
    for s in sorted(R["cenario"].unique()):
        src = pd.read_csv(os.path.join(folder, s, "source.csv"))
        ax = ce.axis_statistic(ce.wide_series(src))
        r = R[R["cenario"] == s]
        row = {"cenario": s, "condicao": "sem AR" if "semAR" in s else "com AR",
               "acf1_mediana": ax["mediana_acf1"], "limiar": ax["limiar"], "regra_sem_eixo": ax["sem_eixo"]}
        for m in r.itertuples():
            row[f"f1_{m.model}"] = ce.f1(m.TP, m.FP, m.FN) if m.status == "ok" else np.nan
        res = os.path.join(folder, s, "patch_tau10", "results.csv")
        if os.path.exists(res):
            d = pd.read_csv(res)
            row["pipeline_links_defasados"] = int((pd.to_numeric(d.get("lag", 0), errors="coerce") > 0).sum())
            row["pipeline_links_lag0"] = int((pd.to_numeric(d.get("lag", 0), errors="coerce") == 0).sum())
        rows.append(row)
    D = pd.DataFrame(rows)
    out = os.path.dirname(folder)
    D.to_csv(os.path.join(out, "por_cenario.csv"), index=False)
    f1c = [k for k in D.columns if k.startswith("f1_")]
    G = D.groupby("condicao").agg(cenarios=("cenario", "count"), regra_sem_eixo=("regra_sem_eixo", "mean"),
                                  acf1=("acf1_mediana", "median"),
                                  **{k: (k, "mean") for k in f1c + ["pipeline_links_defasados",
                                                                    "pipeline_links_lag0"] if k in D})
    txt = ["# R3-1 — Counterexample to the time-axis diagnostic", "", G.round(3).to_markdown(), "",
           "regra_sem_eixo = fraction of scenarios in which the median of |ACF(1)| falls below 2/sqrt(n)."]
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

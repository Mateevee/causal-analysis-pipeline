"""
R3-1b / R2-m1d -- Rerun with the new time-axis rule (criteria C2 and
C3 of PROTOCOL_time_axis_rule.md, frozen; SHA-256 in NOTAS.md).

Design (fixed before running, 2026-11-09)
  Models, with the same code and in the same run:
    patch_tau10            -- paper configuration (old rule, default);
    patch_tau10_eixo_bloco -- the same, with time_axis_rule='block_portmanteau'.
  Scenarios:
    (a) the 5 of the paper in which the decision changes (r2_m1c_regras_eixo):
        BEELINE HSC-2000-1, HSC-2000-2, VSC-2000-2, VSC-2000-10 and
        real_causal_chamber_lt_smooth_polarizers;
    (b) the 40 of the counterexample (r3_1_contraexemplo_eixo: 20 without AR, 20 with AR).
  In the other 85 scenarios of the paper the decision of the two rules is the same (C1,
  r2_m1e), so the output is identical by construction and is not rerun.
  Both versions run now because the pipeline code changed since the paper
  run (the paper-v1 tag is still missing): the comparison that counts is between the two
  current runs; the F1 recorded in the paper for the 5 scenarios is shown separately,
  to show the code drift.
  Scoring: the paper's (main.validate_all, directional entity pairs).
  Reporting, whatever the result (C3): F1, TP, FP, FN and number of lagged
  and lag-0 links of each model in each scenario; means per group; on the 5
  of the paper, the F1 recorded in the paper; and the mean over the 90 with the 5 swapped
  (development, held-out and total), using the F1 recorded in the paper on the 85
  unchanged -- marked as an approximation because it mixes two code versions.
  Expected, not assumed: on the 20 without AR the pipeline F1 stops being 0.

Usage
  python revision/execution/r3_1b_new_rule_rerun.py rodar      (~1 h, sequential)
  python revision/execution/r3_1b_new_rule_rerun.py avaliar
Output: Results_Revisao/r3_1b_regra_nova_reexecucao/
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common_exec as ce  # noqa: E402

NAME = "r3_1b_regra_nova_reexecucao"
NEW_MODEL = "patch_tau10_eixo_bloco"
PAPER_5 = ["benchmark_beeline_HSC_HSC-2000-1", "benchmark_beeline_HSC_HSC-2000-2",
           "benchmark_beeline_VSC_VSC-2000-2", "benchmark_beeline_VSC_VSC-2000-10",
           "real_causal_chamber_lt_smooth_polarizers"]


def models():
    return [ce.model("patch_tau10"), ce.model("patch_tau10", NEW_MODEL, time_axis_rule="block_portmanteau")]


def run():
    import r3_1_axis_counterexample as r31
    folder = ce.output_dir(os.path.join(NAME, "run"))
    dfs = [d for d in ce.paper_scenarios() if d["name"] in PAPER_5]
    missing = set(PAPER_5) - {d["name"] for d in dfs}
    if missing:
        raise RuntimeError(f"paper scenarios not found: {sorted(missing)}")
    dfs += r31.scenarios()
    ce.run_resuming(dfs, models(), folder)
    evaluate(folder)


def evaluate(folder=None):
    import common as c
    folder = folder or os.path.join(ce.OUTPUT_ROOT, NAME, "run")
    R = ce.summary(folder)
    rows = []
    for s in sorted(R["cenario"].unique()):
        group = ("artigo" if s in PAPER_5 else "contraexemplo sem AR" if "semAR" in s else "contraexemplo com AR")
        row = {"cenario": s, "grupo": group}
        for m in R[R["cenario"] == s].itertuples():
            tag = "novo" if m.model == NEW_MODEL else "antigo"
            row[f"f1_{tag}"] = ce.f1(m.TP, m.FP, m.FN) if m.status == "ok" else np.nan
            row[f"tp_{tag}"], row[f"fp_{tag}"], row[f"fn_{tag}"] = m.TP, m.FP, m.FN
            res = os.path.join(folder, s, m.model, "results.csv")
            if os.path.exists(res):
                d = pd.read_csv(res)
                lag = pd.to_numeric(d["lag"], errors="coerce") if "lag" in d else pd.Series(dtype=float)
                row[f"defasados_{tag}"] = int((lag > 0).sum())
                row[f"lag0_{tag}"] = int((lag == 0).sum())
        if group == "artigo":
            row["f1_gravado_artigo"] = c.f1_model(c.scenario_key(s) if hasattr(c, "scenario_key") else s, c.PIPE)
            row["heldout"] = c.is_heldout(s)
        rows.append(row)
    D = pd.DataFrame(rows)
    out = os.path.dirname(folder)
    D.to_csv(os.path.join(out, "por_cenario.csv"), index=False)

    txt = ["# R3-1b — Rerun with the new time-axis rule", "",
           "Old = patch_tau10 (original rule); new = patch_tau10_eixo_bloco (block Hosking). "
           "Same code, same run. Paper scoring.", ""]
    G = D.groupby("grupo").agg(cenarios=("cenario", "count"), f1_antigo=("f1_antigo", "mean"),
                               f1_novo=("f1_novo", "mean"), defasados_antigo=("defasados_antigo", "mean"),
                               defasados_novo=("defasados_novo", "mean"), lag0_antigo=("lag0_antigo", "mean"),
                               lag0_novo=("lag0_novo", "mean"))
    txt += ["## Means per group", "", G.round(3).to_markdown(), ""]
    A = D[D["grupo"] == "artigo"]
    cols = ["cenario", "heldout", "f1_gravado_artigo", "f1_antigo", "f1_novo", "tp_antigo", "fp_antigo", "fn_antigo",
            "tp_novo", "fp_novo", "fn_novo", "defasados_antigo", "defasados_novo", "lag0_antigo", "lag0_novo"]
    txt += ["## The 5 scenarios of the paper", "", A[[k for k in cols if k in A]].round(3).to_markdown(index=False), ""]
    drift = A[(A["f1_gravado_artigo"] - A["f1_antigo"]).abs() > 1e-9]
    txt += [f"Code drift (F1 recorded in the paper ≠ F1 of patch_tau10 now): {len(drift)} of {len(A)} "
            f"({', '.join(drift['cenario']) or 'none'}).", ""]
    # means over the 90 with the 5 swapped (approximation)
    base = {s: c.f1_model(s, c.PIPE) for s in c.scenarios()}
    swapped = dict(base)
    for r in A.itertuples():
        key = next((k for k in base if k == r.cenario or c.scenario_key(k) == r.cenario), None) \
            if hasattr(c, "scenario_key") else r.cenario
        if key in swapped:
            swapped[key] = r.f1_novo
    body = ["| Part | Scenarios | Paper F1 | F1 with the 5 swapped (approximation) |", "|---|---|---|---|"]
    for part in ("todos", "dev", "ho"):
        sc = c.scenarios(part)
        body.append(f"| {part} | {len(sc)} | {np.mean([base[s] for s in sc]):.3f} | "
                    f"{np.mean([swapped[s] for s in sc]):.3f} |")
    txt += ["## Mean over the 90 scenarios", "",
            "Approximation: on the 85 unchanged it uses the F1 recorded in the paper; on the 5, the F1 of the new model run "
            "now (two code versions mixed; see the drift above).", ""] + body + [""]
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

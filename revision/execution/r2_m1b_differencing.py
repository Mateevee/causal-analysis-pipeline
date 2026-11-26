"""
R2-m1b (variant) -- Differencing of the unit-root series before the pipeline.

Request
  R2: "heterogeneous multivariate networks, where some series are strongly
       non-stationary and others behave as white noise".

The pipeline neither tests stationarity nor differences series. This variant,
outside the pipeline and without changing any of its defaults, tests whether
treating the unit-root series changes the result.

Design (fixed before running)
  For each scenario of the paper and each series (entity x property, with at
  least 30 values): "unit root" when the ADF (AIC) does not reject and the KPSS
  (level) rejects, at 5%. These series are replaced by their first
  difference; the first date is dropped in all series, to keep the
  alignment. Only the scenarios with at least one differenced series are
  run again (the others stay identical to the paper run).
  Model: patch_tau10, unchanged. Caveat: on a differenced series, theta
  measures the effect on the change, not on the level.
  Reporting: how many scenarios and series are affected, per family; F1 with and
  without differencing on those scenarios (paired Wilcoxon), on development and on held-out.

Usage
  python revision/execution/r2_m1b_differencing.py rodar [--filtro name ...]
  python revision/execution/r2_m1b_differencing.py avaliar
Output: Results_Revisao/r2_m1b_diferenciacao/
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

NAME = "r2_m1b_diferenciacao"


def unit_root(v) -> bool:
    from statsmodels.tsa.stattools import adfuller, kpss
    v = v[~np.isnan(v)]
    if len(v) < 30 or np.std(v) < 1e-12:
        return False
    try:
        return adfuller(v, autolag="AIC")[1] > 0.05 and kpss(v, regression="c", nlags="auto")[1] < 0.05
    except Exception:
        return False


def difference(d):
    """Returns (new entry or None, list of differenced series)."""
    df = d["df"].copy()
    props = [k for k in df.columns if k not in ("Date", "Name")]
    df = df.sort_values(["Name", "Date"])
    flagged = []
    for ent, g in df.groupby("Name"):
        for p in props:
            if unit_root(g[p].to_numpy(float)):
                flagged.append(f"{ent}.{p}")
                idx = g.index
                df.loc[idx, p] = g[p].diff().to_numpy()
    if not flagged:
        return None, []
    first = sorted(df["Date"].unique())[0]
    df = df[df["Date"] != first].reset_index(drop=True)
    return {**d, "df": df}, flagged


def run():
    import common_exec as ce
    name_filter = sys.argv[sys.argv.index("--filtro") + 1:] if "--filtro" in sys.argv else None
    base = ce.output_dir(NAME)
    folder = os.path.join(base, "run")
    new_entries, record = [], {}
    for d in ce.paper_scenarios(name_filter):
        nd, flagged = difference(d)
        record[d["name"]] = flagged
        if nd is not None:
            new_entries.append(nd)
    with open(os.path.join(base, "series_diferenciadas.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print(f"[{NAME}] {len(new_entries)} of {len(record)} scenarios with differenced series -> {folder}")
    ce.run_resuming(new_entries, ce.models(["patch_tau10"]), folder)


def evaluate():
    import common as c
    import common_exec as ce
    rel = c.Report(NAME, "R2-m1b — Differencing of the unit-root series")
    base = os.path.join(c.OUTPUT_ROOT, NAME)
    reg = json.load(open(os.path.join(base, "series_diferenciadas.json"), encoding="utf-8"))
    R = ce.summary(os.path.join(base, "run"))
    R = R[R.model == "patch_tau10"]
    rows = []
    for s, flg in reg.items():
        k = c.scenario_key(s)
        if not k:
            continue
        r = R[R["cenario"].map(c.O.canonical) == c.O.canonical(s)]
        f_dif = ce.f1(r.TP.iloc[0], r.FP.iloc[0], r.FN.iloc[0]) if len(r) else None
        rows.append({"cenario": k, "familia": c.family(k), "heldout": c.is_heldout(k), "series_dif": len(flg),
                     "f1_artigo": c.f1_model(k, c.PIPE), "f1_diferenciado": f_dif if flg else c.f1_model(k, c.PIPE)})
    D = pd.DataFrame(rows)
    rel.csv("por_cenario.csv", D)
    A = D[D["series_dif"] > 0]
    rel.p(f"Scenarios with at least one differenced series: {len(A)} of {len(D)}.")
    body = []
    for f, g in A.groupby("familia"):
        body.append([f, len(g), int(g["series_dif"].sum()), c.fmt(g["f1_artigo"].mean()), c.fmt(g["f1_diferenciado"].mean())])
    rel.table(["Family", "Affected scenarios", "Differenced series", "Paper F1", "F1 with differencing"], body)
    for part, m in (("development", ~A["heldout"]), ("held-out", A["heldout"])):
        g = A[m].dropna(subset=["f1_diferenciado"])
        if len(g):
            w, t, l = c.wins(g["f1_diferenciado"], g["f1_artigo"])
            rel.p(f"- {part}: {len(g)} scenarios; F1 {g['f1_artigo'].mean():.3f} -> {g['f1_diferenciado'].mean():.3f}; "
                  f"better/equal/worse {w}/{t}/{l}; Wilcoxon p = {c.wilcoxon_p(g['f1_diferenciado'], g['f1_artigo']):.3g}.")
    rel.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        if len(sys.argv) > 1 and sys.argv[1] == "rodar":
            run()
        else:
            evaluate()

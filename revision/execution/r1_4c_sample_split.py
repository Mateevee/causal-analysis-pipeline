"""
R1-4c / R3-2 -- Post-selection re-estimation on a separate sample.

Request
  R1: "The post-selection re-estimation you suggest to users should be
       demonstrated, not only recommended."
  R3: "selection of pairs, lags, models, and controls on the same data still
       limits the interpretation of the reported intervals."

Design (fixed before running)
  Scenarios: the five configurations of the paper's internal VAR generator
    (Config.execution.SYNTHETIC_VAR_SCENARIOS), with twice the observations and
    new seeds (4101-4110): 50 scenarios, none seen during development.
  Half A (selection): the paper's pipeline (patch_tau10) runs only on the first
    half, which has the same T as the paper's scenario; it yields the pair, the lag
    and the controls (the target's other candidate parents, from original.csv).
  Half B (inference): for each reported link with lag >= 1, theta is
    re-estimated on the second half by the Stage 6 DML (parent design,
    Bayesian Ridge, K = 3 contiguous blocks, Newey-West error), with the
    controls chosen in A. Nothing from B enters the selection.
  Reference: the standardized true coefficient (coef · sd(source)/sd(target),
    deviations of half B), as in Table S9.
  Reporting: on the true links reported at the true lag, coverage of the
    95% interval, correct sign, mean absolute error and standardized bias,
    for (i) the pipeline's own interval in A (selection and inference on the
    same data) and (ii) the re-estimation in B; on the false links, the fraction
    whose interval in B excludes zero.

Usage
  python revision/execution/r1_4c_sample_split.py rodar [--verificar]   (pipeline on half A; ~1 h)
  python revision/execution/r1_4c_sample_split.py avaliar
Output: Results_Revisao/r1_4c_divisao_amostra/
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
from Pipeline import dml_engine as E  # noqa: E402

NAME = "r1_4c_divisao_amostra"
SEEDS = list(range(4101, 4111))
K = 3
MAX_LAG = 10


def generate(check=False):
    """Scenarios with 2T observations; returns (list of A halves, {A name: df of half B})."""
    halves_a, halves_b = [], {}
    confs = cfg_exec.SYNTHETIC_VAR_SCENARIOS[:1] if check else cfg_exec.SYNTHETIC_VAR_SCENARIOS
    seeds = SEEDS[:1] if check else SEEDS
    for (prefix, _seed, n_points, n_ent, prob, mp, ml, me) in confs:
        tag = re.sub(r"^benchmark_", "", prefix)
        for d in ce.generate_var(seeds, 2 * n_points, n_ent, prob, mp, ml, me, name_prefix=f"revisao_split_{tag}"):
            dates = sorted(d["df"]["Date"].unique())
            cut = dates[len(dates) // 2]
            a = {**d, "name": d["name"] + "_A", "df": d["df"][d["df"]["Date"] < cut].reset_index(drop=True)}
            halves_a.append(a)
            halves_b[a["name"]] = d["df"][d["df"]["Date"] >= cut].reset_index(drop=True)
    return halves_a, halves_b


def run():
    check = "--verificar" in sys.argv
    base = ce.output_dir(NAME + ("_verificacao" if check else ""))
    a, b = generate(check)
    os.makedirs(os.path.join(base, "metade_B"), exist_ok=True)
    for name, df in b.items():
        df.to_csv(os.path.join(base, "metade_B", f"{name}.csv"), index=False)
    ce.run_resuming(a, ce.models(["patch_tau10"]), os.path.join(base, "run"))
    evaluate(base)


def lagged(x, l):
    out = np.full(len(x), np.nan)
    out[l:] = x[:-l]
    return out


def estimate(proc, controls, own, s, t, lag):
    W = E._parent_controls(s, t, lag, proc, controls, own, K, True)
    Tr, Ou = lagged(proc[s], lag), proc[t].copy()
    ok = ~np.isnan(Tr) & ~np.isnan(Ou) & ~np.any(np.isnan(W), axis=1)
    rT, rO, _ = E._run_crossfit("bayesian_ridge", Tr[ok], Ou[ok], W[ok], K)
    r = E._compute_theta_and_se(rT, rO, int(ok.sum()), None)
    return r["effect"], r["se"]


def evaluate(base=None):
    base = base or os.path.join(ce.OUTPUT_ROOT, NAME)
    run_dir = os.path.join(base, "run")
    rows = []
    for s in sorted(os.listdir(run_dir)):
        pa = os.path.join(run_dir, s)
        res = os.path.join(pa, "patch_tau10", "results.csv")
        fb = os.path.join(base, "metade_B", f"{s}.csv")
        if not (os.path.exists(res) and os.path.exists(fb)):
            continue
        txt = open(os.path.join(pa, "scenario_params.txt"), encoding="utf-8").read()
        edges = ast.literal_eval(re.search(r"causal_edges\s*:\s*(\[.*\])", txt).group(1))
        src = pd.read_csv(fb)
        col = [k for k in src.columns if k not in ("Date", "Name")][0]
        wide = src.pivot(index="Date", columns="Name", values=col)
        proc = {k: (wide[k].values - wide[k].values.mean()) / wide[k].values.std() for k in wide.columns}
        sd = src.groupby("Name")[col].std()
        true_edges = {(e["source"], e["target"]): (int(e["lag"]), e["coef"] * sd[e["source"]] / sd[e["target"]])
                      for e in edges}
        orig = pd.read_csv(os.path.join(pa, "patch_tau10", "original.csv"),
                           usecols=["source_entity", "target_entity", "lag"])
        cand = {}
        for r in orig.itertuples():
            if int(r.lag) >= 1:
                cand.setdefault(r.target_entity, []).append((r.source_entity, int(r.lag)))
        own = {k: E._ar_order_bic(proc[k], MAX_LAG) for k in proc}
        d = pd.read_csv(res)
        d = d[(d["significant"].astype(str) == "True") & (pd.to_numeric(d["lag"], errors="coerce") >= 1)]
        for r in d.itertuples():
            k = (r.source_entity, r.target_entity)
            lag = int(r.lag)
            eb, seb = estimate(proc, cand, own, k[0], k[1], lag)
            v = true_edges.get(k)
            rows.append({"cenario": s, "fonte": k[0], "alvo": k[1], "lag": lag,
                         "verdadeiro": v is not None, "lag_correto": bool(v and v[0] == lag),
                         "ref": v[1] if v and v[0] == lag else 0.0,
                         "theta_A": r.causal_effect, "lo_A": r.ci_95_lower, "hi_A": r.ci_95_upper,
                         "theta_B": eb, "lo_B": eb - 1.96 * seb, "hi_B": eb + 1.96 * seb, "se_B": seb})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(base, "links.csv"), index=False)
    V = D[D["lag_correto"]]
    F = D[~D["verdadeiro"]]
    txt = ["# R1-4c / R3-2 — Re-estimation on a separate sample", "",
           f"{D['cenario'].nunique()} scenarios; {len(D)} links reported with lag >= 1 on half A; "
           f"{len(V)} true at the true lag; {len(F)} false.", "",
           "| Interval | Links | 95% coverage | Correct sign | Mean absolute error | Mean standardized bias |",
           "|---|---|---|---|---|---|"]
    for name, th, lo, hi in (("pipeline on A (selection and inference on the same data)", "theta_A", "lo_A", "hi_A"),
                             ("re-estimation on B (data that did not enter the selection)", "theta_B", "lo_B", "hi_B")):
        if len(V):
            cov = ((V[lo] <= V["ref"]) & (V["ref"] <= V[hi])).mean()
            sign = (np.sign(V[th]) == np.sign(V["ref"])).mean()
            se = (V[hi] - V[lo]) / 3.92
            bias = (((V[th] - V["ref"]) / se) * np.sign(V["ref"])).mean()
            txt.append(f"| {name} | {len(V)} | {cov:.1%} | {sign:.1%} | {(V[th] - V['ref']).abs().mean():.3f} | {bias:+.2f} |")
    if len(F):
        txt += ["", f"False links: interval on B excludes zero in {((F['lo_B'] > 0) | (F['hi_B'] < 0)).mean():.1%} "
                    f"(on A, by construction, in {((F['lo_A'] > 0) | (F['hi_A'] < 0)).mean():.1%})."]
    open(os.path.join(base, "relatorio.md"), "w", encoding="utf-8").write("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        if len(sys.argv) > 1 and sys.argv[1] == "rodar":
            run()
        else:
            evaluate()

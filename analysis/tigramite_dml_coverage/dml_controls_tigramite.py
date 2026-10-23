"""
Coverage of the Stage 6 DML intervals on the tigramite SCMs.

Extends generators/dml_controls.py (the controls panel of Table S9, which
covers only the synthetic VAR scenarios) to the 15 tigramite scenarios (five
SCMs x seeds 11, 22, 33), at both windows. Hypothesis: the reference of each
link (the coefficient of X_{t-tau} in the least-squares regression of Y_t on
X_{t-tau} and Y_{t-1}, on a 500,000-point simulation; Table S9, panels 7-8)
and the pipeline's theta condition on different controls, and on a nonlinear
link that changes the quantity being estimated.

Steps:
  1. Reproduction. For each scenario, runs Stages 0-2 of the pipeline itself
     (Pipeline/pipeline.py) on the stored source.csv, takes the candidates of
     original.csv and re-runs Stage 6 (estimate_effects_by_control_set, with
     the run's configuration: BR -> RF -> KR -> NW cascade, selection by
     nuisance loss, parent-based or blanket design chosen from the data).
     The re-estimated theta must match the reported one on every evaluated
     link; otherwise the diagnosis is void.
  2. Re-estimates each true link reported at its true lag with the SAME
     nuisance learner the pipeline reported for it, under three control sets:
       pipeline     those of Stage 6 (parent-based: the target's past up to
                    its BIC order + the other candidate parents; blanket:
                    lags 1..window of every series) -- must reproduce the
                    reported value;
       Y_{t-1}      the target's first lag only (the reference's controls);
       true         the target's true past and its other true parents (in
                    these SCMs every target has a single cross-parent, so
                    this set coincides with Y_{t-1}).
     The three are repeated with Bayesian ridge on every link (with Y_{t-1},
     Bayesian ridge is the exact sample analogue of the reference).

Only the DML of these links is re-run. Writes results/RESULTS.md,
results/links_tau{10,24}.csv and results/stage6_reproduction_tau{10,24}.csv.

Usage (from the repository root):
    python analysis/tigramite_dml_coverage/dml_controls_tigramite.py
"""
import contextlib
import inspect
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "results")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "generators"))
import origins as O  # noqa: E402
from Pipeline import pipeline as PL  # noqa: E402
from Pipeline import dml_engine as E  # noqa: E402
from Config import execution as X  # noqa: E402
from Data import tigramite_scm as TG  # noqa: E402
from tigramite.toymodels import structural_causal_processes as toys  # noqa: E402

RUN = O.require("MAIN_RUN")
SEEDS = [11, 22, 33]
WINDOWS = [("10", "patch_tau10"), ("24", "patch")]
T_LONG = 500_000
TOL = 1e-4          # original.csv / results.csv store theta with 4 decimals
FORM = {"lin": "linear", "quad": "sign-preserving quadratic", "threshold": "step (threshold)",
        "tanh": "saturating (tanh)"}

_DEFAULTS = {k: v.default for k, v in inspect.signature(PL.run_pipeline).parameters.items()
             if v.default is not inspect._empty}


def params(model):
    """Arguments of run_pipeline for `model`, as main.py builds them."""
    m = next(m for m in X.ALL_MODELS if m["name"] == model)
    return {**_DEFAULTS, **X.PIPELINE_CONFIG, **m["config_override"], "n_jobs": 1}


def key(ent, prop):
    return f"{ent}__{prop}"


def _ols(y, X_):
    A = np.column_stack([np.ones(len(y))] + X_)
    return np.linalg.lstsq(A, y, rcond=None)[0][1:]


def references():
    """(scenario, source, target) -> dict(lag, form, b, c), the same computation
    as generators/additional_analyses.py (Table S9, panel 8); and the true
    parents of every target."""
    ref, truth = {}, {}
    for sc, sdef in TG._BENCHMARK_SCENARIO_DEFINITIONS.items():
        links, npr = sdef["links"], sdef["n_properties"]
        long, _ = toys.structural_causal_process(links=links, T=T_LONG, seed=0)
        name = lambda v: key(f"ENT_{v // npr}", f"F_{v % npr}")  # noqa: E731
        truth[sc] = {name(t): [(name(s), abs(tau)) for (s, tau), _, _ in par] for t, par in links.items()}
        for tgt, par in links.items():
            for (s, tau), coef, fn in par:
                if s == tgt:
                    continue
                tau = abs(tau)
                y, x = long[:, tgt], long[:, s]
                b = _ols(y[tau + 1:], [x[1:len(x) - tau], y[tau:-1]])[0]
                ref[(sc, name(s), name(tgt))] = dict(lag=tau, form=fn.__name__.replace("_f_", ""), b=b, c=coef)
    return ref, truth


def stage6(base, model):
    """Pipeline Stages 0-2 + Stage 6 on the stored candidates."""
    p = params(model)
    p["df"] = pd.read_csv(os.path.join(base, "source.csv"))
    run = PL._Run(p)
    with contextlib.redirect_stdout(io.StringIO()):
        PL._stage0_structure(run); PL._stage1_active_properties(run); PL._stage2_preprocessing(run)
    orig = pd.read_csv(os.path.join(base, model, "original.csv"))
    cands = [dict(source=key(r.source_entity, r.source_property), target=key(r.target_entity, r.target_property),
                  lag=int(r.lag)) for r in orig.itertuples()]
    with contextlib.redirect_stdout(io.StringIO()):
        est, lean, _ = E.estimate_effects_by_control_set(
            cands, run.processed, run.pairs,
            control_set=p["dml_control_set"], source_block=p["dml_source_block"],
            auto_max_outcome_r2=p["dml_auto_max_outcome_r2"], max_lag=p["max_lag"], n_splits=p["dml_splits"],
            n_jobs=1, dml_model=p["dml_model"], var_res_T_threshold=p["var_res_T_threshold"],
            nuisance_max_controls=p["dml_nuisance_max_controls"],
            auto_nuisance_screening=p["dml_auto_nuisance_screening"],
            estimate_selection=p["dml_estimate_selection"], cascade_decision=p["dml_cascade_decision"])
    return p, run, cands, est, lean


def dml(p, run, src, tgt, lag, model, W=None, blanket=False):
    """One link through the pipeline's _dml_single, with a single nuisance learner."""
    proc = run.processed
    n = len(proc[tgt])
    if blanket:
        others = [k for k in run.pairs if k not in (src, tgt)]
        controls = np.column_stack([proc[k] for k in others]) if others else np.zeros((n, 1))
        opts = {"cascade_decision": p["dml_cascade_decision"]}
    else:
        controls = np.zeros((n, 0))
        opts = {"cascade_decision": p["dml_cascade_decision"], "W": W}
    task = (src, tgt, lag, proc[src].copy(), proc[tgt].copy(), controls, p["max_lag"], p["dml_splits"], None,
            model, p["var_res_T_threshold"], p["dml_estimate_selection"], None, opts)
    r = E._dml_single(task)
    return r["effect"], r["se"]


def window_links(model, ref, truth):
    rows, repro = [], []
    for sc in TG._BENCHMARK_SCENARIO_DEFINITIONS:
        for sd in SEEDS:
            name = f"benchmark_tigramite_{sc}_seed{sd}"
            base = os.path.join(RUN, name)
            p, run, cands, est, lean = stage6(base, model)
            proc = run.processed
            idx = {(c["source"], c["target"], c["lag"]): i for i, c in enumerate(cands)}
            orig = pd.read_csv(os.path.join(base, model, "original.csv"))
            d_all = np.abs(np.array([e["effect"] for e in est]) - orig.causal_effect.values)
            repro.append(dict(scenario=name, design="parent-based" if lean else "blanket", n_candidates=len(cands),
                              max_abs_diff_all_candidates=d_all.max()))
            src_df = pd.read_csv(os.path.join(base, "source.csv")).set_index("Name")
            res = pd.read_csv(os.path.join(base, model, "results.csv"))
            res = res[res["significant"].astype(str) == "True"]
            parents_of = {}
            for c in cands:
                parents_of.setdefault(c["target"], []).append((c["source"], c["lag"]))
            own = {k: E._ar_order_bic(proc[k], p["max_lag"])
                   for k in {c["source"] for c in cands} | {c["target"] for c in cands}}
            for (sc_, s, t), rf in ref.items():
                if sc_ != sc:
                    continue
                (se_, sp), (te, tp) = s.split("__"), t.split("__")
                m = res[(res.source_entity == se_) & (res.source_property == sp)
                        & (res.target_entity == te) & (res.target_property == tp)]
                if not len(m) or int(m.iloc[0].lag) != rf["lag"]:
                    continue        # only the links reported at the true lag (panel 7)
                r = m.iloc[0]
                lag = rf["lag"]
                sx, sy = src_df.loc[se_, sp].std(), src_df.loc[te, tp].std()
                i = idx[(s, t, lag)]
                rep_model = str(r.dml_model_used).split("[")[0]
                W_pipe = (E._parent_controls(s, t, lag, proc, parents_of, own, p["dml_splits"], True)
                          if lean else None)
                W_y1 = E._lagged(proc[t], 1).reshape(-1, 1)
                W_true = np.column_stack([E._lagged(proc[q], l) for q, l in truth[sc][t] if q != s])
                row = dict(scenario=name, design="parent-based" if lean else "blanket", source=s, target=t, lag=lag,
                           form=FORM[rf["form"]], ref=rf["b"] * sx / sy, learner=rep_model,
                           rep=r.causal_effect, rep_lo=r.ci_95_lower, rep_hi=r.ci_95_upper,
                           cascade=est[i]["effect"], cascade_se=est[i]["se"],
                           n_controls_pipeline=(W_pipe.shape[1] if lean else None))
                for prefix, mod in (("", rep_model), ("br_", "bayesian_ridge")):
                    row[f"{prefix}pipeline"], row[f"{prefix}pipeline_se"] = dml(p, run, s, t, lag, mod, W_pipe,
                                                                                blanket=not lean)
                    row[f"{prefix}y1"], row[f"{prefix}y1_se"] = dml(p, run, s, t, lag, mod, W_y1)
                    row[f"{prefix}true"], row[f"{prefix}true_se"] = dml(p, run, s, t, lag, mod, W_true)
                row["true_equals_y1"] = bool(np.allclose(np.nan_to_num(W_true), np.nan_to_num(W_y1)))
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(repro)


def summary(d, est, se=None):
    """Coverage, signed z (negative = shrinkage toward zero), SD of z, relative error of |theta|, MAE."""
    if se is None:      # as reported: the stored interval
        cov = (d.rep_lo <= d.ref) & (d.ref <= d.rep_hi)
        s = (d.rep_hi - d.rep_lo) / 3.92
        th = d.rep
    else:
        th, s = d[est], d[se]
        cov = (th - d.ref).abs() <= 1.96 * s
    z = (th - d.ref) / s * np.sign(d.ref)
    return dict(n=len(d), cov=cov.mean(), z=z.mean(), zsd=z.std(),
                shrink=((th.abs() - d.ref.abs()) / d.ref.abs()).mean(), mae=(th - d.ref).abs().mean())


ROWS = [("As reported (pipeline output)", "rep", None),
        ("Pipeline controls, reported learner (re-estimated)", "pipeline", "pipeline_se"),
        ("Only Y_{t−1} (the reference's controls), reported learner", "y1", "y1_se"),
        ("True parents, reported learner", "true", "true_se"),
        ("Pipeline controls, Bayesian ridge", "br_pipeline", "br_pipeline_se"),
        ("Only Y_{t−1}, Bayesian ridge", "br_y1", "br_y1_se"),
        ("True parents, Bayesian ridge", "br_true", "br_true_se")]


def table(d, title):
    L = [f"**{title}** ({len(d)} links)", "",
         "| Controls | 95% CI coverage | Mean (θ̂ − θ)/SE, sign-adjusted | SD of (θ̂ − θ)/SE "
         "| Mean relative error of \\|θ̂\\| | MAE |", "|---|---|---|---|---|---|"]
    for lab, est, se in ROWS:
        s = summary(d, est, se)
        L.append(f"| {lab} | {s['cov']:.1%} | {s['z']:+.2f} | {s['zsd']:.2f} | {s['shrink']:+.1%} | {s['mae']:.3f} |")
    return L + [""]


def _counts(series):
    return ", ".join(f"{k} {v}" for k, v in series.value_counts().items())


def main():
    os.makedirs(OUT, exist_ok=True)
    ref, truth = references()
    out = ["# Coverage of the DML intervals on the tigramite SCMs — results", "",
           f"Written by `dml_controls_tigramite.py` from `{os.path.basename(os.path.normpath(RUN))}` "
           "(15 scenarios: 5 SCMs × seeds 11, 22, 33). Engine: `Pipeline/dml_engine.py` (`_dml_single`), "
           "K = 3 contiguous blocks, HAC standard error, as in the run.", "",
           "Reference θ of each link: the coefficient of X_{t−τ} in the least-squares regression of Y_t on X_{t−τ} "
           f"and Y_{{t−1}}, on a {T_LONG:,}-point simulation of the same SCM (seed 0), times σ_X/σ_Y of the sample "
           "— the computation of Table S9, panels 7–8.", ""]
    for w, model in WINDOWS:
        d, repro = window_links(model, ref, truth)
        d.to_csv(os.path.join(OUT, f"links_tau{w}.csv"), index=False)
        repro.to_csv(os.path.join(OUT, f"stage6_reproduction_tau{w}.csv"), index=False)
        diff_casc = (d.rep - d.cascade).abs()
        diff_pipe = (d.rep - d["pipeline"]).abs()
        ok = bool((diff_casc <= TOL).all() and (diff_pipe <= TOL).all())
        out += [f"## τ_max = {w}", "",
                "### 1. Reproduction of the reported θ̂", "",
                f"- True links reported at the true lag: **{len(d)}** (reported learner: {_counts(d.learner)}).",
                f"- Control design chosen by Stage 6: scenarios {_counts(repro.design)}; links {_counts(d.design)}.",
                f"- Stage 6 re-run (full cascade) vs reported: max |Δθ| = {diff_casc.max():.1e}.",
                f"- Re-estimated with the reported learner and the pipeline's controls vs reported: "
                f"max |Δθ| = {diff_pipe.max():.1e}.",
                f"- Every candidate of the 15 scenarios (not only the true links): max |Δθ| = "
                f"{repro.max_abs_diff_all_candidates.max():.1e}; scenarios with some |Δθ| > {TOL:g}: "
                f"{int((repro.max_abs_diff_all_candidates > TOL).sum())} (false links estimated by random forest, "
                "whose feature sampling depends on the column order of the controls).",
                f"- **Reproduction on the evaluated links: {'OK' if ok else 'FAILED'}** (tolerance {TOL:g}, the "
                "4-decimal rounding of the CSV).", "",
                f"- True parents = Y_{{t−1}} only on {int(d.true_equals_y1.sum())} of {len(d)} links (every target "
                "of these SCMs has a single cross-parent and first-order autodependence).", "",
                "### 2. Coverage by control set", ""]
        out += table(d, "All links")
        out += table(d[d.form == "linear"], "Linear links")
        out += table(d[d.form != "linear"], "Nonlinear links")
        for des in ("parent-based", "blanket"):
            if (d.design == des).any():
                out += table(d[d.design == des], f"{des.capitalize()} design")
        out += ["95% CI coverage by functional form:", "",
                "| Form | n | Reported | Pipeline, reported learner | Only Y_{t−1}, reported learner "
                "| Pipeline, BR | Only Y_{t−1}, BR |", "|---|---|---|---|---|---|---|"]
        for f_, g in d.groupby("form", sort=False):
            c = [summary(g, e, s)["cov"] for e, s in [("rep", None), ("pipeline", "pipeline_se"), ("y1", "y1_se"),
                                                     ("br_pipeline", "br_pipeline_se"), ("br_y1", "br_y1_se")]]
            out.append(f"| {f_} | {len(g)} | " + " | ".join(f"{x:.0%}" for x in c) + " |")
        out += ["", "Coverage of the reported intervals by seed:", "",
                "| Seed | Links covered |", "|---|---|"]
        cov = (d.rep_lo <= d.ref) & (d.ref <= d.rep_hi)
        for sd, g in cov.groupby(d.scenario.str.extract(r"seed(\d+)")[0]):
            out.append(f"| {sd} | {int(g.sum())}/{len(g)} |")
        out.append("")
    open(os.path.join(OUT, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("ok ->", os.path.join(OUT, "RESULTS.md"))


if __name__ == "__main__":
    main()

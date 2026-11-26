"""
R1-2a / R2-3a -- Tuned baselines.

Request
  R1: "Please add (a) per-benchmark tuned baselines (or at least a small grid)".
  R2: "the tuning protocol is asymmetric ... the baselines ... were run with their
       default published parameters."

Design (fixed before running)
  The same access the pipeline had: for each baseline, the configuration is
  chosen by mean F1 on the 51 development scenarios and applied, unchanged, to
  the 39 held-out ones. An upper bound (configuration chosen per scenario, with
  the scenario's own ground truth) is reported apart, like the oracles of
  Table VI. Window tau_max = 10. Ties in the selection: the default stays.

  Grid (around the published values; each method's default is in it):
    PCMCI       final alpha {0.01; 0.025; 0.05; 0.1} x correction {BH, none}
                (offline, from the stored MCI p-values of every link) and
                pc_alpha {0.05; 0.1; 0.2; 0.4} (rerun, alpha 0.05 with BH);
    PCMCI+      pc_alpha {0.001; 0.005; 0.01; 0.05; 0.1} (rerun);
    DYNOTEARS   lambda {0.01; 0.05; 0.1} x (tau_W, tau_A) {(0.3; 0.1); (0.2; 0.05);
                (0.4; 0.2)} (rerun);
    CUTS+       score threshold {0.3; 0.4; 0.5; 0.6; 0.7} (rerun);
    VARLiNGAM   order by {AIC, BIC} (rerun); and, for DYNOTEARS, CUTS+ and
                VARLiNGAM, the post hoc F-test gate of each wrapper,
                alpha {0.01; 0.05} x correction {none, BH}, on the stored native
                graph (offline; it only removes edges).

  Report: chosen configuration; mean F1 on the development suite and on the
  held-out datasets; pipeline against each tuned baseline on the held-out
  datasets (two-sided paired Wilcoxon, Holm over the five); mean rank of the six
  with the tuned baselines, Friedman and Nemenyi critical difference; and the
  upper bound.

Usage
  python revision/execution/r1_2a_tuned_baselines.py rodar   [--filtro name] (hours; resumes)
  python revision/execution/r1_2a_tuned_baselines.py rodar --paralelo 20
      (one process per scenario, up to 20 at a time, one thread each; ~1-2 h on a 24-core machine;
       progress in Results_Revisao/r1_2a_baselines_ajustados/progresso.log; resumes)
  python revision/execution/r1_2a_tuned_baselines.py avaliar [--verificar]
Output: Results_Revisao/r1_2a_baselines_ajustados/ (run/ with the reruns)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

NAME = "r1_2a_baselines_ajustados"

# ---------------------------------------------------------------- rerun grid
GRID_RERUN = (
    [("pcmci_tigramite_tau10", f"pcmci_pca{a}_tau10", {"pcmci_pc_alpha": a}) for a in (0.05, 0.1, 0.4)]
    + [("pcmciplus_tau10", f"pcmciplus_pca{a}_tau10", {"pcmciplus_pc_alpha": a})
       for a in (0.001, 0.005, 0.05, 0.1)]
    + [("dynotears_tau10", f"dynotears_l{lam}_w{w}_a{a}_tau10",
        {"dynotears_lambda_w": lam, "dynotears_lambda_a": lam, "dynotears_w_threshold": w,
         "dynotears_a_threshold": a})
       for lam in (0.01, 0.05, 0.1) for (w, a) in ((0.3, 0.1), (0.2, 0.05), (0.4, 0.2))
       if not (lam == 0.05 and (w, a) == (0.3, 0.1))]
    + [("cuts_plus_tau10", f"cuts_plus_th{t}_tau10", {"cuts_causal_threshold": t}) for t in (0.3, 0.4, 0.6, 0.7)]
    + [("varlingam_tau10", "varlingam_bic_tau10", {"varlingam_criterion": "bic"})]
)
FAMILY_OF = {"pcmci": "pcmci_tigramite", "pcmciplus": "pcmciplus", "dynotears": "dynotears",
             "cuts_plus": "cuts_plus", "varlingam": "varlingam"}


def base_of(model_name: str) -> str:
    for pre, b in (("pcmciplus", "pcmciplus"), ("pcmci", "pcmci_tigramite"), ("dynotears", "dynotears"),
                   ("cuts_plus", "cuts_plus"), ("varlingam", "varlingam")):
        if model_name.startswith(pre):
            return b
    raise KeyError(model_name)


# ---------------------------------------------------------------- run
def run():
    import common_exec as ce
    name_filter = sys.argv[sys.argv.index("--filtro") + 1:] if "--filtro" in sys.argv else None
    folder = ce.output_dir(os.path.join(NAME, "run"))
    dfs = ce.paper_scenarios(name_filter)
    mods = [ce.model(b, n, **o) for b, n, o in GRID_RERUN]
    print(f"[{NAME}] {len(dfs)} scenarios x {len(mods)} configurations -> {folder}")
    if "--paralelo" in sys.argv:
        ce.run_parallel(dfs, GRID_RERUN, folder, int(sys.argv[sys.argv.index("--paralelo") + 1]))
    else:
        ce.run_resuming(dfs, mods, folder)


# ---------------------------------------------------------------- evaluate
def f1_of_set(S, T):
    tp, fp, fn = len(S & T), len(S - T), len(T - S)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def offline_grid(c, scen) -> dict:
    """{config: F1} of the variants that can be computed from the stored files of the paper's run."""
    T, _ = c.truth(scen)
    out = {}
    # PCMCI: final alpha x correction. With BH it uses the stored q-values (pvalue_corrected),
    # which are tigramite's own (family N^2 * tau_max, self-edges included) and
    # do not depend on alpha; without correction, the raw MCI p-value.
    d = c.original(scen, "pcmci_tigramite_tau10", c.folder(scen, "pcmci_tigramite"))
    if not d.empty and "pvalue" in d.columns:
        p = pd.to_numeric(d["pvalue"], errors="coerce").fillna(1.0).to_numpy()
        pc = pd.to_numeric(d["pvalue_corrected"], errors="coerce").fillna(1.0).to_numpy()
        lag = pd.to_numeric(d["lag"], errors="coerce").fillna(0).to_numpy()
        pr = c.pairs(d).to_numpy()
        for corr in ("fdr_bh", "none"):
            q = pc if corr == "fdr_bh" else p
            for a in (0.01, 0.025, 0.05, 0.1):
                S = {pr[i] for i in range(len(p)) if q[i] < a and lag[i] >= 1}
                out[f"pcmci_alpha{a}_{corr}"] = ("pcmci_tigramite", f1_of_set(S, T))
    # post hoc F-test gate on the stored native graph
    for b in ("dynotears", "cuts_plus", "varlingam"):
        if b not in c.rows()[scen]:
            continue
        d = c.original(scen, f"{b}_tau10", c.folder(scen, b))
        r = pd.read_csv(os.path.join(c.folder(scen, b), f"{b}_tau10", "results.csv"))
        if d.empty or r.empty or "pvalue" not in r.columns:
            continue
        r["src"], r["dst"] = r["source_entity"].astype(str).str.strip(), r["target_entity"].astype(str).str.strip()
        p = pd.to_numeric(r["pvalue"], errors="coerce").fillna(1.0).to_numpy()
        for corr in ("none", "fdr_bh"):
            q = multipletests(p, method="fdr_bh")[1] if corr == "fdr_bh" and len(p) else p
            for a in (0.01, 0.05):
                S = {(s, t) for s, t, qq in zip(r["src"], r["dst"], q) if qq < a}
                out[f"{b}_ftest{a}_{corr}"] = (b, f1_of_set(S, T))
    return out


def f1_table(c, scens) -> pd.DataFrame:
    """Rows (scenario, baseline, config, F1): the paper's defaults, the offline grid and the reruns."""
    rows = []
    for s in scens:
        for b in c.BASELINES:
            f = c.f1_model(s, b)
            if f is not None:
                rows.append((s, b, "default", f))
        for cfg, (b, f) in offline_grid(c, s).items():
            rows.append((s, b, cfg, f))
    run = os.path.join(c.OUTPUT_ROOT, NAME, "run")
    if os.path.isdir(run):
        for s in scens:
            ps = c.folder_in(run, s)
            p = os.path.join(ps, "validation_summary.csv") if ps else ""
            if not os.path.exists(p):
                continue
            v = pd.read_csv(p)
            for r in v.itertuples():
                if r.status == "ok":
                    rows.append((s, base_of(r.model), r.model, f1_of_counts(r.TP, r.FP, r.FN)))
    return pd.DataFrame(rows, columns=["cenario", "baseline", "config", "f1"])


def f1_of_counts(tp, fp, fn):
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def ranks(M):
    rk = np.array([rankdata(-r) for r in M]).mean(axis=0)
    k, n = M.shape[1], M.shape[0]
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    p = friedmanchisquare(*M.T).pvalue if n > 1 else float("nan")
    return rk, cd, p


def evaluate():
    import common as c
    check = "--verificar" in sys.argv
    rel = c.Report(NAME + ("_verificacao" if check else ""), "R1-2a / R2-3a — Tuned baselines")
    scens = c.scenarios()
    if check:
        scens = c.scenarios("dev")[:4] + c.scenarios("ho")[:4]
    dev = [s for s in scens if not c.is_heldout(s)]
    ho = [s for s in scens if c.is_heldout(s)]
    F = f1_table(c, scens)
    rel.csv("f1_por_config.csv", F)

    # check: the offline variant (0.05, BH) of PCMCI must reproduce the PCMCI of the paper
    a = F[F.config == "pcmci_alpha0.05_fdr_bh"].set_index("cenario")["f1"]
    b = F[(F.baseline == "pcmci_tigramite") & (F.config == "default")].set_index("cenario")["f1"]
    common_idx = a.index.intersection(b.index)
    rep = float((np.abs(a[common_idx] - b[common_idx]) < 1e-6).mean()) if len(common_idx) else float("nan")
    rel.p(f"Check: offline PCMCI (alpha 0.05, BH) reproduces the paper's F1 on {100 * rep:.0f}% "
          f"of the {len(common_idx)} scenarios.")
    rel.data["reproducao_pcmci"] = rep

    rel.h("## Configuration chosen on development and applied to held-out")
    choice, body, F_tuned = {}, [], {}
    for bl in c.BASELINES:
        G = F[F.baseline == bl]
        med = G[G.cenario.isin(dev)].groupby("config")["f1"].agg(["mean", "count"])
        if med.empty:
            rel.p(f"- {c.LABEL.get(bl, bl)}: did not run on any development scenario of this selection.")
            F_tuned[bl] = {}
            continue
        n_max = med["count"].max()
        med = med[med["count"] == n_max]        # only configurations that ran on every scenario
        best = med["mean"].idxmax()
        if abs(med.loc[best, "mean"] - med.loc["default", "mean"]) < 1e-12:
            best = "default"
        choice[bl] = best
        piv = G.pivot_table(index="cenario", columns="config", values="f1")
        F_tuned[bl] = piv[best].to_dict()
        oracle = piv.max(axis=1)
        fd = lambda sc, col: c.fmt(np.nanmean([piv[col].get(s, np.nan) for s in sc]))  # noqa: E731
        body.append([c.LABEL.get(bl, bl), best, len(piv.columns), fd(dev, "default"), fd(dev, best),
                     fd(ho, "default"), fd(ho, best), c.fmt(np.nanmean([oracle.get(s, np.nan) for s in ho]))])
    rel.table(["Baseline", "Chosen configuration", "Configurations in the grid", "F1 dev (default)",
               "F1 dev (tuned)", "F1 held-out (default)", "F1 held-out (tuned)",
               "F1 held-out (upper bound: best per scenario)"], body)
    rel.data["escolha"] = choice

    rel.h("## Pipeline against each tuned baseline")
    pip = {s: c.f1_model(s, c.PIPE) for s in scens}
    for part, sc in (("development", dev), ("held-out", ho)):
        pv, body = {}, []
        for bl in c.BASELINES:
            ss = [s for s in sc if s in F_tuned[bl] and not np.isnan(F_tuned[bl][s])]
            x, y = [pip[s] for s in ss], [F_tuned[bl][s] for s in ss]
            pv[bl] = c.wilcoxon_p(x, y)
            w, t, l = c.wins(x, y)
            body.append([c.LABEL.get(bl, bl), len(ss), c.fmt(np.mean(x)), c.fmt(np.mean(y)),
                         f"{np.mean(x) - np.mean(y):+.3f}", f"{w} / {t} / {l}"])
        hp = c.holm(pv)
        for r, bl in zip(body, c.BASELINES):
            r += [f"{pv[bl]:.3g}", f"{hp[bl]:.3g}"]
        rel.p(f"**{part}**")
        rel.table(["Tuned baseline", "Scenarios", "F1 pipeline", "F1 baseline", "Difference",
                   "Pipeline higher / equal / lower", "Wilcoxon p", "Holm p"], body)
        rel.p("")
        ss = [s for s in sc if all(s in F_tuned[bl] and not np.isnan(F_tuned[bl][s]) for bl in c.BASELINES)]
        if len(ss) > 1:
            M = np.array([[pip[s]] + [F_tuned[bl][s] for bl in c.BASELINES] for s in ss])
            rk, cd, pf = ranks(M)
            rel.p(f"Mean rank ({len(ss)} scenarios on which all six run; Friedman p = {pf:.2g}; "
                  f"Nemenyi critical difference {cd:.2f}): pipeline {rk[0]:.2f}; "
                  + "; ".join(f"{c.LABEL.get(bl, bl)} {r:.2f}" for bl, r in zip(c.BASELINES, rk[1:])))
            rel.p("")
    rel.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        if len(sys.argv) > 1 and sys.argv[1] == "rodar":
            run()
        else:
            evaluate()

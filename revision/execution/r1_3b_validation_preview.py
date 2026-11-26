"""
R1-3b -- PREVIEW on the VALIDATION data of R1-2c (request of the author, 2026-11-14:
"let's do this simplified test with the smoothing and compare with the original model").

Design (fixed before running; a PREVIEW -- it gives an idea and is NOT R1-2c nor the evaluation of R1-3b)
  Scenarios: the VALIDATION set of the R1-2c protocol (section 3), whole, without choice:
    wind tunnel, walks 1 and 4 to 9 (7), and DREAM3 size 50, the five networks (5) = 12.
    No method of the paper had run these data. By the rule of the protocol, they never enter the report
    of R1-2c; the TEST set is neither loaded nor touched here.
  Loading: the readers and parameters of section 4 of the R1-2c protocol (Causal Chambers with
    n_points = 500, offset = 0, step = 1; DREAM3 with sizes = (50,), the rest by default).
  Methods (8): patch_tau10 (the original default configuration), patch_tau10_final, patch_tau10_graded (the
    R1-3b variant, hash 90a38422...) and the 5 baselines of the paper at tau = 10.
  Execution: as in the runs of the paper (sequential; main.run_all + main.validate_all). The 6 methods of
    the paper run WITHOUT the guard swap; the final and graded ones run with it installed, through the same
    path as R1-3b (_run_list, with diagnostics), whose identity with the final one was checked
    byte by byte (revision/variants/check_r1_3b.py). method_code_hash must be d0473069ebef1978 (in this
    release, a different hash prints a warning; see revision/README.md, Notes).
  Report (fixed beforehand):
    - F1 per scenario and per method; mean per family (wind tunnel, DREAM3-50) and over the set;
    - graded - original and graded - final, per scenario; number of scenarios on which they differ;
    - mean rank per scenario among the 8;
    - paired (two-sided) Wilcoxon test of the original, the final and the graded against each baseline, with Holm
      over the 5 baselines; and of the graded and the final against the original. With n = 12 the power is low:
      it is descriptive;
    - weights of the graded variant (w_inc, w_lean) per scenario, to say where it can differ.
  Limitations: 12 scenarios of two families already seen (DREAM3 and the Causal Chambers device);
    without the external regime; a single run.
  Its run folder is also an input of the offline analyses in revision/offline/incoherent_improvement_prototype/
  (the set "val12").

Usage
  python revision/execution/r1_3b_validation_preview.py carregar   # counts only
  python revision/execution/r1_3b_validation_preview.py rodar
  python revision/execution/r1_3b_validation_preview.py avaliar
Output: Results_Revisao/r1_3b_previa_validacao/ (file and column names kept in Portuguese; see revision/README.md,
Notes)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
for _p in (HERE, REV, os.path.dirname(REV)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r1_3b_graded_transitions as R  # noqa: E402

NAME = "r1_3b_previa_validacao"
R.NAME = NAME
WT_EXPERIMENTS = [("wt_walks_v1", f"actuators_random_walk_{i}") for i in (1, 4, 5, 6, 7, 8, 9)]
ORIGINAL = "patch_tau10"
BASELINES = ["pcmci_tigramite_tau10", "pcmciplus_tau10", "dynotears_tau10", "cuts_plus_tau10", "varlingam_tau10"]
METHODS = [ORIGINAL, "patch_tau10_final", "patch_tau10_graded"] + BASELINES


def scenarios():
    import common_exec as ce  # noqa: F401  (chdir to the root)
    from Data.causal_chamber import generate_causal_chamber_benchmark_scenarios
    from Data.dream3 import generate_dream3_benchmark_scenarios
    wt = generate_causal_chamber_benchmark_scenarios(experiments=WT_EXPERIMENTS, n_points=500, offset=0, step=1)
    d3 = generate_dream3_benchmark_scenarios(sizes=(50,))
    dfs = list(wt) + list(d3)
    assert len(dfs) == 12, [d["name"] for d in dfs]
    return dfs


def family(name):
    return "tunel_vento" if "causal_chamber" in name else "dream3_50"


def load():
    for d in scenarios():
        df = d.get("df")
        gt = d.get("gt")
        print(d["name"], "| shape", getattr(df, "shape", None), "| ground-truth edges",
              None if gt is None else len(gt))


def run():
    import common_exec as ce
    from variants import r1_3b_graded as g
    R._check_hash()
    dfs = scenarios()
    folder = ce.output_dir(os.path.join(NAME, "run"))
    # 1) the 6 methods of the paper, without the guard swap
    ce.run_resuming(dfs, ce.models([ORIGINAL] + BASELINES), folder)
    # 2) final and graded, with the swap installed (path of R1-3b, with diagnostics)
    orig = g.install()
    try:
        R._run_list(ce, g, [(d, v, "padrao") for d in dfs for v in R.VERSIONS], folder,
                    os.path.join(folder, "diag.jsonl"))
    finally:
        g.uninstall(orig)
    evaluate()


def _holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj


def _wilcoxon(a, b):
    from scipy.stats import wilcoxon
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)


def evaluate():
    import common_exec as ce
    folder = os.path.join(ce.OUTPUT_ROOT, NAME, "run")
    b = os.path.join(ce.OUTPUT_ROOT, NAME)
    scens = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
    F = pd.DataFrame({m: {s: (R._f1(c) if (c := R._counts(folder, s, m)) else np.nan) for s in scens}
                      for m in METHODS})
    F.index.name = "cenario"
    F.insert(0, "familia", [family(s) for s in F.index])
    F["grad_menos_orig"] = F["patch_tau10_graded"] - F[ORIGINAL]
    F["grad_menos_final"] = F["patch_tau10_graded"] - F["patch_tau10_final"]
    ranks = F[METHODS].rank(axis=1, ascending=False, method="average")
    # weights of the graded variant
    diag = {}
    dp = os.path.join(folder, "diag.jsonl")
    if os.path.exists(dp):
        for l in open(dp, encoding="utf-8"):
            r = json.loads(l)
            if r["versao"] == "graduada":
                diag[r["cenario"]] = r
    weights = []
    for s in scens:
        w = {"cenario": s}
        p = os.path.join(folder, s, "patch_tau10_graded", "results.csv")
        if os.path.exists(p):
            res = pd.read_csv(p)
            for c in ("regime_weight_stage6_raw", "regime_weight_stage6", "regime_weight_lean", "regime_lean_path"):
                if c in res.columns and len(res):
                    w[c] = res[c].iloc[0]
        r = diag.get(s, {})
        w["regime"] = (r.get("coerencia") or {}).get("regime")
        w["n_coerencia"] = (r.get("coerencia") or {}).get("n")
        w["w_lean_diag"] = (r.get("dml") or {}).get("w_lean")
        w["caminho_dml"] = (r.get("dml") or {}).get("caminho")
        weights.append(w)
    W = pd.DataFrame(weights)
    # tests
    T = []
    ok = F[METHODS].dropna()
    for m in (ORIGINAL, "patch_tau10_final", "patch_tau10_graded"):
        ps = [_wilcoxon(ok[m], ok[bl]) for bl in BASELINES]
        for bl, p, ph in zip(BASELINES, ps, _holm(ps)):
            T.append({"metodo": m, "contra": bl, "n": len(ok), "media_dif": float((ok[m] - ok[bl]).mean()),
                      "vitorias": int((ok[m] > ok[bl] + 1e-9).sum()), "derrotas": int((ok[m] < ok[bl] - 1e-9).sum()),
                      "p": p, "p_holm": ph})
    for m in ("patch_tau10_final", "patch_tau10_graded"):
        T.append({"metodo": m, "contra": ORIGINAL, "n": len(ok), "media_dif": float((ok[m] - ok[ORIGINAL]).mean()),
                  "vitorias": int((ok[m] > ok[ORIGINAL] + 1e-9).sum()),
                  "derrotas": int((ok[m] < ok[ORIGINAL] - 1e-9).sum()), "p": _wilcoxon(ok[m], ok[ORIGINAL]),
                  "p_holm": np.nan})
    T = pd.DataFrame(T)
    F.to_csv(os.path.join(b, "f1_por_cenario.csv"))
    W.to_csv(os.path.join(b, "pesos_graduada.csv"), index=False)
    T.to_csv(os.path.join(b, "testes.csv"), index=False)
    med = F.groupby("familia")[METHODS].mean().T
    med["todos"] = F[METHODS].mean()
    med["posto_medio"] = ranks.mean()
    med["sem_resultado"] = F[METHODS].isna().sum()
    with open(os.path.join(b, "relatorio_previa.md"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write("# R1-3b — PREVIEW on the validation data of R1-2c (12 scenarios; it gives an idea; it is not R1-2c)\n\n")
        f.write("## Mean F1 per method\n\n" + med.round(4).to_markdown() + "\n\n")
        f.write("## F1 per scenario\n\n" + F.round(4).to_markdown() + "\n\n")
        f.write(f"- graded ≠ original on {int((F.grad_menos_orig.abs() > 1e-9).sum())} of {len(F)} scenarios; "
                f"graded ≠ final on {int((F.grad_menos_final.abs() > 1e-9).sum())}\n\n")
        f.write("## Paired tests (two-sided Wilcoxon; Holm over the 5 baselines; n = 12, descriptive)\n\n"
                + T.round(4).to_markdown(index=False) + "\n\n")
        f.write("## Weights of the graded variant per scenario\n\n" + W.round(4).to_markdown(index=False) + "\n")
    print(open(os.path.join(b, "relatorio_previa.md"), encoding="utf-8").read())


if __name__ == "__main__":
    from run_log import execution
    mode = sys.argv[1] if len(sys.argv) > 1 else "avaliar"
    if mode == "carregar":
        load()
    else:
        with execution(NAME, __file__):
            run() if mode == "rodar" else evaluate()

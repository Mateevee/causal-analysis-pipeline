"""
R1-7b -- "What if you remove all but 3 stages": stage ladder.

Request
  R1: "A concise main-text algorithm box and a short "what if you remove all but
       3 stages" analysis would help readers judge what actually matters."

Design (fixed before running)
  Offline (from the columns of patch_tau10/original.csv, 90 scenarios, tau_max = 10;
  pairs that did not reach Stage 6 are negatives on every rung):
    D1  detection (Stages 3-5): every candidate confirmed by the pre-filtering;
    D2  + Stage 6: only those significant in the cascade (significant_stage6);
    D3  + Stage 7: accepted by the fusion (causal_penalty_score = 1);
    D4  full pipeline (+ Stage 8: significant).
  Also, from the standalone detectors already run (Table S26), Granger and TE as
  rung zero, when the run is configured (GRANGER_TE_RUN).
  Re-run (rodar mode): the pipeline without Stage 7 -- the fusion accepts every
  candidate and Stage 8 acts on the output of Stage 6 --, the only "three
  layers out of four" that cannot be derived from the stored columns.
  Reporting: mean F1, precision and recall per rung, on development and on
  held-out, and the gain of each rung over the previous one (paired Wilcoxon).

Usage
  python revision/execution/r1_7b_stage_ladder.py rodar     (the pipeline without Stage 7, 90 scenarios; hours)
  python revision/execution/r1_7b_stage_ladder.py avaliar [--verificar]
Output: Results_Revisao/r1_7b_escada_estagios/
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

NAME = "r1_7b_escada_estagios"
MODEL_NO_S7 = "patch_semS7_tau10"


def run():
    import common_exec as ce
    from Pipeline.penalties import UniformCausalPenalty
    folder = ce.output_dir(os.path.join(NAME, "run"))
    # UniformCausalPenalty is the native way to switch off Stage 7 (Pipeline/pipeline.py,
    # _stage7_causal_filter): the fusion does not run and Stage 8 receives the output of Stage 6.
    m = ce.model("patch_tau10", MODEL_NO_S7, causal_penalty_fn=UniformCausalPenalty())
    name_filter = sys.argv[sys.argv.index("--filtro") + 1:] if "--filtro" in sys.argv else None
    ce.run_resuming(ce.paper_scenarios(name_filter), [m], folder)


def prf(S, T):
    tp, fp, fn = len(S & T), len(S - T), len(T - S)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))


def evaluate():
    import common as c
    check = "--verificar" in sys.argv
    rep = c.Report(NAME + ("_verificacao" if check else ""), "R1-7b — Stage ladder")
    scens = c.scenarios()
    if check:
        scens = c.scenarios("dev")[:4] + c.scenarios("ho")[:4]
    rules = {
        "D1 detecção (Stages 3-5)": lambda d: pd.Series(True, index=d.index),
        "D2 + Stage 6": lambda d: d["significant_stage6"].astype(str) == "True",
        "D3 + Stage 7": lambda d: pd.to_numeric(d["causal_penalty_score"], errors="coerce") == 1,
        "D4 completo (+ Stage 8)": lambda d: d["significant"].astype(str) == "True",
    }
    R = {k: {} for k in rules}
    for s in scens:
        T, _ = c.truth(s)
        d = c.original(s)
        if d.empty:
            continue
        lag = pd.to_numeric(d["lag"], errors="coerce")
        pr = c.pairs(d)
        for k, fn in rules.items():
            m = fn(d).fillna(False)
            if not k.startswith("D4"):
                m &= lag > 0          # D1-D3 are stages of the lagged graph
            R[k][s] = prf(set(pr[m]), T)
    # extra rungs: standalone detectors and the pipeline without Stage 7
    extras = {}
    gte = getattr(c.O, "GRANGER_TE_RUN", None)
    if gte:
        for name, mod in (("D0 Granger sozinho", "granger_tau10"), ("D0 TE sozinho", "transfer_entropy_tau10")):
            extras[name] = (gte, mod)
    run_s7 = os.path.join(c.OUTPUT_ROOT, NAME, "run")
    if os.path.isdir(run_s7):
        extras["Sem Stage 7 (3-6 + 8)"] = (run_s7, MODEL_NO_S7)
    for name, (run_dir, mod) in extras.items():
        R[name] = {}
        for s in scens:
            ps = c.folder_in(run_dir, s)
            p = os.path.join(ps, mod, "results.csv") if ps else ""
            if os.path.exists(p):
                T, _ = c.truth(s)
                r = pd.read_csv(p, skipinitialspace=True)
                # entities as text, as in common.original/truth (fixed on 2026-11-14: with numeric
                # IDs -- PeMS-Bay -- no pair matched the truth and F1 came out 0)
                for col in ("source_entity", "target_entity"):
                    if col in r.columns:
                        r[col] = r[col].astype(str).str.strip()
                R[name][s] = prf(set(c.pairs(r)) if len(r) else set(), T)
    order = [k for k in R if k.startswith("D0")] + list(rules) + [k for k in R if k.startswith("Sem")]
    for part, sc in (("all", scens), ("development", [s for s in scens if not c.is_heldout(s)]),
                     ("held-out", [s for s in scens if c.is_heldout(s)])):
        body, prev = [], None
        for k in order:
            ss = [s for s in sc if s in R[k]]
            if not ss:
                continue
            P, Rc, F = (np.mean([R[k][s][i] for s in ss]) for i in range(3))
            gain = p = ""
            if prev and k.startswith("D") and not k.startswith("D0"):
                common = [s for s in ss if s in R[prev]]
                a = [R[k][s][2] for s in common]; b = [R[prev][s][2] for s in common]
                gain, p = f"{np.mean(a) - np.mean(b):+.3f}", f"{c.wilcoxon_p(a, b):.2g}"
            body.append([k, len(ss), c.fmt(P), c.fmt(Rc), c.fmt(F), gain, p])
            if k.startswith("D") and not k.startswith("D0"):
                prev = k
        rep.p(f"**{part}**")
        rep.table(["Rung", "Scenarios", "Precision", "Recall", "F1", "Gain over the previous rung",
                   "Wilcoxon p"], body)
        rep.p("")
    pd.DataFrame([{"cenario": s, "degrau": k, "precisao": v[0], "revocacao": v[1], "f1": v[2]}
                  for k, d in R.items() for s, v in d.items()]).to_csv(os.path.join(rep.dir, "por_cenario.csv"),
                                                                     index=False)
    rep.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        if len(sys.argv) > 1 and sys.argv[1] == "rodar":
            run()
        else:
            evaluate()

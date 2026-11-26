"""
A3 -- final method on the paper's 90 scenarios (new column of Table III). Read-only; no method is run here.

Request: Table III and R1-7b ask for the final method (patch_tau10_final = patch_tau10 + time_axis_rule =
'block_portmanteau' + evidence_coherence_overlap = 'coherent') evaluated on the same 90 scenarios, next to the
submitted method and the 5 baselines.

Source (fixed before seeing any result of patch_tau10_final on the 90): the default-mode run of the full evaluation
of R1-3b (revision/execution/r1_3b_graded_transitions.py rodar padrao), which runs patch_tau10_final through the
logging module (output identical, byte for byte, to that of patch_tau10_final; check_r1_3b.py, item b2) and the
graded variant, on the 90 scenarios. A3 has no run of its own.

Design (fixed beforehand):
  - paper scoring: TP/FP/FN counts from validation_summary.csv, F1 per scenario, as in
    generators/paper_numbers.py; the submitted method (patch) and the 5 baselines come from the paper's runs
    (common.rows());
  - T1: mean F1 per part (all, development 51, held-out 39), with the number of scenarios;
  - T2: patch_tau10_final against each baseline: two-sided paired Wilcoxon, Holm over the 5, wins / ties /
    losses, per part; only scenarios where both have a result (as in Table III);
  - T3: the same for the submitted method (reproduces Table III; check);
  - T4: patch_tau10_final against the submitted method: mean difference, wins / ties / losses, Wilcoxon, per part,
    and the list of scenarios where the F1 changed;
  - the graded variant enters only T1, descriptively (its evaluation is that of R1-3b);
  - scenario without a result for the final method: listed; left out of the paired comparisons.
  - Reading: there is no success criterion; everything is reported, including where the final method falls below
    the submitted one or a baseline.

Usage
  python revision/offline/a3_final_method_90.py [<R1-3b run/padrao folder>]
  (default: Results_Revisao/r1_3b_transicoes_graduais/run/padrao)
Output: Results_Revisao/a3_metodo_final_90/ (relatorio.md, por_cenario.csv).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
for _p in (REV, os.path.join(REV, "execution"), os.path.dirname(REV)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common as c  # noqa: E402

NAME = "a3_metodo_final_90"
FINAL, GRAD = "patch_tau10_final", "patch_tau10_graded"
PARTS = (("all", "todos"), ("development", "dev"), ("held-out", "ho"))


def _f1_run(scen_folder, model):
    if not scen_folder:
        return None
    p = os.path.join(scen_folder, "validation_summary.csv")
    if not os.path.exists(p):
        return None
    v = pd.read_csv(p)
    r = v[(v["model"] == model) & (v["status"] == "ok")]
    return None if r.empty else c.f1(int(r.TP.iloc[0]), int(r.FP.iloc[0]), int(r.FN.iloc[0]))


def _pair(a: dict, b: dict, sc):
    s = [x for x in sc if a.get(x) is not None and b.get(x) is not None]
    xa, xb = np.array([a[x] for x in s]), np.array([b[x] for x in s])
    return s, xa, xb


def main(run):
    rel = c.Report(NAME, "A3 — final method (patch_tau10_final) on the paper's 90 scenarios")
    rel.p(f"Source of the final and graded methods: `{run}` (default-mode run of the full R1-3b evaluation). "
          "Submitted method and baselines: the paper's runs, paper scoring (tau 10).")
    sc_all = c.scenarios()
    V = {"Submetido (patch_tau10)": {s: c.f1_model(s, c.PIPE) for s in sc_all},
         "Final (patch_tau10_final)": {s: _f1_run(c.folder_in(run, s), FINAL) for s in sc_all},
         "Graduada (R1-3b, descritiva)": {s: _f1_run(c.folder_in(run, s), GRAD) for s in sc_all}}
    for m in c.BASELINES:
        V[c.LABEL.get(m, m)] = {s: c.f1_model(s, m) for s in sc_all}
    D = pd.DataFrame({k: pd.Series(v) for k, v in V.items()})
    D.insert(0, "held-out", [c.is_heldout(s) for s in D.index])
    D.index.name = "cenario"
    rel.csv("por_cenario.csv", D.reset_index())
    sub, fin = V["Submetido (patch_tau10)"], V["Final (patch_tau10_final)"]
    missing = [s for s in sc_all if fin[s] is None]
    rel.p(f"Final method with a result on {len(sc_all) - len(missing)} of {len(sc_all)} scenarios. "
          f"Without a result: {missing or 'none'}.")

    rel.h("## T1. Mean F1 per part")
    body = []
    for lbl, part in PARTS:
        sc = c.scenarios(part)
        for name, v in V.items():
            xs = [v[s] for s in sc if v.get(s) is not None]
            body.append([lbl, name, len(xs), c.fmt(float(np.mean(xs))) if xs else "—"])
    rel.table(["Part", "Method", "Scenarios", "Mean F1"], body)

    for title, target in (("## T2. Final method against each baseline (two-sided Wilcoxon; Holm over the 5)", fin),
                          ("## T3. Submitted method against each baseline (reproduces Table III; check)", sub)):
        rel.h(title)
        body = []
        for lbl, part in PARTS:
            sc = c.scenarios(part)
            ps, lines = {}, []
            for m in c.BASELINES:
                lab = c.LABEL.get(m, m)
                s, xa, xb = _pair(target, V[lab], sc)
                w, e, l = c.wins(xa, xb)
                ps[lab] = c.wilcoxon_p(xa, xb)
                lines.append([lbl, lab, len(s), c.fmt(float(np.mean(xa - xb))) if len(s) else "—", f"{w} / {e} / {l}",
                              f"{ps[lab]:.3g}"])
            hp = c.holm(ps)
            body += [r + [f"{hp[r[1]]:.3g}"] for r in lines]
        rel.table(["Part", "Baseline", "Scenarios", "Mean F1 difference", "Wins / ties / losses",
                   "Wilcoxon p", "Holm p (5)"], body)

    rel.h("## T4. Final method against the submitted one")
    body = []
    for lbl, part in PARTS:
        s, xa, xb = _pair(fin, sub, c.scenarios(part))
        w, e, l = c.wins(xa, xb)
        body.append([lbl, len(s), c.fmt(float(np.mean(xb))), c.fmt(float(np.mean(xa))), c.fmt(float(np.mean(xa - xb))),
                     f"{w} / {e} / {l}", f"{c.wilcoxon_p(xa, xb):.3g}"])
    rel.table(["Part", "Scenarios", "Submitted", "Final", "Difference", "Wins / ties / losses", "Wilcoxon p"],
              body)
    changed = [(s, sub[s], fin[s]) for s in sc_all if fin[s] is not None and sub[s] is not None
               and abs(fin[s] - sub[s]) > c.EPS]
    rel.p(f"Scenarios where the F1 changed: {len(changed)}.")
    if changed:
        rel.table(["Scenario", "Held-out", "Submitted", "Final", "Difference"],
                  [[s, "yes" if c.is_heldout(s) else "no", c.fmt(a, 4), c.fmt(b, 4), c.fmt(b - a, 4)]
                   for s, a, b in sorted(changed, key=lambda x: x[2] - x[1])])
    rel.save()


if __name__ == "__main__":
    default = os.path.join(c.OUTPUT_ROOT, "r1_3b_transicoes_graduais", "run", "padrao")
    main(sys.argv[1] if len(sys.argv) > 1 else default)

"""
R1-m6b -- AUROC and AUPRC of PCMCI with ParCorr, GPDC and CMIknn (and of the pipeline, as reference) on the 36 scenarios
of the GPDC check. Only scores existing runs; no model is run.

Request
  R1 (minor): "The fully nonparametric CMIknn ... were not evaluated (you say so); I would add them since the claim
  is domain-agnosticism."
  R1 (point 5): "a metric that does not depend on a single operating point".

Why (decided on 2026-11-19, AFTER seeing the CMIknn F1, and declared as such)
  The F1 of PCMCI + CMIknn (0.007) is dominated by the decision rule: the floor of the permutation p-value (1/501) does
  not survive FDR-BH over N^2 * tau_max links. A metric that does not depend on the operating point separates "the test
  does not order the pairs" from "the decision rule lets nothing through".

Design (fixed before running; nothing here is chosen by the result)
  - Metric and score: those of Table S7 and R1-5d, unchanged (Auxiliary/ranking_metrics.py). Directed pairs at the
    entity level; baselines with the two_tier score (final graph > tested > not tested; within the level,
    1 - min p across lags and properties); pipeline with max |penalized_effect|. Sensitivity: *_pvalue_only.
  - Runs scored (copies in Results_Revisao/r1_m6b_auprc_36/<test>/, to avoid writing into the paper's runs):
      ParCorr  = pcmci_tigramite_tau10, the folder from which paper_numbers reads the paper's F1;
      GPDC     = pcmci_gpdc_tau10, likewise (Table S21);
      pipeline = patch_tau10 of METRICS_RUN, the source of the published AUPRC (Table S7, R1-5d); the F1 is the same as
                 in the paper_numbers run (correction after the 1st attempt; see NOTAS.md, 2026-11-19);
      CMIknn   = pcmci_cmiknn_tau10_dividido, the official run from the runner (Results_Revisao/_outra_maquina/
                 r1_m6b_cmiknn_36_2026-11-19/).
  - Checks: (1) recounted TP/FP/FN match the validation_summary (ranking_metrics checks this); (2) the F1 of
    each copy equals that of the official R1-m6b report (ParCorr, GPDC, CMIknn) and that of paper_numbers (pipeline);
    (3) the AUPRC of the ParCorr and pipeline copies equals that of METRICS_RUN (Table S7), where the scenario exists there.
  - Reporting: mean AUROC and AUPRC per test, overall and per group; better / equal / worse and Wilcoxon of CMIknn
    against ParCorr, GPDC and pipeline, with Holm over the three comparisons of each metric; the same for the
    pvalue_only variant. No result is omitted.
  - Time: the duration of each run (validation_summary) only as a description. Machines and parallelism differ
    (CMIknn split into up to 20 processes on the runner), so the times are not comparable one to one.

Usage:  python revision/offline/r1_m6b_auprc_36.py
Output: Results_Revisao/r1_m6b_auprc_36/
"""
from __future__ import annotations

import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c  # noqa: E402
from Auxiliary import ranking_metrics as rm  # noqa: E402

NAME = "r1_m6b_auprc_36"
OFFICIAL = os.path.join(c.OUTPUT_ROOT, "_outra_maquina", "r1_m6b_cmiknn_36_2026-11-19", "r1_m6b_cmiknn_36")
RUN_CMI = os.path.join(OFFICIAL, "r1_m6b_cmiknn", "run_36_padrao")
REP_CMI = os.path.join(OFFICIAL, "r1_m6b_cmiknn_36", "por_cenario.csv")
CMI_FOLDER = "pcmci_cmiknn_tau10_dividido"
TESTS = [("ParCorr", "pcmci_tigramite"), ("GPDC", "pcmci_gpdc"), ("pipeline", c.PIPE)]
FILES = ("ground_truth.csv", "validation_summary.csv")
EPS = 1e-9


def group(s):
    if s.startswith("benchmark_tigramite"):
        return "tigramite"
    if "logistic" in s:
        return "logisticos"
    if "lorenz" in s:
        return "lorenz"
    return "reais"


def copy_run(src_scen, model_folder, dst_scen):
    os.makedirs(dst_scen, exist_ok=True)
    for a in FILES:
        shutil.copy2(os.path.join(src_scen, a), os.path.join(dst_scen, a))
    d = os.path.join(dst_scen, model_folder)
    if not os.path.isdir(d):
        shutil.copytree(os.path.join(src_scen, model_folder), d)


def prepare(rel, official):
    """Copies the 4 runs (only ground truth, validation_summary and the model folder) and returns {test: model_folder}."""
    folders = {"CMIknn": CMI_FOLDER}
    for s in official.cenario:
        copy_run(os.path.join(RUN_CMI, s), CMI_FOLDER, os.path.join(rel.dir, "CMIknn", s))
        sc = c.scenario_key(s)
        if sc is None:
            raise SystemExit(f"scenario without a key in paper_numbers: {s}")
        for t, model in TESTS:
            run, pm = c.rows()[sc][model]["run"].rsplit(":", 1)
            if t == "pipeline":
                # Published AUROC/AUPRC (Table S7, R1-5d) come from METRICS_RUN, not from the run from which
                # paper_numbers reads the F1 (same graph, same F1; different scores). Correction of 2026-11-19, after
                # the 1st attempt, whose check (3) flagged the difference; see NOTAS.md.
                run = c.O.require("METRICS_RUN")
            if folders.setdefault(t, pm) != pm:
                raise SystemExit(f"{t}: model folder differs between scenarios ({pm} vs {folders[t]})")
            copy_run(c.folder_in(run, sc), pm, os.path.join(rel.dir, t, s))
    return folders


def score(rel, folders):
    tabs = []
    for t, pm in folders.items():
        root = os.path.join(rel.dir, t)
        rm.run(root)
        summary = open(os.path.join(root, rm.SUMMARY_FILE), encoding="utf-8").read()
        if "Sanity check against the existing validation summaries: OK" not in summary:
            raise SystemExit(f"{t}: recounted TP/FP/FN do not match the validation_summary")
        d = pd.read_csv(os.path.join(root, rm.DETAIL_FILE))
        d = d[d.model == pm].copy()
        d["teste"] = t
        tabs.append(d)
    return pd.concat(tabs, ignore_index=True)


def check(D, official):
    errors = []
    for t in ("ParCorr", "GPDC", "CMIknn"):
        x = D[D.teste == t].set_index("scenario").f1.reindex(official.cenario).to_numpy(float)
        y = official[t].to_numpy(float)
        if np.nanmax(np.abs(x - y)) > 1e-3:
            errors.append(f"F1 of {t} differs from the official report (max {np.nanmax(np.abs(x - y)):.4f})")
    for s in official.cenario:
        f = c.f1_model(c.scenario_key(s), c.PIPE)
        g = D[(D.teste == "pipeline") & (D.scenario == s)].f1.iloc[0]
        if abs(f - g) > 1e-3:
            errors.append(f"pipeline F1 differs from paper_numbers on {s}: {g} vs {f}")
    # AUPRC of the copies against METRICS_RUN (Table S7), where the scenario exists
    base = pd.read_csv(os.path.join(c.O.require("METRICS_RUN"), "auroc_auprc_shd_per_scenario.csv"))
    base["cen"] = base.scenario.map(c.O.canonical)
    n_checked = 0
    for t, pm in (("ParCorr", "pcmci_tigramite_tau10"), ("pipeline", "patch_tau10")):
        for s in official.cenario:
            b = base[(base.cen == c.O.canonical(s)) & (base.model == pm)]
            if len(b):
                a = D[(D.teste == t) & (D.scenario == s)]
                n_checked += 1
                for m in ("auroc", "auprc"):
                    if abs(float(a[m].iloc[0]) - float(b[m].iloc[0])) > 1e-4:
                        errors.append(f"{t} {s} {m}: {a[m].iloc[0]} vs METRICS_RUN {b[m].iloc[0]}")
    return errors, n_checked


def duration_s(txt):
    if not isinstance(txt, str):
        return np.nan
    h = re.search(r"(\d+)h", txt); m = re.search(r"(\d+)min", txt); s = re.search(r"([\d.]+)s", txt)
    return (int(h.group(1)) * 3600 if h else 0) + (int(m.group(1)) * 60 if m else 0) + (float(s.group(1)) if s else 0)


def main():
    rel = c.Report(NAME, "R1-m6b — AUROC and AUPRC: PCMCI with ParCorr, GPDC and CMIknn, and the pipeline, on the 36 scenarios")
    official = pd.read_csv(REP_CMI)
    if len(official) != 36:
        raise SystemExit("the official report does not have 36 scenarios")
    folders = prepare(rel, official)
    D = score(rel, folders)
    D["grupo"] = D.scenario.map(group)
    rel.csv("por_cenario.csv", D)
    errors, n_checked = check(D, official)
    rel.p("Declared post hoc: decided after seeing the CMIknn F1; metric and score those of Table S7 / R1-5d, "
          "unchanged. Directed pairs at the entity level. Baselines: two_tier score; pipeline: max |penalized_effect|. "
          f"Folders scored: {folders}.")
    rel.p(f"Checks: TP/FP/FN match the validation_summary in the 4 runs; F1 against the official report and "
          f"paper_numbers; AUROC/AUPRC against METRICS_RUN on {n_checked} scenario × model pairs. "
          f"{'OK' if not errors else 'ERRORS: ' + '; '.join(errors)}")
    order = ["ParCorr", "GPDC", "CMIknn", "pipeline"]
    for met in ("auprc", "auroc", "auprc_pvalue_only", "auroc_pvalue_only"):
        X = D.pivot_table(index="scenario", columns="teste", values=met)[order]
        g = D.drop_duplicates("scenario").set_index("scenario").grupo.reindex(X.index)
        rel.p(f"**{met}** — mean per test ({X.dropna().shape[0]} scenarios with all four)")
        body = [["all", len(X)] + [c.fmt(X[t].mean()) for t in order]]
        body += [[gr, int((g == gr).sum())] + [c.fmt(X.loc[g == gr, t].mean()) for t in order]
                 for gr in ("tigramite", "logisticos", "lorenz", "reais")]
        rel.table(["Group", "Scenarios"] + order, body)
        pv, line = {}, {}
        for ref in ("ParCorr", "GPDC", "pipeline"):
            ss = X[["CMIknn", ref]].dropna()
            w, e, l = c.wins(ss["CMIknn"], ss[ref])
            pv[ref] = c.wilcoxon_p(ss["CMIknn"], ss[ref])
            line[ref] = [f"CMIknn × {ref}", len(ss), c.fmt((ss["CMIknn"] - ss[ref]).mean()), f"{w} / {e} / {l}"]
        ph = c.holm(pv)
        rel.table(["Comparison", "Scenarios", "Mean difference", "Better / equal / worse", "Wilcoxon p", "Holm p (3)"],
                  [line[r] + [c.fmt(pv[r]), c.fmt(ph[r])] for r in ("ParCorr", "GPDC", "pipeline")])
        rel.p("")
    # time, descriptive only
    T = D[["teste", "scenario", "grupo"]].copy()
    vs = {}
    for t, pm in folders.items():
        for s in official.cenario:
            v = pd.read_csv(os.path.join(rel.dir, t, s, "validation_summary.csv")).set_index("model")
            vs[(t, s)] = duration_s(v.loc[pm, "duration"]) if pm in v.index and "duration" in v.columns else np.nan
    T["s"] = [vs[(t, s)] for t, s in zip(T.teste, T.scenario)]
    body = [[t, int(T[T.teste == t].s.notna().sum()), f"{T[T.teste == t].s.sum() / 3600:.2f}",
             f"{T[T.teste == t].s.median() / 60:.1f}"] for t in order]
    rel.p("**Time per run** (duration from validation_summary; different machines and parallelism, descriptive only)")
    rel.table(["Test", "Scenarios with time", "Total (h)", "Median (min)"], body)
    rel.csv("tempo_por_cenario.csv", T)
    rel.save()
    if errors:
        raise SystemExit("checks failed: " + "; ".join(errors))


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        main()

"""
Decision trail per candidate (Section 5.5, Table 7; complete in Table S27) and re-execution of Stage 7
from the stored record. Generates Decision_Trail_tau10.md.

Unlike paper_numbers.py, this script imports the pipeline's code
(Pipeline/ of this repository), because the masses of the Stage 7 fusion are not stored in
original.csv: only the binary weight (causal_penalty_score) and the inputs of the fusion.
The masses are recomputed here by causal_pen_diagnostics on the stored row,
with the support floor recalibrated as in the pipeline
(AdaptiveCausalPenalty, without the old_sig column, which only exists after
calibration).

Two outputs:
  1. The trail of the candidates of one scenario (SCENARIO): how each one came in
     (Stages 3-5), what each model of the cascade decided and why (Stage 6), the
     support and opposition of the fusion (Stage 7) and the outcome.
  2. On how many candidates with lag >= 1 of the 58 scenarios the re-execution
     reproduces the stored decision, and how far from a threshold the ones it does
     not reproduce are.

How to use:  python generators/decision_trail.py
"""
import ast
import glob
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import origins as O  # noqa: E402
# Re-execution (section 2): the scenarios of the main run (in the paper, the 58 of
# Run_20261004_232204). The scenario of the trail is also looked up in the held-out runs.
RUN = O.require("MAIN_RUN")
OUT = O.out("Decision_Trail_tau10.md")
FOLDER = "patch_tau10"
SCENARIO = "real_tigramite_climate_walker"

sys.path.insert(0, O.REPO)
from Config.execution import ADAPTIVE_CAUSAL_PENALTY as CFG  # noqa: E402
from Pipeline.SelectorFunc import (  # noqa: E402
    MAX_OPPOSITION_THRESHOLD, NET_RATIO_MIN_THRESHOLD, AdaptiveCausalPenalty,
    causal_pen_diagnostics)


def calibrate(df):
    """Support floor of the scenario, recalibrated as in the pipeline's Stage 7."""
    pen = AdaptiveCausalPenalty(
        percentile=CFG.percentile, min_absolute_floor=CFG.min_absolute_floor,
        max_absolute_floor=CFG.max_absolute_floor, min_n_candidates=CFG.min_n_candidates,
        domain_gate_fn=CFG.domain_gate_fn, fusion_rule=CFG.fusion_rule)
    floor = pen.calibrate(df.drop(columns=["old_sig"], errors="ignore"))
    return floor, pen.n_calibration_candidates_


def candidates(scen, run=RUN):
    """Rows of original.csv that went through Stage 7 (lag >= 1)."""
    df = pd.read_csv(os.path.join(run, scen, FOLDER, "original.csv"))
    if df.empty or "causal_penalty_score" not in df.columns:
        return df.iloc[0:0]
    lag = pd.to_numeric(df["lag"], errors="coerce")
    return df[(lag > 0) & df["causal_penalty_score"].notna()]


def trail(out):
    folder = O.find(SCENARIO)
    df = candidates(os.path.basename(folder), os.path.dirname(folder))
    gt = pd.read_csv(os.path.join(folder, "ground_truth.csv"))
    truth = {(r.source_entity, r.target_entity): bool(r.causal_gt) for r in gt.itertuples()}
    floor, n_cal = calibrate(df)
    out.append(f"## 1. Decision trail -- {SCENARIO} ({FOLDER})\n")
    out.append(f"Candidates: {len(df)}. Stage 7 support floor: {floor:.4f} "
               f"({n_cal} candidates in the calibration; with fewer than {CFG.min_n_candidates} the fixed floor applies). "
               f"Minimum net ratio {NET_RATIO_MIN_THRESHOLD}; opposition ceiling {MAX_OPPOSITION_THRESHOLD}.\n")
    for r in df.itertuples():
        s = causal_pen_diagnostics(r._asdict(), min_support_threshold=floor)
        out.append(f"### {r.source_entity} -> {r.target_entity}, lag {r.lag} "
                   f"(truth: {truth.get((r.source_entity, r.target_entity))})\n")
        out.append(f"- Stages 3-5: came in through `{r.filter_method}`; TE = {r.transfer_entropy:.3f}; "
                   f"raw p {r.pvalue_discovery_raw}, corrected {r.pvalue_discovery_corrected}; "
                   f"lag penalization {r.lag_penalty_score}")
        out.append("- Stage 6 (cascade), model by model:")
        for e in ast.literal_eval(r.cascade_log):
            reason = "; ".join(e.get("reasons_fail") or []) or "passes every condition"
            out.append(f"    - {e['model']}: {'accepted' if e['accepted'] else 'rejected'} "
                       f"(t = {e['t_stat']}, p = {e['pvalue']}) -- {reason}")
        out.append(f"- Stage 6 (result): model `{r.dml_model_used}`, confidence {r.cascade_confidence}, "
                   f"{r.n_models_agree} of 4 models; theta = {r.causal_effect} (SE {r.standard_error}; "
                   f"95% CI [{r.ci_95_lower}, {r.ci_95_upper}]; p = {r.pvalue}); "
                   f"significant at Stage 6: {r.significant_stage6}")
        out.append(f"- Stage 7 (recomputed): support {s['support']:.4f}, opposition {s['opposition']:.4f}, "
                   f"uncertainty {s['uncertainty']:.4f}, net ratio {s['net_ratio']:.4f} -> "
                   f"{'accepted' if s['accepted'] else 'rejected'}; stored: causal_penalty_score = {r.causal_penalty_score}")
        masses = s["primary_experts"] + s["active_meta_experts"] + s["active_interaction_experts"]
        out.append("    - experts (H / not-H): " + "; ".join(
            f"{n} {m['H']:.2f}/{m['notH']:.2f}" for n, m in masses if m["H"] > 0.005 or m["notH"] > 0.005))
        out.append(f"- Stage 8: evidence regime `{r.evidence_regime}`; stability {r.stability_score}; "
                   f"structural rejection: {r.structural_rejection_reason}")
        out.append(f"- Outcome: reported = {r.significant}\n")


def rerun(out):
    total = same = near = far = scenarios = complete = 0
    lines = []
    for d in sorted(glob.glob(os.path.join(RUN, "*", FOLDER, "original.csv"))):
        scen = d.split(os.sep)[-3]
        if "dream3" in scen:   # outside the paper's suite
            continue
        df = candidates(scen)
        if df.empty:
            continue
        scenarios += 1
        floor, _ = calibrate(df)
        diff = 0
        for r in df.itertuples():
            s = causal_pen_diagnostics(r._asdict(), min_support_threshold=floor)
            total += 1
            if s["accepted"] == (r.causal_penalty_score > 0):
                same += 1
                continue
            diff += 1
            margin = min(abs(s["support"] - floor), abs(s["net_ratio"] - NET_RATIO_MIN_THRESHOLD),
                         abs(s["opposition"] - MAX_OPPOSITION_THRESHOLD))
            if margin < 0.01:
                near += 1
            else:
                far += 1
        complete += diff == 0
        if diff:
            lines.append(f"| {scen} | {len(df)} | {diff} | {floor:.4f} |")
    out.append("## 2. Re-execution of Stage 7 from the record (58 scenarios, tau_max = 10)\n")
    out.append(f"Candidates with lag >= 1: {total}. Stored decision reproduced: {same} "
               f"({100 * same / total:.1f}%). Not reproduced: {total - same}, of which {near} less "
               f"than 0.01 from an acceptance threshold (support floor, net ratio or opposition ceiling) and "
               f"{far} further. Scenarios: {scenarios}; fully reproduced: {complete}.\n")
    out.append("| Scenario | Candidates | Not reproduced | Floor |\n|---|---|---|---|")
    out.extend(lines)


if __name__ == "__main__":
    text = ["# Decision trail and re-execution of Stage 7 (generated by generators/decision_trail.py)\n"]
    trail(text)
    rerun(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(O.canonical("\n".join(text) + "\n"))
    print("ok ->", OUT)

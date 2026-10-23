# Stage 7 against a simple adaptive rule — pre-registration

The rule and the decision criterion below were written down and approved on 2026-10-23, before the
rule was computed on any data. No F1 of this rule had been seen when they were fixed. This file is the
English version of that record.

## Data

- Run: `MAIN_RUN` of `generators/origins.py` (in the paper, `Results/Run_20261004_232204`), the 58
  scenarios of Tables VI, S25 and S17.
- τ_max = 10: `<scenario>/patch_tau10/original.csv`; τ_max = 24: `<scenario>/patch/original.csv`.
- Ground truth: `<scenario>/ground_truth.csv`, column `causal_gt`.
- Metric and the other rules: exactly those of `generators/paper_numbers.py::ablation()` (ordered
  entity pair, F1 per scenario, mean over scenarios).

## Rule 11 — |t| with a per-dataset adaptive cut-off

The counterpart of Stage 7's adaptive Support floor (`AdaptiveCausalPenalty(percentile=45,
min_n_candidates=10)`, `Config/execution.py`), with |t| in place of the fused Support and 1.96 in place
of the fixed floor.

For each scenario and window, without looking at the ground truth:

1. Unit = ordered entity pair. Score of a pair = the largest |t_stat| over the pair's rows in
   `original.csv` (the same aggregation as rule 7). A missing `t_stat` rejects the pair.
2. Candidates = pairs with score > 1.96. Let n be their number.
3. If n < 10: accept the n candidates (the fixed cut-off alone, as in rule 5).
4. If n ≥ 10: c = 45th percentile of the candidates' scores (`numpy.percentile`, linear interpolation,
   as in `AdaptiveCausalPenalty`); accept a pair if score > 1.96 **and** score > c.

No parameter is adjusted after seeing the result (45 and 10 come from Stage 7).

## Test

- Two-sided paired Wilcoxon test of per-scenario F1 against Stage 7 (`causal_penalty_score == 1`), as
  for the other rules.
- Holm correction over 11 comparisons (the 10 existing ones plus rule 11), in each window.
- The 10 existing rules must reproduce Tables VI/S25 (τ = 10) and S17 (τ = 24); if they do not, the
  computation stops.

## Decision criterion (fixed in advance)

Primary window: τ = 10 (that of Table VI). τ = 24 is a sensitivity check and is reported alongside.

- **Stage 7 wins**: Stage 7's mean F1 > rule 11's and Holm p < 0.05.
- **Rule 11 wins (Stage 7 loses)**: rule 11's mean F1 > Stage 7's and Holm p < 0.05.
- **Tie**: Holm p ≥ 0.05.

If τ = 10 and τ = 24 disagree, the text says so and the conclusion follows τ = 10.

Consequence for the paper:
- Stage 7 wins → "no simpler rule given such an operating point was tested" is removed from
  contribution 1 and Section VI-C.
- Tie or loss → contribution 1 and Section V-D state that the gain comes from choosing the operating
  point per dataset, and the Dempster-Shafer fusion becomes one way of doing so.

# Protocol — graded transitions in the two regime switches (R1-3b)

**Status: APPROVED and frozen** (authors' decision by delegation, 2026-11-13; see `revision/NOTAS.md`, entry
"R1-3b — graded transitions protocol"). Hash recorded in NOTAS.md. Nothing was implemented or run before the
freeze.

## 1. The request

R1, point 3: "Stability of the regime switches. Only 53% of the cut-off perturbations meet the pre-specified
tolerance (target 90%), driven by ρ and the outcome R² that switch the treatment of an entire dataset (Sec. V-B).
That means small changes in two thresholds can change the output graph wholesale. This should be treated as a main
result, not a footnote. Graded rather than hard switches (which you propose as future work) look necessary for a
'fixed-configuration' claim."

What S20 showed (Sec. V-F of the manuscript): 79 of 150 joint configurations within the tolerance (52.7%), against
the 90% required. The failures are concentrated at low ρ (8% within, in the lower third) and at high outcome R² (26%, in the
upper third). The new evidence from the revision (E5, `NOTAS.md`): under 80% subsampling, the coherence regime changes
in 32 of the 90 scenarios, and in 10 of them in half or more of the subsamples.

## 2. The two switches today (final method, `patch_tau10_final`)

1. **Coherence guard (ρ).** `Pipeline/coherence.py::assess_evidence_coherence`: Spearman's ρ between the TE and the DML |t|,
   with a Fisher interval. In the "incoherent" regime (upper bound < `min_rho` = 0.3, without the overlap of
   G1), the rejections of Stages 7-8a are not applied and the decision reverts to that of Stage 6 (`keep_stage6_pairs`). The
   switch applies to the whole dataset.
2. **Control design by the outcome R².** `Pipeline/dml_engine.py::estimate_effects_by_control_set`, with
   `control_set='auto'`: the lean design (candidate parents and source block) runs first. If the median of the
   out-of-sample R² of the outcome nuisance is ≥ `auto_max_outcome_r2` = 0.5, the whole dataset is
   re-estimated with the blanket design. The switch also applies to the whole dataset.

## 3. The variant: graded transitions, with no new numerical parameter

New control parameter, **off by default**: `regime_transition` ∈ {`'hard'` (default; today's
behaviour), `'graded'`}. With `'hard'`, the output is identical, byte for byte, to that of `patch_tau10_final` (identity test,
section 5).

### 3.1 Weight of each regime
- **Coherence guard.** The Stage 6 weight, w_inc, is **the probability that the hard rule itself (with G1)
  would declare "incoherent" if ρ were drawn again from its sampling distribution**, through the same Fisher approximation
  that the guard already uses:
  - let ẑ = atanh(ρ̂), se = √(1.06 / (n − 3)) and h = z₁₋α · se;
  - the hard rule declares "incoherent" when the upper bound < `min_rho` and the lower bound ≤ 0, that is,
    when z < min(atanh(`min_rho`) − h, h);
  - hence, w_inc = Φ((min(atanh(`min_rho`) − h, h) − ẑ) / se). The weight of Stages 7-8a is 1 − w_inc.
  - The transition is centred exactly on the boundary of the hard rule and coincides with it far from the cut.
    Perturbing `min_rho` moves the centre, so the perturbation of the primary outcome measures the effect for real.
  - Special cases, the same as today: n < `min_pairs` or ρ not computable ("undetermined") → w_inc = 0; no
    candidate with TE evidence (today "incoherent") → w_inc = 1.
- **Control design.** Let k be the pairs with outcome R² < `auto_max_outcome_r2`, among the n pairs evaluated
  in the lean design. w_lean = P(q > 0.5 | k, n), with q ~ Beta(k + 1, n − k + 1). It is the probability that the median
  is below the cut. It has a closed analytical form, with no bootstrap.
- The cuts (`min_rho` = 0.3, `auto_max_outcome_r2` = 0.5, α = 0.05) and `min_pairs` are today's. There is no
  new numerical parameter.

### 3.2 How the weight enters the decision: interpolation by evidence ranking
For each switch there are two binary decisions per pair: that of regime A and that of regime B.
- Pairs that both regimes keep: they stay. Pairs that both reject: they go.
- Pairs kept by only one regime: they enter in proportion to that regime's weight, ordered by the Stage 6 evidence
  (DML |t|, from largest to smallest). If m pairs are kept only by regime B, the ⌊w_B · m + 0.5⌋ with the
  largest |t| enter.
- **Guard:** regime A = decision of Stages 7-8a (weight 1 − w_inc); regime B = decision of Stage 6 (weight w_inc).
- **Control design:** both designs are estimated whenever 0.001 < w_lean < 0.999. The decision follows the
  interpolation above, and the θ and standard error of the pair come from the weighted mean (w_lean · lean + (1 − w_lean) ·
  blanket), a conservative composition for correlated estimates. Outside this interval, only the design with
  weight ~1 is estimated, as today.
- With w = 1 or 0, the variant gives exactly the output of the hard switch. The difference only appears near the cut, which is
  where R1 pointed out the problem.
- **Cost:** the second Stage 6 estimation only happens in the datasets near the R² cut.

## 4. Evaluation, fixed before running

All evaluations compare the variant (`patch_tau10_graded` = `patch_tau10_final` + `regime_transition='graded'`)
with `patch_tau10_final` (hard switch), on the same code.

### 4.1 Primary outcome: perturbation of the two cuts (R1's own metric)
- **Configurations:** 12, not counting the default:
  - `min_rho` ∈ {0.10; 0.20; 0.40; 0.50}, with R² at the default;
  - `auto_max_outcome_r2` ∈ {0.35; 0.45; 0.55; 0.65}, with ρ at the default;
  - the 4 corners: (0.10; 0.35), (0.10; 0.65), (0.50; 0.35) and (0.50; 0.65).
  - The range covers the lower third of ρ and the upper third of R², where S20 concentrated the failures.
- **Scenarios:** the 90 of the paper.
- **Tolerance, the same as in S20** (Sec. V-F), against the default of the same version:
  - (R1) mean F1 within 0.02 of the default on the whole set and on each part;
  - (R2) the number of scenarios within 0.10 of the best of the six methods (runs of the paper) falls by at most 3;
  - (R3) the mean deficit stays below that of each baseline.
- **Success criterion,** R1's: the variant within the tolerance in **≥ 90% of the 12 configurations (≥ 11 of
  12)**. The same number is reported for `patch_tau10_final`.
- **Savings without changing the result, fixed beforehand:** a scenario is only re-run in a configuration if, according to the
  diagnostics of the execution at the default (ρ, interval and outcome R²), its regime decision can change:
  - in the hard switch: the regime changes with the new cut;
  - in the variant: w leaves [0.001; 0.999] at the default or at the new cut.

  In the other scenarios, the default output is reused. The rule is applied by script, from the recorded
  diagnostics, and the list of re-run scenarios goes into the report. Caveat: the reused output comes from another
  process; the variation between processes (round 3: 0 to 1 edge) is declared.

### 4.2 Secondary outcome 1: non-inferiority at the default
- On the 90 scenarios, with the default cuts: mean F1 of the variant minus that of `patch_tau10_final`, on development and
  on held-out.
- Fixed margin: −0.01. Non-inferior if the mean difference is ≥ −0.01 in both parts.
- Paired Wilcoxon, reported alongside.

### 4.3 Secondary outcome 2: stability under subsampling
- **Scenarios:** the 10 with frequent switching from E5 (`NOTAS.md`): `rcaeval productcatalogservice_cpu_2`,
  `logistic_maps_with_isolated_seed2`, `logistic_maps_chain_seed1`, `beeline VSC-2000-1`, `easyvista_mom_1`,
  `easyvista_web_2`, `synthetic_var_2_seed22`, `beeline HSC-2000-2`, `beeline VSC-2000-2` and `easyvista_antivirus_1`.
  **The selection came from a computation done after seeing the R2-m2 data, and this is declared.**
- **Subsamples:** the 20 of R2-m2 (same `subsamples` function, same seed per scenario). Both
  `patch_tau10_final` and the variant run: 400 executions.
- **Measures:**
  - Jaccard against the full sample of the same version;
  - Jaccard between subsamples;
  - fraction of stable edges (≥ 80%);
  - in the hard switch, the fraction of subsamples with a regime different from that of the full sample.
- **Reading:** descriptive, with a per-scenario paired Wilcoxon on the two Jaccards (n = 10, low power; declared).

### 4.4 Inclusion in R1-2c
The variant enters R1-2c (`PROTOCOL_new_benchmark_R1-2c.md`, section 5, item 3) after passing the
identity test (section 5), whatever the result of 4.1 to 4.3. R1-2c runs only once, with it.

### 4.5 Reading, fixed beforehand
- **4.1 with ≥ 11 of 12:** the variant meets the criterion that S20 did not reach. The text says that the graded
  transitions make the graph robust to the position of the two cuts, with the cost measured in 4.2 and 4.3. The
  "fixed configuration" claim can be kept for the variant.
- **4.1 with fewer than 11:** report the number and how much it improved or worsened relative to the hard switch. The
  "fixed configuration" claim is softened in the text and in the letter, as decided on 2026-11-11.
- **4.2 inferior:** report it and weigh it against 4.1. Neither version replaces the other without both numbers side by
  side.
- **In any case:** the submitted `patch_tau10` remains the main comparison of Table III. The variant is
  reported alongside, as a response to R1-3, and declared as post-review.

## 5. Implementation and identity test
- Code only in `Pipeline/coherence.py` (the weight w_inc and the guard interpolation), in `Pipeline/dml_engine.py` (w_lean
  and the two designs) and in `Pipeline/pipeline.py` (the parameter). No default changes.
- **Identity test,** before any execution of the evaluation: with `regime_transition='hard'`, the `original.csv`
  and the `results.csv` of `patch_tau10_final` come out identical, byte for byte, to those of an execution with the previous code,
  in the same process and with a fixed `PYTHONHASHSEED` (round 3), on 5 scenarios:
  - `benchmark_synthetic_var_1_seed11`;
  - a scenario in which the guard fires;
  - one in which the blanket design is chosen;
  - Walker;
  - a BEELINE one.
  - The last four are chosen from the recorded diagnostics, by the rule "the first in alphabetical order that
    satisfies the condition".
- **Unit tests of w:**
  - w_inc = 0.5 when ẑ = min(atanh(`min_rho`) − h, h);
  - w_inc tends to 0 far above the boundary and to 1 far below;
  - w_inc rounded (< 0.5 → 0; ≥ 0.5 → 1) reproduces the decision of the hard rule with G1 in all 90
    scenarios, from the recorded diagnostics;
  - w_lean = 0.5 when k = n/2;
  - the interpolation with w = 1 and w = 0 returns the hard decisions.
- The `method_code_hash` changes with the new code. The default behaviour does not change, by the test above.

## 6. Order
1. This protocol frozen with a hash.
2. Implementation and unit tests, here.
3. Identity test, on the runner, in a short round.
4. Evaluation: 4.1, 4.2 and 4.3 in an overnight round; the same round can carry R1-2c, already with the variant.
5. Evaluation script `revision/execution/r1_3b_graded_transitions.py`, with its hash noted before running.

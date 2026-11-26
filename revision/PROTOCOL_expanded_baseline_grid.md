# Protocol — expanded baseline grid, run from scratch (R1-2a, R1-2b, R2-3a)

**Status: FROZEN on 2026-11-10, with the decisions of section 10 taken by delegation (to be confirmed by the
authors).** SHA-256 hash of this file in `revision/NOTAS.md`. After the freeze, this file does not change;
deviations go to the diary.

## 1. The request

- R1, point 2: "each baseline runs untuned with library defaults (and PCMCI with a non-default BH correction ...).
  The comparison is therefore not like-for-like on the development suite, and even on held-out data the baselines
  are not tuned. Please add (a) per-benchmark tuned baselines (or at least a small grid), (b) a
  leave-one-family-out protocol, where cut-offs are frozen using other families only, and (c) a truly fresh
  benchmark added after the code freeze."
- R2, point 3: "the tuning protocol is asymmetric".

This protocol covers (a) and extends (b) to the baselines. **It does not cover (c)**, which is the new benchmark (R1-2c).

## 2. Why expand, and what this is

- **Authors' decision (2026-11-10):** "I want a test run again with the expanded version".
- The `r1_2a` grid (fixed on 2026-11-09) has 5 to 13 configurations per baseline and almost always varies one
  parameter at a time. The check of 2026-11-10 (NOTAS.md) showed that it meets the "small grid", but not the
  "per-benchmark tuned".
- **This is a deviation from the registered design, made after seeing the results of the original grid** (section 8).
  The expansion only adds configurations to the baselines: it gives them more chances, never fewer.
- The original grid is a subset of the expanded one, and the original `r1_2a` report continues to be published.

## 3. Grid rule (fixed before choosing any value)

- **Scale parameters** (significance levels, penalties, weight thresholds, network size): the paper's
  value × {1/4, 1/2, 1, 2, 4}, discarding what falls outside the parameter's domain.
- **CUTS+ threshold** (score in [0, 1]): from 0.1 to 0.9, in steps of 0.1.
- **Categorical parameters:** all options the wrapper accepts, listed in section 4.
- **Full crossing** of the parameters within each method.
- **Plus** every value of the original grid that the rule does not generate, so that it remains a subset.
- Window tau_max = 10 and independence test (ParCorr) fixed: they are not tuned. See section 7.

## 4. The grid

| Baseline | Re-executed parameters | Re-executions per scenario | Decision variants computed offline | Configurations |
|---|---|---|---|---|
| PCMCI | pc_alpha {0.05; 0.1; 0.2; 0.4; 0.8} | 5 | final alpha {0.01; 0.0125; 0.025; 0.05; 0.1; 0.2} × correction {BH, none} | 60 |
| PCMCI+ | pc_alpha {0.001; 0.0025; 0.005; 0.01; 0.02; 0.04; 0.05; 0.1} × collider rule {majority, conservative, none} | 24 | — | 24 |
| DYNOTEARS | lambda_W × lambda_A, each in {0.0125; 0.025; 0.05; 0.1; 0.2} × thresholds (W; A) {(0.075; 0.025); (0.15; 0.05); (0.3; 0.1); (0.6; 0.2); (1.2; 0.4)}; plus the 7 configurations of the original grid outside this crossing | 132 | F-test gate: none, or alpha {0.01; 0.05} × correction {none, BH} | 660 |
| CUTS+ | threshold {0.1 … 0.9} × network size {8; 16; 32; 64; 128}; each configuration with 3 seeds {0; 1; 2} | 135 | F-test gate, as above | 225 |
| VARLiNGAM | order criterion {AIC, BIC, HQIC, FPE, none} × pruning {yes, no} | 10 | F-test gate, as above | 50 |

- **Total: 306 re-executions per scenario × 90 scenarios = 27,540 executions; 1,019 configurations.**
- **The paper's values** (the "default" of each method):
  - PCMCI: pc_alpha 0.2, final alpha 0.05, BH;
  - PCMCI+: pc_alpha 0.01, majority;
  - DYNOTEARS: lambda_W = lambda_A = 0.05, thresholds (0.3; 0.1), no gate;
  - CUTS+: threshold 0.5, network 32, seed 0, no gate;
  - VARLiNGAM: AIC, with pruning, no gate.
- **The 7 extra DYNOTEARS configurations** (lambda_W = lambda_A):
  - 0.01 with (0.3; 0.1), (0.2; 0.05) and (0.4; 0.2);
  - 0.05 with (0.2; 0.05) and (0.4; 0.2);
  - 0.1 with (0.2; 0.05) and (0.4; 0.2).
- **CUTS+ and seeds:** the F1 of a configuration is the mean over the 3 seeds. Seed 0 alone (the paper's) is
  reported alongside.
- **From scratch:** all configurations run on the same machine and in the same execution, including the defaults. Nothing is
  read from the paper's run or from the original grid. The re-executed default is compared with the paper's F1, scenario by
  scenario; every difference is reported (the BEELINE scenarios depend on the environment; NOTAS.md, R3-1b).
- **DYNOTEARS and CUTS+ thresholds:** the wrappers record only the edges that pass the threshold, not the weights. That is
  why each threshold is a re-execution. The wrappers are not changed.

## 5. The analyses (all offline, on the expanded grid)

**A1. Configuration chosen on development** (the `r1_2a` report, with no change of rule).
- For each baseline, the configuration with the highest mean F1 over the 51 development scenarios, applied to the 39 held-out ones.
- Per-scenario upper bound, reported separately.

**A2. Per-family tuning** (`PROTOCOL_per_family_tuning.md`, with this grid in place of the original one).
- Conditions: default; tuned per family (ceiling); tuned per family with one scenario left out.
- Pipeline: the 216 configurations of the sensitivity run.

**A3. Leave-one-family-out for the six methods.**
- For each family, the configuration with the highest mean per-scenario F1 over the scenarios of the other families, measured on the
  left-out family, against the default. It is the `r1_2b` rule, unchanged.
- Baselines: 17 families, 90 scenarios. Pipeline: `r1_2b` itself (16 families, 86 scenarios, 216 configurations).

**Common rules.**
- Metric: F1 per directed pair, as in Table III.
- Tie in the selection: the default is kept. Among tied non-default configurations: the first in grid order.
- A configuration only competes if it has a value in all scenarios of the selection set in which the default of the same
  method has a value.
- **Empty native graph:** the F-test gate variants are worth the same empty graph. In the original grid these
  variants had no value in that scenario and dropped out of the selection (VARLiNGAM, 1 development scenario). It is a
  correction of the rule; the selection under the old rule is reported alongside.
- Comparisons: two-sided paired Wilcoxon with Holm over the five; mean rank of the six, Friedman and Nemenyi critical
  difference. On held-out and overall.

## 6. Reporting (whatever the result)

- **Supplement:**
  - the grid of each method and the mean F1 of each configuration, on development and on held-out;
  - the tables of A1, A2 and A3;
  - A1 also on the original grid, side by side.
- **Main text, no new table:** in V-A, the held-out numbers of A1 and A2; in VI-C, the "untuned baselines"
  threat rewritten. The sentences go in even if some baseline beats the pipeline.
- **Letter:** the same numbers; what of the request is met and what is not (section 7).
- **Executions with errors** enter the count and the report, per method and configuration. No configuration is removed
  from the grid after running.

## 7. Limitations to declare

- **(c) is not covered.** The clean test of the comparison remains the new benchmark.
- **The literal LOFO of the pipeline remains impossible** for the development families (R1-2d): the cut-offs were
  fixed by hand while looking at the 51 scenarios. A3 measures transfer within a grid, for the six methods. For the
  held-out families, which do not appear in development, the cut-offs were already fixed without them.
- **The pipeline grid** (216 configurations of nine cut-offs) is the paper's sensitivity run: old time-axis
  rule, old guard, without Causal Chambers. It is smaller than that of DYNOTEARS (660), close to that of CUTS+ (225)
  and larger than those of PCMCI (60), PCMCI+ (24) and VARLiNGAM (50). The ceiling favours the larger grid.
- **Not tuned in the baselines:** tau_max, independence test (GPDC and CMIknn are other items), CUTS+ learning
  rates, epochs and sparsity penalty, `reset_lagged_links` of PCMCI+, DYNOTEARS iterations.
- **VARLiNGAM does not run on 13 scenarios**, with any configuration of the original grid.
- **Ceilings and upper bounds** use the ground truth of the scenario itself: they are not usable results.

## 8. Context, declared

Written after seeing:
- the paper's results;
- the complete original `r1_2a` grid (another machine, 86 scenarios) and its report;
- the mean F1 of each configuration of the original grid on development and on held-out, including which
  configurations do better on held-out (PCMCI without correction; CUTS+ with threshold 0.4; DYNOTEARS with lambda 0.01);
- `r1_2b`.

The values of the expanded grid come from the rule of section 3, and not from these observations.

## 9. Execution

- Another machine (`runner`, 24 CPUs), one process per scenario, heaviest scenarios first.
- Estimate: 10 to 13 hours, from the recorded times of the original grid (8.0 CPU hours for 20
  configurations; the heaviest scenario sets the wall time). The new configurations of CUTS+ (network 64 and 128) and
  of PCMCI+ (conservative rule) have no measured time.
- It needs the paper's runs there, including `Results_Extra/Run_20261009_causal_chamber`, for the analyses to cover
  the 90 scenarios.
- Outputs brought to `Results_Revisao/_outra_maquina/rodada2_grade_ampliada_<data>/`.
- Scripts to write: `execution/r1_2g_expanded_grid.py` (run and tabulate) and `offline/r1_2f_per_family_tuning.py`
  (A1 to A3).

## 10. Decisions, taken by delegation on 2026-11-10 (to be confirmed by the authors)

Criterion given by the author: "always prioritize the fullest resolution of the reviewers' points and honesty".

1. **Rule and grid:** those of sections 3 and 4.
2. **Thresholds:** each one is re-executed; the wrappers do not change.
3. **Original `r1_2a` report:** kept, alongside the new one.
4. **Pipeline on Causal Chambers** (216 configurations): not run; the family stays out of the comparisons with the
   tuned pipeline and is reported separately.
5. **Pipeline grid:** that of the sensitivity run, with the old axis rule; declared as a limitation.

## Log

- Draft: 2026-11-10.
- Freeze: 2026-11-10; hash in NOTAS.md.

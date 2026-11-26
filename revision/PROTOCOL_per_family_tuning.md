# Protocol — per-family tuning, for the six methods (R1-2a, R2-3a)

**Status: FROZEN on 2026-11-10, with the decisions of section 9 taken by delegation (to be confirmed by the
authors).** SHA-256 hash of this file in `revision/NOTAS.md`. Nothing has been run or evaluated under this design.
Where this file and `PROTOCOL_expanded_baseline_grid.md` differ, the expanded grid one prevails.

## 1. The request

- R1: "Please add (a) per-benchmark tuned baselines (or at least a small grid)". Reason given by him: the
  comparison "is not like-for-like", because the pipeline was refined while looking at 51 scenarios and the baselines ran with
  the library parameters.
- R2 (point 3): "the tuning protocol is asymmetric".

## 2. Why this protocol, in addition to the already registered design of `r1_2a`

The design of `r1_2a_tuned_baselines.py` (fixed on 2026-11-09) chooses one configuration per baseline on the 51
development scenarios and applies it to the 39 held-out ones, with a per-scenario upper bound. Two points are left
out:

- **"Per-benchmark" is the family.** The literal request is one configuration per benchmark; the registered design gives one
  configuration for all, or one per scenario.
- **The pipeline is not held to the same yardstick.** The pipeline configuration is a single one for the 17 families. Comparing
  per-family tuned baselines with the pipeline in a single configuration is unequal in the opposite direction.

**Authors' decision (2026-11-10):** per-family tuning ("per-family tuning, indeed"), applied to the baselines and to the
pipeline. All tables go in the supplement; the main text receives only sentences with the held-out numbers and the
cross-reference (section 5).

**This is an addition to the registered design, not a replacement.** The `r1_2a` reports (configuration chosen on
development and per-scenario upper bound) continue to be produced and reported.

## 3. Unit and grids

- **Unit:** the family, as in `common.family` (17 families, 90 scenarios). One configuration per method and per
  family, the same in all scenarios of the family.
- **Families with several scenarios (12):** synthetic VAR (5), tigramite (15), Lorenz-96 (6), logistic maps (12),
  CausalTime (9), NetSim (3), BEELINE (6), Causal Chambers (4), DREAM3 (5), EasyVista (7), PetShop (3),
  RCAEval (10).
- **Single-scenario families (5):** ALARM, PeMS-BAY, Sachs, Tennessee Eastman, Walker.
- **Baseline grid:** the expanded grid of `PROTOCOL_expanded_baseline_grid.md` (1,019 configurations), run
  from scratch. Each method's default is in it. The same analysis on the original `r1_2a` grid is reported alongside.
- **Pipeline grid:** the 216 configurations of the sensitivity run (Table S20: default, 65 one at a time, 150
  joint) of the nine cut-offs of Table I, seed 0, as in `r1_2b`.
- **Metric:** F1 per directed pair, tau_max = 10, as in Table III.
- A configuration only competes if it has a value in all scenarios of the family in which the default of the same method has
  a value.

## 4. The three conditions, for each method

1. **Default.** Baselines: the paper's parameters. Pipeline: the paper's configuration.
2. **Tuned per family (ceiling).** The configuration with the highest mean F1 over the scenarios of the family itself. It uses the ground
   truth of the same scenarios on which it is measured: it is an upper bound, not a usable result.
3. **Tuned per family, with one scenario left out.** In the 12 families with several scenarios: for each scenario, the
   configuration with the highest mean F1 over the other scenarios of the family, measured on the left-out scenario. The 5 single-scenario
   families do not enter this condition.

Tie in the selection: the default is kept. Among tied non-default configurations: the first in grid order.

## 5. Reporting (whatever the result)

- **Per family (supplement):** for each method, the configuration chosen in condition 2, the F1 in the three conditions
  and the number of configurations in the grid. Single-scenario families marked as pure ceiling.
- **Summary (supplement):** mean F1 of each method in the three conditions, overall, on development and on
  held-out; and each method's gain from tuning (condition 2 minus 1; condition 3 minus 1).
- **Main text, no table:** in V-A, the held-out mean F1 of each method, default and tuned per family, and
  whether the comparison with the pipeline changes; in VI-C, the "untuned baselines" threat rewritten with that result. The
  sentences go in whatever the result.
- **Response letter:** the same numbers and a pointer to the tables.
- **Comparisons, each on held-out and overall** (two-sided paired Wilcoxon, Holm over the five; mean rank
  of the six, Friedman and Nemenyi critical difference):
  - default pipeline against tuned baselines (condition 2 and condition 3): the comparison R1 asked for;
  - tuned pipeline against tuned baselines, in the same condition: the like-for-like comparison;
  - default against default: the paper's, repeated here on the same scenarios, as a check.
- **Scenarios of each comparison:** those in which all compared methods have a value in the condition. VARLiNGAM and CUTS+
  do not run on all scenarios; the number of scenarios goes in each row.

## 6. Limitations to declare in the text

- **The grids are not equal in size.** Pipeline: 216 configurations of nine cut-offs. Baselines, in the expanded
  grid: DYNOTEARS 660, CUTS+ 225, PCMCI 60, VARLiNGAM 50, PCMCI+ 24. The ceiling favours the larger grid. The size
  of each grid goes in the table.
- **The pipeline grid covers 86 of the 90 scenarios.** The sensitivity run does not have Causal Chambers. In that
  family the pipeline only has condition 1; it is left out of the comparisons with the tuned pipeline and reported
  separately.
- **The pipeline grid uses the old time-axis rule** and the old guard (it is the paper's run).
- **The pipeline F1 in the sensitivity run** (seed 0, default configuration) may differ from the F1 of the main
  run; the two are reported side by side.
- **The pipeline grid covers nine cut-offs**, around the paper's values. It does not cover the other parameters.
- **Condition 2 is not a usable result** for any method: nobody chooses parameters knowing the graph.

## 7. Context, declared

Written after seeing:
- the paper's results and `r1_2b` (leave-one-family-out of the cut-offs);
- the complete original `r1_2a` grid (another machine, 86 scenarios) and its report;
- the mean F1 of each configuration of the original grid on development and on held-out (NOTAS.md, check of
  2026-11-10), including which configurations do better on held-out.

Not tabulated per family: the per-configuration F1 of the pipeline and those of the baselines. None of the three conditions
of section 4 was computed before the freeze.

## 8. Implementation

- New offline script: `revision/offline/r1_2f_per_family_tuning.py`. It reads the `f1_por_config.csv` of the expanded grid
  (`r1_2g_grade_ampliada`) and, for the side-by-side report, that of the original grid (`r1_2a ... avaliar`); for the pipeline, the
  sensitivity run, through the `load` function of `r1_2b`.
- Depends on the expanded grid finishing and being tabulated.
- No change to the pipeline or to the baseline wrappers.

## 9. Decisions, taken by delegation on 2026-11-10 (to be confirmed by the authors)

Criterion given by the author: "always prioritize the fullest resolution of the reviewers' points and honesty".

1. **Original `r1_2a` report** (configuration chosen on development; per-scenario upper bound): kept.
2. **Pipeline on Causal Chambers** (216 configurations): not run. In that family the pipeline only has condition 1.
3. **Pipeline grid:** that of the sensitivity run, with the old time-axis rule; declared as a limitation.

## Log

- Draft: 2026-11-10, from the authors' decision to tune per family.
- Updated on 2026-11-10: expanded grid, context (section 7) and decisions (section 9).
- Freeze: 2026-11-10; hash in NOTAS.md.

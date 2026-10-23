# Analyses

Follow-up analyses of the paper's results. Each one reads a finished run and re-runs at most the DML
of a few links, never the whole pipeline. Like the scripts in [`generators/`](../generators/README.md),
they hold no path: they read `MAIN_RUN` (and, for one check, `OUT_DIR`) from
[`generators/origins.py`](../generators/origins.py). Each script writes its reports and tables to a
`results/` folder next to it.

Run every script from the repository root.

| Folder | Question | Scripts | Run time |
|---|---|---|---|
| [`stage7_vs_adaptive_rule/`](stage7_vs_adaptive_rule/) | Does Stage 7 beat a simple rule that also sets its operating point per dataset? | `adaptive_rule.py` | seconds |
| [`tigramite_dml_coverage/`](tigramite_dml_coverage/) | Why do the DML intervals under-cover on the tigramite SCMs? | `dml_controls_tigramite.py`, then `monte_carlo_coverage.py` | about 4 min + 3 min |

---

## 1. Stage 7 against a simple adaptive rule

**Question.** The paper's Stage 7 ablation (Table VI; Tables S25 and S17) compares the
Dempster-Shafer fusion with acceptance rules whose threshold is fixed in advance, and with oracle rules
that pick their threshold per scenario from the ground truth. Stage 7 beats every fixed rule, and the
oracles beat Stage 7. The paper reads this as follows: Stage 7 contributes the choice of an operating
point per dataset, not a better ranking of the pairs. That left a gap. No simpler rule that also picks
its operating point per dataset, without ground truth, had been tested.

**Rule tested (rule 11).** It is the counterpart of Stage 7's own adaptive Support floor
(`AdaptiveCausalPenalty(percentile=45, min_n_candidates=10)` in `Config/execution.py`), with the DML
t-statistic in place of the fused Support:
- Within each scenario, take the pairs with |t| > 1.96 as candidates.
- Accept the candidates above the 45th percentile of their |t|.
- With fewer than 10 candidates, keep the fixed 1.96 cut-off alone.

The rule and the decision criterion were fixed before the computation; see
[`PREREGISTRATION.md`](stage7_vs_adaptive_rule/PREREGISTRATION.md).

**Method.** `adaptive_rule.py`:
- Rescores the pipeline's stored per-pair output (`patch_tau10/original.csv` and
  `patch/original.csv`) on the 58 scenarios of `MAIN_RUN`.
- Uses the code of `generators/paper_numbers.py::ablation()`: same metric, same oracles, two-sided
  paired Wilcoxon test against Stage 7.
- Adds rule 11 and applies the Holm correction over 11 comparisons.
- Checks that the 10 existing rules reproduce the paper's tables (section 8 of
  `Paper_Numbers_tau{10,24}.md` in `OUT_DIR`). If that report is absent, it skips the check.

**Result** (`results/RESULTS.md`).

| Window | Stage 7 F1 | Rule 11 F1 | ΔF1 | Rule wins / ties / losses | Holm p | Pre-registered verdict |
|---|---|---|---|---|---|---|
| τ_max = 10 (primary) | 0.591 | 0.542 | −0.049 | 19 / 8 / 31 | 0.025 | Stage 7 wins |
| τ_max = 24 | 0.594 | 0.544 | −0.050 | 23 / 5 / 30 | 0.16 | tie |

- The 10 existing rules reproduce the published tables at both windows.
- Rule 11 reaches Stage 7's recall (0.716 against 0.714) with lower precision (0.459 against 0.539).
- The percentile cut-off applies on 40 of the 58 scenarios.
- With an eleventh comparison, every Holm-adjusted p of the tables changes slightly. No earlier
  conclusion is reversed.

**Outputs.**
- `results/RESULTS.md`: the full tables at both windows.
- `results/rules_tau{10,24}.csv`.
- `results/f1_per_scenario_tau{10,24}.csv`.
- `results/adaptive_cutoff_tau{10,24}.csv`: the cut-off per scenario.

---

## 2. Coverage of the DML intervals on the tigramite SCMs

**Question.** On the 15 `tigramite` scenarios (five SCMs, seeds 11, 22, 33), the 95% intervals of θ̂
cover the reference for 79% of the true links at τ_max = 10 (Table S9, panel 7). The internal VAR
generator also under-covers, but there `generators/dml_controls.py` traced the shortfall to the
controls. On tigramite the cause had not been isolated.

The hypothesis tested here is that the two computations condition on different controls:
- the reference regresses Y_t on X_{t−τ} and Y_{t−1} only, on a 500,000-point simulation (panel 8);
- the pipeline conditions on more: the target's past up to its BIC order and the other candidate
  parents, or every series in the blanket design.

On a nonlinear link, different controls change the quantity being estimated.

**Method, step 1: `dml_controls_tigramite.py`.**
1. **Reproduction.** For each scenario and window, the script:
   - runs Stages 0–2 of the pipeline on the stored `source.csv`;
   - takes the candidates of `original.csv`;
   - re-runs Stage 6 (`estimate_effects_by_control_set`) with the run's configuration, so the
     cascade, the learner selection and the parent-based/blanket choice are those of the run.

   The re-estimated θ̂ matches the reported one on every evaluated link, 48 at τ_max = 10 and 49 at
   τ_max = 24, to 5 × 10⁻⁵ (the CSV's rounding). This includes the links whose reported learner is
   Nadaraya-Watson.
2. **Re-estimation.** Each true link reported at its true lag is re-estimated with the learner the
   pipeline reported for it, and again with Bayesian ridge, under three control sets:
   - the pipeline's;
   - Y_{t−1} only (the reference's);
   - the true parents.

   In these SCMs every target has one cross-parent and first-order autodependence, so the true
   parents are the same set as Y_{t−1}.

**Method, step 2: `monte_carlo_coverage.py`.** Each SCM is simulated again with 300 seeds the run did
not use (T = 300, as in the suite). Every true link is estimated with the same DML (Bayesian ridge,
K = 3, HAC standard error) and the reference's controls. The script checks that its estimator
reproduces the run's links (to 10⁻⁷). It then asks how often three random seeds give a pooled coverage
as low as the suite's. The seed is the unit here, because within a seed the SCMs share their noise.

**Result** (`results/RESULTS.md`, `results/MONTE_CARLO.md`).

| | τ_max = 10 | τ_max = 24 |
|---|---|---|
| Coverage as reported (pipeline controls) | 79.2% | 83.7% |
| Coverage with the reference's controls (Y_{t−1}) | 79.2% | 79.6% |
| Same, Bayesian ridge on every link | 79.2% | 79.6% |
| Coverage by seed, as reported (11 / 22 / 33) | 12/17, 16/16, 10/15 | 14/17, 16/16, 11/16 |
| Monte Carlo, reference's controls, 300 new seeds per SCM | 93.5% (5,100 estimates) | — |
| Share of three-seed Monte Carlo draws with coverage ≤ the suite's | 0.9% | 0.9% |

What these numbers show:
- **The hypothesis is not supported.** Controlling only for what the reference controls for does not
  raise coverage. It lowers the bias toward zero (mean signed z from −0.30 to −0.18 at τ_max = 10), but
  the standard deviation of (θ̂ − θ)/SE stays at 1.25–1.30.
- **The nuisance learner is not the cause either.** Using Bayesian ridge on every link changes nothing.
- **The misses fall on two of the three seeds.** With the same seed, the five SCMs share noise; two
  links are identical copies across SCMs.
- **The estimator is close to calibrated over repeated realizations.** With the reference's controls
  it covers 93.5% (one quadratic link 84%). A coverage as low as the suite's occurs in about 1% of
  three-seed draws.
- **Not tested.** Whether the pipeline's own control set is calibrated over repeated realizations. That
  would require running the whole pipeline on new seeds.

**Outputs.**
- `results/RESULTS.md`: every control set, by functional form, by design and by seed.
- `results/MONTE_CARLO.md`.
- `results/links_tau{10,24}.csv`: one row per link, with every estimate and its SE.
- `results/stage6_reproduction_tau{10,24}.csv`.
- `results/monte_carlo_links.csv`.

# Protocol — new time-axis rule (R3-1, R2-m1)

**Status: APPROVED and frozen (see Log at the end).** After the freeze, any change to this design
enters as a recorded deviation in `revision/NOTAS.md`, with the reason.

Authors' decisions (2026-11-09): option (a), replace the rule; after the `r2_m1c_regras_eixo` comparison, adopt the
"block-wise Hosking" version (R2 of that comparison) in place of the (A)+(B) version of the first draft. Motivation and
evidence in `revision/NOTAS.md` (sections R2-m1/R3-1, R3-1 counterexample and R2-m1c).

---

## 1. The current rule

`Pipeline/discovery.py::_detect_time_axis(processed, alpha_white_noise=2.0)`: on the preprocessed series
(output of Stage 2) with at least 30 values, it computes the median of |ACF(1)|; if it falls below 2/sqrt(n)
(n = median of the number of values), it declares **no time axis**.

The decision is used at two points:
1. **Stage 5** (`discovery.py`, line ~164): with `rescue_correct_lag_multiplicity='auto'`, the Šidák correction of the
   rescue is only applied if there is a time axis.
2. **Stage 8** (`contemporaneous.py::drop_lagged_without_time_axis`, enabled by `lagged_requires_time_axis`):
   without a time axis, all lagged links are discarded (`structural_rejection_reason = 'no_time_axis'`).

## 2. The defects

1. **R3's case.** ACF(1) measures the dependence of each series on its own past; it does not see the dependence of one series
   on the past of another. In Y_t = c·X_{t-1} + noise with white X, no series has autocorrelation and the rule
   discards precisely the effect being sought. Evidence: simulated counterexample (VAR without AR) — the rule fires in
   20/20, pipeline F1 = 0 against 0.85-0.96 for the baselines.
2. **R2's case.** The median is a summary statistic: in a heterogeneous network in which most series are white and
   a few have their own dynamics (strong AR, unit root), the median says "white". Evidence: arm S5 of
   `r2_m1c_regras_eixo` — the current rule is right in 0 to 25% of the scenarios.

On the paper's data, the rule fires in 7 scenarios, 5 of them with detectable lagged cross-dependence.

## 3. The new rule: multivariate portmanteau tested block by block

**Hypothesis tested:** the series form a multivariate white noise (none depends on its own past or on the
past of another). **No time axis ⇔ none of the d² blocks rejects this hypothesis.**

1. Series: the same as in the current rule (output of Stage 2, at least 30 non-missing values, deviation > 0); NaNs that
   Stage 2 left (gaps > `gap_limit`) are linearly interpolated for the test only; standardized.
2. **Diagonal blocks (each series with itself):** Ljung-Box of the series, without prewhitening,
   Q_ii = n(n+2) Σ_{k=1..m} r_ii(k)² / (n−k) ~ chi-squared(m).
3. **Off-diagonal blocks (ordered pair i → j, i ≠ j):** Haugh test on the prewhitened residuals —
   each series is replaced by the residuals of an AR(p) fitted by least squares, p ∈ {0, ..., 5} by BIC,
   standardized; Q_ij = n_r Σ_{k=1..m} r_ij(k)² ~ chi-squared(m), where r_ij(k) is the correlation between the residual of
   i at t−k and that of j at t, and n_r the number of residuals.
4. A single family with the d² p-values, Benjamini-Hochberg correction at q.
5. Parameters: **m = 5, q = 0.05** (fixed before the `r2_m1c` comparison; the sensitivity over m ∈ {3, 5, 10} and
   q ∈ {0.01; 0.05; 0.10} has already been measured in it).

Consequence: a single series with its own dynamics, or a single pair with lagged dependence, is enough to keep the
time axis. The rule only declares "no axis" in a subset of the cases in which the current one does (on the paper's data,
2 of 7: ALARM and `lt_actuators_white`).

Cost: d² tests and m products of d × d matrices; negligible compared with the pre-filter.

## 3a. Scientific basis and justification

**What is to be detected.** "No time axis" means that the observations behave as independent
in time — the vector of series is a multivariate white noise. The current rule tests only part of this (the
autocorrelation, through a median); the new one tests the whole hypothesis, block by block.

**Where each component comes from:**
- **Multivariate portmanteau** — Hosking (1980), *JASA* 75(371), 602-608 (already cited in the paper as [109], in
  limitation L3). It tests the multivariate white-noise hypothesis at once, with d² · m degrees of freedom; on the
  paper's data it is unstable in 53 of the 90 scenarios (d² · m > n). The new rule tests the same d² blocks one by one, which
  remains usable with many series.
- **Diagonal: Ljung-Box** — Ljung and Box (1978), *Biometrika* 65(2), 297-303; white-noise band of Bartlett and
  Box-Jenkins ([51] in the paper).
- **Off-diagonal: cross-correlation of prewhitened residuals** — Haugh (1976), *JASA*, "Checking the
  independence of two covariance-stationary time series: a univariate residual cross-correlation approach";
  variant with a finite-order AR: Hong (1996), *Biometrika* 83(3), 615-625; multivariate extension: El Himdi and Roy
  (1997), *Canadian Journal of Statistics* 25(2), 233-256.
- **Why prewhiten off the diagonal:** autocorrelation in one of the series produces apparent cross-correlation
  even between independent series; prewhitening (Box-Jenkins) removes this effect.
- **AR order by BIC** — Schwarz (1978), *Annals of Statistics* 6(2), 461-464.
- **Correction for many tests** — Benjamini and Hochberg (1995), *JRSS B* 57(1), 289-300 (already used in the pipeline).

**Who asked:** Reviewer 2 ("how robust this single summary statistic is in heterogeneous multivariate networks
... whether pair-specific portmanteau tests were considered as an alternative") and Reviewer 3 ("individual series
can have zero autocorrelation while exhibiting strong cross-series lagged dependence ... revisions or targeted
validation against such cases"); limitation L3 of the paper itself already said that the rule had not been compared with
a multivariate portmanteau.

**What is a design choice, not a result from the literature (declare in the text):**
- Using the test as a regime gate in a causal discovery pipeline: no work was found that does this;
  the components are classical, the combination is new.
- Off the diagonal, the sum runs only over positive lags, per ordered pair (the two directions come from (i, j) and
  (j, i)); Haugh sums over lags on both sides.
- The per-pair result is not used to decide the links of that pair (another possible reading of R2's request): that
  would duplicate the per-pair screening of the pre-filter (Stages 3-5).
- m = 5 is the same as in the Hosking test already computed in `r2_m1_portmanteau`; q = 0.05 is the usual level. Alternative with more
  power recorded, not adopted: Hong's (1996) statistic with kernel weights over many lags.
- Benign failure mode: with non-stationary series, the diagonal block rejects (ACF close to 1) and the axis is
  kept, which is the conservative behaviour.

The references Haugh (1976), Hong (1996), El Himdi and Roy (1997), Ljung and Box (1978), Schwarz (1978) and Benjamini
and Hochberg (1995) go into the paper's reference list (check which are already there); the bibliographic data of Haugh, Hong and
El Himdi and Roy were checked in a search on 2026-11-09; those of Ljung and Box, Schwarz and Benjamini and Hochberg, to be checked
at the source.

## 4. Implementation

Repository convention: no default of the pipeline changes.
- `Pipeline/discovery.py`: new function `_block_portmanteau_is_white(processed, m=5, p_max=5, q=0.05)` that
  returns (is it white noise?, statistics); `_detect_time_axis` gains `rule: str = 'median_acf1'` — with the default,
  the behaviour is exactly the current one; with `rule='block_portmanteau'`, it applies section 3.
- `filter_causal_candidates` (Stage 5) and `contemporaneous.drop_lagged_without_time_axis` (Stage 8) receive and
  pass on the rule; `run_pipeline` gains `time_axis_rule: str = 'median_acf1'`.
- New model variant, without touching `Config/execution.py`: `patch_tau10_eixo_bloco` =
  `patch_tau10` with `time_axis_rule='block_portmanteau'` (`revision/execution/common_exec.model`).
- The decision log prints the rule, the number of series and of pairs that reject.
- Promoting the variant to the paper's configuration is a decision of the authors after the re-execution (C3); until then,
  `patch_tau10` remains the paper's model.

## 5. Criteria fixed before running

- **C1 (implementation).** Unit tests: (i) with the default, the decision is identical to the current one in the 90 scenarios;
  (ii) with `rule='block_portmanteau'`, the decision is identical to R2 of `r2_m1c_regras_eixo` in the 90 scenarios and in the
  simulated scenarios; (iii) a synthetic i.i.d. case → no axis; (iv) Y_t = c·X_{t-1} + noise, white X → with axis.
- **C2 (re-execution).** `patch_tau10_eixo_bloco` on the 5 scenarios of the paper whose decision changes (BEELINE HSC-1, HSC-2,
  VSC-2, VSC-10; `lt_smooth_polarizers`) and on the 40 of the counterexample (`r3_1_contraexemplo_eixo`); on the other 85, the
  decision is the same as under the current rule, so the output is identical by construction (checked in C1).
- **C3 (reporting, whatever the result).** F1 of `patch_tau10` and of `patch_tau10_eixo_bloco` in each of the
  45 scenarios, and the means on development, on held-out and over the total of 90 (with the 85 unchanged). Success
  expected, not assumed: in the 20 scenarios without AR of the counterexample, the pipeline F1 stops being 0. In the 5 scenarios
  of the paper, the F1 may rise or fall; both cases go into the text.
- **C4 (sensitivity).** Already measured in `r2_m1c_regras_eixo` (decision stable over the 9 (m, q) pairs).
- **C5 (power, additional simulation, decision only).** `r2_m1d_poder_eixo`: weak cross-dependence (the repository's VAR
  generator without AR, `coef_strength` (0.05; 0.15) and (0.15; 0.30), T ∈ {50, 100, 300}, 5 and 12 entities) and series
  with a linear trend or seasonality without cross-dependence; report the fraction of "with axis" per cell for the
  current rule and the new one, with no pass criterion (it is a power curve).
- **C6 (text).** Describe the new rule (Section III-D2 and Table I), the comparison of the rules, the counterexample and the
  before/after numbers of the affected scenarios; the old numbers stay in the supplement. Declare that the change was
  motivated by the reviewers' criticism and by the simulation, after the held-out ones had already been seen.

## 6. Related proposal, pending decision: coherence guard at scale

It is not part of this protocol until the authors decide. The guard compares the upper bound of rho(TE, |t|) with 0.3;
rho falls with the number of candidates (0.6 → 0.26 from 10 to 160 entities) and the guard fires wrongly at n = 160.
Options to design if the correction is approved: compute rho only over the k strongest candidates (e.g. k = 10 · d), or
also require that the fraction of candidates confirmed by TE stay below a threshold. If approved, it goes into its own
protocol, with the same criteria, and the two re-executions can be done together.

---

## Log

- Draft 1 ((A)+(B) version): 2026-11-09, morning/afternoon.
- Comparison of the three rules (`r2_m1c_regras_eixo`, valid execution f4eed02966c7, 16:01): R2 passed in all
  cells of the simulation; R1 failed in R2's heterogeneous case.
- Draft 2 (this version, block-wise Hosking): 2026-11-09, after the comparison.
- Approval: instruction from one of the authors on 2026-11-09, after seeing the comparison and the recommendation of R2:
  "let's pursue all the answers for the reviewers". Recorded as approval to proceed; **the authors must
  confirm** — if they do not confirm, the re-executions done with the variant remain valid as an experiment, but not
  as the paper's rule.
- Freeze (date and SHA-256 of this file): recorded in `revision/NOTAS.md` and in
  `Results_Revisao/registro_execucoes.jsonl` (event `congelamento`), because the hash cannot be inside the file
  itself.

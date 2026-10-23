# Causal Analysis Pipeline

One file per stage. `pipeline.py` holds only the high-level
orchestration — no math lives in it: `run_pipeline` runs the stage
functions listed in `_STAGES` one after the other, each delegating to
its own module.

## Stage map

The numbering and names are those of the paper (Section 3, Figure 1). The
stages form three layers: the *detection and estimation layer* (Stages 1–6),
the *evidence layer* (Stage 7) and the *structural layer* (Stage 8).

| Stage | What it does | File | Entry point |
|-------|--------------|------|-------------|
| — | Input structuring, before Stage 1 (long → wide, one series per entity×property) | `pivot.py` | `load_and_pivot` |
| 1 | Active-property detection | `property_scanner.py` | `scan_entity_properties`, `get_active_pairs` |
| 2 | Preprocessing | `preprocessing.py` | `preprocess_all` |
| 3–5 | Correlation pre-filter, Transfer Entropy and rescue; lag selection with a pluggable λ(τ) (`penalties.py`) | `discovery.py` | `filter_causal_candidates` |
| 3 | — lagged cross-correlation screening | `xcorr.py` | `xcorr_filter_pair` |
| 4 | — Transfer Entropy with permutation test | `transfer_entropy.py` | `te_filter_pair` |
| 5 | — rescue by Granger causality and Pearson bootstrap | `rescue.py` | `granger_rescue_pair`, `pearson_rescue_pair` |
| 6 | Cascaded DML for lag-specific effect estimation (BR→RF→KRR→NadarayaWatson); control-set switch (default or blanket design) | `dml_engine.py` | `estimate_effects_by_control_set` |
| 6 | — results table + statistical checks | `results.py` | `build_results`, `apply_statistical_checks` |
| 6 | — evidence-coherence check (TE vs. DML \|t\|), applied by the Stage 8 guard | `coherence.py` | `assess_evidence_coherence` |
| 7 | Evidence-based filtering of effect estimates (Dempster-Shafer fusion of seventeen specialists, pluggable) | `SelectorFunc.py`, `penalties.py` | `dml_engine.apply_causal_penalty` |
| 8a | Graph-level structural adjustment — structural pruning (lagged edges) | `structural.py` | `apply_structural_filter` |
| 8a-bis | — predictive parent competition (near-deterministic regime) | `predictive.py` | `predictive_competition` |
| 8 | Regime guards (evidence-coherence guard, no time axis) | `coherence.py`, `contemporaneous.py` | `keep_stage6_pairs`, `drop_lagged_without_time_axis` |
| 8b | Graph-level structural adjustment — contemporaneous completion (lag-0 adjacencies) | `contemporaneous.py` | `append_contemporaneous_links` |
| 8 | Output invariant — a reported pair always carries its effect estimate | `results.py` | `require_effect_estimate` |

Three regime checks, decided from the data alone (no ground truth, no
dataset name), change how the stages behave on a given dataset:

- **near-deterministic dynamics** (Stage 6): the DML falls back from the
  default control set (the target's own past and its other candidate
  parents) to the blanket design, and the predictive competition (8a-bis)
  is attempted;
- **incoherent evidence** (Stage 6): the Stage 7–8a rejections are not
  applied — the pairs with a significant Stage 6 effect are kept — and the
  output is flagged; except, with `evidence_coherence_fallback='auto'`, when
  the past explains little of the series (R² at or below the Stage 8b gate):
  the dependence is then mostly contemporaneous, the Stage 7–8a decision
  stands and 8b adds the lag-0 links;
- **no time axis** (Stage 8 guards): every lagged edge is dropped and 8b
  describes the whole graph.

Utilities outside the stage chain: `reporting.py` (`print_report`),
`synthetic.py` (`make_synthetic_df`), `diagnostics_aux.py` (extra residual
diagnostics used by the DML cascade).

## DML cascade (BR → RF → KRR → NadarayaWatson)

`dml_engine.py` natively runs the bayesian_ridge → random_forest →
kernel_ridge → nadaraya_watson cascade (accepted by agreement, Šidák
correction, dCor/RESET nonlinearity diagnostics, extra diagnostics via
`diagnostics_aux.py`, `lag_median` injection before the causal penalty
— see the module's own docstring). Just call `run_pipeline`.

```python
from Pipeline.pipeline import run_pipeline

results = run_pipeline(df, ...)    # uses BR -> RF -> KRR -> NadarayaWatson
```

# Protocol — new benchmark after the freeze (R1-2c)

**Status: part A APPROVED and frozen** (decision of the authors by delegation, 2026-11-13; see `revision/NOTAS.md`,
entry "R1-2c — protocol of the new benchmark"). Hash recorded in NOTAS.md. **Annex B (external regime) is NOT yet
frozen:** it is completed and frozen with its own hash before any run (section 8).

## 1. The request

R1, point 2(c): "a truly fresh benchmark added after the code freeze". The meaning: evaluate the method on data that
no method of the paper has seen, with the code already frozen, a single time, and report the result whatever it is.

The scope was fixed on 2026-11-11 (decisions by delegation, item 8): data on disk never run, plus at least one
external regime, under a protocol with a hash before the download. The runs cover `patch_tau10`, `patch_tau10_final`,
the R1-3b variant and the 5 baselines.

## 2. What counts as "never run"

A dataset enters the test set only if no scenario folder with its name exists in any `Results*`
of the `Final2` repository. The search was done on 2026-11-13, before this protocol, by folder name:
- **never run:** DREAM3 sizes 50 and 100; DREAM4 size 100; wind tunnel, random walks 1 and 4 to 16;
  `loads_hatch_mix` (10 runs); `color_mix` (light tunnel);
- **already run (left out):**
  - wind tunnel random walks 2 and 3, `actuators_white` and `smooth_polarizers`, which are in the held-out set of the
    paper;
  - CIPCaD UF, in `Results_Real/Run_20261004_133152_cipcad_uf`, with 3 windows. The dataset survey of
    2026-11-09 listed it as "to be confirmed"; it is confirmed that it has already run.

## 3. Split of the never-run data, fixed before running

The split follows the proposal of `estudo/11`, written before any result of the revision. It separates a
**validation** set, for developing the improvement variants (workstream B of `PLANO_ACAO_2026-11-13.md`), from the
**test** set, which is R1-2c.

| Set | Data | Scenarios | Use |
|---|---|---|---|
| **Test (R1-2c)** | Wind tunnel, random walks 10 to 16 | 7 | Single evaluation |
| | Wind tunnel, `loads_hatch_mix` (fast_run_1-5, slow_run_1-5) | 10 | Single evaluation |
| | Light tunnel, `color_mix` | 1 | Single evaluation |
| | DREAM3 size 100 (Ecoli1, Ecoli2, Yeast1, Yeast2, Yeast3) | 5 | Single evaluation |
| | DREAM4 size 100 (networks 1 to 5) | 5 | Single evaluation |
| | External regime (annex B) | to be fixed in the annex | Single evaluation |
| **Validation (workstream B)** | Wind tunnel, random walks 1 and 4 to 9 | 7 | Development of the variants; **never** enters the R1-2c report |
| | DREAM3 size 50 (five networks) | 5 | Same |

- **Separation rule:**
  - no test data is loaded, looked at or run to develop anything;
  - no validation data enters the R1-2c report;
  - the only check allowed on the test data before the run is the loading check (count of series,
    points, pairs and ground-truth edges).
- **Caveat to declare:** the data on disk are from families or devices already seen. DREAM3 and the
  Causal Chambers device are in the held-out set; DREAM4 size 10 ran among the candidates. They are new realizations
  and conditions, not a new regime. **The new regime is that of annex B.** The two blocks are reported separately.

## 4. Loading (fixed parameters, the same as for the held-out set of the paper)

| Data | Reader | Parameters |
|---|---|---|
| Causal Chambers | `Data.causal_chamber.generate_causal_chamber_benchmark_scenarios` | `experiments` = the (dataset, experiment) pairs of section 3; `n_points` = 500, `offset` = 0, `step` = 1 (those of `main_heldout.py`, `CAUSAL_CHAMBER_N_POINTS`) |
| DREAM3 | `Data.dream3.generate_dream3_benchmark_scenarios` | `sizes` = (100,) in the test and (50,) in the validation; the five networks; other parameters at their defaults |
| DREAM4 | `Data.dream4.generate_dream4_benchmark_scenarios` | `sizes` = [100]; `networks` = 1 to 5; `mode` = "concatenated" (the default) |

No reader is changed. The ground truth is that of each reader, the same convention as in the paper.

## 5. Methods (all with `tau_max` = 10, as in Table III)

1. `patch_tau10`, the submitted method: **the main comparison**.
2. `patch_tau10_final` = `patch_tau10` + `time_axis_rule='block_portmanteau'` + `evidence_coherence_overlap='coherent'`
   (the definition of `r1_3d`, of `r0_beeline_empates` and of `r2_2a`).
3. The R1-3b variant, as it comes out of its own protocol, frozen **before** the R1-2c run.
4. The 5 baselines in the configurations of the paper: `pcmci_tigramite_tau10`, `pcmciplus_tau10`, `dynotears_tau10`,
   `cuts_plus_tau10` and `varlingam_tau10` (`common_exec.PAPER_MODELS_TAU10`).

- **Code:** `method_code_hash` = `d0473069ebef1978` for items 1, 2 and 4. The R1-3b variant adds
  code that is switched off by default, and the hash changes. Its protocol must show that, with the variant switched
  off, the output of `patch_tau10_final` is identical, byte for byte, on the test scenarios of R1-3b itself.
- **Execution:** as in the runs of the paper (`common_exec.run_resuming`: sequential, with the default threads of the
  libraries and `PYTHONHASHSEED` unset). The thread configuration and the environment go to the report (D2b; round 4).
- **Improvement variants (workstream B):** they enter only if they are frozen before the R1-2c run. Otherwise, they
  stay out, and the report says so. No variant is added after the results of R1-2c have been seen.

## 6. Order and single run

1. This protocol (part A) frozen with a hash.
2. Annex B completed (license, access, format and selection rule checked without looking at any result) and frozen
   with a hash.
3. The R1-3b protocol frozen, and the variant implemented and tested.
4. The script `revision/execution/r1_2c_new_benchmark.py`, written after this protocol and with its hash noted, with the
   modes `carregar` (only the count of section 3), `rodar` and `avaliar`.
5. **One run**, on the runner, following the daily plan, all methods at once.
   - If a run fails for an external cause (machine, memory), it is resumed (`time.txt`), and the report says so.
   - Nothing is run again because of the result.

**Why R1-2c waits for R1-3b, instead of running now:** running before and adding the variant afterwards would make it
an evaluation done with the results already seen, and R1-2c would no longer be a blind test for it. The wait is 1 to 2
days.

## 7. Analysis (fixed in advance)

- **Scoring:** the same as in the paper (`main.validate_all`: directional pairs, `SOFT_THRESHOLD` = 1,
  `DETECT_INTRA_ENTITY` = True, `USE_CANONICAL_PAIR` = False). F1, precision and recall per scenario.
- **Run without result** (error or method guard): left out of the mean of that method, like VARLiNGAM on the 3
  held-out scenarios of the paper. The count is shown alongside. The paired tests use the scenarios in which both
  methods have a result.
- **Tables:**
  - **T1:** F1 per scenario and per method;
  - **T2:** mean F1 per family, with the number of scenarios of each family alongside;
  - **T3:** mean F1 per block: (i) data on disk from families already seen; (ii) external regime; (iii) both together.
- **Tests:**
  - main: `patch_tau10` against each baseline, paired Wilcoxon over the test scenarios, with Holm correction
    for the 5 baselines, as in Table III. Done on (iii) and, separately, on (i) and (ii);
  - secondary, descriptive and with the same test: `patch_tau10_final` and the R1-3b variant against `patch_tau10`.
- **Descriptive counts:** scenarios below the trivial floor ("all edges", as in step P), per method; the
  mean rank per scenario, as in the paper.
- **Interpretation, fixed in advance:** there is no "success" criterion for the pipeline. Any result is reported.
  - If the pipeline is above the baselines, or with no significant difference: this is consistent with the reading of
    the held-out set ("strong general-purpose choice"), and the text says so.
  - If a baseline is significantly above: report it with the same prominence, in the text and in the letter, and
    delimit the scope with what the result shows.
  - Blocks (i) and (ii) are reported separately; (ii) answers the "truly fresh" more directly.
- **What is not done:** choosing the scenarios, the metric or the block after seeing the result, or running again.

## 8. Annex B — external regime (TO BE COMPLETED AND FROZEN BEFORE RUNNING)

- **Order of preference,** fixed now, by the criterion of the survey of 2026-11-09 (new regime, reliable ground truth,
  close to the proposed use):
  1. **CausalDynamics** (NeurIPS 2025): coupled systems with a known graph;
  2. **causalAssembly** (Bosch, 2023): semi-synthetic manufacturing;
  3. **Krebs** (2024), long variant (KrebsL).
- **The first one that passes the four conditions goes in, and only it:**
  - (a) a license that allows research use;
  - (b) access without a request for special credentials;
  - (c) time series with lagged dependence, not only i.i.d. samples;
  - (d) ground truth published by the dataset itself, not assembled by us.
- **What the annex must fix,** without looking at any result (only metadata and the loading count):
  - the version or the commit of the dataset;
  - the subset selection rule, for example "the first K systems of the catalogue, in order, with a number of
    variables between 5 and 40", decided before opening the data;
  - the length of the series and the sampling: in continuous time, the step defines the lag, and this must be fixed
    beforehand;
  - the new reader (`Data/<dataset>.py`), with the loading check.
- **If none passes the four conditions:** R1-2c runs only with block (i), and the report says, with the words
  "new regime absent", that the request was met in part, and why.

## 9. Outside this protocol

- The measured interventions of the light tunnel (`lt_interventions_standard_v1`), for R1-4e, go into a protocol of
  their own, if they are done (dataset survey, section 4).
- The scale (R2-4b) has already been addressed; DREAM 100 is not used here as a scale analysis.

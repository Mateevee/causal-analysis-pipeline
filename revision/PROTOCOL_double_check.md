# Protocol — double check of round 2 (2026-11-11)

## 1. Objective and the author's decisions

- **Author's decisions (2026-11-11):**
  - "the purpose now is to have an overview of how to respond, then run the tests for a double check; what
    came out different can be run again";
  - "let's prioritize the runs that take the least time first";
  - "can we put more scenarios in today's run?" → D2 goes from the 18 scenarios that differed to the 90, and
    D4 gains the DYNOTEARS and CUTS+ grid in 14 families (approved by delegation of the author, entry of
    2026-11-11 in `NOTAS.md`).
- **Question:** do the numbers of round 2 (another machine, `runner`) reproduce on this machine? And is what does not
  reproduce the paper variation between executions, between machines, or something else?
- **Scope:**
  - nothing in the method changes;
  - no number of the paper is recomputed;
  - no result of round 2 is replaced.
  - Everything that comes out of here is reported alongside, whatever the result.
- **Fixed before running:** this document is frozen with two hashes (disk in CRLF and content in LF) before the
  first execution. Any deviation is noted in `NOTAS.md` with the reason.

## 2. Inputs

- **Round 2:** `Results_Revisao/_outra_maquina/rodada2_grade_ampliada_2026-11-10/`. Copy checked by SHA-256
  on 2026-11-11: 141,075 files.
- **Round 1:** `Results_Revisao/_outra_maquina/rodada1_faseA_2026-11-09/`. It is where `r1_2a_baselines_ajustados/run`,
  `r2_4b_escala/run` and `r1_3d_guarda_reexecucao/run` come from, which round 2 did not bring.
- **Runs of the paper and sensitivity run:** those of this machine, read-only.
- **Code:** that of this machine.
  - `method_code_hash` must be `d0473069ebef1978`, the same recorded there. If it is not, stop and take note.
  - Scripts `r1_2g_expanded_grid.py` (CRLF `9e9a3e63…893738`) and `r1_2f_per_family_tuning.py`
    (CRLF `216c45a4…2e13bd9`): the same as there.
- **Scenario names:** the tables use the paper's names (`real_alarm_uti`, `real_pems_bay_trafego`); the `run/`
  folders, the loader's names (`real_alarm_icu`, `real_pems_bay_traffic`).

## 3. Steps, in order of execution (shortest time first)

The times are estimated from the durations recorded in round 2.

### P — floor count on held-out (offline, ~5 min)
- **What:** on the 39 held-out scenarios, the F1 of the trivial floor ("all edges"), as in
  `generators/paper_numbers.py`, and the F1 of each of the six methods in the paper's configuration.
- **Output:** per scenario, which methods fall below the floor. Counts: per method; with all six below; with at
  least one above.
- **Use:** correct R2's reading ("all methods ... in 23 of the 39"). It is descriptive, with no pass criterion.
- **Check:** the pipeline count has to be 23 of 39 (V-A). If it is not, stop and take note.

### D5 — CUTS+ with threshold below 0.5 (offline, ~5 min)
- **What:** for the 20 native configurations with threshold 0.1 to 0.4, on the 90 scenarios of round 2, check whether
  `results.csv` contains every ordered pair (i, j), i ≠ j, at least once ("complete graph").
- **Output:** fraction of executions with a complete graph, per threshold, network and part (development or held-out). List
  of those that are not complete.
- It is descriptive. It answers the open item of round 2: "I did not check on the 90 whether the graph below 0.5 is always the
  complete one".

### D3a — R2-4b, n = 160, seed 6102, comparison between machines (offline, ~10 min)
- **What:** compare `r2_4b_escala/run/revisao_escala_n160_seed6102/` of this machine (F1 0.796) with that of round 1
  (F1 0.831) in:
  1. `source.csv` (byte by byte);
  2. number of candidates in Stage 6, ρ and guard interval, regime;
  3. TP, FP and FN of `original.csv` and `results.csv`;
  4. the edges that only one of the two machines has.
- **Reading fixed beforehand:**
  - **`source.csv` different:** the generator is not reproducible across machines. Take note and stop here.
  - **`source.csv` equal and output different:** variation of the pipeline across environments. Proceed to D3b.

### D1 — offline reproduction of the round 2 reports (~45 min: ~30 of preparation, ~10 of execution)
- **Environment:** short root `C:\dc\F2`, because this machine has `LongPathsEnabled = 0`.
  - Copy of the current Final2 code: `revision/`, `Pipeline/`, `ModelComparison/`, `Config/`, `Data/`,
    `Auxiliary/`, `generators/`, `main*.py`.
  - Directory junctions, read-only by use, for the input `Results*` folders, except `Results_Revisao`, and
    for `external_data/`.
  - `Results_Revisao/` assembled by **copy**, never by junction, so that nothing is written in the round 2 folders: first
    the round 1 folders, then those of round 2 on top.
  - Check before running: `method_code_hash` = `d0473069ebef1978`.
- **Commands, in the order in which they ran there:**
  1. `offline/r1_3c_guard_rho_decision.py`
  2. `execution/r1_3d_guard_rerun.py testar`
  3. `execution/r1_2a_tuned_baselines.py avaliar`
  4. `execution/r1_3d_guard_rerun.py avaliar`
  5. `offline/r1_2b_lofo_cutoffs.py`
  6. `execution/r1_2g_expanded_grid.py tabular`
  7. `offline/r1_2f_per_family_tuning.py --grade-original`
  8. `offline/r1_2f_per_family_tuning.py`
- **Criterion:**
  - Every output CSV byte-for-byte equal to the one delivered by round 2 (line endings normalized before the
    comparison).
  - `dados.json` and `relatorio.md` equal after removing the lines with date, time or machine name.
  - If anything differs: numerical comparison with a tolerance of 1e-9 and a list of every difference. Nothing is corrected.
- **Log:** the lines that `run_log.py` writes to `C:\dc\F2\Results_Revisao\registro_execucoes.jsonl` go back
  to the log of this machine, annotated.
- **Afterwards:** the folder `C:\dc\F2` is kept until the end of the revision and is only deleted with the author's ok.

### D3b — only if D3a shows `source.csv` equal and output different (~30 min)
- **What:** `patch_tau10` on `revisao_escala_n160_seed6102`, twice on this machine, with the same code.
- **Reading:**
  - **the two equal to each other and equal to the one from here:** the difference is between machines;
  - **the two different from each other:** non-determinism within the machine.

### D2 — re-execution of the DYNOTEARS and CUTS+ defaults on the 90 scenarios (~40 min; ~10 min per repetition)
- **New script:** `revision/execution/r0_double_check.py`.
  - It uses the specifications (`SPECS`) of `r1_2g_expanded_grid.py` without changing them.
  - The SHA-256 of the script is noted before the first execution.
  - The script is tested on 1 scenario (`--verificar`), and the test is not a result.
- **Executions:**
  - `dynotears_lw0.05_la0.05_w0.3_a0.1_tau10` (default) and `cuts_plus_th0.5_h32_tau10` (default), seeds 0, 1
    and 2, **on the 90 scenarios** (1.4 h of CPU per pass in round 2). Those that did not reproduce the paper are the ones
    highlighted in the report:
  - DYNOTEARS, 6 scenarios: `benchmark_causaltime_pm25_sample0`,
    `benchmark_causaltime_pm25_sample2`, `benchmark_netsim_sim3_subj1`,
    `benchmark_rcaeval_RE1-OB_adservice_delay_5`, `real_pems_bay_trafego` and `real_petshop_high_traffic`.
  - CUTS+, 12 scenarios:
    `benchmark_causaltime_traffic_sample1`, `benchmark_rcaeval_RE1-OB_adservice_cpu_3`,
    `benchmark_rcaeval_RE1-OB_adservice_delay_5`, `benchmark_rcaeval_RE1-OB_cartservice_delay_3`,
    `benchmark_rcaeval_RE1-OB_checkoutservice_disk_1`, `benchmark_rcaeval_RE1-OB_checkoutservice_mem_5`,
    `benchmark_synthetic_var_5_seed55`, `real_alarm_uti`, `real_causal_chamber_lt_actuators_white`,
    `real_easyvista_web_2`, `real_petshop_low_traffic` and `real_petshop_temporal_traffic1`.
  - **Two repetitions** (`rep1`, `rep2`), in separate folders, with one thread per process, as in `r1_2g`.
  - **D2b, a third repetition with the libraries' default number of threads** (without `uma_thread`). It tests the
    hypothesis that the difference comes from numerical parallelism. Another ~15 min.
- **Comparison:** TP, FP and FN of each execution, compared as follows:
  - `rep1` against `rep2`: within the machine;
  - against round 2: between machines;
  - against the paper's run;
  - `rep1` against D2b: effect of the threads.
- **Reading fixed beforehand:**
  - **`rep1` = `rep2` = round 2 ≠ paper:** the re-execution is stable; the paper was generated in another environment, or
    with another thread configuration if D2b reproduces the paper. Declare it.
  - **`rep1` = `rep2` ≠ round 2:** variation between machines. Declare it.
  - **`rep1` ≠ `rep2`:** non-determinism within the machine. Declare it and measure the F1 range.
- **Also:** how many of the other scenarios (84 for DYNOTEARS, 78 for CUTS+), equal to the paper in round 2, come out
  equal here.
- **Output:** `Results_Revisao/r0_dupla_conferencia/` (`run_rep1/`, `run_rep2/`, `run_threads/`, `d2.csv`,
  `relatorio.md`).

### D4 — grid of the deterministic methods on this machine (~1 to 1.2 h)
- **What:** the 39 re-executions of PCMCI, PCMCI+ and VARLiNGAM of `r1_2g`, on the 90 scenarios: 3,510 executions.
  - There it took 11.6 h of CPU; the heaviest scenario is ALARM, with 71 min.
  - Command: `r0_double_check.py d4 --paralelo 20`.
- **Plus (B):** the complete DYNOTEARS and CUTS+ grid (267 re-executions) on the scenario with the shortest summed duration
  of these two grids in round 2, in each family, except the three in which that scenario exceeds 6 h of CPU
  (PetShop, ALARM, Tennessee Eastman): 14 scenarios, ~7 h of CPU there, ~1 h of wall time (RCAEval, with 1.5 h,
  sets the time). Scenarios: `benchmark_synthetic_var_1_seed11`, `benchmark_dream3_size10_Ecoli2`,
  `benchmark_logistic_maps_convergence_seed3`, `benchmark_tigramite_saturation_seed22`, `real_easyvista_mom_1`,
  `benchmark_lorenz96_K8_seed1`, `benchmark_netsim_sim3_subj2`, `benchmark_beeline_VSC_VSC-2000-10`,
  `benchmark_causaltime_traffic_sample1`, `real_sachs_proteins`, `real_tigramite_climate_walker`,
  `real_pems_bay_trafego`, `real_causal_chamber_wt_actuators_random_walk_3`,
  `benchmark_rcaeval_RE1-OB_productcatalogservice_cpu_2`.
- **Criterion:**
  - **PCMCI, PCMCI+, VARLiNGAM:** TP, FP and FN equal to those of round 2 in the 3,510 executions. The errors must be the
    same (VARLiNGAM on the same 13 scenarios, with the same message).
  - **DYNOTEARS and CUTS+ on the 14 scenarios:** descriptive (how many configurations equal, range of the F1 difference).
- **Not today:** the whole grid on the 90 (C, ~6.5 to 10 h) does not run today; the night goes to phase C. It remains
  an option if the machine becomes free.
- **Output:** `Results_Revisao/r0_dupla_conferencia/run_d4/` and `d4.csv`.

## 4. What to do with the result

- Every result goes into `NOTAS.md`, favourable or not, with what changes in the text (`[texto]`).
- **If D1 fails:** the round 2 reports cannot be cited until the cause is understood. Notify the author
  before proceeding.
- **If D2 or D4 show variation:** the round 2 numbers remain valid as the result of one execution.
  - The text and the supplement declare the variation and its range.
  - No new re-execution of the grid is decided without the author.
- The catalogue of `status.py` gains `r0_dupla_conferencia`.

## 5. Estimated total time

| Step | Time |
|---|---|
| P, D5, D3a | ~20 min |
| D1 | ~45 min |
| D3b (conditional) | ~30 min |
| D2 + D2b (90 scenarios) | ~40 min |
| D4 | ~1 to 1.2 h |
| B (14 families) | ~1 h |
| Annotation | ~30 min |
| **Total** | **~4.5 h** after the freeze (~5 h with D3b) |

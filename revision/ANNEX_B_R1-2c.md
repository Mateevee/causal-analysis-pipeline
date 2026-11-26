# Annex B of the R1-2c protocol — external regime: CausalRivers, East Germany

**Status: FROZEN on 2026-11-15, before any method was run on the test data.** Hash of this file in
`revision/NOTAS.md` (entry "R1-2c — annex B frozen"). It completes section 8 of `PROTOCOL_new_benchmark_R1-2c.md`;
part A of the protocol was not changed (CRLF hash `c4ed4b4d…`).

## 1. Deviation from section 8, recorded on 2026-11-13

Section 8 fixed the order CausalDynamics → causalAssembly → Krebs, with "the first one that passes the four conditions
goes in". On 2026-11-13, before any result, the choice was redone with seven criteria fixed before the
comparison (NOTAS, entry "choice of the external regime"):
- causalAssembly fails condition (c), time series;
- coupled CausalDynamics passes (a) to (c), but its regime is close to the Lorenz-96 and logistic maps of the
  development set; the form of the ground truth in the files and the number of systems had not been checked;
- Krebs is a single graph, so the statistical unit is 1;
- CausalRivers passes all seven criteria, and East Germany is the **test** split designated by the authors themselves
  (Bavaria, already used among the candidates of the supplement, is their training split).

## 2. Dataset, version and integrity

- **Source:** CausalRivers (Stein, Shadaydeh, Blunk, Penzel and Denzler, ICLR 2025). Release `First_release`,
  published on 2025-02-19 (`github.com/CausalRivers/causalrivers`; the old repository, `CausalRivers/benchmark`,
  redirects to it). Data license dl-de/by-2-0. The authors' code declares no license: **it is not imported,
  executed or redistributed**; it was read and reimplemented (section 4).
- **Integrity:** the release publishes no checksum. On 2026-11-15 it was downloaded again (`product.zip`,
  133 228 548 bytes, the same size that the GitHub API reports for the asset) and compared with the file on disk:
  - SHA-256 of the new download and of the zip on disk: `4f779fa9222653af092046eacba39a955cc65638bfb70fb7e31a32d9d8373b55`
    (identical);
  - the East Germany files extracted in `external_data/causal_rivers/` are byte-for-byte identical to the members of
    the zip:
    - `rivers_east_germany.p`: `d6ff7df27325be8e1a8b7194ee2d1dd915a29e253561e389db66338a6711ade3`;
    - `rivers_ts_east_germany.csv`: `c0b2202df0c69eefd3174160d8d537aa4977452bd0545c77e3ef41052da8abc5`.
  - The "~570 MB" in the docstring of `Data/causal_rivers.py` is wrong: the zip is 133 MB and expands to ~1.1 GB. It
    is not a data discrepancy.
- **Structure** (metadata only): `networkx` directed acyclic graph with 666 stations and 651 edges. The edge u → v
  means that u is the next measured point upstream of v; edges carry `geo_distance`. The series starts on 2019-01-01,
  at 15 min, one column per station.

## 3. Window, resolution and ground truth

- **The same as in the Bavaria CausalRivers runs of the supplement** (`Run_20261002_013625`), that is, parameters
  already fixed on the authors' training split:
  - start at 2021-01-01 00:00;
  - 500 points at 1 h (mean of the 15 min readings);
  - a station is eligible if it has ≤ 5% missing readings in the window and is not constant;
  - remaining gaps linearly interpolated, as in the reader `Data/causal_rivers.py`, whose functions are
    reused unchanged.
  - The 1 h resolution comes from the lag measured in Bavaria: median ~4.5 h between neighbouring stations; at 6 h (the
    default of the authors' preprocessing), ~60% of the edges would fall in the same time step.
- **Ground truth:** the subgraph induced on the chosen stations, as in the authors' `save_subgraphs_to_pickle` and in
  the paper's reader. Scoring: that of the paper (directional pairs between entities).

## 4. Subgraphs: the authors' strategies, reimplemented

- **Strategies** (commit `a30154fbb9a2e03bae133c643d7e978df63b3766`, `tools/graph_sampling_tools.py` and
  `0_generate_datasets.py`; `config/data_sampling.yaml`: `dist_measure` = `geo_distance`, `max_distance` = 0.3):
  - `root_cause_5`: connected sets of 5 stations whose longest directed path has 4 edges (an upstream
    chain);
  - `close_5`: connected sets of 5 stations whose longest path, in `geo_distance`, is shorter than 0.3.
- **Reimplementation:** `revision/execution/r1_2c_causalrivers.py`, SHA-256 in NOTAS. It is a line-by-line translation of
  `get_all_subgraphs`, `all_extensions` and `get_longest_path`, and of the filters of `0_generate_datasets.py`.
- **Check:**
  - running the authors' downloaded code locally, to compare the lists, was **blocked by the permission policy** of
    this session and was not circumvented;
  - instead, the enumeration of the connected sets was checked against an independent enumerator (Wernicke's ESU,
    with a different algorithm): **7 521 sets in both, identical**;
  - the `root_cause` and `close` filters are direct translations, checked by reading.

## 5. Selection rule (approved by the author on 2026-11-13: 30 per strategy; detailed here)

1. **Canonical order:** the list of sets, each as a sorted tuple of IDs, in lexicographic order.
   - The "order that the authors' tool produces" is the iteration order of a Python `set`, which is neither specified
     nor stable across versions, and the tool has no seed (the authors' own TODO says so).
   - The approved text ("the first 30 in the tool's order, with a fixed seed") is implemented as canonical
     order plus a draw with a fixed seed. It is the reproducible form of the same rule.
2. **Eligibility:** all 5 stations eligible in the window (section 3).
3. **`root_cause_5`:** 30 drawn without replacement among the eligible ones, with `numpy.random.default_rng(20261115)`.
4. **`close_5`:** 30 drawn with `default_rng(20261116)`, among the eligible ones not drawn in `root_cause_5`, so that
   no scenario is repeated.
5. If a strategy has fewer than 30 eligible sets, all of them go in.

**Counts** (structure and eligibility only; no method was run):

| | Stations | Edges | Eligible in the window |
|---|---|---|---|
| Graph | 666 | 651 | 606 stations |

| Strategy | Candidates | Eligible | Chosen | Ground-truth edges per scenario |
|---|---|---|---|---|
| `root_cause_5` | 655 | 505 | 30 | 4 |
| `close_5` | 637 | 423 | 30 | 4 or 5 |

There are 60 scenarios: 5 stations × 500 points, 20 ordered pairs. The names are
`real_causal_rivers_east_germany_<strategy>_<00..29>`, and the stations of each scenario are in `scenario_params`.

## 6. What goes into the single run (complements sections 5 and 6 of part A)

- **Blocks and order:** `chambers` (18) → `rivers` (60) → `dream4` (5) → `dream3` (5). The cheap ones go first; the
  order does not change the results.
- **Methods:**
  - `patch_tau10` (main), `patch_tau10_final`, `patch_tau10_graded` and the 5 baselines;
  - plus the **exploratory** variant `patch_semS7_tau10`, without Stage 7 (`UniformCausalPenalty`, the native
    switch-off). Decision by delegation, recorded in NOTAS before running. It runs in a final pass, after all the
    blocks, and does not enter the main test.
- **Secondary tests added on 2026-11-14, before running:**
  - `patch_tau10_final` and `patch_tau10_graded` against each baseline, with Wilcoxon and Holm over the 5;
  - the main test is still `patch_tau10` against the baselines;
  - no version is promoted to main after the result is seen.
- **Reader:** the protocol (section 8) foresaw a new reader in `Data/`. It was written in `revision/execution/`, so as
  not to change the `method_code_hash` (`d0473069ebef1978`). It is a deviation of form, with no effect on the data.
- **Script:** `revision/execution/r1_2c_new_benchmark.py`, SHA-256 in NOTAS. It refuses to run if the `method_code_hash`
  changes.
- **Estimate:** ~50 to 65 h sequential (from 40 to 90 h); DREAM3-100 is ~70% of the total (NOTAS, 2026-11-15).

## 7. Preparation environment

Python 3.12.14, networkx 3.6.1, numpy 2.0.0, pandas 2.3.3 (main machine). The environment of the run goes to the
log (`registro_execucoes.jsonl`) and to the report.

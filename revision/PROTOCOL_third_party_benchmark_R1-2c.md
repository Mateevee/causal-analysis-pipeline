# Protocol — second benchmark after the freeze: two third-party generators never run (R1-2c, block iii)

**Status: part A, written on 2026-11-22 and frozen with a hash BEFORE any download of these data** (hash in
`revision/NOTAS.md`, entry "R1-2c, block (iii) — protocol"). **Annex C** (structure of the files) is completed
after the download, only with reading of the structure and the loading count, and frozen with its own hash before
any method run (section 6).

## 1. Why this protocol exists

- **The request.** R1, point 2(c): "a truly fresh benchmark added after the code freeze".
- **The problem with the first protocol.** The 88-scenario benchmark (`PROTOCOL_new_benchmark_R1-2c.md`, part A
  frozen on 2026-11-13) has as its external regime CausalRivers from East Germany. The same family, in the region of
  Bavaria (the training split of the benchmark's authors), had already run among the candidate families of Table S15 of
  the original supplement. The East Germany data had never run, but the family was not new, and the text went as far
  as saying "a family the pipeline had never seen" (corrected; `SubmissaoIEEE/NOTAS_FINAIS.md`, item 54).
- **The author's decision (2026-11-22):** "I want a new dataset to run what was asked and to resolve this
  correctly"; "we are not choosing after seeing, the first protocol was wrong"; "let's do it with these 2,
  use a protocol that satisfies r1. do everything carefully".
- **Mandatory statement (goes into the text and the letter):** this protocol was written after the results of the
  88-scenario benchmark had been seen. Therefore:
  1. the 88-scenario benchmark is still reported in full, as it is; nothing of it is replaced or withdrawn;
  2. this benchmark is reported in full, next to the first one, whatever the result;
  3. the two datasets were chosen by the criteria of section 2, before any of their data was downloaded, and the
     scenario selection rule is fixed here, with only the file listing (metadata).

## 2. Datasets and what counts as "never run"

**Criteria** (the seven of 2026-11-13, with the first one stricter):
1. no data of the **family** has run in any execution of this work (original paper, supplement, revision);
2. a regime different from the families of the development set;
3. ground truth published by the dataset itself, in machine-readable form (not assembled by us);
4. fits the format of the pipeline, with non-degenerate density;
5. open license and verifiable provenance;
6. published in a recognized venue;
7. enough scenarios for the paired test.

**Check of "never run" (2026-11-22, before this protocol):** no folder with `timegraph`,
`causaldynamics`, `causal_dynamics` or `coupled` in its name exists in any `Results*` of the `Final2` repository.
TimeGraph is only cited in the literature review (`docs/`). Of CausalDynamics, on 2026-11-13, only the
`simple` and `climate` files were downloaded, to read the structure; no method was run on them, and they do not enter
here. The **coupled** files were never downloaded.

| Dataset | Fixed source | License | Publication |
|---|---|---|---|
| **TimeGraph** | `github.com/hferdous/TimeGraph`, commit `4d93403b002664fa55607f308ad0e80623472ded` (2025-06-03), folder `Datasets/` | code MIT; data CC BY 4.0 (README and `LICENSE-CC-BY-4.0.txt`) | Ferdous, Hossain and Gani, KDD 2025, Datasets and Benchmarks Track; arXiv 2506.01361; DOI 10.1145/3711896.3737439 |
| **CausalDynamics, coupled** | `huggingface.co/datasets/kausable/CausalDynamics`, revision `e42a6eebb4736f982be5332dbba91670b21e861e`, file `inputs/coupled.tar.gz`, 6,514,628,417 bytes, official SHA-256 (LFS) `92e6a64b34a9908c2c9af5d8b5b753eb057afb7463e4650771ea21c90569f6c0` | MIT (code and dataset) | NeurIPS 2025, Datasets and Benchmarks Track; arXiv 2505.16620 |

**Caveats to declare in the report:**
- both are **synthetic**, from generators of other groups; they are not measured data;
- coupled CausalDynamics consists of coupled chaotic dynamical systems, a regime related to the Lorenz-96 and the
  logistic maps of the development set (criterion 2 only in part);
- TimeGraph uses structural processes with lags, related to the `tigramite` models of the development set; what
  changes is what the dataset adds (irregular sampling, missing values, latent confounder, mixed noise, depending on the
  variant), described in annex C from the authors' paper.

## 3. Selection of the scenarios (fixed now, before the download)

### 3.1 TimeGraph

- **Universe:** every CSV of `Datasets/` in the fixed commit that has **machine-readable ground truth published in the
  same folder**, paired by name: the suffix `_complete` or `_missing`, if any, and the extension are removed from the
  name of the CSV; the ground truth is `<base>_structure.txt` or `<base>_causal_structure.txt`.
  - From the listing of the commit (GitHub API, 2026-11-22, names only), this gives **1,188 bases**: B2 36, C2C 72,
    D2C 216, D3 432 and D3C 432.
  - The variants whose ground truth exists only as a **figure** (PNG) are left out: transcribing an image would make
    the ground truth ours (criterion 3). These are A1, A1C, A2, A2C, B1, B1C, B2C, C1, C1C, C2, D1, D1C and D2, and
    the CSVs of B2 without a structure file.
- **Version:** where a base has `_complete` and `_missing`, the `_missing` one goes in (the condition the variant was
  made to test). The other file is not used.
- **Draw:** per variant, the bases in lexicographic order of the full path; **20 drawn without replacement** with
  `numpy.random.default_rng(20261122 + k)`, k = position of the variant in the order (B2, C2C, D2C, D3, D3C) = 0 to 4;
  drawn indices in increasing order. If a variant has fewer than 20 bases, all of them go in. **Total: 100
  scenarios.**
- **Preprocessing, in the reader, the same for all methods:**
  - the rows, in file order, are the time axis; a timestamp column, if present, is not used as a
    variable (the irregular sampling is read as an ordered sequence, which is what a user of these methods does);
  - columns that the dataset itself designates as **latent** (the confounder of the variants with C) are removed before
    running; the designation comes from the authors' documentation, code or structure file (annex C), never from a
    result;
  - missing values: linear interpolation along the index, in both directions (as in the CausalRivers reader);
  - columns that are constant after that are removed (`Data.base.drop_constant_columns`), with a count in the report.
- **Ground truth:** the directed edges between **distinct** observed variables listed in the structure file, at
  any lag; the self-lag is left out. Each variable is an entity with one property
  (`array_to_pipeline_format_single_property`).

### 3.2 CausalDynamics, coupled

- **Universe:** all the coupled graphs of `inputs/coupled.tar.gz` at the fixed revision.
- **Canonical order:** path of the file inside the tar, in lexicographic order; inside the file, the index of the graph
  in the order in which it is stored. As the internal organization has not yet been seen, annex C records how the
  graphs are enumerated (one per file or several per file), before any method runs; the draw rule below applies
  to that enumeration.
- **Strata:** the number of nodes N ∈ {3, 5, 10} (the authors' Table 1). **30 graphs drawn per stratum**, without
  replacement, with `numpy.random.default_rng(20261122 + 10 + k)`, k = 0, 1, 2 for N = 3, 5, 10; indices in increasing
  order. **Total: 90 scenarios.** Within each stratum, all the configurations of the generator (noise, confounder,
  lag, standardization, coupling type) compete in the draw; none is chosen by us.
- **Trajectory:** the first one stored for each graph (index 0), with all the steps.
- **Entities and properties:** each node is an entity; its d dimensions are the properties `F_0`…`F_{d−1}`,
  the format of the `tigramite` models with several properties.
- **Ground truth:** the summary adjacency matrix published with each graph, off the diagonal, between observed nodes.
  The orientation (row → column or the reverse) comes from the authors' code or documentation (annex C). A node that the
  dataset designates as a latent confounder is removed, with its columns.
- **Scoring for this dataset:** the pairs **within the same node** (between dimensions of the same node) are removed
  from the ground truth and from the predictions before the metrics, for all methods (`main._filter_intra_entity`, the
  same effect as `ignore_intra_entity=True`). Reason: the published ground truth is between nodes; the dimensions of a
  node form a single dynamical system, whose internal couplings the ground truth does not list as links (the diagonal of
  the summary matrix).
- **Exclusion:** a graph whose ground truth has no edge between distinct observed nodes is removed (F1 undefined),
  with a count in the report; there is no replacement.

## 4. Methods and execution

- **Methods** (all with `tau_max` = 10, as in Table III):
  1. `patch_tau10`, the default configuration: **the main comparison**;
  2. `patch_tau10_final`, the corrected variant (`time_axis_rule='block_portmanteau'`,
     `evidence_coherence_overlap='coherent'`): secondary;
  3. the 5 baselines in the configurations of the paper: `pcmci_tigramite_tau10`, `pcmciplus_tau10`, `dynotears_tau10`,
     `cuts_plus_tau10` and `varlingam_tau10`.
  - Neither the graded variant (rejected on the stability criterion; not a claim of the paper) nor the variant
    without Stage 7 (exploratory in the first benchmark) is run. The decision is one of cost and is recorded here,
    before running.
- **Code:** `method_code_hash` = `d0473069ebef1978`; the script refuses to run if the hash is different.
- **Execution:** each run (scenario × method) in its own process, with a **60 min limit** plus 30 s of slack, as
  in the final stages of the first benchmark (decision of 2026-11-17, for cost); past the limit, the run has no
  result, with the reason in the report. `PYTHONHASHSEED=0`, as in the reproduction instructions of the paper. Default
  threads of the libraries. Resumption via `time.txt`. The environment goes to the report.

## 5. Analysis (fixed in advance)

- **Scoring:** that of the paper (`main.validate_all`: directional pairs between entities, `SOFT_THRESHOLD` = 1,
  `DETECT_INTRA_ENTITY` = True, `USE_CANONICAL_PAIR` = False), with the exception of section 3.2 for CausalDynamics.
  F1, precision and recall per scenario.
- **Run without result** (error, method guard or time limit): out of the mean of that method, with a count
  alongside. The paired tests use the scenarios in which both methods have a result.
- **Tables:** T1, F1 per scenario and method; T2, mean F1 per TimeGraph variant and per CausalDynamics stratum N,
  with the number of scenarios; T3, mean F1 per block: (iii-a) TimeGraph, (iii-b) CausalDynamics and
  (iii) both together.
- **Tests:**
  - **main:** `patch_tau10` against each baseline, two-sided paired Wilcoxon, Holm over the 5 baselines, on
    (iii) and, separately, on (iii-a) and (iii-b);
  - **secondary:** `patch_tau10_final` against `patch_tau10` and against each baseline (Holm over the 5).
- **Descriptive:** scenarios below the trivial floor ("all edges"), per method; mean rank among the 7 methods.
- **This benchmark is not pooled with the 88-scenario one** in the main test: the protocols are different, and the two
  are reported side by side.
- **Interpretation, fixed in advance:** there is no "success" criterion for the pipeline. Any result is reported with
  the same prominence, in the text and in the letter. If a baseline is significantly above, this goes into the summary
  of the benchmark and delimits the scope of the paper.

## 6. Order and single run

1. This part A frozen with a hash (CRLF and LF), recorded in `NOTAS.md`, **before any download**.
2. Downloads, with the provenance checked in both directions (paper → repository and repository → paper):
   - TimeGraph: the repository archive at the fixed commit; SHA-256 of the downloaded file recorded;
   - CausalDynamics: `inputs/coupled.tar.gz` at the fixed revision; the SHA-256 must match the official one above.
   - Inspection of the archive listing before extracting (no absolute path, path with `..` or link; only the expected
     data types); extraction with `tarfile` and `filter='data'`; no script of the authors is executed.
3. **Annex C:** only structure and loading count (names, shapes, field and format of the ground truth, designation of
   the latents, enumeration of the graphs, orientation of the matrix; series, points and ground-truth edges per
   scenario). No method runs. Frozen with its own hash.
4. Readers and the script `revision/execution/r1_2c_third_party_benchmark.py` (modes `carregar`, `rodar` and `avaliar`),
   written after this protocol, with their hashes noted; a manifest with the SHA-256 of each input file.
5. **One run**, on the runner, following the execution plan (`INSTRUCOES_<date>.md` and queue). A run that fails
   for an external cause (machine, memory) is resumed, and the report says so. Nothing is run again because of the
   result.
6. `avaliar`, report, and application to the text and to the supplement, whatever the result.

## 7. What is not done

- Choosing scenarios, metric, block or dataset after seeing a result; withdrawing one of the two datasets after seeing
  a result; running again; adding variants afterwards.
- Looking at series values or running any method on these data before annex C is frozen.

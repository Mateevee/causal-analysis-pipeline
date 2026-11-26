# Annex C — structure of the files of block (iii): TimeGraph and coupled CausalDynamics

*Completed on the runner on 2026-11-22, after the download and BEFORE any method was run, using only the structure
report (`revision/offline/r1_2c_third_party_structure.py estrutura`, output in
`Results_Revisao/r1_2c_benchmark_terceiros/estrutura.md`), the structure census of CausalDynamics
(`revision/offline/r1_2c_third_party_cd_census.py`, output in `censo_causaldynamics.md` and `.csv` in the same folder) and
the count of `carregar` (`carregamento.csv`). Each decision gives its source (documentation, code or file of the
authors), never a result. Protocol: `revision/PROTOCOL_third_party_benchmark_R1-2c.md` (part A). All the fields of the
template are filled in; the `congelar` mode refuses while any of them remains open. **Section 5 has four deviations
from part A, decided by the author on this date, before any method was run.***

## 1. Download and provenance
- TimeGraph: SHA-256 of the zip of commit `4d93403b002664fa55607f308ad0e80623472ded`:
  `964dc047a3b8bd8b027734c241cbe4d4fea949988123c2d7005208e8967ad4f7`; size: 1,024,063,241 bytes; date and time:
  2026-11-22, 21:58 to 21:59 (from `codeload.github.com/hferdous/TimeGraph/zip/<commit>`).
- CausalDynamics: SHA-256 of `inputs/coupled.tar.gz` (revision `e42a6eebb4736f982be5332dbba91670b21e861e`):
  `92e6a64b34a9908c2c9af5d8b5b753eb057afb7463e4650771ea21c90569f6c0`, equal to the official one; size: 6,514,628,417
  bytes, equal to the official one; downloaded on 2026-11-22, 21:59 to 22:00. `download.json`: `causaldynamics_confere: true`.
- Provenance in both directions, checked on 2026-11-22 in the full text of the papers (`arxiv.org/html/<id>`):
  - TimeGraph, paper → repository: arXiv 2506.01361 (on the arXiv page, DOI 10.1145/3711896.3737439): "Source code
    is available at https://github.com/hferdous/TimeGraph".
  - TimeGraph, repository → paper (lines of the `README.md` of the fixed commit): "This repository supports our KDD 2025
    paper to promote transparent and reproducible research."; in the "Citation" section: `booktitle = {Proceedings of the 31st
    ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '25)}` and `doi = {10.1145/3711896.3737439}`.
    Licenses present in the commit: `LICENSE` (MIT, code) and `LICENSE-CC-BY-4.0.txt` (data).
  - CausalDynamics, paper → repository: arXiv 2505.16620: "The code is available at
    https://github.com/kausable/CausalDynamics and the dataset can be downloaded from
    https://huggingface.co/datasets/kausable/CausalDynamics".
  - CausalDynamics, repository → paper: the file came from that Hugging Face repository, at the fixed revision, with the
    official SHA-256 (LFS). The tar contains no README and no code (only `.nc` files).
- Source of the CausalDynamics code, only read (nothing executed): `github.com/kausable/CausalDynamics`, commit
  `c094bf9dd6e2c725c61af4ea57d9eb16f6ac3372` (2026-10-08, the most recent on 2026-11-22), zip of 50,302,410 bytes, SHA-256
  `b959de456562497703b6f31bdf19311b367bc1006feab190239a5e540415d920`, in
  `external_data/causaldynamics/codigo_autores/`. Caveat: it is today's code, not necessarily the code that generated
  the fixed revision of the data; for that reason, what was taken from it was also checked against the structure of the
  files (section 4).

## 2. Extraction
- TimeGraph (zip): 10,077 entries; extensions: `.csv` 4,320, `.png` 4,044, `.txt` 1,405, `.ini` 25, `.ipynb` 18, `.py`
  18, `.cff` 1, `.md` 1, no extension 1. No suspicious path (absolute, with `..` or outside the destination folder).
  Extracted to `external_data/timegraph/extraido/` (1.7 GB).
- CausalDynamics (tar.gz): 14,096 members, all `.nc`. No suspicious path, link or device. Extracted with
  `tarfile` and `filter='data'` to `external_data/causaldynamics/extraido_coupled/` (9.1 GB).
- No script of the authors was executed. No long-path error.

## 3. TimeGraph
- Bases with machine-readable ground truth: B2 36, C2C 72, D2C 216, D3 432, D3C 432 (1,188): this matches the listing
  of the commit used in part A.
- Columns of the CSVs, in the 1,188 bases: `X1..Xn` (n = 4, 6 or 8), then `U` only in the variants with C (C2C, D2C, D3C)
  and `time` last. There is a timestamp column: `time` (irregular sampling); it is not used as a variable.
  - Version: D2C, D3 and D3C have `_missing.csv` and `_complete.csv` (the `_missing` one goes in); B2 and C2C have only
    the CSV without a suffix.
- Variants with a confounder (C2C, D2C, D3C): the confounder appears as the column `U`. Designation by the dataset:
  - `README.md`: "Confounders: Incorporate latent variables influencing multiple observed variables"; "`a1c.ipynb` /
    `a1c.py`: Linear with confounder `U[t]`"; "Add `U[t]` terms to multiple variable equations";
  - code (`Codes/c2c.py`, `Codes/d2c.py`, `Codes/d3c.py`): `df['U'] = U  # Add confounder column`; label of the
    plots "U (confounder)";
  - structure files: titles "… with Confounder" and the equation `U[t] = eU`.
  - The column `U` is removed before running; the edges with source in `U` are left out of the ground truth.
- Format of the structure file. Three forms, all with the direction explicit (source first, target second):
  - B2 (`_structure.txt`), header "Format: (source, lag, target) => weight"; example: `('X1', -2, 'X4') => 0.25`;
  - D2C (`_causal_structure.txt`), header "Format: (source, lag, target) => coefficient", section "Equations" and section
    "Nonlinear Causal Links"; example: `(X1, -2, X4) => 0.25 [base coefficient]`;
  - C2C, D3 and D3C (`_structure.txt`), section "Equations" and section "Causal Links"; example:
    `X1 --(cos, lag=2)--> X4: 0.25`.
  - Lag: the second field (negative in the tuples, positive after `lag=`); from 0 to 4 in the 1,188 bases. The ground
    truth of the benchmark is the pair (source, target) at any lag.
  - Self-lag: none in the 1,188 bases (no edge with source equal to target).
  - Check on the 1,188 bases: no line with `=>` or `-->` was left unread (apart from the "Format:" line); no
    edge mentions a variable outside the header of the CSV; where there is an "Equations" section (all but B2), the
    pairs of the edges are equal to those of the equations (target = left-hand side; sources = variables on the
    right-hand side) in all of them.
  - From 4 to 9 edges between distinct observed variables per base.
- How `_tg_read_series` and `_tg_read_truth` were written:
  - `_tg_read_series`: reads the CSV; requires the header `X1..Xn` + (`U` in the variants whose name ends in C) + `time`,
    otherwise it stops; returns only the `X#` columns, in file order; reports `coluna_tempo = "time"` and
    `latentes_retiradas = ["U"]` (or empty). It does not interpolate.
  - `_tg_read_truth`: reads the edge lines in the three forms above; discards the lag; returns the pairs between
    distinct observed variables (edges with `U` and self-edges are left out). It stops if there is an unparsed edge
    line, no edge, an unknown variable, or a disagreement between the edges and the "Equations" section.
- Selection: `selecionar` reproduced `revision/r1_2c_third_party_timegraph_expected.json`: **yes** (the 100 items equal
  in name, variant, index, base, CSV and ground truth).
- From `carregar` (100 drawn scenarios):
  - points: 500 in 21, 1,000 in 27, 3,000 in 25, 5,000 in 27; variables: 4 in 24, 6 in 40, 8 in 36;
  - missing values interpolated: 366,593 in total; by variant: B2 0, C2C 0, D2C 142,603, D3 84,862, D3C 139,128;
  - constant columns removed: none, in any scenario;
  - scenarios with an empty ground truth: **none** (from 4 to 9 edges);
  - **infinite value:** one drawn scenario, `timegraph_D2C_06`
    (`Datasets/D2C/lag 2/nonlinear_ts_n5000_vars8_lag2_student_t_block10_missing.csv`), has infinite values in
    rows 4,766 to 4,999 of X1 to X5 (355 in the CSV; count and position, no value read beyond that). This happens in
    9 of the 1,188 bases, all D2C with n = 5,000, 8 variables and `student_t` noise. The interpolation of part A does
    not handle infinity. See section 5, deviation 3: the scenario is removed. **TimeGraph: 99 valid scenarios** (B2 20,
    C2C 20, D2C 19, D3 20, D3C 20), with 352,995 missing values interpolated (D2C 129,005).

## 4. CausalDynamics, coupled
- Organization: 14,096 `.nc` files, in 720 folders `coupled/<configuration>/data/`. **One graph per file** (index 0).
  The configuration of the generator is in the folder name:
  `coupling=<linear|nonlinear|periodic>_noise=<0.00|0.50|1.00|1.50|2.00>_systems=<3|5|10>_confounder=<True|False>_standardize=<True|False>_timelag=<0|1>_activation=<mixed|none>`.
  The file name is `S<SYSTEM>_N10_T1000_seed<k>.nc` (SLORENZ 3,600, SRANDOM 3,296, SRAYLEIGHBENARD 3,600, SROSSLER
  3,600); the `N10` in the name is the number of trajectories, not of nodes. Files per `systems=`: 3: 4,729; 5: 4,706;
  10: 4,661.
- Fields of each file (netCDF4/HDF5):
  - `time_series`, float32, dimensions (`time`, `system`, `node`, `dim`) = (1000, 10, nodes, 3) in all 14,096: 1,000
    steps, **10 trajectories** (`system`), d = 3 dimensions per node;
  - `adjacency_matrix` and `adjacency_matrix_summary`, (nodes, nodes), dimensions (`node_in`, `node_out`), values 0 and 1;
  - in the 7,052 files with `timelag=1`: also `adjacency_matrix_regular_edges`, `adjacency_matrix_time_edges` and the
    `lag` dimension; in the 7,044 with `timelag=0`: the scalar `time_lag`;
  - `weights`, `biases`, `magnitudes`, `root_nodes` and the coordinates (`time`, `system`, `node`, `node_in`, `node_out`,
    `dim`, `dim_in`, `dim_out`).
- Summary adjacency matrix: field `adjacency_matrix_summary` (the one the authors' `eval.py` uses: "if
  'adjacency_matrix_summary' in input_ds"), shape (nodes, nodes). The diagonal holds the self-edges of the roots
  (`data_io.create_output_dataset`: `add_root_self_edges(adjacency_matrix)`); the ground truth of the benchmark uses
  only what is off the diagonal. No NaN in any file.
  - **ORIENTATION: row → column, no transposition** (`A[i, j] = 1`: node i causes node j). Sources: paper, section 2:
    "a_ki = 1 if x_i is causally impacted by x_k, else a_ki = 0"; `src/causaldynamics/scm.py`: "A[i,j] indicates an edge
    from i to j" (in `create_scm_graph`, `get_root_nodes_mask`, `add_root_self_edges`). Check against the structure of
    the files: in the 3,513 files with `confounder=False` and `timelag=0`, `root_nodes` (node with no incoming edge)
    coincides with the null columns of `adjacency_matrix` in 3,513 and with the null rows in 0.
- Number of nodes N of each graph and count by N:
  - **the number of nodes stored is not always the N of the generator.** By `systems=` × `confounder=` × nodes in the
    file: (3, False): 3 in 2,358; (3, True): 2 in 1,657 and 3 in 714; (5, False): 5 in 2,352; (5, True): 4 in 2,354;
    (10, False): 10 in 2,328; (10, True): 9 in 2,333. By the number of nodes stored: 2: 1,657; 3: 3,072; 4: 2,354;
    5: 2,352; 9: 2,333; 10: 2,328;
  - N used in the draw: the N of the generator, `systems=` of the path (3: 4,729; 5: 4,706; 10: 4,661). See section 5,
    deviation 1.
- Confounder: in the files with `confounder=True` the authors **had already removed one node before saving**
  (`scripts/generate_coupled.py`: "if confounder: dataset = process_confounders(dataset)", with the help text "If passed,
  drop the identified confounder"; `src/causaldynamics/utils.py::process_confounders`: "This induces psuedo-A and dropped
  timeseries of the identified confounding variable", with `ds.drop_sel` of the node in `node_in`, `node_out` and
  `node`). For this reason the file has N − 1 nodes in 6,344 of the 7,058 graphs with `confounder=True` (and N in the
  714 in which the post-processing found no node to remove). **No latent node remains in the files; the reader removes
  none.**
  - Note for the report: from the code read, the removed node is the first one with two or more off-diagonal edges in
    its column (`child_counts = ((ds.adjacency_matrix != 0) & off_diag).sum(dim=in_dim)`), which, with the orientation
    row → column, counts incoming edges. The counts in the files agree with this reading (in `systems=3`,
    `confounder=True`: 714 without removal; 1,657 with 2 nodes, of which 720 without any off-diagonal edge). The report
    describes these graphs as "the authors' `confounder=True` configuration, with one node removed by them", without
    claiming that the removed node is a common cause. The published ground truth among the remaining nodes is used as is.
- Non-finite values (count and position; no value read): in 3,948 of the 14,096 graphs (28.0%) trajectory 0 has
  non-finite values, in all of them from step 0 to the end; in 3,912 a whole node has no value. By `systems=`: 3: 854
  (18.1%); 5: 1,204 (25.6%); 10: 1,890 (40.5%). All ten trajectories are entirely finite in 7,500 graphs; none is
  in 1,927 (13.7%; by `systems=`: 301, 481, 1,145). See section 5, deviation 2.
- Graphs without any off-diagonal edge: 720, all with `systems=3`, `confounder=True` and 2 nodes.
- How `_cd_enumerate` and `_cd_read_graph` were written:
  - `_cd_enumerate`: the 14,096 relative paths, with `/`, in lexicographic order; one graph per file (`indice` 0);
    `N` = `systems=` of the path; it stops if the path has no `systems=` in {3, 5, 10} or if the count is not
    14,096. It opens no file.
  - `_cd_read_graph`: reads `adjacency_matrix_summary` (not transposed; binary) and `time_series`; checks the dimensions
    (`time`, `system`, `node`, `dim`), that the matrix is square, without NaN and with values 0 and 1, and the number of
    nodes (N without confounder; N or N − 1 with confounder), otherwise it stops; returns the first entirely finite
    trajectory (the one with index 0 when it is finite); reports the trajectory used, how many of the 10 are finite, how
    many series are constant in it, the nodes in the file and whether the authors removed a node.
- Selection (list saved on 2026-11-22 22:24:46, `revision/r1_2c_third_party_selection.json`): 30 drawn per stratum
  (N = 3, 5, 10 of the generator), 90 in total; **68 valid** (N = 3: 19; N = 5: 27; N = 10: 22) and 22 excluded, without
  replacement:
  - without an edge between distinct nodes (section 3.2 of part A), 11: `causaldynamics_N03_03`, `_N03_05`, `_N03_08`,
    `_N03_10`, `_N03_13`, `_N03_14`, `_N03_16`, `_N03_17` (this one also without a finite trajectory), `_N03_20`,
    `_N03_26`, `_N03_28`;
  - without any entirely finite trajectory (deviation 2), 6 besides `_N03_17`: `causaldynamics_N05_15`, `_N10_04`,
    `_N10_08`, `_N10_09`, `_N10_16`, `_N10_23`;
  - with a constant series in the trajectory used (deviation 4), 5: `causaldynamics_N05_14` (1 series), `_N05_21` (2),
    `_N10_13` (2), `_N10_14` (2), `_N10_18` (1).
  - In the 68 valid ones: observed nodes 2 in 6, 3 in 13, 4 in 12, 5 in 15, 9 in 14, 10 in 8; `confounder=True` in 33
    (N = 3: 7; N = 5: 12; N = 10: 14); trajectory used: 0 in 58, 1 in 4, 3 in 3, 4 in 1, 6 in 2; 1,000
    points and 3 properties per node in all; no non-finite value and no constant series.

## 5. Deviations from part A
Four deviations, all decided by the author on 2026-11-22 (answers given in the conversation with Claude on the runner),
before any method was run on these data and without any result. What was known at each decision is stated in each item.
1. **CausalDynamics stratum N = N of the generator (`systems=` of the path), not the number of nodes stored.**
   - Reason: part A fixes the strata N ∈ {3, 5, 10} "(the authors' Table 1)" and says that all the configurations of
     the generator, including the confounder, compete in the draw; the docstring of `_cd_enumerate` said to take N from
     the file. The two cannot be reconciled, because the files with `confounder=True` have one node fewer (section 4):
     with the N of the file, 6,344 graphs (45%) would never enter, and strata 5 and 10 would have only
     `confounder=False`.
   - Decision of the author: "N of the generator". A scenario of stratum N has N or N − 1 observed nodes.
   - Known at the time: only the structure census (counts); the draw had not yet been computed.
   - In the code: `_cd_enumerate` (N from the path) and **one line outside the 4 functions**, in `_scenario_cd`: the
     shape check went from `N != it["N"]` to `N not in (it["N"], it["N"] - 1)`.
2. **CausalDynamics trajectory = the first entirely finite one, not necessarily the one with index 0; a graph without
   any entirely finite trajectory is removed, without replacement.**
   - Reason: part A says to use trajectory 0 and has no rule for missing values for this dataset; in 28.0% of the
     graphs trajectory 0 has non-finite values from step 0 to the end, almost always a whole node (section 4), which
     interpolation does not resolve.
   - Decision of the author: "1st finite trajectory". Options rejected: keep trajectory 0 and exclude; restrict the
     universe before the draw; replace with zero as the authors' `eval.py` does.
   - Known at the time: only the structure census; the draw had not yet been computed.
   - In the code: `_cd_read_graph` and, **outside the 4 functions**, the exclusion loop of `selection("causaldynamics")`.
3. **TimeGraph: a scenario with a non-finite value after interpolation is removed, without replacement** (part A did
   not foresee any exclusion in TimeGraph).
   - Reason: `timegraph_D2C_06` has infinite values (section 3), which the interpolation of part A does not handle.
   - Decision of the author: "Exclude, without replacement". Options rejected: keep as is; treat infinity as missing.
   - Known at the time: the drawn list and the counts of a trial load (non-finite values and constant series per
     scenario); no method had run.
   - In the code, **outside the 4 functions**: `selection("timegraph")` now marks the exclusion.
4. **CausalDynamics: a graph with any constant series (one dimension of a node without variation) in the trajectory
   used is removed, without replacement** (section 3.2 has no rule for constant series; 3.1 removes the constant column
   in TimeGraph, but the node × dimension format does not allow removing one dimension of a single node).
   - Decision of the author: "Exclude the 5 scenarios". Option rejected: keep as is (the one recommended by Claude).
   - Known at the time: the drawn list and the counts of the trial load (5 scenarios, 8 series); no method
     had run.
   - In the code: `_cd_read_graph` counts the constant series; **outside the 4 functions**, `selection("causaldynamics")`
     excludes.

Consequences to declare in the report:
- Totals: TimeGraph 99 valid scenarios of 100; CausalDynamics 68 of 90 (19, 27 and 22); block (iii) with 167 scenarios
  and 167 × 7 = 1,169 runs (693 + 476), not the 700 + 630 that the queue shows as expected (that number is only for
  the display of progress).
- The four exclusions use the same mark in the list, `excluido_sem_aresta`, because it is the only one that
  `r1_2c_third_party_benchmark.py` reads (the script was not changed). The field `motivo_exclusao` of each item of the
  list gives the reason. Therefore, **in the automatic `relatorio.md`, the sentence "excluded for having no edge
  between distinct nodes" lists the 22 excluded from CausalDynamics, for any of the three reasons, and does not mention
  the one excluded from TimeGraph**; the correct reasons are those of section 4 and of the list.
- Fields added to each item of the list: `motivo_exclusao`; in CausalDynamics, also `trajetoria` and
  `nos_no_arquivo`.
- Deviations 3 and 4 were decided after the drawn list was known, with loading counts and without any method
  result. The rules are general (they apply to any scenario), not a list of names.
- Apart from the deviations, nothing in part A changed: datasets, sources, seeds and sizes of the draw, methods,
  60 min limit, scoring, tests and interpretation.

## 6. Freezing
- Hashes of `revision/FROZEN_third_party.json` (protocol, annex C, readers, script, list), date and time: noted
  in `NOTAS_outra_maquina.md` and in the `RESUMO.md` of the zip after the `congelar` mode (this annex is frozen before
  receiving its own hashes).

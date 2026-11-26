# Analyses added in the revision of the paper

This folder holds the code and the protocols of the analyses added in the
revision of *A Multistage Pipeline for Predictive Causal Discovery and
Exploratory Effect Estimation in Multivariate Time Series* (IEEE Access,
manuscript Access-2026-54916): the tuned baselines, the two benchmarks fixed
after the code freeze, the targeted check of the time-axis rule, the
series-length sweep, the re-estimation on data not used for selection, the
comparison with randomized interventions, the graded regime switches, the
stage ladder, the stability under subsampling, the run times, and the other
analyses reported in Supplementary Tables S28 to S46.

The Supplementary Reproducibility Index of the paper maps every one of those
tables to the script in this folder that produces it.

## Layout

| Path | Contents |
|---|---|
| `execution/` | Scripts that run models (the pipeline, its variants, or the baselines). Each writes its output to `Results_Revisao/<experiment>/`. |
| `offline/` | Scripts that only read stored outputs (the runs of the paper and of `execution/`) and compute tables; they re-run no model. |
| `variants/` | The module of the graded regime switches (Table S41), kept apart so that the method's own code is unchanged. |
| `PROTOCOL_*.md`, `ANNEX_*.md` | The protocols of the analyses, each written and frozen before its run (English translations; see the Notes below). |
| `*.json` | Frozen inputs of the protocols (the data manifest and the seeded selection of the second benchmark). |
| `common.py`, `execution/common_exec.py`, `run_log.py` | Helpers shared by the scripts: loading and scoring exactly as the paper does (through `generators/paper_numbers.py`), and the log of every execution. |

## Running

Run every script from the repository root, with the reproduction settings of
the main `README.md` (`PYTHONHASHSEED=0`), for example:

```
python revision/offline/r1_m4_bootstrap.py
python revision/execution/r2_2a_T_sweep.py
```

The packages are those of the main `requirements.txt`, including its section
"Revision analyses" (`networkx`, `h5py`, `causal-learn`). The data of the two
benchmarks fixed after the code freeze and of the randomized interventions are
described in section 4 of `Data/README.md`.

The scripts in `offline/` need the runs of the paper, located through
`generators/origins.py`, and, for some tables, the outputs of the scripts in
`execution/`. Run folders (`Results*/`) are not distributed: the code here
produces them.

## Notes

- Code, comments, docstrings and protocols were written in Portuguese, the
  working language of the authors, and have been translated into English,
  file and folder names included (this folder was `revisao/`; for example,
  `execucao/r1_2c_benchmark_novo.py` is now `execution/r1_2c_new_benchmark.py`).
  Command-line arguments (`rodar`, `avaliar`, `--verificar`, ...), experiment
  names and the names of output folders, files and columns were kept in
  Portuguese, so that they match the stored results: an experiment therefore
  writes to `Results_Revisao/<Portuguese experiment name>/` (for example,
  `r1_2c_new_benchmark.py` writes to `Results_Revisao/r1_2c_benchmark_novo/`).
  Script names follow the reviewer comment each one answers (for example,
  `r1_2c_*` answers comment 2(c) of Reviewer 1).
- Some files mention the authors' internal lab notes and run instructions
  (`NOTAS.md`, `INSTRUCOES_*`), and the name of the second machine on which
  part of the runs were made; some protocols name `Final2`, the authors'
  working repository, of which this one is the release. Those notes are not
  part of this release; every
  result they recorded is reported in the paper and its Supplementary
  Material.
- Every table of the Supplementary Material names the script that produces
  it, and those scripts are in this folder, together with the scripts that the
  protocols here name and the runs whose folders other scripts read. Not
  distributed: the authors' bookkeeping scripts (the run queue and its status
  and progress reports, the checks of result files copied between the two
  machines, run-to-run determinism checks), pilots and previews superseded by
  the reported runs, and exploratory prototypes.
- The protocols in this folder are English translations of the frozen
  Portuguese originals. The SHA-256 of each protocol was computed on the
  Portuguese original when it was frozen, and is kept in the run registry
  (`Results_Revisao/registro_execucoes.jsonl`, not distributed); the same
  holds for the values recorded in `FROZEN_third_party.json` and for those
  quoted inside the protocols and scripts. They do not match the translated
  files, so the paper does not print them. The SHA-256 of the input data
  (`r1_2c_data_manifest.json`, `Data/README.md`) are unaffected by the
  translation and can be checked.
- Before each run, the scripts that run methods checked that the hash of the
  method code (`run_log.method_code_hash()`) equalled that of the frozen code,
  `d0473069ebef1978`; `r1_2c_third_party_benchmark.py rodar` also checked the
  five files of `FROZEN_third_party.json`, and three offline scripts checked
  the SHA-256 of the frozen script they import. The released method code
  differs from the frozen code only in three comment lines, in
  `Pipeline/coherence.py` and `Pipeline/discovery.py`, which name translated
  protocol files; its hash is `5c10222886dcc287`. In this release, these
  checks print a warning instead of stopping the run.

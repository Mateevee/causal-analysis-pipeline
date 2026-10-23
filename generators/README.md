# Paper generators

The scripts in this folder read finished runs and write every result-dependent
number, table and figure of the paper. None of them re-runs a model, and none
holds a number entered by hand. The Supplementary Reproducibility Index of the
paper maps each figure and table to the script that produces it.

## Origins

The scripts do not hold any path. They read their runs from
[`origins.py`](origins.py), where every origin starts as `None`. Two of these
origins feed the main results:

| Origin | Run written by | Content |
|---|---|---|
| `MAIN_RUN` | `python main.py` (`Results/Run_<timestamp>/`) | the development suite |
| `HELDOUT_RUN` | `python main_heldout.py` (`Results_HeldOut/Run_<timestamp>/`) | the held-out datasets; may be a list of run folders |

A scenario is looked up in the held-out runs first and then in the main run.

Each separate experiment of the paper has its own origin, which only the
script that reports it reads:

| Origin | Run written by | Read by |
|---|---|---|
| `GPDC_RUN` | `main_gpdc.py` | `table_s10_gpdc.py`, `paper_numbers.py` (optional) |
| `ABLATION_DIR` | `main_ablation_patch.py` (`Results_Ablation/`) | `table_additive_ablation.py`, `additional_analyses.py` (optional) |
| `SENSITIVITY_RUN` | `main_sensitivity.py` | `sensitivity_analysis.py` |
| `PCMCI_NOBH_RUN` | `main_pcmci_nobh.py` | `table_s25_pcmci_no_bh.py` |
| `GRANGER_TE_RUN` | `main_granger_te.py` | `table_s26_granger_te.py` |
| `METRICS_RUN` | a run scored by `Auxiliary/ranking_metrics.py` | `metrics_shd_auc.py`, `paper_numbers.py` (optional) |
| `CANDIDATES_RUN` | `main_heldout.py` with every held-out candidate family | `extra_numbers.py` (optional) |
| `SEEDS_RUN` | `main_seed_replication.py` | `coverage_new_seeds.py` |

A script stops with a message when an origin it needs is not set. When an
optional origin is missing, the script leaves out the part that uses it.

There are two ways to set the origins. If a value is set in both places, the
environment variable wins:

- **In `generators/origins_local.py`.** Git ignores this file. Write plain
  assignments in it:

  ```python
  MAIN_RUN = r"D:\runs\Results\Run_20261004_232204"
  HELDOUT_RUN = [r"D:\runs\Results_Extra\Run_20261008_221315",
                 r"D:\runs\Results_Extra\Run_20261009_causal_chamber"]
  ```

- **Through environment variables named `CAP_<ORIGIN>`.** For example,
  `CAP_MAIN_RUN=/data/Run_20261004_232204`. A list is separated by
  `os.pathsep` (`;` on Windows, `:` elsewhere).

Runs written before a scenario ID was renamed keep the old folder names. To
read them, map each old folder name to the current ID in
`LEGACY_SCENARIOS = {"real_old_name": "real_new_name"}` in
`generators/origins_local.py`; the reports then show the current IDs.

Reports go to `OUT_DIR` and figures to `FIG_DIR`. By default these are
`paper_outputs/` and `paper_outputs/figures/` at the repository root.

### The runs of the paper

The paper was produced from these runs:

| Origin | Run |
|---|---|
| `MAIN_RUN` | `Results/Run_20261004_232204` |
| `HELDOUT_RUN` | `Results_Extra/Run_20261008_221315` (BEELINE, DREAM3, RCAEval, EasyVista) and `Results_Extra/Run_20261009_causal_chamber` |
| `GPDC_RUN` | `Results_GPDC/Run_20261002_210059` |
| `ABLATION_DIR` | `Results_Ablation` |
| `SENSITIVITY_RUN` | `Results_Sensitivity/Run_20261010_233817` |
| `PCMCI_NOBH_RUN` | `Results_PCMCI_NoBH/Run_20261011_202543` |
| `GRANGER_TE_RUN` | `Results_GrangerTE/Run_20261012_182259` |
| `METRICS_RUN` | `Results/Run_20261011_132346` |
| `CANDIDATES_RUN` | `Results_Extra/Run_20261007_213826` |
| `SEEDS_RUN` | `Results_Seeds/Run_20261014_222930` |

These run folders are not distributed. The repository contains everything
needed to produce them, and producing them is up to whoever reproduces the
results: run the entry points listed in the main README (the seeds are fixed
in the code), then point the origins above at the folders they write.

`Run_20261004_232204` predates `main_heldout.py`. Besides the 51 scenarios of
the development suite, it also holds five held-out datasets: TEP, PeMS-BAY,
Sachs, the Walker Circulation and PetShop. Scripts that read only `MAIN_RUN`
therefore cover 58 scenarios with these runs, as the paper reports. Examples
are the Stage 7 ablation of `paper_numbers.py` and the Stage 7 re-execution
of `decision_trail.py`. With a fresh `main.py` run, the same scripts cover the
51 development scenarios.

The scripts tell the development suite and the held-out datasets apart by
dataset family, as the paper defines them (`HELDOUT` in `paper_numbers.py`),
and not by the run a scenario comes from.

## Running

Run every script from the repository root. Run `paper_numbers.py` first,
because `table_s8_ranking.py` reads the JSON files it writes.

| Script | Writes |
|---|---|
| `python generators/paper_numbers.py` | `Paper_Numbers_tau10.md`, `Paper_Numbers_tau24.md`, `paper_numbers_tau{10,24}.json` |
| `python generators/table_s8_ranking.py` | `Table_S8_Ranking.md` |
| `python generators/additional_analyses.py` | `Additional_Analyses.md` |
| `python generators/dml_controls.py` | `Table_S9_Controls.md` (re-estimates with `Pipeline/dml_engine.py`) |
| `python generators/dml_controls_tigramite.py` | `Table_S9_Tigramite_Controls.md`, `tigramite_links_tau{10,24}.csv` (Table S9, panel 9; re-runs Stage 6) |
| `python generators/monte_carlo_coverage.py` | `Table_S9_Tigramite_MonteCarlo.md` (Table S9, panel 9; needs the links above) |
| `python generators/coverage_new_seeds.py` | summary on the console, `tigramite_coverage_new_seeds.csv` (Table S9, panel 9) |
| `python generators/decision_trail.py` | `Decision_Trail_tau10.md` (re-runs Stage 7 with `Pipeline/SelectorFunc.py`) |
| `python generators/metrics_shd_auc.py` | `Metrics_SHD_AUC.md`, `Table_S7_Metrics.md` |
| `python generators/extra_numbers.py` | `Extra_Numbers.md`, `Table_S15_Extra.md` |
| `python generators/table_additive_ablation.py` | `Table_Additive_Ablation.md` |
| `python generators/table_s10_gpdc.py` | `Table_S21_GPDC.md` |
| `python generators/table_s25_pcmci_no_bh.py` | `Table_S22_PCMCI_NoBH.md` |
| `python generators/table_s26_granger_te.py` | `Table_S26_GrangerTE.md` |
| `python generators/sensitivity_analysis.py` | `Sensitivity.md`, `Table_S20_Sensitivity.md` |
| `python generators/make_figure2_lag_penalty.py [10\|24]` | `lag-pen-en_tau{10,24}.svg` (no run needed) |
| `python generators/make_figure3_dempster_shafer.py` | `causal_effect_penalty_dempster_shafer_v2.svg` (no run needed) |
| `python generators/make_figure4_synthetic_v2.py [10\|24]` | `synthetic_performance_tau{10,24}.svg` |
| `python generators/make_figure5_expanded_v2.py [10\|24]` | `benchmark_performance_tau{10,24}.svg` |

The figures also need `matplotlib` (listed in `requirements.txt`). Figure 1
of the paper is a hand-drawn diagram, and no script generates it.

"""
Where the generators read their runs and write their reports.

Two origins feed every number of the paper:

    MAIN_RUN     the run folder written by `python main.py`
                 (Results/Run_<timestamp>/), the development suite;
    HELDOUT_RUN  the run folder written by `python main_heldout.py`
                 (Results_HeldOut/Run_<timestamp>/), the held-out datasets.
                 It may also be a list of folders, searched in order.

A scenario is looked up in HELDOUT_RUN first and then in MAIN_RUN, so a
scenario present in both is read from the held-out run.

The separate experiments of the paper have their own runs, used only by the
script that reports them (see generators/README.md):

    GPDC_RUN          main_gpdc.py                -> table_s10_gpdc.py, paper_numbers.py
    ABLATION_DIR      main_ablation_patch.py      -> table_additive_ablation.py, additional_analyses.py
    SENSITIVITY_RUN   main_sensitivity.py         -> sensitivity_analysis.py
    PCMCI_NOBH_RUN    main_pcmci_nobh.py          -> table_s25_pcmci_no_bh.py
    GRANGER_TE_RUN    main_granger_te.py          -> table_s26_granger_te.py
    METRICS_RUN       a run scored by Auxiliary/ranking_metrics.py
                                                  -> metrics_shd_auc.py, paper_numbers.py
    CANDIDATES_RUN    main_heldout.py with every held-out candidate family
                                                  -> extra_numbers.py
    SEEDS_RUN         main_seed_replication.py    -> coverage_new_seeds.py

Every origin starts as None. Set them, in increasing priority:
  1. in generators/origins_local.py (not versioned), as plain assignments,
     for example  MAIN_RUN = r"D:\\runs\\Results\\Run_20261004_232204";
  2. through environment variables named CAP_<NAME>, for example
     CAP_MAIN_RUN=/data/Run_20261004_232204. A list is separated by
     os.pathsep (';' on Windows, ':' elsewhere).

OUT_DIR receives the reports (.md, .json) and FIG_DIR the figures (.svg).

LEGACY_SCENARIOS, set only in origins_local.py, maps scenario folder names of
runs written before a scenario ID was renamed to its current ID, for example
{"real_old_name": "real_new_name"}; the scripts then read those runs as if
the folders had the current names.
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The reports print τ, θ and other non-ASCII characters; a console in a legacy
# code page (cp1252 on Windows) would otherwise stop the script at the print.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

MAIN_RUN = None
HELDOUT_RUN = None

GPDC_RUN = None
ABLATION_DIR = None
SENSITIVITY_RUN = None
PCMCI_NOBH_RUN = None
GRANGER_TE_RUN = None
METRICS_RUN = None
CANDIDATES_RUN = None
SEEDS_RUN = None

OUT_DIR = os.path.join(REPO, "paper_outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")

_NAMES = ("MAIN_RUN", "HELDOUT_RUN", "GPDC_RUN", "ABLATION_DIR", "SENSITIVITY_RUN",
          "PCMCI_NOBH_RUN", "GRANGER_TE_RUN", "METRICS_RUN", "CANDIDATES_RUN", "SEEDS_RUN",
          "OUT_DIR", "FIG_DIR")

LEGACY_SCENARIOS = {}

try:
    import origins_local as _local
    for _n in _NAMES:
        if hasattr(_local, _n):
            globals()[_n] = getattr(_local, _n)
    LEGACY_SCENARIOS = dict(getattr(_local, "LEGACY_SCENARIOS", {}))
except ImportError:
    pass

for _n in _NAMES:
    _v = os.environ.get("CAP_" + _n)
    if _v:
        globals()[_n] = _v.split(os.pathsep) if _n == "HELDOUT_RUN" and os.pathsep in _v else _v


def heldout_runs():
    """HELDOUT_RUN as a list (empty when unset)."""
    if not HELDOUT_RUN:
        return []
    return [HELDOUT_RUN] if isinstance(HELDOUT_RUN, str) else list(HELDOUT_RUN)


def scenario_runs():
    """Runs searched for a scenario, in order: the held-out runs, then MAIN_RUN."""
    return heldout_runs() + ([MAIN_RUN] if MAIN_RUN else [])


def canonical(text):
    """`text` with every legacy scenario ID replaced by the current one, with or
    without its benchmark_/real_ prefix; used on names read from a run and on
    report text before it is written."""
    bare = lambda s: re.sub(r"^(benchmark_|real_)", "", s)  # noqa: E731
    for old, new in LEGACY_SCENARIOS.items():
        text = text.replace(bare(old), bare(new))
    return text


def find(scen):
    """Folder of a scenario in the first run of scenario_runs() that has it, or None.
    Also finds a folder that still carries a legacy ID of the scenario."""
    names = [scen] + [old for old, new in LEGACY_SCENARIOS.items() if new == scen]
    for run in scenario_runs():
        for name in names:
            path = os.path.join(run, name)
            if os.path.isdir(path):
                return path
    return None


def scenarios():
    """{scenario: folder} over scenario_runs(); a scenario in several runs keeps the first."""
    found = {}
    for run in scenario_runs():
        for scen in sorted(os.listdir(run)):
            path = os.path.join(run, scen)
            if os.path.isdir(path) and scen not in found:
                found[scen] = path
    return found


def require(name):
    """Value of an origin; stops the script with a clear message when it is unset."""
    value = globals()[name]
    if not value:
        sys.exit(f"[generators] {name} is not set. Set it in generators/origins_local.py "
                 f"or in the environment variable CAP_{name} (see generators/origins.py).")
    return value


def out(name):
    """Path of a report in OUT_DIR (created if needed)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def fig(name):
    """Path of a figure in FIG_DIR (created if needed)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)


def label(path):
    """Short name of a run for the reports: its parent folder and its own name."""
    p = os.path.normpath(path)
    return os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)

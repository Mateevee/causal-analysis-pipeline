"""
main_run_one_case.py -- runs ONE isolated benchmark scenario (by default,
'benchmark_tigramite_large_mixed', 3 seeds) with the SAME production config
(Config.execution.PIPELINE_CONFIG, including correct_lag_multiplicity=True)
and the same models as main.py (Config.execution.MODELS: 'patch' and every
baseline, each at tau=24 and tau=10), saving to Results/ with a folder name
that makes it clear it is a diagnostic run for ONE scenario, not the full
suite.

Runs a single scenario without the entire suite (~hours), e.g. to inspect
the per-stage pair summary (Pipeline/pipeline.py::_print_stage_pair_summary)
on one dataset.

Usage: edit SCENARIOS/SEEDS below if you want a different scenario, and run
    python main_run_one_case.py
"""
from __future__ import annotations

import sys
import datetime

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import main as main_mod
from Config import execution as cfg_exec
from Data.tigramite_scm import generate_tigramite_benchmark_scenarios

SCENARIOS = ['large_mixed']
SEEDS = [11, 22, 33]

_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"./Results/Run_{_ts}_single_case_large_mixed"


def main():
    scenarios = generate_tigramite_benchmark_scenarios(seeds=SEEDS, scenarios=SCENARIOS)
    print(f"\n[main_run_one_case] {len(scenarios)} scenario(s) generated: "
          f"{[c['name'] for c in scenarios]}")

    main_mod.run_all(
        dataframes=scenarios,
        models=cfg_exec.MODELS,
        config=cfg_exec.PIPELINE_CONFIG,
        output_dir=OUTPUT_DIR,
    )
    main_mod.validate_all(
        OUTPUT_DIR, show_pairs=True,
        ignore_intra_entity=not cfg_exec.DETECT_INTRA_ENTITY,
        use_canonical_pair=cfg_exec.USE_CANONICAL_PAIR,
    )
    print(f"\n[main_run_one_case] done. Output at: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

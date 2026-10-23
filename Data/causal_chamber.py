"""
Data.causal_chamber
===============================================================================
REAL BENCHMARK — CAUSAL CHAMBERS (Gamella, Peters & Bühlmann 2025, "Causal
chambers as a real-world physical testbed for AI methodology", Nature
Machine Intelligence).

Two computer-controlled physical devices built specifically to produce REAL
data with a KNOWN causal graph:

    wind tunnel  ("wt")  two fans (loads `load_in`/`load_out`), a hatch,
                         barometers (upwind/downwind/ambient/intake),
                         fan speeds and currents, a microphone, and two
                         potentiometer-driven signal circuits. 32 variables.
    light tunnel ("lt")  an RGB light source, two rotating polarizers,
                         three light sensors (infrared + visible each),
                         auxiliary LEDs, current/angle sensors. 38 variables.

GROUND TRUTH: the graph published by the authors for each chamber
configuration -- an edge exists where the PHYSICAL CONSTRUCTION of the
device makes one variable act on another (an actuator on the sensors it can
reach, a sensor parameter on the reading of that sensor), each one justified
in the paper and validated by the authors with randomized interventions.
Among the real-data sources of this project it is the one whose ground truth
is closest to interventional evidence, rather than a topology proxy.

The edges are embedded below (WT_EDGES / LT_EDGES) instead of being read
from the authors' `causalchamber` package at run time: they were copied from
`causalchamber.ground_truth.graph(chamber, "standard")` (package version
0.2.8: 42 edges for "wt", 57 for "lt"), which avoids a hard dependency for a
static table -- and importing that package fails on a Windows console with a
non-UTF-8 code page (its start-up banner cannot be encoded). The
"pressure-control" configuration of the wind tunnel only adds the two
feedback edges in WT_PRESSURE_CONTROL_EXTRA_EDGES.

WHICH DATA IS A TIME SERIES
-------------------------------------------------------------------------------
Only the "walks" datasets are meant for time-series methods: in them the
actuators follow a random walk / smooth trajectory, so consecutive rows are
dependent (wt_walks_v1, lt_walks_v1). The "interventions"/"uniform" datasets
draw every row independently and have no time axis (same caveat as
Data/sachs.py) -- they can be loaded, but lag results on them mean nothing.

CAVEAT ON LAGS: each row holds the actuator values AND the sensor readings
taken right after setting them, so most effects are CONTEMPORANEOUS at the
row resolution (light sensors react instantly); only variables with physical
inertia (fan speed, pressures after a load change) carry over to later rows.

Variables held fixed during an experiment (sensor parameters such as
oversampling rates and reference voltages, unused actuators) are constant in
the file and are removed, with the ground truth projected onto the remaining
ones (see Data.base.drop_constant_columns). Missing readings are stored by
the source as -9999.

HOW TO GET THE DATA
-------------------------------------------------------------------------------
Not auto-downloaded. The default experiments (DEFAULT_EXPERIMENTS) need
two datasets, wt_walks_v1 and lt_walks_v1. Each one is a zip whose content
is a folder named after the dataset; extract both into <base_dir>/ (default:
env CAUSAL_CHAMBER_DIR or "./external_data/causal_chamber/") so that each
experiment is at <base_dir>/<dataset>/<experiment>.csv:

    https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/wt_walks_v1.zip
        (MD5 19bb4e92cbe0b8dff49299b9b509ac36)
    https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/lt_walks_v1.zip
        (MD5 dcad019186661a56de7a2d1db97fc2f0)

Expected layout (only these four files are read by default):

    ./external_data/causal_chamber/wt_walks_v1/actuators_random_walk_2.csv
    ./external_data/causal_chamber/wt_walks_v1/actuators_random_walk_3.csv
    ./external_data/causal_chamber/lt_walks_v1/actuators_white.csv
    ./external_data/causal_chamber/lt_walks_v1/smooth_polarizers.csv

The links and checksums are listed on each dataset's page,
https://github.com/juangamella/causal-chamber/tree/main/datasets/<dataset>.
Alternatively, let the authors' package (`pip install causalchamber`)
download and extract them into the same place:

    import causalchamber.datasets as datasets
    for name in ("wt_walks_v1", "lt_walks_v1"):
        datasets.Dataset(name, root="./external_data/causal_chamber", download=True)

(on a Windows console, set PYTHONIOENCODING=utf-8 first -- see the note on
the package's start-up banner above; the package also leaves the
causal_chamber_<md5>.zip archives in root, which can be deleted).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .base import (
    ScenarioEntry,
    register,
    array_to_pipeline_format_single_property,
    drop_constant_columns,
)

_BASE_DEFAULT = Path(os.environ.get("CAUSAL_CHAMBER_DIR", "./external_data/causal_chamber"))

_DOWNLOAD_URL = "https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables"
_MISSING_VALUE = -9999

# Bookkeeping columns of every dataset (not chamber variables).
_META_COLUMNS = ["timestamp", "config", "counter", "flag", "intervention"]

# (dataset, experiment) pairs loaded by default -- time-series ("walks")
# experiments of both chambers.
DEFAULT_EXPERIMENTS: List[Tuple[str, str]] = [
    ("wt_walks_v1", "actuators_random_walk_2"),
    ("wt_walks_v1", "actuators_random_walk_3"),
    ("lt_walks_v1", "actuators_white"),
    ("lt_walks_v1", "smooth_polarizers"),
]

# ── Wind tunnel, "standard" configuration: 32 variables, 42 edges ───────────
WT_VARIABLES = [
    "hatch", "pot_1", "pot_2", "osr_1", "osr_2", "osr_mic", "osr_in",
    "osr_out", "osr_upwind", "osr_downwind", "osr_ambient", "osr_intake",
    "v_1", "v_2", "v_mic", "v_in", "v_out", "load_in", "load_out",
    "current_in", "current_out", "res_in", "res_out", "rpm_in", "rpm_out",
    "pressure_upwind", "pressure_downwind", "pressure_ambient",
    "pressure_intake", "mic", "signal_1", "signal_2",
]

_WT_AIRFLOW_EFFECTS = ["rpm_in", "rpm_out", "pressure_upwind",
                       "pressure_downwind", "pressure_intake", "mic"]

WT_EDGES = (
    # hatch opening and the two fan loads act on the airflow
    [("hatch", effect) for effect in _WT_AIRFLOW_EFFECTS]
    + [(load, effect) for load in ("load_in", "load_out")
       for effect in ["current_in", "current_out"] + _WT_AIRFLOW_EFFECTS]
    # potentiometer circuits
    + [("pot_1", "mic"), ("pot_1", "signal_1"), ("pot_1", "signal_2"),
       ("pot_2", "signal_2")]
    # sensor parameters (osr_*, v_*, res_*) act on the reading of their own
    # sensor
    + [("osr_1", "signal_1"), ("osr_2", "signal_2"), ("osr_mic", "mic"),
       ("osr_in", "current_in"), ("osr_out", "current_out"),
       ("osr_upwind", "pressure_upwind"), ("osr_downwind", "pressure_downwind"),
       ("osr_ambient", "pressure_ambient"), ("osr_intake", "pressure_intake"),
       ("v_1", "signal_1"), ("v_2", "signal_2"), ("v_mic", "mic"),
       ("v_in", "current_in"), ("v_out", "current_out"),
       ("res_in", "rpm_in"), ("res_out", "rpm_out")]
)

# "pressure-control" configuration: the fan loads are set by a controller
# that reads the downwind barometer (44 edges in total).
WT_PRESSURE_CONTROL_EXTRA_EDGES = [
    ("pressure_downwind", "load_in"), ("pressure_downwind", "load_out"),
]

# ── Light tunnel, "standard" configuration: 38 variables, 57 edges ──────────
LT_VARIABLES = [
    "red", "green", "blue", "osr_c", "v_c", "current", "pol_1", "pol_2",
    "osr_angle_1", "osr_angle_2", "v_angle_1", "v_angle_2", "angle_1",
    "angle_2", "ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3", "l_11",
    "l_12", "l_21", "l_22", "l_31", "l_32", "diode_ir_1", "diode_vis_1",
    "diode_ir_2", "diode_vis_2", "diode_ir_3", "diode_vis_3", "t_ir_1",
    "t_vis_1", "t_ir_2", "t_vis_2", "t_ir_3", "t_vis_3",
]

_LT_LIGHT_SENSORS = ["ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3"]

LT_EDGES = (
    # light source: drawn current and every light sensor
    [(color, effect) for color in ("red", "green", "blue")
     for effect in ["current"] + _LT_LIGHT_SENSORS]
    + [("osr_c", "current"), ("v_c", "current")]
    # polarizers: their angle sensors and the third light sensor (behind both)
    + [(f"pol_{k}", effect) for k in (1, 2)
       for effect in (f"angle_{k}", "ir_3", "vis_3")]
    + [(f"{param}_angle_{k}", f"angle_{k}") for param in ("osr", "v") for k in (1, 2)]
    # auxiliary LEDs l_<k><m> of each light sensor k
    + [(f"l_{k}{m}", f"{band}_{k}") for k in (1, 2, 3) for m in (1, 2)
       for band in ("ir", "vis")]
    # sensor parameters (diode_*, t_*) of each light sensor
    + [(f"{param}_{band}_{k}", f"{band}_{k}") for param in ("diode", "t")
       for band in ("ir", "vis") for k in (1, 2, 3)]
)

_CHAMBERS = {
    "wt": ("wind tunnel", WT_VARIABLES, {
        "standard": WT_EDGES,
        "pressure-control": WT_EDGES + WT_PRESSURE_CONTROL_EXTRA_EDGES,
    }),
    "lt": ("light tunnel", LT_VARIABLES, {"standard": LT_EDGES}),
}


def _read_experiment(path: Path, chamber: str) -> Tuple[pd.DataFrame, str]:
    """Reads one experiment CSV and returns (the chamber variables only, in
    the order of the ground-truth graph, with -9999 as NaN; configuration)."""
    variables = _CHAMBERS[chamber][1]
    df = pd.read_csv(path)
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(
            f"'{path}' lacks variables of the '{chamber}' chamber: {missing}."
        )
    configurations = df["config"].unique()
    if len(configurations) != 1:
        raise ValueError(
            f"'{path}' mixes chamber configurations {list(configurations)} -- "
            f"each one has its own ground-truth graph."
        )
    return df[variables].replace(_MISSING_VALUE, np.nan), str(configurations[0])


@register("causal_chamber")
def generate_causal_chamber_benchmark_scenarios(
    *,
    experiments: Optional[Sequence[Tuple[str, str]]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    n_points: int = 500,
    offset: int = 0,
    step: int = 1,
    name_prefix: str = "real_causal_chamber",
) -> List[dict]:
    """
    Generates scenarios from the Causal Chambers (real physical testbeds,
    ground truth known from the construction of the device), in the format
    consumed by run_all()/validate_dataset() (same signature described in
    Data.base). See the module docstring for the source, the ground truth
    and the caveats about the time axis.

    Parameters
    ----------
    experiments : (dataset, experiment) pairs, one scenario each -- the file
                  read is <base_dir>/<dataset>/<experiment>.csv, and the
                  chamber ("wt"/"lt") is the dataset's prefix (default:
                  DEFAULT_EXPERIMENTS, the time-series "walks" experiments).
    base_dir    : root with one subfolder per dataset (default: env
                  CAUSAL_CHAMBER_DIR or "./external_data/causal_chamber").
    n_points    : rows per scenario.
    offset      : index of the first row used.
    step        : keeps one row every `step` (1 = every row; the walks
                  datasets are sampled a few times per second).
    name_prefix : prefix for the name/folder of each generated scenario.

    Returns a list of dicts (ScenarioEntry.to_dict()), one per experiment.
    """
    root = Path(base_dir) if base_dir is not None else _BASE_DEFAULT
    experiments = list(experiments) if experiments is not None else DEFAULT_EXPERIMENTS

    scenarios: List[dict] = []
    for dataset, experiment in experiments:
        chamber = dataset.split("_")[0]
        if chamber not in _CHAMBERS:
            raise ValueError(
                f"Dataset '{dataset}' does not start with a known chamber "
                f"prefix {sorted(_CHAMBERS)}."
            )
        chamber_label, variables, graphs = _CHAMBERS[chamber]

        path = root / dataset / f"{experiment}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Causal Chambers file not found: '{path}'. Download "
                f"{_DOWNLOAD_URL}/{dataset}.zip and extract it into '{root}' "
                f"(the archive already holds the '{dataset}/' folder), or "
                f"point base_dir= / the CAUSAL_CHAMBER_DIR environment "
                f"variable -- see this module's docstring."
            )

        df_raw, configuration = _read_experiment(path, chamber)
        if configuration not in graphs:
            raise ValueError(
                f"'{path}' was recorded in configuration '{configuration}', "
                f"for which no ground-truth graph is embedded in this module "
                f"(available for '{chamber}': {sorted(graphs)})."
            )

        window = df_raw.iloc[offset:offset + n_points * step:step]
        if len(window) < n_points:
            raise ValueError(
                f"offset={offset} + n_points={n_points} x step={step} exceeds "
                f"the {len(df_raw)} rows of '{path}'."
            )
        window = window.interpolate(method="linear", limit_direction="both")

        index = {v: k for k, v in enumerate(variables)}
        full_pairs = {(index[a], index[b]) for a, b in graphs[configuration]}
        data, names, pairs, removed = drop_constant_columns(
            window.to_numpy(dtype=float), variables, full_pairs,
        )
        name = f"{name_prefix}_{chamber}_{experiment}"

        # The source's timestamps are seconds since the chamber was switched
        # on, irregularly spaced -- 'Date' is a synthetic regular grid.
        df_long, df_gt = array_to_pipeline_format_single_property(
            data, names, pairs,
            start_date="2015-01-01", freq="s", property_name="VAL",
        )

        params = {
            "source": "causal_chamber",
            "reference": "Gamella, Peters & Bühlmann 2025, Nature Machine "
                         "Intelligence (Causal Chambers)",
            "chamber": chamber_label,
            "configuration": configuration,
            "dataset": dataset,
            "experiment": experiment,
            "data_file": str(path),
            "n_points": n_points,
            "offset": offset,
            "step": step,
            "n_entities": len(names),
            "n_causal_links": len(pairs),
            "n_causal_links_full_graph": len(full_pairs),
            "variables_removed_zero_variance": removed,
            "has_time_axis": "walks" in dataset,
            "ground_truth_type": "physical construction of the device, "
                                  "published by the authors and validated "
                                  "with randomized interventions",
        }

        if not params["has_time_axis"]:
            print(f"  [WARNING] {name}: '{dataset}' is not a 'walks' dataset "
                  f"-- its rows are independent draws (NO real time axis); "
                  f"lag results on it are not meaningful.")

        print(f"  [real-benchmark] {name} loaded — {chamber_label}, "
              f"{len(names)} variables x {n_points} points | {len(pairs)} "
              f"causal links (ground truth by construction, Causal Chambers) "
              f"| {len(removed)} constant variable(s) removed")

        scenarios.append(ScenarioEntry(
            name=name, df=df_long, gt=df_gt,
            _scenario=f"{chamber}_{configuration}", _params=params,
        ).to_dict())

    return scenarios


if __name__ == "__main__":
    # Quick manual test (needs the files in CAUSAL_CHAMBER_DIR / the default path).
    for c in generate_causal_chamber_benchmark_scenarios(n_points=200):
        n_edges = int(c["gt"]["causal_gt"].sum())
        print(f"  {c['name']}: df={c['df'].shape}, "
              f"{n_edges}/{len(c['gt'])} real edges")

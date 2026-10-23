"""
Data.base
===============================================================================
Common contract for benchmark/validation scenario "providers" of the causal
pipeline (Lorenz-96, tigramite SCM, logistic maps, real climate, ...)
and shared conversion utilities.

COMMON SIGNATURE (to add a new case)
-------------------------------------------------------------------------------
A provider is any callable that returns a list of entries in the format
below. There is no mandatory base class to inherit from; it is a structural
contract (duck typing), the same one run_all()/validate_dataset() already
consume:

    def generate_<source>_scenarios(
        *,
        seeds: Optional[List[int]] = None,
        n_points: int = 300,
        name_prefix: str = "benchmark_<source>",
        **kwargs,
    ) -> List[dict]:
        ...

    # single-source loaders (not parameterized by seed) may return a single
    # dict instead of a 1-element list -- main.py accepts both formats
    # (see DATAFRAMES.append(...) vs DATAFRAMES += ...)

Each returned entry (a "ScenarioEntry") is a dict with the keys:

    name      : str            unique identifier (becomes a folder name on disk)
    df        : pd.DataFrame   columns (Date, Name, <property_1>, <property_2>, ...)
    gt        : pd.DataFrame   columns (source_entity, source_property, target_entity,
                                target_property, causal_gt)
    _scenario : Optional[str]  internal scenario/topology name (log only)
    _params   : dict           free-form metadata (seed, source, etc.), saved to
                                disk by _save_scenario_params()

Use the `ScenarioEntry` dataclass below to build these entries with type
safety (`.to_dict()` returns exactly the dict above), or build the dict
directly -- both work, since the runner only accesses it via `.get(...)`.

REGISTRATION (optional)
-------------------------------------------------------------------------------
Providers can register themselves with the `@register("key")` decorator to
show up in `list_providers()` / `get_provider()`. This is purely a discovery
convenience (e.g. for a CLI or a generic loop in main.py) -- it is NOT
required for the pipeline to work: run_all() only needs the final list of
dicts, so a provider can also be called directly, without registering.

HOW TO ADD A NEW SCENARIO
-------------------------------------------------------------------------------
1. Create a new file in Data/, e.g.: Data/my_benchmark.py
2. Implement a function with the signature above, returning
   `ScenarioEntry(...).to_dict()` (or equivalent dicts) for each generated
   case/seed.
3. If the source produces data as a (T, N) array with a single "property"
   per entity (common case: chaotic systems, univariate real series per
   station/node), reuse `array_to_pipeline_format_single_property()` below
   instead of duplicating the conversion to (df_long, df_gt).
4. Decorate the function with `@register("my_benchmark")` (optional).
5. Export the function in Data/__init__.py and import it in main.py.
No change is needed in run_all()/validate_dataset().
"""
from __future__ import annotations

import datetime
import itertools
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# Log of scenario loading failures (source unavailable in this environment:
# missing dependency, file/folder not found, no internet, sensitive data
# that only exists on another machine, etc.). Path relative to the Data/
# package (i.e. <main.py directory>/logs/...), in the same spirit as
# logs/failures/ used by the runner (pipeline_logging.py).
SCENARIO_FAILURES_LOG = Path(__file__).resolve().parent.parent / "logs" / "scenarios" / "loading_failures.log"


# =============================================================================
# CONTRACT
# =============================================================================

@dataclass
class ScenarioEntry:
    """A scenario entry, in the format run_all()/validate_dataset() expect.
    See the module docstring for the meaning of each field."""
    name: str
    df: pd.DataFrame
    gt: Optional[pd.DataFrame] = None
    _scenario: Optional[str] = None
    _params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name":     self.name,
            "df":       self.df,
            "gt":       self.gt,
            "_scenario": self._scenario,
            "_params":  self._params,
        }


# Signature every scenario provider must follow: any number of source-specific
# kwargs, returning a list of ScenarioEntry/dicts.
ScenarioProvider = Callable[..., List[dict]]

_REGISTRY: Dict[str, ScenarioProvider] = {}


def register(source_name: str):
    """Decorator: registers a scenario provider under the `source_name` key.

    Usage:
        @register("lorenz96")
        def generate_lorenz96_benchmark_scenarios(...) -> List[dict]:
            ...
    """
    def _decorator(func: ScenarioProvider) -> ScenarioProvider:
        if source_name in _REGISTRY:
            raise ValueError(
                f"Scenario provider '{source_name}' already registered "
                f"(name collision in Data.base.register)."
            )
        _REGISTRY[source_name] = func
        return func
    return _decorator


def list_providers() -> List[str]:
    """Keys of all providers registered via @register(...)."""
    return sorted(_REGISTRY.keys())


def get_provider(source_name: str) -> ScenarioProvider:
    try:
        return _REGISTRY[source_name]
    except KeyError:
        raise KeyError(
            f"Scenario provider '{source_name}' not registered. "
            f"Available: {list_providers()}"
        )


# =============================================================================
# LOADING WITH FALLBACK — "try it, and if it doesn't work, move on and log it"
# =============================================================================
# Some providers only work in certain environments: 'tigramite' needs the
# `tigramite` package installed; 'real_climate' needs internet on first
# run; 'sachs'/'tep' need a manually downloaded CSV.
# load_with_fallback() encapsulates the pattern "try to load; if it fails for
# ANY reason, skip this source, log the reason (console + log file) and let
# the remaining sources continue normally" -- instead of scattering an ad-hoc
# try/except per source across the entry point.

def log_scenario_failure(source_name: str, exc: BaseException,
                          log_path: Optional[Union[str, Path]] = None) -> None:
    """Logs a scenario loading failure to both the console AND a log file on
    disk (default: SCENARIO_FAILURES_LOG). Does not raise an exception -- it
    is safe to call from within an except block already in progress."""
    path = Path(log_path) if log_path is not None else SCENARIO_FAILURES_LOG
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"  [WARNING] scenario source '{source_name}' could not be loaded "
          f"in this environment: {exc}")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("=" * 79 + "\n")
            f.write(f"[{timestamp}] source: {source_name}\n")
            f.write(f"error: {exc!r}\n")
            f.write(traceback.format_exc())
            f.write("\n")
    except Exception as log_err:
        # A failure while writing the log must never take down the loading
        # of the remaining scenarios -- just warn on the console and move on.
        print(f"  [WARNING] could not write the failure log to "
              f"'{path}': {log_err}")


def load_with_fallback(
    source_name: str,
    provider: ScenarioProvider,
    *args,
    log_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> List[dict]:
    """
    Calls `provider(*args, **kwargs)` guarded by try/except: if the call
    fails for any reason (missing dependency, file/folder not found, no
    internet, sensitive data unavailable in this environment, etc.), the
    source is skipped -- without interrupting the loading of the others --
    and the reason is logged via log_scenario_failure() (console + log).

    Accepts a provider that returns List[dict], a single dict, or None --
    always normalizes to List[dict] (empty list on failure or empty return).

    Typical usage at the entry point (main.py), instead of a manual
    try/except per source:

        DATAFRAMES += load_with_fallback(
            "tigramite", generate_tigramite_benchmark_scenarios,
            seeds=[11, 22, 33], n_points=300,
        )
    """
    try:
        result = provider(*args, **kwargs)
    except Exception as e:
        log_scenario_failure(source_name, e, log_path=log_path)
        return []

    if result is None:
        return []
    if isinstance(result, dict):
        return [result]
    return list(result)


# =============================================================================
# SHARED CONVERSION HELPER
# =============================================================================
# Used by any source whose generator produces a raw (T, N) array where EACH
# COLUMN is its own entity with a single "property" -- the common case for
# classic benchmarks (Lorenz-96, logistic maps) and univariate real series
# per station/node (climate). Sources with multiple properties per
# entity (e.g. tigramite SCM, which groups columns into blocks of N
# properties per entity) have their own converter -- see
# Data/tigramite_scm.py: _scm_to_pipeline_format().

def _array_to_pipeline_format_single_property(
    data: np.ndarray,
    entity_names: List[str],
    causal_pair_idx: set,
    start_date: str = "2015-01-01",
    freq: str = "MS",
    property_name: str = "VAL",
) -> tuple:
    """
    Generic converter for benchmarks where each COLUMN of the array is its
    own entity with a single "property" (unlike the tigramite SCM benchmark
    above, which groups columns into blocks of N properties per entity).
    Used by the real climate, Lorenz-96 and logistic maps loaders.

    data              : array (T, N)
    entity_names      : list of N names (becomes the 'Name' column)
    causal_pair_idx   : set of (i, j) -- causal column index i -> j
    """
    T, N = data.shape
    if len(entity_names) != N:
        raise ValueError(f"entity_names has {len(entity_names)} items, "
                          f"but data has {N} columns.")

    dates = pd.date_range(start_date, periods=T, freq=freq)
    records = []
    for t in range(T):
        for i, name in enumerate(entity_names):
            records.append({"Date": dates[t], "Name": name,
                             property_name: float(data[t, i])})
    df_long = pd.DataFrame(records)

    gt_records = []
    for i, j in itertools.permutations(range(N), 2):
        gt_records.append({
            "source_entity":    entity_names[i],
            "source_property":  property_name,
            "target_entity":    entity_names[j],
            "target_property":  property_name,
            "causal_gt":        (i, j) in causal_pair_idx,
        })
    df_gt = pd.DataFrame(gt_records)
    return df_long, df_gt


# Public name (no underscore) -- use this when importing from another
# module; the underscore-prefixed name remains for internal use within
# this module.
array_to_pipeline_format_single_property = _array_to_pipeline_format_single_property


def drop_constant_columns(
    data: np.ndarray,
    entity_names: List[str],
    causal_pair_idx: set,
) -> tuple:
    """
    Removes the columns of `data` with zero variance (or entirely NaN) and
    projects the ground truth onto the remaining ones -- the pipeline
    contract requires that `gt` only reference entities that exist in `df`,
    and a constant series carries no information for any method. Common in
    real plant/testbed data cut into a window (standby pumps, valves that
    never move, actuators held fixed during an experiment); same treatment
    CIPCaD-Bench already applied to the TEP (see Data/tep.py).

    Returns (data, entity_names, causal_pair_idx, removed_names), with
    `causal_pair_idx` re-indexed to the columns that were kept.
    """
    def _varies(column: np.ndarray) -> bool:
        valid = column[np.isfinite(column)]
        return valid.size > 1 and valid.max() > valid.min()

    keep = [i for i in range(data.shape[1]) if _varies(data[:, i])]
    new_index = {old: new for new, old in enumerate(keep)}
    removed = [entity_names[i] for i in range(data.shape[1]) if i not in new_index]
    pairs = {(new_index[i], new_index[j]) for (i, j) in causal_pair_idx
             if i in new_index and j in new_index}
    return data[:, keep], [entity_names[i] for i in keep], pairs, removed

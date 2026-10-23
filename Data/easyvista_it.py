"""
Data.easyvista_it
===============================================================================
REAL BENCHMARK — IT MONITORING (EasyVista), from Aït-Bachir, Assaad, de
Bignicourt, Devijver, Ferreira, Gaussier, Mohanna & Zan 2023, "Case Studies
of Causal Discovery from IT Monitoring Time Series" (UAI 2023 Workshop on
The History and Development of Search Methods for Causal Structure,
arXiv:2307.15678).

Metrics collected by the monitoring system of real IT infrastructures, in
four case studies (seven datasets):

    antivirus_1, antivirus_2   13 series x ~1320 samples (5 min) — impact of
                               antivirus activity on two servers (Portal,
                               VDI) and on user-facing transactions
    mom_1, mom_2               7 series x 288 / 364 samples — a
                               message-oriented middleware (publish rate,
                               consumers, queued messages) and the resources
                               of its broker
    storm                      8 series x 991 samples — a Storm ingestion
                               topology (one series per bolt)
    web_1, web_2               10 series x ~7500 samples (5 min) — a web
                               server (network, http/php processes, mysql
                               connections, cpu, ram, disk)

GROUND TRUTH: the summary causal graph of each system described by
EasyVista's system experts (`structure.txt`, one "cause effect" pair per
line). Expert knowledge of the architecture — not estimated, and not
validated by intervention.

PREPROCESSING: the files are used as published (the Antivirus and Web
files are the authors' "preprocessed" versions). The first column
(timestamp/index) is dropped and the rows are taken as equally spaced.
Constant series are removed. Series longer than `max_points` are cut to
their first `max_points` samples.

HOW THE DATA IS OBTAINED
-------------------------------------------------------------------------------
Automatically downloads (with on-disk caching) from the authors'
repository (no license file is published with it):
    https://github.com/ckassaad/Case_Studies_of_Causal_Discovery_from_IT_Monitoring_Time_Series
        data/<case study>/...
"""
from __future__ import annotations

import os
import urllib.request as _urlreq_bench
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .base import array_to_pipeline_format_single_property, register

_BASE_URL = ("https://raw.githubusercontent.com/ckassaad/"
             "Case_Studies_of_Causal_Discovery_from_IT_Monitoring_Time_Series/"
             "main/data")
_BASE_DEFAULT = Path(os.environ.get("EASYVISTA_IT_DIR", "./external_data/easyvista_it"))

# dataset -> (folder, data file, structure file, sampling)
_DATASETS = {
    "antivirus_1": ("Antivirus_Activity", "preprocessed_1.csv", "structure.txt", "5min"),
    "antivirus_2": ("Antivirus_Activity", "preprocessed_2.csv", "structure.txt", "5min"),
    "mom_1": ("Middleware_oriented_message_Activity", "monitoring_metrics_1.csv",
              "structure.txt", "1min"),
    "mom_2": ("Middleware_oriented_message_Activity", "monitoring_metrics_2.csv",
              "structure.txt", "1min"),
    "storm": ("Storm_Ingestion_Activity", "storm_data_normal.csv",
              "storm_structure.txt", "1min"),
    "web_1": ("Web_Activity", "preprocessed_1.csv", "structure.txt", "5min"),
    "web_2": ("Web_Activity", "preprocessed_2.csv", "structure.txt", "5min"),
}


def _cached(cache_dir: Path, folder: str, file: str, timeout: int = 120) -> Path:
    path = cache_dir / folder / file
    if path.exists():
        return path
    url = f"{_BASE_URL}/{folder}/{file}"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _urlreq_bench.urlopen(url, timeout=timeout) as resp:
            path.write_bytes(resp.read())
    except Exception as e:
        raise RuntimeError(
            f"Failed to download '{url}'. Check your internet connection, "
            f"or download it manually to '{path}'. Original error: {e}"
        ) from e
    return path


@register("easyvista_it")
def generate_easyvista_it_benchmark_scenarios(
    datasets: Sequence[str] = tuple(_DATASETS),
    max_points: Optional[int] = 1500,
    cache_dir: Optional[str] = None,
    name_prefix: str = "real_easyvista",
) -> List[dict]:
    """
    Loads the EasyVista IT monitoring case studies — see the module
    docstring for the source, the ground truth (expert summary graphs) and
    the preprocessing.

    Parameters
    ----------
    datasets    : which of the seven datasets to load, one scenario each.
    max_points  : series longer than this are cut to their first
                  `max_points` samples (None = no cut).
    cache_dir   : download/cache folder (default: env EASYVISTA_IT_DIR or
                  "./external_data/easyvista_it").
    name_prefix : prefix for the name/folder of each generated scenario.
    """
    cache = Path(cache_dir) if cache_dir else _BASE_DEFAULT

    scenarios: List[dict] = []
    for dataset in datasets:
        if dataset not in _DATASETS:
            raise ValueError(
                f"dataset='{dataset}' is invalid. Use one of {sorted(_DATASETS)}.")
        folder, data_file, structure_file, sampling = _DATASETS[dataset]
        data_path = _cached(cache, folder, data_file)
        structure_path = _cached(cache, folder, structure_file)

        df_raw = pd.read_csv(data_path, index_col=0)
        df_raw.columns = df_raw.columns.str.replace(" ", "_")
        n_total = len(df_raw)
        if max_points is not None:
            df_raw = df_raw.iloc[:max_points]
        if df_raw.isna().to_numpy().any():
            raise ValueError(f"'{data_path}' has missing values.")
        entity_names = list(df_raw.columns)

        edges = np.loadtxt(structure_path, delimiter=" ", dtype=str)
        unknown = sorted(set(edges.ravel()) - set(entity_names))
        if unknown:
            raise ValueError(
                f"'{structure_path}' references series absent from "
                f"'{data_path}': {unknown}.")
        full_pairs = {(entity_names.index(c), entity_names.index(e))
                      for c, e in edges}

        # A constant series carries no information for any method: it is
        # removed and the ground truth projected onto the remaining ones.
        keep = [i for i, c in enumerate(entity_names) if df_raw[c].nunique() > 1]
        new_index = {old: new for new, old in enumerate(keep)}
        removed = [c for i, c in enumerate(entity_names) if i not in new_index]
        names = [entity_names[i] for i in keep]
        pairs = {(new_index[i], new_index[j]) for i, j in full_pairs
                 if i in new_index and j in new_index}
        data = df_raw[names].to_numpy(dtype=float)

        df_long, df_gt = array_to_pipeline_format_single_property(
            data, names, pairs,
            start_date="2021-01-01", freq=sampling, property_name="VAL",
        )

        name = f"{name_prefix}_{dataset}"
        params = {
            "source": "easyvista_it",
            "reference": "Aït-Bachir, Assaad et al. 2023, UAI Workshop "
                         "(arXiv:2307.15678)",
            "dataset": dataset,
            "data_file": str(data_path),
            "structure_file": str(structure_path),
            "n_points": len(data),
            "n_points_in_file": n_total,
            "n_entities": len(names),
            "n_causal_links": len(pairs),
            "variables_removed_zero_variance": removed,
            "ground_truth_type": "summary causal graph described by the "
                                  "system experts (expert knowledge, not "
                                  "validated by intervention)",
        }

        print(f"  [real-benchmark] {name} loaded — {len(names)} series x "
              f"{len(data)} points"
              + (f" (first {len(data)} of {n_total})" if len(data) < n_total else "")
              + f" | GT: {len(pairs)} links (expert summary graph, "
              f"Aït-Bachir et al. 2023)"
              + (f" | removed (constant): {removed}" if removed else ""))

        scenarios.append({"name": name, "df": df_long, "gt": df_gt,
                          "_scenario": None, "_params": params})

    return scenarios


if __name__ == "__main__":
    for c in generate_easyvista_it_benchmark_scenarios():
        n_edges = int(c["gt"]["causal_gt"].sum())
        print(f"  {c['name']}: df={c['df'].shape}, {n_edges}/{len(c['gt'])} real edges")

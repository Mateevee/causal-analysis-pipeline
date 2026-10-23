"""
Data.petshop
===============================================================================
REAL BENCHMARK — PETSHOP (Hardt, Orchard, Blöbaum, Kirschbaum &
Kasiviswanathan 2024, "The PetShop Dataset — Finding Causes of Performance
Issues across Microservices", CLeaR 2024, PMLR 236:957-978).

Metrics of a microservice application (a pet adoption site) RUNNING on AWS:
about 40 components (containers, lambdas, databases, queues, load
balancers), with latency, request and availability metrics aggregated over
5-minute windows. The dataset was built for root-cause analysis: besides the
normal-operation period it holds 68 injected performance issues. This
loader uses ONLY the normal-operation period ("noissue") of each traffic
regime, as an observational time series.

TRAFFIC REGIMES (one scenario each)
-------------------------------------------------------------------------------
    low_traffic        589 samples (2023-08-21..23)
    high_traffic       589 samples (2023-09-05..07)
    temporal_traffic1  1652 samples (2023-04-22..28), time-varying load
(temporal_traffic2 shares the normal-operation period of temporal_traffic1
and is not loaded.) The three are load profiles of the SAME application, not
three independent systems.

VARIABLE: one series per component, its AVERAGE LATENCY. The causal graph
used by the authors is a statement about latency: a slow callee makes its
caller slow.

GROUND TRUTH: `graph.csv` is the CALL graph (row -> column means "row calls
column"), obtained from the application's tracing; the causal graph is its
REVERSE (callee -> caller). It is the service dependency topology, by
construction. The authors themselves note that the service map "does not
describe the causal graph that connects the metrics" (it "can be useful in
constructing" one), and the injected issues validate root causes, not
edges: report it as a topology proxy, not as a causal graph validated by
intervention. The original paper evaluates root-cause analysis only; using
the dataset for graph discovery follows CausalArena (arXiv:2609.11897).

SAMPLING. The metrics are 5-minute aggregates, while a slow callee delays
its caller within seconds: the relations are essentially CONTEMPORANEOUS at
this resolution (temporal aggregation, Breitung & Swanson 2002).

PREPROCESSING
-------------------------------------------------------------------------------
Components that receive no traffic in a period have no latency sample
there. A component is kept if at most `max_nan_frac` of its samples are
missing in the window and its latency varies; the remaining gaps are
linearly interpolated. The ground truth is restricted to the kept
components. Series longer than `max_points` are cut to their first
`max_points` samples. Component names are sanitized to [0-9A-Za-z_] (the
originals are kept in the scenario parameters).

HOW THE DATA IS OBTAINED
-------------------------------------------------------------------------------
Automatically downloads (with on-disk caching) from the official
repository (dataset under CC-BY-4.0):
    https://github.com/amazon-science/petshop-root-cause-analysis
        dataset/<regime>/graph.csv
        dataset/<regime>/noissue/metrics.csv
"""
from __future__ import annotations

import os
import re
import urllib.request as _urlreq_bench
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .base import array_to_pipeline_format_single_property, register

_BASE_URL = ("https://raw.githubusercontent.com/amazon-science/"
             "petshop-root-cause-analysis/main/dataset")
_BASE_DEFAULT = Path(os.environ.get("PETSHOP_DIR", "./external_data/petshop"))
_REGIMES = ("low_traffic", "high_traffic", "temporal_traffic1")
_METRIC, _STATISTIC = "latency", "Average"


def _download(cache_dir: Path, regime: str, timeout: int = 300) -> tuple:
    graph_path = cache_dir / regime / "graph.csv"
    metrics_path = cache_dir / regime / "noissue" / "metrics.csv"
    for path, url in [
        (graph_path, f"{_BASE_URL}/{regime}/graph.csv"),
        (metrics_path, f"{_BASE_URL}/{regime}/noissue/metrics.csv"),
    ]:
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with _urlreq_bench.urlopen(url, timeout=timeout) as resp:
                path.write_bytes(resp.read())
        except Exception as e:
            raise RuntimeError(
                f"Failed to download '{url}'. Check your internet "
                f"connection, or download it manually to '{path}'. "
                f"Original error: {e}"
            ) from e
    return graph_path, metrics_path


def _sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


@register("petshop")
def generate_petshop_benchmark_scenarios(
    regimes: Sequence[str] = _REGIMES,
    max_nan_frac: float = 0.05,
    max_points: Optional[int] = 1500,
    cache_dir: Optional[str] = None,
    name_prefix: str = "real_petshop",
) -> List[dict]:
    """
    Loads the normal-operation period of each PetShop traffic regime — see
    the module docstring for the source, the variable, the ground truth
    (reversed call graph) and the preprocessing.

    Parameters
    ----------
    regimes      : traffic regimes to load, one scenario each.
    max_nan_frac : maximum fraction of missing latency samples for a
                   component to be kept.
    max_points   : series longer than this are cut to their first
                   `max_points` samples (None = no cut).
    cache_dir    : download/cache folder (default: env PETSHOP_DIR or
                   "./external_data/petshop").
    name_prefix  : prefix for the name/folder of each generated scenario.
    """
    cache = Path(cache_dir) if cache_dir else _BASE_DEFAULT

    scenarios: List[dict] = []
    for regime in regimes:
        graph_path, metrics_path = _download(cache, regime)

        call_graph = pd.read_csv(graph_path, index_col=0)
        metrics = pd.read_csv(metrics_path, header=[0, 1, 2], index_col=0)
        latency = metrics.loc[:, (slice(None), [_METRIC], [_STATISTIC])]
        latency.columns = [c[0] for c in latency.columns]
        latency.index = pd.to_datetime(latency.index, unit="s")
        if max_points is not None:
            latency = latency.iloc[:max_points]

        nan_frac = latency.isna().mean()
        kept = [c for c in latency.columns
                if c in call_graph.index and nan_frac[c] <= max_nan_frac
                and latency[c].nunique() > 1]
        dropped = [c for c in call_graph.index if c not in kept]

        window = latency[kept]
        missing_frac = float(window.isna().to_numpy().mean())
        window = window.interpolate(method="linear", limit_direction="both")
        data = window.to_numpy(dtype=float)

        names = [_sanitize(c) for c in kept]
        if len(set(names)) != len(names):
            raise ValueError(f"{regime}: sanitized component names collide.")

        # graph.csv: row calls column. Causal direction: callee -> caller.
        causal_pairs_idx = {
            (j, i)
            for i, caller in enumerate(kept) for j, callee in enumerate(kept)
            if i != j and call_graph.loc[caller, callee] != 0
        }

        df_long, df_gt = array_to_pipeline_format_single_property(
            data, names, causal_pairs_idx,
            start_date=str(window.index[0]), freq="5min",
            property_name="LATENCY",
        )

        name = f"{name_prefix}_{regime}"
        params = {
            "source": "petshop",
            "reference": "Hardt, Orchard, Blöbaum, Kirschbaum & "
                         "Kasiviswanathan 2024, CLeaR (PMLR 236:957-978)",
            "regime": regime,
            "graph_file": str(graph_path),
            "metrics_file": str(metrics_path),
            "variable": f"{_METRIC} / {_STATISTIC}",
            "n_points": len(window),
            "n_entities": len(kept),
            "start_timestamp": str(window.index[0]),
            "end_timestamp": str(window.index[-1]),
            "max_nan_frac": max_nan_frac,
            "missing_fraction_interpolated": round(missing_frac, 4),
            "components": dict(zip(names, kept)),
            "components_dropped": dropped,
            "n_causal_links": len(causal_pairs_idx),
            "n_call_edges_full_graph": int((call_graph.to_numpy() != 0).sum()),
            "ground_truth_type": "reversed call graph (callee -> caller), "
                                  "service dependency topology from tracing "
                                  "-- a proxy, not validated edge by edge "
                                  "by intervention (see the module "
                                  "docstring)",
        }

        print(f"  [real-benchmark] {name} loaded — {len(kept)} components x "
              f"{len(window)} points (5min, average latency) | GT: "
              f"{len(causal_pairs_idx)} links (reversed call graph, Hardt et "
              f"al. 2024) | window {window.index[0]}..{window.index[-1]} | "
              f"{len(dropped)} components dropped (no traffic/constant), "
              f"{missing_frac:.1%} missing readings interpolated")

        scenarios.append({"name": name, "df": df_long, "gt": df_gt,
                          "_scenario": None, "_params": params})

    return scenarios


if __name__ == "__main__":
    for c in generate_petshop_benchmark_scenarios():
        n_edges = int(c["gt"]["causal_gt"].sum())
        print(f"  {c['name']}: df={c['df'].shape}, {n_edges}/{len(c['gt'])} real edges")

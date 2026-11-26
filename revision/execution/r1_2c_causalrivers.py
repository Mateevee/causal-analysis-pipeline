"""
R1-2c, annex B -- CausalRivers scenarios, East Germany region (the authors' TEST split).

Subgraphs: faithful reimplementation of the benchmark's `root_cause_5` and `close_5` strategies
(Stein et al., ICLR 2025; repository CausalRivers/causalrivers, commit
a30154fbb9a2e03bae133c643d7e978df63b3766, `tools/graph_sampling_tools.py` and
`0_generate_datasets.py`; `config/data_sampling.yaml`: dist_measure = geo_distance,
max_distance = 0.3). The authors' code declares no license; it is neither imported nor
redistributed here -- it was read and reimplemented, and the equivalence of the lists of
subgraphs was checked locally (see NOTAS.md, annex B).

  - get_all_subgraphs(G, n): every connected set of n stations (extension by
    predecessors and successors, from each station), without repetition;
  - root_cause_n: the sets whose longest directed path in the induced subgraph has n - 1
    edges (an upstream chain);
  - close_n: the sets whose longest path (sum of geo_distance over every simple path)
    is < 0.3; a missing distance on some path -> infinity (excluded).

Selection rule (annex B, fixed before running any method):
  1. canonical order: each set as a sorted tuple of IDs; the list sorted
     lexicographically (the order of the authors' tool is the iteration order of a set, which is
     not specified; the tool has no seed -- the authors' TODO says so);
  2. eligibility, that of the reader `Data/causal_rivers.py` in the fixed window: every station
     with <= 5% missing readings and not constant;
  3. root_cause_5: draw of 30 among the eligible, without replacement, with
     numpy.random.default_rng(SEED); if there are fewer than 30, all enter;
  4. close_5: the same draw, with the same seed + 1, among the eligible NOT drawn in
     root_cause_5 (no duplicated scenarios between the strategies).
Window and resolution: those of the runs of CausalRivers Bavaria (the authors' training split) in the
supplement: start 2021-01-01 00:00, 500 points at 1 h (mean of the 15-min readings), linear interpolation of the
remaining gaps. Ground truth: the induced subgraph, as the authors' `save_subgraphs_to_pickle` and the
paper's reader.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

STRATEGIES = ("root_cause_5", "close_5")
N_PER_STRATEGY = 30
SEED = 20261115
START = "2021-01-01"
N_POINTS = 500
RESAMPLE = "1h"
MAX_NAN = 0.05
MAX_DIST = 0.3
REGION = "east_germany"
BASE = Path("./external_data/causal_rivers")


# ----------------------------------------------------------------------------- the authors' strategies
def _extensions(current: list, G) -> list:
    ext = []
    for node in current:
        ext += [x for x in G.predecessors(node) if x not in current]
        ext += [x for x in G.successors(node) if x not in current]
    return [current + [e] for e in set(ext)]


def all_subgraphs(G, n: int) -> List[Tuple[int, ...]]:
    all_stacks = []
    for start in list(G.nodes):
        dead_end = False
        stack = [[start]]
        for _ in range(n - 1):
            res = [it for sub in (_extensions(g, G) for g in stack) for it in sub]
            if res:
                stack = res
            else:
                dead_end = True
        if not dead_end:
            all_stacks.append(stack)
    return sorted(set(tuple(sorted(i)) for sub in all_stacks for i in sub))


def longest_geo_path(sub, measure: str = "geo_distance") -> float:
    import networkx as nx
    lengths = []
    for a in sub.nodes:
        for b in sub.nodes:
            for path in nx.all_simple_edge_paths(sub, a, b):
                d = [sub.edges[u, v][measure] for u, v in path]
                if None in d:
                    return np.inf
                lengths.append(sum(d))
    return max(lengths)


def candidates(G, strategy: str) -> List[Tuple[int, ...]]:
    import networkx as nx
    n = int(strategy.rsplit("_", 1)[1])
    all_sets = all_subgraphs(G, n)
    if strategy.startswith("root_cause"):
        return [x for x in all_sets if nx.dag_longest_path_length(G.subgraph(x)) == n - 1]
    if strategy.startswith("close"):
        return [x for x in all_sets if longest_geo_path(nx.subgraph(G, x)) < MAX_DIST]
    raise ValueError(strategy)


# ----------------------------------------------------------------------------- data
def graph():
    with open(BASE / f"rivers_{REGION}.p", "rb") as f:
        return pickle.load(f)


def resampled_window():
    """(resampled series of the eligible stations, set of the eligible), through the reader's path."""
    from Data.causal_rivers import _load_window, _RAW_SAMPLING
    G = graph()
    step = int(pd.Timedelta(RESAMPLE) / _RAW_SAMPLING)
    w = _load_window(BASE / f"rivers_ts_{REGION}.csv", pd.Timestamp(START), N_POINTS * step)
    nan = w.isna().mean()
    eligible = {node for node in G.nodes if node in w.columns and nan[node] <= MAX_NAN and w[node].nunique() > 1}
    r = (w[sorted(eligible)].interpolate(method="linear", limit_direction="both").resample(RESAMPLE).mean())
    return G, r, eligible


def choose(G, eligible) -> Dict[str, List[Tuple[int, ...]]]:
    out, taken = {}, set()
    for k, est in enumerate(STRATEGIES):
        cand = [c for c in candidates(G, est) if set(c) <= eligible and c not in taken]
        rng = np.random.default_rng(SEED + k)
        if len(cand) <= N_PER_STRATEGY:
            chosen = cand
        else:
            idx = sorted(rng.choice(len(cand), size=N_PER_STRATEGY, replace=False).tolist())
            chosen = [cand[i] for i in idx]
        out[est] = chosen
        taken |= set(chosen)
    return out


def counts() -> dict:
    """Counts of annex B (structure and eligibility; no method runs)."""
    G, r, eligible = resampled_window()
    c = {"estacoes": G.number_of_nodes(), "arestas": G.number_of_edges(), "elegiveis": len(eligible),
         "janela": [str(r.index[0]), str(r.index[-1]), len(r)]}
    taken = set()
    for est, chosen in choose(G, eligible).items():
        all_sets = candidates(G, est)
        c[est] = {"candidatos": len(all_sets), "elegiveis": sum(set(x) <= eligible for x in all_sets),
                  "escolhidos": len(chosen), "arestas_verdade": [G.subgraph(x).number_of_edges() for x in chosen]}
    return c


def scenarios() -> List[dict]:
    from Data.base import ScenarioEntry, array_to_pipeline_format_single_property
    G, r, eligible = resampled_window()
    out = []
    for est, chosen in choose(G, eligible).items():
        for i, station_ids in enumerate(chosen):
            st = list(station_ids)
            pos = {node: k for k, node in enumerate(st)}
            pairs = {(pos[u], pos[v]) for u, v in G.subgraph(st).edges}
            names = [f"station_{node}" for node in st]
            name = f"real_causal_rivers_{REGION}_{est}_{i:02d}"
            df, gt = array_to_pipeline_format_single_property(
                r[st].to_numpy(dtype=float), names, pairs, start_date=str(r.index[0]), freq=RESAMPLE,
                property_name="DISCHARGE")
            params = {"source": "causal_rivers", "region": REGION, "strategy": est, "index": i, "stations": st,
                      "rivers": {f"station_{node}": str(G.nodes[node].get("R")) for node in st},
                      "n_points": N_POINTS, "resample": RESAMPLE, "start_timestamp": str(r.index[0]),
                      "end_timestamp": str(r.index[-1]), "max_nan_frac": MAX_NAN, "n_eligible_stations": len(eligible),
                      "n_causal_links": len(pairs), "seed": SEED,
                      "authors_commit": "a30154fbb9a2e03bae133c643d7e978df63b3766"}
            out.append(ScenarioEntry(name=name, df=df, gt=gt, _scenario=REGION, _params=params).to_dict())
    return out


if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, os.getcwd())
    print(json.dumps(counts(), indent=1, default=str))

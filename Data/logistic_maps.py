"""
Data.logistic_maps
===============================================================================
COUPLED LOGISTIC MAPS BENCHMARK (no download) — classic discrete chaotic
benchmark, used since the original transfer entropy paper (Schreiber, PRL
2000) up to today to validate causality methods on discrete chaotic time
series.

GROUND TRUTH: explicitly defined by the coupling topology (coupling_pairs
in `_LOGISTIC_MAP_TOPOLOGIES`) — known by construction, not estimated. To
add a new topology, add an entry to `_LOGISTIC_MAP_TOPOLOGIES`.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import array_to_pipeline_format_single_property, register

def _simulate_logistic_maps(N: int, T: int, r: float, eps: float,
                             coupling_pairs: List[tuple], seed: int,
                             burn_in: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parents: dict = {}
    for src, dst in coupling_pairs:
        parents.setdefault(dst, []).append(src)

    n_steps = T + burn_in
    x = rng.uniform(0.1, 0.9, N)
    traj = np.zeros((n_steps, N))
    f = lambda v: r * v * (1 - v)
    for t in range(n_steps):
        x_new = np.zeros(N)
        for i in range(N):
            if i in parents:
                coupling = np.mean([f(x[s]) for s in parents[i]])
                x_new[i] = (1 - eps) * f(x[i]) + eps * coupling
            else:
                x_new[i] = f(x[i])
            x_new[i] = np.clip(x_new[i], 1e-6, 1 - 1e-6)
        x = x_new
        traj[t] = x
    return traj[burn_in:]


_LOGISTIC_MAP_TOPOLOGIES = {
    "chain": {"N": 5, "pairs": [(0, 1), (1, 2), (2, 3), (3, 4)]},
    "convergence": {"N": 5, "pairs": [(0, 1), (1, 2), (2, 3), (0, 3)]},
    "bifurcation": {"N": 5, "pairs": [(0, 1), (0, 2), (1, 3), (2, 4)]},
    "with_isolated": {"N": 6, "pairs": [(0, 1), (1, 2), (2, 0), (3, 4)]},
}


@register("logistic_maps")
def generate_logistic_maps_benchmark_scenarios(
    topologies: Optional[List[str]] = None,
    r: float = 3.8,
    eps: float = 0.3,
    n_points: int = 300,
    seeds: Optional[List[int]] = None,
    name_prefix: str = "benchmark_logistic_maps",
) -> List[dict]:
    """
    Generates scenarios using coupled logistic maps (x_(t+1) = r*x*(1-x)),
    a classic discrete chaotic benchmark used since the original transfer
    entropy paper (Schreiber, PRL 2000) up to today to validate causality
    methods on time series.

    GROUND TRUTH: explicitly defined by the coupling topology
    (coupling_pairs) — known by construction, not estimated.

    Available topologies: "chain", "convergence", "bifurcation",
    "with_isolated" (see _LOGISTIC_MAP_TOPOLOGIES).
    """
    if topologies is None:
        topologies = list(_LOGISTIC_MAP_TOPOLOGIES.keys())
    if seeds is None:
        seeds = [1, 2, 3]

    result = []
    for topology_name in topologies:
        if topology_name not in _LOGISTIC_MAP_TOPOLOGIES:
            print(f"  [benchmark] unknown topology '{topology_name}' — "
                  f"skipping. Available: {list(_LOGISTIC_MAP_TOPOLOGIES.keys())}")
            continue

        topology = _LOGISTIC_MAP_TOPOLOGIES[topology_name]
        N = topology["N"]
        pairs = topology["pairs"]
        causal_pair_idx = set(pairs)

        for seed in seeds:
            name = f"{name_prefix}_{topology_name}_seed{seed}"
            try:
                data = _simulate_logistic_maps(
                    N=N, T=n_points, r=r, eps=eps,
                    coupling_pairs=pairs, seed=seed, burn_in=200,
                )
                if np.any(~np.isfinite(data)) or np.any((data < 0) | (data > 1)):
                    print(f"  [benchmark] {name} FAILED: trajectory left the "
                          f"[0,1] interval or diverged.")
                    continue

                entity_names = [f"M{i}" for i in range(N)]
                df_long, df_gt = array_to_pipeline_format_single_property(
                    data, entity_names, causal_pair_idx, property_name="X",
                )

                params = {
                    "source": "coupled_logistic_maps",
                    "topology": topology_name, "r": r, "eps": eps, "seed": seed,
                    "n_points": n_points, "n_causal_links": len(causal_pair_idx),
                    "ground_truth_type": "coupling topology (defined by construction)",
                }
                result.append({"name": name, "df": df_long, "gt": df_gt,
                                "_scenario": None, "_params": params})

                print(f"  [benchmark] {name} generated — topology={topology_name} "
                      f"({N} variables) x {n_points} points | "
                      f"{len(causal_pair_idx)} causal links")

            except Exception as e:
                print(f"  [benchmark] {name} FAILED to generate: {e}")

    return result

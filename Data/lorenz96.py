"""
Data.lorenz96
===============================================================================
LORENZ-96 BENCHMARK (coupled chaotic system, no download) — classic
spatio-temporal chaotic benchmark (Lorenz 1996), widely used to validate
causal discovery methods on time series (CCM, transfer entropy, convergent
cross mapping and several PCMCI/Granger papers).

Equation: dx_k/dt = (x_(k+1) - x_(k-2)) * x_(k-1) - x_k + F  (k cyclic)

GROUND TRUTH: each variable k is causally influenced by its three spatial
neighbors (k-1, k-2, k+1 mod K) — a structure known analytically from the
equation itself, not estimated.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import array_to_pipeline_format_single_property, register

def _simulate_lorenz96(K: int, F: float, T: int, dt: float, downsample: int,
                        seed: int, burn_in: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = F * np.ones(K) + rng.normal(0, 0.5, K)

    def deriv(x):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

    n_steps = T * downsample + burn_in
    traj = np.zeros((n_steps, K))
    for t in range(n_steps):
        k1 = deriv(x)
        k2 = deriv(x + dt / 2 * k1)
        k3 = deriv(x + dt / 2 * k2)
        k4 = deriv(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[t] = x
    return traj[burn_in::downsample]


@register("lorenz96")
def generate_lorenz96_benchmark_scenarios(
    Ks: Optional[List[int]] = None,
    F: float = 8.0,
    n_points: int = 300,
    dt: float = 0.01,
    downsample: int = 10,
    seeds: Optional[List[int]] = None,
    name_prefix: str = "benchmark_lorenz96",
) -> List[dict]:
    """
    Generates scenarios using the Lorenz-96 system — a classic
    spatio-temporal chaotic benchmark (Lorenz 1996), widely used to validate
    causal discovery methods on time series (CCM, transfer entropy,
    convergent cross mapping and several PCMCI/Granger papers).

    Equation: dx_k/dt = (x_(k+1) - x_(k-2)) * x_(k-1) - x_k + F  (k cyclic)

    GROUND TRUTH: each variable k is causally influenced by its three
    spatial neighbors (k-1, k-2, k+1 mod K) — a structure known
    analytically from the equation itself, not estimated.

    For small K (<6) the graph becomes too dense (almost every pair is
    causal); the default uses K=[8, 10] to guarantee truly non-causal pairs
    in the GT.
    """
    if Ks is None:
        Ks = [8, 10]
    if seeds is None:
        seeds = [1, 2, 3]

    result = []
    for K in Ks:
        causal_pair_idx = set()
        for k in range(K):
            for j in [(k - 1) % K, (k - 2) % K, (k + 1) % K]:
                causal_pair_idx.add((j, k))

        for seed in seeds:
            name = f"{name_prefix}_K{K}_seed{seed}"
            try:
                data = _simulate_lorenz96(K=K, F=F, T=n_points, dt=dt,
                                           downsample=downsample, seed=seed,
                                           burn_in=500)
                if np.any(~np.isfinite(data)):
                    print(f"  [benchmark] {name} FAILED: trajectory diverges "
                          f"(NaN/Inf) — consider a smaller dt or F.")
                    continue

                entity_names = [f"X{k}" for k in range(K)]
                df_long, df_gt = array_to_pipeline_format_single_property(
                    data, entity_names, causal_pair_idx, property_name="X",
                )

                params = {
                    "source": "lorenz96",
                    "K": K, "F": F, "seed": seed, "n_points": n_points,
                    "n_causal_links": len(causal_pair_idx),
                    "ground_truth_type": "analytical structure (spatial neighborhood)",
                }
                result.append({"name": name, "df": df_long, "gt": df_gt,
                                "_scenario": None, "_params": params})

                print(f"  [benchmark] {name} generated — K={K} variables x "
                      f"{n_points} points | {len(causal_pair_idx)} causal "
                      f"links (spatial neighborhood)")

            except Exception as e:
                print(f"  [benchmark] {name} FAILED to generate: {e}")

    return result

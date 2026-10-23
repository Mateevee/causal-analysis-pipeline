"""
Data.tigramite_scm
===============================================================================
SYNTHETIC BENCHMARK — causal discovery scenarios generated via the structural
causal model (SCM) generators from the `tigramite` package (Runge et al.) —
the same library used to validate PCMCI/PCMCI+ in the original literature.
Requires `pip install tigramite`.

Motivation: the synthetic VAR generator
(Data/synthetic_var.py) adds every causal relationship LINEARLY to the target
series. That tests multi-stage detection and lag selection, but not the need
for the DML's nonlinear nuisance models. The tigramite scenarios:
  (a) were not designed by this pipeline's authors, so the generator and the
      evaluated method are not co-designed;
  (b) include GENUINELY NONLINEAR dependencies (sign-preserving quadratic,
      threshold/step, tanh-type saturation), not just a linear lag;
  (c) are PCMCI's own validation ground.

The generator is `structural_causal_process(links, T, seed)`, an additive
noise process
    X^j_t = noise^j_t + sum_over_parents( coef * func(X^i_{t-tau}) )
with `func` any function of one argument. Only LAGGED links (tau >= 1) are
used, the regime the pipeline was designed to detect (X_{t-tau} -> Y_t).
The N variables of the process are grouped into consecutive blocks of
`n_properties`; each block becomes an entity with `n_properties` properties,
so a link is either intra-entity (same entity, different properties) or
inter-entity (the main case).

Each scenario is defined by a `links` dict in tigramite's native format:
    {target_var: [((source_var, -tau), coefficient, function), ...]}
and groups columns into blocks of N properties per entity (unlike the
single-property benchmarks such as Lorenz-96/logistic maps — see Data.base
for the generic converter used by those).

To add a new SCM scenario: add an entry to
`_BENCHMARK_SCENARIO_DEFINITIONS` (key = scenario name) following the same
`links` format as the existing ones, reusing `_f_lin`/`_f_quad`/
`_f_threshold`/`_f_tanh` or defining a new function. No changes are needed
to `generate_tigramite_benchmark_scenarios()`.
"""
from __future__ import annotations

import itertools
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import ScenarioEntry, register

def _f_lin(x):
    """Identity — used for autocorrelation and as a linear baseline."""
    return x


def _f_quad(x):
    """Sign-preserving quadratic: sign(x) * x^2. Monotonic, but curved
    (not captured by linear correlation/Xcorr, requires an auxiliary
    non-linear model in DML to correctly remove the confounding)."""
    return np.sign(x) * (x ** 2)


def _f_threshold(x, c: float = 0.3):
    """Step/threshold: zero below c, identity above. Non-smooth,
    non-monotonic in derivative — a hard regime for Kernel Ridge (RBF) and
    better suited to Random Forest / Extra Trees."""
    return np.where(x > c, x, 0.0)


def _f_tanh(x, scale: float = 2.0):
    """Tanh-style saturation — smooth non-linearity with a plateau, common
    in physical systems with a capacity limit (e.g. flow saturation)."""
    return scale * np.tanh(x)


_BENCHMARK_SCENARIO_DEFINITIONS = {
    # ── Linear baseline: linear links only, a reference that separates
    #    the effect of non-linearity from the rest of the SCM ─────────────
    "baseline_linear": {
        "type": "linear",
        "n_properties": 3,
        "links": {
            0: [((0, -1), 0.30, _f_lin)],
            1: [((1, -1), 0.30, _f_lin), ((0, -1), 0.55, _f_lin)],
            2: [((2, -1), 0.30, _f_lin)],
            3: [((3, -1), 0.30, _f_lin), ((2, -2), 0.55, _f_lin)],
            4: [((4, -1), 0.30, _f_lin), ((1, -1), 0.50, _f_lin)],
            5: [((5, -1), 0.30, _f_lin)],  # isolated (true negative)
        },
    },

    # ── Quadratic non-linear: intra-entity and inter-entity relation via a
    #    sign-preserving quadratic function ──────────────────────────────
    "quadratic": {
        "type": "nonlinear",
        "n_properties": 3,
        "links": {
            0: [((0, -1), 0.25, _f_lin)],
            1: [((1, -1), 0.25, _f_lin), ((0, -1), 0.55, _f_quad)],   # intra-entity
            2: [((2, -1), 0.25, _f_lin)],
            3: [((3, -1), 0.25, _f_lin), ((2, -2), 0.60, _f_quad)],   # inter-entity
            4: [((4, -1), 0.25, _f_lin), ((1, -1), 0.50, _f_quad)],   # inter-entity
            5: [((5, -1), 0.25, _f_lin)],  # isolated
        },
    },

    # ── Threshold/step: non-smooth, stress-tests tree-based models ───────
    "threshold": {
        "type": "nonlinear",
        "n_properties": 3,
        "links": {
            0: [((0, -1), 0.25, _f_lin)],
            1: [((1, -1), 0.25, _f_lin), ((0, -1), 0.60, _f_threshold)],
            2: [((2, -1), 0.25, _f_lin)],
            3: [((3, -1), 0.25, _f_lin), ((2, -1), 0.60, _f_threshold)],
            4: [((4, -1), 0.25, _f_lin), ((1, -2), 0.55, _f_threshold)],
            5: [((5, -1), 0.25, _f_lin)],  # isolated
        },
    },

    # ── Tanh saturation: smooth non-linearity with a physically plausible
    #    plateau ──────────────────────────────────────────────────────────
    "saturation": {
        "type": "nonlinear",
        "n_properties": 3,
        "links": {
            0: [((0, -1), 0.25, _f_lin)],
            1: [((1, -1), 0.25, _f_lin), ((0, -1), 0.50, _f_tanh)],
            2: [((2, -1), 0.25, _f_lin)],
            3: [((3, -1), 0.25, _f_lin), ((2, -1), 0.50, _f_tanh)],
            4: [((4, -1), 0.25, _f_lin), ((1, -2), 0.45, _f_tanh)],
            5: [((5, -1), 0.25, _f_lin)],  # isolated
        },
    },

    # ── Large mixed: combines all three non-linearity types in a scenario
    #    with more entities, to stress the pipeline's discrimination at a
    #    larger scale ─────────────────────────────────────────────────────
    "large_mixed": {
        "type": "nonlinear",
        "n_properties": 3,
        "links": {
            0:  [((0, -1), 0.25, _f_lin)],
            1:  [((1, -1), 0.25, _f_lin), ((0, -1), 0.55, _f_quad)],
            2:  [((2, -1), 0.25, _f_lin)],
            3:  [((3, -1), 0.25, _f_lin), ((2, -2), 0.55, _f_threshold)],
            4:  [((4, -1), 0.25, _f_lin), ((1, -1), 0.50, _f_tanh)],
            5:  [((5, -1), 0.25, _f_lin)],
            6:  [((6, -1), 0.25, _f_lin), ((3, -1), 0.50, _f_quad)],
            7:  [((7, -1), 0.25, _f_lin)],
            8:  [((8, -1), 0.25, _f_lin), ((6, -2), 0.55, _f_threshold)],
            9:  [((9, -1), 0.25, _f_lin)],   # isolated
            10: [((10, -1), 0.25, _f_lin)],  # isolated
            11: [((11, -1), 0.25, _f_lin)],  # isolated
        },
    },
}

def _build_gt_from_links(links: dict, n_properties: int) -> tuple:
    """
    From the `links` dict in tigramite's format ({target_var: [((source_var,
    -tau), coef, func), ...]}), returns:
        - causal_links: set of (source_var, target_var, abs_tau, func_name)
          with tau >= 1 (autocorrelation, source_var == target_var, is
          excluded)
        - n_vars: total number of variables in the process
    """
    n_vars = max(links.keys()) + 1
    causal_links = set()
    for target_var, parents in links.items():
        for (source_var, tau), coeff, func in parents:
            if source_var == target_var:
                continue  # autocorrelation is not a "pair" in this framework
            if abs(tau) < 1:
                continue  # contemporaneous links are not this pipeline's regime
            causal_links.add((source_var, target_var, abs(tau), func.__name__))
    return causal_links, n_vars


def _scm_to_pipeline_format(
    data: np.ndarray,
    links: dict,
    n_properties: int,
    start_date: str = "2015-01-01",
    entity_prefix: str = "ENT",
) -> tuple:
    """
    Converts the output of structural_causal_process (T x N array) and the
    known-links dict into the (df_long, df_gt) format expected by
    run_all()/validate_dataset() — the same contract as
    Data.base.array_to_pipeline_format_single_property().

    Returns:
        df_long : pd.DataFrame  — columns (Date, Name, F_0, F_1, ..., F_{n_properties-1})
        df_gt   : pd.DataFrame  — columns (source_entity, source_property,
                                            target_entity, target_property, causal_gt)
    """
    import itertools

    T, n_vars = data.shape
    if n_vars % n_properties != 0:
        raise ValueError(
            f"n_vars={n_vars} is not divisible by n_properties={n_properties}; "
            f"adjust the scenario definition so that every entity has "
            f"the same number of properties."
        )
    n_entities = n_vars // n_properties

    entity_names = [f"{entity_prefix}_{e}" for e in range(n_entities)]
    property_names = [f"F_{f}" for f in range(n_properties)]

    def var_to_entity_property(var_idx):
        return var_idx // n_properties, var_idx % n_properties

    # ── df_long ──────────────────────────────────────────────────────────
    dates = pd.date_range(start_date, periods=T, freq="MS")
    records = []
    for t in range(T):
        for e in range(n_entities):
            row = {"Date": dates[t], "Name": entity_names[e]}
            for f in range(n_properties):
                var_idx = e * n_properties + f
                row[property_names[f]] = float(data[t, var_idx])
            records.append(row)
    df_long = pd.DataFrame(records)

    # ── df_gt: full cartesian product (entity,property) x (entity,property),
    #    except the pair with itself (same entity AND same property) ──────
    causal_links, _ = _build_gt_from_links(links, n_properties)
    causal_pairs = {(sv, tv) for sv, tv, tau, fname in causal_links}

    all_vars = list(range(n_vars))
    gt_records = []
    for sv, tv in itertools.permutations(all_vars, 2):
        src_entity_idx, src_prop_idx = var_to_entity_property(sv)
        tgt_entity_idx, tgt_prop_idx = var_to_entity_property(tv)
        gt_records.append({
            "source_entity":    entity_names[src_entity_idx],
            "source_property":  property_names[src_prop_idx],
            "target_entity":    entity_names[tgt_entity_idx],
            "target_property":  property_names[tgt_prop_idx],
            "causal_gt":        (sv, tv) in causal_pairs,
        })
    df_gt = pd.DataFrame(gt_records)

    return df_long, df_gt


@register("tigramite")
def generate_tigramite_benchmark_scenarios(
    seeds: Optional[List[int]] = None,
    n_points: int = 300,
    scenarios: Optional[List[str]] = None,
    name_prefix: str = "benchmark_tigramite",
    transient_fraction: float = 0.2,
) -> List[dict]:
    """
    Generates causal discovery benchmark scenarios using the structural
    causal process generators from the `tigramite` package (Runge et al.) —
    see the module docstring for the motivation.

    Requires: pip install tigramite

    Parameters
    ----------
    seeds : list[int], optional
        Realization seeds (each seed generates a distinct time series for
        the SAME known causal graph). Default: [11, 22, 33] — three
        realizations per scenario.
    n_points : int
        Series length (T). Default 300 — well above the short-series
        sparse regime (24-120), deliberately: this benchmark tests non-linear
        discovery capability in itself, not the short-data regime (covered
        by the synthetic VAR scenarios).
    scenarios : list[str], optional
        Subset of names from _BENCHMARK_SCENARIO_DEFINITIONS to generate.
        Default: all ("baseline_linear", "quadratic", "threshold",
        "saturation", "large_mixed").
    name_prefix : str
        Prefix used in the 'name' field of each returned entry.
    transient_fraction : float
        Passed through to structural_causal_process — fraction of
        transient periods discarded at the start of the series. Default
        0.2.

    Returns
    -------
    List[dict] — same contract as generate_synthetic_var_benchmark_scenarios():
        {'name', 'df', 'gt', '_scenario', '_params'}
    Compatible with run_all(dataframes=..., ...) with no changes to the
    runner.
    """
    try:
        from tigramite.toymodels import structural_causal_processes as toys
    except ImportError as e:
        raise ImportError(
            "Package 'tigramite' not found. Install it with:\n"
            "    pip install tigramite\n"
            "This function reuses tigramite's structural causal process "
            "generators (the same library used to validate PCMCI in the "
            "original literature) to produce benchmark scenarios "
            "independent of this project's internal generator."
        ) from e

    if seeds is None:
        seeds = [11, 22, 33]

    scenario_names = scenarios if scenarios is not None else list(
        _BENCHMARK_SCENARIO_DEFINITIONS.keys()
    )

    result = []
    for scenario_name in scenario_names:
        if scenario_name not in _BENCHMARK_SCENARIO_DEFINITIONS:
            print(f"  [benchmark] unknown scenario '{scenario_name}' — skipping. "
                  f"Available: {list(_BENCHMARK_SCENARIO_DEFINITIONS.keys())}")
            continue

        scenario_def = _BENCHMARK_SCENARIO_DEFINITIONS[scenario_name]
        links = scenario_def["links"]
        n_properties = scenario_def["n_properties"]
        scenario_type = scenario_def["type"]

        causal_links, n_vars = _build_gt_from_links(links, n_properties)

        for seed in seeds:
            name = f"{name_prefix}_{scenario_name}_seed{seed}"
            try:
                data, invalid = toys.structural_causal_process(
                    links=links,
                    T=n_points,
                    seed=seed,
                    transient_fraction=transient_fraction,
                )
                if invalid:
                    print(f"  [benchmark] {name} FAILED: generated data contains "
                          f"NaN/Inf (structural process instability — "
                          f"consider reducing the coupling coefficients).")
                    continue

                df_long, df_gt = _scm_to_pipeline_format(
                    data=data, links=links, n_properties=n_properties,
                )

                params = {
                    "source":            "tigramite_benchmark",
                    "scenario":          scenario_name,
                    "type":              scenario_type,
                    "seed":              seed,
                    "n_points":          n_points,
                    "n_variables":       n_vars,
                    "n_entities":        n_vars // n_properties,
                    "n_properties":      n_properties,
                    "n_causal_links":    len(causal_links),
                    "functions_used":    sorted({fname for *_, fname in causal_links}),
                }

                result.append({
                    "name":      name,
                    "df":        df_long,
                    "gt":        df_gt,
                    "_scenario": None,
                    "_params":   params,
                })

                print(
                    f"  [benchmark] {name} generated — type={scenario_type} | "
                    f"{n_vars} variables ({n_vars // n_properties} entities x "
                    f"{n_properties} properties) x {n_points} points | "
                    f"{len(causal_links)} causal links | "
                    f"functions: {sorted({fname for *_, fname in causal_links})}"
                )

            except Exception as e:
                print(f"  [benchmark] {name} FAILED to generate/convert: {e}")

    return result

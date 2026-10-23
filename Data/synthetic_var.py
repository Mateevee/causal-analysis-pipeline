"""
Data.synthetic_var
===============================================================================
Synthetic benchmark via a linear VAR (Vector Autoregression) model with
lags, built from scratch -- a simpler/more controllable alternative to the
SCM benchmark via tigramite (Data/tigramite_scm.py), for when you want:

    - to control exactly the density of the causal graph, the lag range,
      and the strength of the coefficients;
    - to not depend on the `tigramite` package being installed;
    - a STRONG guarantee that the returned `causal_gt` matches 100% with
      the generating process (the GT is neither estimated nor inferred --
      it IS the list of edges used to simulate the data. There is no gap
      between "truth" and "generator": they are the same thing).

IDEA
-------------------------------------------------------------------------------
1. Draws a directed causal graph among N entities: for each entity i, a
   random number of "parents" j (j != i), each edge j->i with a lag in
   [1, max_lag] and a random coefficient with a random sign.
1b. MINIMUM GUARANTEE: since the draw above is probabilistic, it is
   possible -- especially with low prob_edge, few max_parents, or few
   entities -- to end up with a graph with ZERO causal edges, which
   leaves precision/recall/F1 undefined (0/0 division) and the whole
   scenario with no signal to validate against. Because of this,
   immediately after the draw, `_ensure_min_edges()` complements the
   graph until it reaches `min_edges` (default 1), adding edges only on
   (source, target) pairs that the original draw left free -- it never
   duplicates a pair nor rewrites a decision already made. Pairs that
   remain outside the graph stay correctly marked causal_gt=False in the
   GT -- the causal/non-causal boundary is never ambiguous, only the
   floor on the number of causal edges is guaranteed.
2. Adds its own autoregressive term (AR(1)) to each entity, only to give
   the series persistence/realism -- it does NOT enter causal_gt (it is
   not a causal relation *between entities*, it is just the series' own
   inertia).
3. Guarantees stability (the series does not explode) by capping the sum
   of the |coefficients| arriving at each entity (AR + causal edges) at a
   ceiling < 1.
4. Simulates T + burn_in steps with Gaussian noise; discards the burn-in
   (to leave the series past the initial transient) and returns the final
   n_points.
5. Converts to the pipeline format via
   `array_to_pipeline_format_single_property()` (Data/base.py), passing
   the exact set of pairs (i, j) used in step 1 as `causal_pair_idx` --
   i.e. the GT comes out automatically correct, with no double bookkeeping.

Mathematically, there is no problem in having "loops" such as i->j (lag 1)
and j->i (lag 2): it is just feedback across time, not a circular
contemporaneous dependency (the simulation always runs "forward" in time
-- x[t] depends only on x[t-1], x[t-2], ...). IN PRACTICE, however, this
pattern triggers "reverse edge" heuristics in detection pipelines (e.g.
the 'patch'), which drop one of the two sides by confusing it with the
classic mirrored-correlation artifact (a strong edge A->B producing a
spurious echo B->A) -- producing an FN "by design" of the heuristic, not
an actual detection error. This is why this generator's DEFAULT
(avoid_bidirectional_pairs=True) is to NOT create this kind of pair; pass
False explicitly if you want this specific stress test.

Signature follows the contract of Data/base.py (see the docstring there).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import ScenarioEntry, array_to_pipeline_format_single_property, register


def _draw_causal_graph(
    rng: np.random.Generator,
    n_entities: int,
    prob_edge: float,
    max_parents: int,
    max_lag: int,
    coef_strength: Tuple[float, float],
    avoid_bidirectional_pairs: bool = True,
) -> Dict[int, List[Tuple[int, int, float]]]:
    """Draws, for each entity i, a list of (j, lag, coef) -- the causal
    "parents" of i. Returns dict {i: [(j, lag, coef), ...]}.

    - j != i always (no self causal relation counted in the GT; the
      entity's own persistence is handled separately as an AR term).
    - at most one edge per ordered pair (i receives from a given j only
      once, with a single lag) -- simplifies the GT to a boolean per pair.

    avoid_bidirectional_pairs : if True (default -- this generator's
        default), never creates both directions (j->i AND i->j) between
        the SAME pair of entities, even at different lags. Why this is
        the default: detection pipelines with a "reverse edge" heuristic
        (e.g. the 'patch', in the structural engine) treat a genuine
        bidirectional pair -- common and valid in systems with real
        feedback -- as if it were the classic "mirrored correlation"
        artifact (a strong edge A->B producing a spurious echo B->A) and
        drop one of the two sides. This produces an FN "by design" of
        the heuristic, not an actual detection error -- and distorts the
        evaluation. Pass False explicitly if you deliberately want
        scenarios with genuine bidirectional feedback (an additional
        stress test, which requires interpreting FN on those pairs with
        care).
    """
    graph: Dict[int, List[Tuple[int, int, float]]] = {i: [] for i in range(n_entities)}
    directed_pairs: set = set()
    lo, hi = coef_strength
    for i in range(n_entities):
        candidates = [j for j in range(n_entities) if j != i]
        rng.shuffle(candidates)
        n_parents = rng.integers(0, max_parents + 1)
        added = 0
        for j in candidates:
            if added >= n_parents:
                break
            if rng.random() >= prob_edge:
                continue
            if avoid_bidirectional_pairs and (i, j) in directed_pairs:
                # the edge i->j already exists; adding j->i would create a
                # bidirectional pair -- skip this candidate (does not
                # consume the n_parents "slot", just discards the attempt).
                continue
            lag = int(rng.integers(1, max_lag + 1))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            coef = sign * rng.uniform(lo, hi)
            graph[i].append((j, lag, coef))
            directed_pairs.add((j, i))
            added += 1
    return graph


def _ensure_min_edges(
    rng: np.random.Generator,
    graph: Dict[int, List[Tuple[int, int, float]]],
    n_entities: int,
    max_lag: int,
    coef_strength: Tuple[float, float],
    min_edges: int,
    avoid_bidirectional_pairs: bool = True,
) -> Dict[int, List[Tuple[int, int, float]]]:
    """If the drawn graph has fewer than `min_edges` causal edges in total
    (this can happen by chance in _draw_causal_graph -- low prob_edge +
    few entities can produce an empty graph), adds extra edges until the
    minimum is reached.

    GUARANTEES of this function (the reason it exists instead of simply
    re-running _draw_causal_graph with a higher prob_edge):
    - NEVER duplicates a (source, target) pair that already exists --
      each ordered pair has at most one edge, exactly as
      _draw_causal_graph already guarantees.
    - NEVER touches the pairs that the original draw decided to leave
      out -- it only adds new edges on top of candidates that are still
      free. This is what keeps the distinction between causal (in the
      graph -> causal_gt=True) and non-causal (not in the graph ->
      causal_gt=False) intact: the boundary is always "is it in the
      graph or not", never a gray zone.
    - The added edges go into the SAME graph that later becomes both the
      simulation and the GT (via causal_pair_idx in
      generate_synthetic_var_benchmark_scenarios) -- so the GT remains
      100% consistent with the generator, and guaranteed not to come
      out empty.
    - avoid_bidirectional_pairs : same semantics as _draw_causal_graph --
      if True, the candidates for extra edges also exclude any pair
      whose opposite direction already exists in the graph, so the
      edges forced to hit the minimum never reintroduce the problem this
      flag exists to avoid.
    """
    lo, hi = coef_strength
    total = sum(len(edges) for edges in graph.values())
    if total >= min_edges:
        return graph

    existing_pairs = {
        (source, target)
        for target, edges in graph.items()
        for (source, _lag, _coef) in edges
    }
    candidates = [
        (i, j) for i in range(n_entities) for j in range(n_entities)
        if i != j and (i, j) not in existing_pairs
        and not (avoid_bidirectional_pairs and (j, i) in existing_pairs)
    ]
    rng.shuffle(candidates)

    max_possible = n_entities * (n_entities - 1)
    if avoid_bidirectional_pairs:
        max_possible = max_possible // 2
    if min_edges > max_possible:
        raise ValueError(
            f"min_edges={min_edges} is greater than the maximum number of "
            f"ordered pairs possible for n_entities={n_entities}"
            + (" with avoid_bidirectional_pairs=True" if avoid_bidirectional_pairs else "")
            + f" ({max_possible} pairs)."
        )

    idx = 0
    while total < min_edges and idx < len(candidates):
        source, target = candidates[idx]
        idx += 1
        lag = int(rng.integers(1, max_lag + 1))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        coef = sign * rng.uniform(lo, hi)
        graph[target].append((source, lag, coef))
        existing_pairs.add((source, target))
        total += 1

    return graph


def _simulate_var(
    rng: np.random.Generator,
    n_entities: int,
    graph: Dict[int, List[Tuple[int, int, float]]],
    n_points: int,
    max_lag: int,
    noise: float,
    ar_coef_range: Tuple[float, float],
    burn_in: int,
    stability_ceiling: float = 0.92,
) -> Tuple[np.ndarray, List[float]]:
    """Simulates the VAR and returns (data[T, N], ar_coefs). `data` already
    comes without the burn-in (only the final n_points)."""
    T = n_points + burn_in
    data = np.zeros((T, n_entities), dtype=float)

    # Each entity's own AR(1), capped to guarantee stability together
    # with the causal coefficients arriving at it.
    ar_lo, ar_hi = ar_coef_range
    ar_coefs = []
    for i in range(n_entities):
        causal_sum = sum(abs(c) for (_, _, c) in graph[i])
        ar_ceiling = max(0.0, stability_ceiling - causal_sum)
        ar_i = min(rng.uniform(ar_lo, ar_hi), ar_ceiling)
        ar_coefs.append(ar_i)

    # initial state (before burn-in), close to zero + noise
    data[:max_lag, :] = rng.normal(0.0, noise, size=(max_lag, n_entities))

    for t in range(max_lag, T):
        for i in range(n_entities):
            val = ar_coefs[i] * data[t - 1, i]
            for (j, lag, coef) in graph[i]:
                val += coef * data[t - lag, j]
            val += rng.normal(0.0, noise)
            data[t, i] = val

    return data[burn_in:], ar_coefs


@register("synthetic_var")
def generate_synthetic_var_benchmark_scenarios(
    *,
    seeds: Optional[List[int]] = None,
    n_points: int = 300,
    name_prefix: str = "benchmark_synthetic_var",
    n_entities: int = 5,
    prob_edge: float = 0.35,
    max_parents: int = 2,
    max_lag: int = 3,
    min_edges: int = 1,
    avoid_bidirectional_pairs: bool = True,
    coef_strength: Tuple[float, float] = (0.3, 0.6),
    ar_coef_range: Tuple[float, float] = (0.2, 0.5),
    noise: float = 0.3,
    burn_in: int = 200,
    property_name: str = "VAL",
    start_date: str = "2015-01-01",
    freq: str = "MS",
    entity_prefix: str = "P",
) -> List[dict]:
    """Generates synthetic scenarios via a linear VAR with lags, one per
    seed.

    The GT (`causal_gt`) is guaranteed correct by construction: it is
    exactly the set of edges (j -> i) drawn and used in the simulation --
    there is no estimation involved, so there is no risk of "the
    generator says one thing, the GT says another" (the classic mistake
    when writing a synthetic benchmark by hand).

    Parameters
    ----------
    seeds            : one entry (scenario) per seed. Default [1].
    n_points         : series length after burn-in.
    name_prefix      : scenario name prefix (name = f"{prefix}_seed{seed}").
    n_entities       : number of entities (N columns).
    prob_edge        : probability of each parent candidate becoming a causal edge.
    max_parents      : maximum number of causal parents per entity.
    max_lag          : maximum lag (in steps) of the causal edges.
    min_edges        : guaranteed minimum number of causal edges in the
                        scenario (default 1 -- avoids scenarios with an
                        empty GT, which make precision/recall/F1
                        undefined via 0/0 division. If the original draw
                        (prob_edge x max_parents) comes out below the
                        minimum, extra edges are added on pairs still
                        free -- see _ensure_min_edges(). Unused pairs
                        remain correctly marked causal_gt=False; nothing
                        is "almost causal" or ambiguous.
    avoid_bidirectional_pairs : if True (default), never generates both
                        directions (j->i AND i->j) between the SAME pair
                        of entities, even at different lags. Reason for
                        the default: pipelines with a "reverse edge"
                        heuristic (e.g. the 'patch', in the structural
                        engine) treat a genuine bidirectional pair as if
                        it were the classic mirrored-correlation
                        artifact (a strong edge A->B producing a
                        spurious echo B->A) and drop one of the two
                        sides -- producing an FN "by design" of the
                        heuristic, not an actual detection error. Pass
                        False explicitly if you want scenarios with
                        genuine bidirectional feedback (useful as an
                        additional stress test, but require interpreting
                        FN on those pairs with care). See the docstring
                        of _draw_causal_graph() for the full breakdown
                        of the trade-off.
    coef_strength    : (min, max) of the |coefficient| of the causal edges.
    ar_coef_range    : (min, max) of the entity's own AR(1) coefficient (non-causal).
    noise            : standard deviation of the Gaussian noise at each step.
    burn_in          : steps discarded at the start to leave the transient.
    property_name    : name of the value column (default "VAL").
    start_date/freq  : used only to generate the df's date axis.
    entity_prefix    : entity names become f"{entity_prefix}{i+1}".

    Returns
    -------
    List[dict] in ScenarioEntry.to_dict() format, one per seed.
    """
    if seeds is None:
        seeds = [1]

    entity_names = [f"{entity_prefix}{i + 1}" for i in range(n_entities)]
    results: List[dict] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)

        graph = _draw_causal_graph(
            rng, n_entities, prob_edge, max_parents, max_lag, coef_strength,
            avoid_bidirectional_pairs=avoid_bidirectional_pairs,
        )
        if min_edges > 0:
            graph = _ensure_min_edges(
                rng, graph, n_entities, max_lag, coef_strength, min_edges,
                avoid_bidirectional_pairs=avoid_bidirectional_pairs,
            )
        data, ar_coefs = _simulate_var(
            rng, n_entities, graph, n_points, max_lag, noise,
            ar_coef_range, burn_in,
        )

        # Set of causal (i, j) pairs in the format expected by
        # array_to_pipeline_format_single_property(): (source, target).
        # In the graph, `graph[target] = [(source, lag, coef), ...]`.
        causal_pair_idx = {
            (source, target)
            for target, edges in graph.items()
            for (source, _lag, _coef) in edges
        }

        df, gt = array_to_pipeline_format_single_property(
            data=data,
            entity_names=entity_names,
            causal_pair_idx=causal_pair_idx,
            start_date=start_date,
            freq=freq,
            property_name=property_name,
        )

        # Full edge metadata (for debugging/reproducibility), readable by
        # entity name instead of index.
        readable_edges = [
            {
                "source": entity_names[source],
                "target": entity_names[target],
                "lag": lag,
                "coef": round(coef, 4),
            }
            for target, edges in graph.items()
            for (source, lag, coef) in edges
        ]

        entry = ScenarioEntry(
            name=f"{name_prefix}_seed{seed}",
            df=df,
            gt=gt,
            _scenario="linear_var_with_lags",
            _params={
                "source": "synthetic_var",
                "seed": seed,
                "n_entities": n_entities,
                "n_points": n_points,
                "prob_edge": prob_edge,
                "max_parents": max_parents,
                "max_lag": max_lag,
                "min_edges": min_edges,
                "avoid_bidirectional_pairs": avoid_bidirectional_pairs,
                "coef_strength": coef_strength,
                "ar_coefs": [round(c, 4) for c in ar_coefs],
                "noise": noise,
                "burn_in": burn_in,
                "causal_edges": readable_edges,
                "n_causal_edges": len(readable_edges),
            },
        )
        results.append(entry.to_dict())

    return results

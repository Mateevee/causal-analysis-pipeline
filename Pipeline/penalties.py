"""
penalties.py -- Pluggable penalty functions

Two independent contracts:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. LagPenaltyFn  --  penalty on LAG SELECTION (pre-DML)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    LagPenaltyFn = Callable[[int], float]

    Takes a lag (int >= 1) and returns a weight in [0.0, 1.0].
        1.0 = lag fully accepted
        0.0 = lag fully vetoed

    Applied inside the pre-filter (Xcorr, TE, Granger, Pearson).
    Weights which lag wins the argmax before the DML stage.

    Examples:
        from penalties import KneeLagPenalty, ExponentialLagPenalty
        run_pipeline(df, lag_penalty_fn=KneeLagPenalty(knee1=5, knee2=10))
        run_pipeline(df, lag_penalty_fn=lambda lag: 1.0 / lag ** 2)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. CausalPenaltyFn  --  penalty on the CAUSAL RELATIONSHIP (post-DML)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    CausalPenaltyFn = Callable[[pd.Series], float]

    Takes a pd.Series corresponding to ONE ROW of the DML results
    DataFrame (build_results) and returns a weight in [0.0, 1.0]
    (or any positive float, if used in multiplicative mode over
    theta).
        1.0 = relationship not penalized
        0.0 = relationship fully vetoed
        > 1.0 = amplification (use with caution -- see note below)

    Columns available in the received row (pd.Series):
        source_entity       : str   -- name of the source entity
        source_property     : str   -- property of the source entity
        target_entity       : str   -- name of the target entity
        target_property     : str   -- property of the target entity
        lag                 : int   -- selected time lag (in periods)
        lag_penalty_score   : float -- lag weight applied in the pre-filter
        transfer_entropy    : float -- Transfer Entropy (or Pearson corr)
                                      of the pair at the selected lag
        filter_method              : str   -- method that approved the pair in the pre-filter
                                      ('xcorr', 'TE_histogram', 'TE_knn',
                                       'Granger_rescue', 'Pearson_rescue')
        causal_effect       : float -- theta estimated by the DML
        standard_error      : float -- HAC standard error of theta
        direction             : str   -- 'positive' or 'negative'
        ci_95_lower         : float -- lower bound of the 95% CI
        ci_95_upper         : float -- upper bound of the 95% CI
        pvalue              : float -- two-sided p-value of theta
        significant       : bool  -- True if pvalue < 0.05 (original DML)
        old_sig             : bool  -- preserved after apply_causal_penalty
        strength               : float -- abs(causal_effect)
        dml_model_used     : str   -- nuisance model chosen
                                      ('kernel_ridge', 'svr',
                                       'random_forest', 'bayesian_ridge')

    Applied to: the theta effect estimated by the DML.
        penalized_theta  = theta * causal_penalty_fn(row)
        ci_lower_pen     = ci_lower * w
        ci_upper_pen     = ci_upper * w

    Note: values > 1.0 are technically allowed, but they break the
    probabilistic interpretability of the weight. Only use them as a
    boost factor informed by a very strong domain prior
    (e.g.: pair confirmed by a tracer -> w=1.5).

    Ready-made implementations:
        UniformCausalPenalty      -- no penalty (w=1.0 for everything)
        PropertyTypeCausalPenalty -- penalizes/favors by property type
        ComposedCausalPenalty     -- combines multiple penalties

    Examples:
        # Custom lambda -- blocks pairs from the same entity (accesses row directly)
        run_pipeline(df, causal_penalty_fn=lambda row: 0.0 if row['source_entity'] == row['target_entity'] else 1.0)

        # Lambda using the row's causal_effect (boost if theta is very high)
        run_pipeline(df, causal_penalty_fn=lambda row: 1.5 if abs(row['causal_effect']) > 0.5 else 1.0)

        # Composition: multiple penalties
        pen = ComposedCausalPenalty([
            PropertyTypeCausalPenalty(bonus={'I1': {'P1': 1.5}}),
        ])
        run_pipeline(df, causal_penalty_fn=pen)
"""

import numpy as np
from typing import Callable, Optional, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# Public aliases -- use for type hints in your modules
# ─────────────────────────────────────────────────────────────────────────────
LagPenaltyFn    = Callable[[int], float]
CausalPenaltyFn = Callable[['pd.Series'], float]


# =============================================================================
# PART 1 -- LAG PENALTY  (pre-DML)
# =============================================================================

class UniformLagPenalty:
    """
    No penalty -- every lag receives weight 1.0.
    Equivalent to turning off the lag penalty.
    """
    def __call__(self, lag: int) -> float:
        return 1.0

    def __repr__(self) -> str:
        return "UniformLagPenalty()"


class KneeLagPenalty:
    """
    Three-zone penalty with two knees.

    Green zone   lag 1 .. knee1       : almost no penalty     (rate_zone1)
    Yellow zone  lag knee1+1 .. knee2 : mild penalty          (rate_zone2)
    Red zone     lag knee2+1 ..       : aggressive penalty    (rate_zone3)

    The function is continuous at the knees -- each zone inherits the
    final value of the previous zone as its starting point.

    Defaults (knee1=4, knee2=6, rates=0.02/0.15/0.50):
        lag=1  -> 1.000
        lag=4  -> 0.942   (end of green zone)
        lag=6  -> 0.685   (end of yellow zone)
        lag=7  -> 0.414
        lag=9  -> 0.151
        lag=12 -> 0.023
        lag=15 -> 0.003
    """
    def __init__(
        self,
        knee1: int = 4,
        knee2: int = 6,
        rate_zone1: float = 0.02,
        rate_zone2: float = 0.15,
        rate_zone3: float = 0.50,
    ):
        if knee1 >= knee2:
            raise ValueError(f"knee1 ({knee1}) must be < knee2 ({knee2})")
        self.knee1      = knee1
        self.knee2      = knee2
        self.rate_zone1 = rate_zone1
        self.rate_zone2 = rate_zone2
        self.rate_zone3 = rate_zone3
        self._pk1 = float(np.exp(-rate_zone1 * (knee1 - 1)))
        self._pk2 = self._pk1 * float(np.exp(-rate_zone2 * (knee2 - knee1)))

    def __call__(self, lag: int) -> float:
        if lag <= self.knee1:
            return float(np.exp(-self.rate_zone1 * (lag - 1)))
        if lag <= self.knee2:
            return float(self._pk1 * np.exp(-self.rate_zone2 * (lag - self.knee1)))
        return float(self._pk2 * np.exp(-self.rate_zone3 * (lag - self.knee2)))

    def __repr__(self) -> str:
        return (
            f"KneeLagPenalty("
            f"knee1={self.knee1}, knee2={self.knee2}, "
            f"rates={self.rate_zone1}/{self.rate_zone2}/{self.rate_zone3})"
        )


class ExponentialLagPenalty:
    """
    Simple exponential decay: w(lag) = exp(-rate * (lag - 1)).
    No knees -- uniform decay starting from lag 1.
    """
    def __init__(self, rate: float = 0.20):
        if rate < 0:
            raise ValueError(f"rate must be >= 0. Got: {rate}")
        self.rate = rate

    def __call__(self, lag: int) -> float:
        return float(np.exp(-self.rate * (lag - 1)))

    def __repr__(self) -> str:
        return f"ExponentialLagPenalty(rate={self.rate})"


class InverseLagPenalty:
    """
    Inversely proportional decay: w(lag) = 1 / lag^power.
    """
    def __init__(self, power: float = 1.0):
        if power < 0:
            raise ValueError(f"power must be >= 0. Got: {power}")
        self.power = power

    def __call__(self, lag: int) -> float:
        return float(1.0 / max(lag, 1) ** self.power)

    def __repr__(self) -> str:
        return f"InverseLagPenalty(power={self.power})"


class StepLagPenalty:
    """
    Hard filter: weight=1.0 for lag <= max_lag_allowed, weight=0.0 above it.
    """
    def __init__(self, max_lag_allowed: int):
        if max_lag_allowed < 1:
            raise ValueError(f"max_lag_allowed must be >= 1. Got: {max_lag_allowed}")
        self.max_lag_allowed = max_lag_allowed

    def __call__(self, lag: int) -> float:
        return 1.0 if lag <= self.max_lag_allowed else 0.0

    def __repr__(self) -> str:
        return f"StepLagPenalty(max_lag_allowed={self.max_lag_allowed})"


# ── Lag utilities ─────────────────────────────────────────────────────────────

DEFAULT_LAG_PENALTY: LagPenaltyFn = KneeLagPenalty()


def resolve_lag_penalty(
    lag_penalty_fn: Optional[LagPenaltyFn] = None,
    use_lag_penalty: bool = True,
    lag_knee1: int = 4,
    lag_knee2: int = 6,
    lag_rate_zone1: float = 0.02,
    lag_rate_zone2: float = 0.15,
    lag_rate_zone3: float = 0.50,
) -> LagPenaltyFn:
    """
    Resolves which lag penalty function to use.

    Priority:
        1. lag_penalty_fn passed explicitly
        2. use_lag_penalty=False  -> UniformLagPenalty
        3. Knee parameters        -> configured KneeLagPenalty
        4. No argument            -> DEFAULT_LAG_PENALTY
    """
    if lag_penalty_fn is not None:
        return lag_penalty_fn
    if not use_lag_penalty:
        return UniformLagPenalty()
    return KneeLagPenalty(
        knee1=lag_knee1,
        knee2=lag_knee2,
        rate_zone1=lag_rate_zone1,
        rate_zone2=lag_rate_zone2,
        rate_zone3=lag_rate_zone3,
    )


def apply_lag_penalty_to_results(
    results,
    lag_penalty_fn: LagPenaltyFn,
    weight_col: str = 'lag_penalty_score',
    penalize_effect: bool = False,
    drop_zero: bool = False,
) -> 'pd.DataFrame':
    """
    Applies the lag penalty to an already-computed results DataFrame
    (typical use: post-hoc over any external pipeline).

    Parameters:
        results         : DataFrame with a 'lag' column
        lag_penalty_fn  : function (int -> float)
        weight_col      : name of the column created with the weight
        penalize_effect : if True, creates 'penalized_effect', 'ci_lower_pen',
                          'ci_upper_pen' multiplied by the weight
        drop_zero       : removes rows with weight == 0.0
    """
    import pandas as pd

    if results is None or results.empty:
        return results

    out = results.copy()
    if 'lag' not in out.columns:
        raise ValueError("DataFrame has no 'lag' column.")

    weights = out['lag'].apply(lag_penalty_fn)
    out[weight_col] = weights.round(6)

    if penalize_effect:
        if 'causal_effect' in out.columns:
            out['penalized_effect'] = (out['causal_effect'] * weights).round(4)
        if 'ci_95_lower' in out.columns:
            out['ci_lower_pen'] = (out['ci_95_lower'] * weights).round(4)
        if 'ci_95_upper' in out.columns:
            out['ci_upper_pen'] = (out['ci_95_upper'] * weights).round(4)
        if 'penalized_effect' in out.columns:
            out['strength_pen'] = out['penalized_effect'].abs().fillna(0)
            out = out.sort_values('strength_pen', ascending=False).reset_index(drop=True)
    else:
        if 'strength' in out.columns:
            out = out.assign(_sort_key=out['strength'] * weights)
            out = out.sort_values('_sort_key', ascending=False).drop(columns='_sort_key')
            out = out.reset_index(drop=True)

    if drop_zero:
        before = len(out)
        out = out[out[weight_col] > 0.0].reset_index(drop=True)
        removed = before - len(out)
        if removed:
            print(f"  [penalties] {removed} links removed (weight=0 with drop_zero=True)")

    return out


# Alias
apply_penalty_to_results = apply_lag_penalty_to_results


def penalty_table(
    lag_penalty_fn: LagPenaltyFn,
    max_lag: int = 15,
) -> 'pd.DataFrame':
    """
    Returns a DataFrame with the weight of each lag from 1 to max_lag.
    Useful for inspecting and comparing different penalties.
    """
    import pandas as pd
    rows = [
        {'lag': lag, 'weight': round(lag_penalty_fn(lag), 4), 'function': repr(lag_penalty_fn)}
        for lag in range(1, max_lag + 1)
    ]
    return pd.DataFrame(rows)


# =============================================================================
# PART 2 -- CAUSAL RELATIONSHIP PENALTY  (post-DML)
# =============================================================================

class UniformCausalPenalty:
    """
    No penalty -- every pair receives weight 1.0.
    Equivalent to turning off the causal penalty.
    """
    def __call__(self, row: 'pd.Series') -> float:
        return 1.0

    def __repr__(self) -> str:
        return "UniformCausalPenalty()"


class PropertyTypeCausalPenalty:
    """
    Penalizes or favors causal relationships based on the property type
    of the source and/or target.

    Parameters:
        rules : nested dict {source_property: {target_property: float}}
        default_weight : weight for pairs not covered by the rules (default=1.0)

    Example:
        pen = PropertyTypeCausalPenalty(
            rules={
                'I1': {'P1': 1.0, 'P2': 1.0, 'I1': 0.1, 'I2': 0.1},
            },
            default_weight=0.5,
        )
    """
    def __init__(
        self,
        rules: Dict[str, Dict[str, float]],
        default_weight: float = 1.0,
    ):
        self.rules          = rules
        self.default_weight = default_weight

    def __call__(self, row: 'pd.Series') -> float:
        source_property = row['source_property']
        target_property = row['target_property']
        return float(
            self.rules
            .get(source_property, {})
            .get(target_property, self.default_weight)
        )

    def __repr__(self) -> str:
        return (
            f"PropertyTypeCausalPenalty("
            f"n_rules={sum(len(v) for v in self.rules.values())}, "
            f"default={self.default_weight})"
        )


class EntityPairCausalPenalty:
    """
    Directly defines weights for specific entity pairs.

    Parameters:
        weights : dict {(source_entity, target_entity): float}
        default_weight : weight for pairs not listed (default=1.0)

    Example:
        pen = EntityPairCausalPenalty(
            weights={
                ('ENTITY_1', 'ENTITY_3'): 1.5,
                ('ENTITY_2', 'ENTITY_4'): 0.2,
            }
        )
    """
    def __init__(
        self,
        weights: Dict[tuple, float],
        default_weight: float = 1.0,
    ):
        self.weights        = weights
        self.default_weight = default_weight

    def __call__(self, row: 'pd.Series') -> float:
        source_entity = row['source_entity']
        target_entity = row['target_entity']
        return float(
            self.weights.get((source_entity, target_entity), self.default_weight)
        )

    def __repr__(self) -> str:
        return (
            f"EntityPairCausalPenalty("
            f"n_pairs={len(self.weights)}, "
            f"default={self.default_weight})"
        )


class ComposedCausalPenalty:
    """
    Combines multiple CausalPenaltyFn into a single function.
    The final weight is the PRODUCT of all individual weights.

    Example:
        pen = ComposedCausalPenalty([
            PropertyTypeCausalPenalty({'I1': {'P1': 1.5}}),
            EntityPairCausalPenalty({('A', 'B'): 0.5}),
        ])
    """
    def __init__(self, penalties: List[CausalPenaltyFn]):
        if not penalties:
            raise ValueError("penalties cannot be an empty list.")
        self.penalties = penalties

    def __call__(self, row: 'pd.Series') -> float:
        w = 1.0
        for pen in self.penalties:
            w *= pen(row)
        return float(w)

    def __repr__(self) -> str:
        inner = ', '.join(repr(p) for p in self.penalties)
        return f"ComposedCausalPenalty([{inner}])"


# ── Causal penalty utilities ──────────────────────────────────────────────────

DEFAULT_CAUSAL_PENALTY: CausalPenaltyFn = UniformCausalPenalty()


def apply_causal_penalty(
    results: 'pd.DataFrame',
    causal_penalty_fn: CausalPenaltyFn,
    inplace_effect: bool = False,
) -> 'pd.DataFrame':
    """
    Applies a CausalPenaltyFn to the DML results DataFrame.

    For each row of the DataFrame, passes the FULL row (pd.Series) to
    the callable, which can access any field of the DML result:

        w = causal_penalty_fn(row)   # row is a pd.Series with every column

    In addition to the original columns, adds:
        old_sig              : original DML significant (pvalue < 0.05),
                               preserved before any overwrite
        significant        : True if w > 0.0, False if w == 0.0
                               (independent of pvalue -- reflects the
                               decision made by causal_penalty_fn)
        causal_penalty_score : the weight w computed for each pair
        penalized_effect    : causal_effect * w
        ci_lower_pen         : ci_95_lower * w
        ci_upper_pen         : ci_95_upper * w

    If inplace_effect=True, replaces 'causal_effect', 'ci_95_lower' and
    'ci_95_upper' with the penalized values (preserves the original in
    'causal_effect_raw').

    *** NO drop_zero HERE, ON PURPOSE ***
    This function NEVER removes rows from the DataFrame -- it only
    marks significant=False for pairs with weight 0.0. Physically
    removing rows at this stage would erase real competitors BEFORE
    any structural optimization of the graph (Stage 8a, if any) runs,
    contaminating any topology/density analysis and the parent/
    reciprocal-edge competition performed afterwards. The responsibility
    for removing non-significant pairs from the final result belongs to
    whoever orchestrates the pipeline (e.g. pipeline.py, AFTER running
    Stage 8, using the `significant` it has already
    resolved) -- not to this function. If you need to filter by
    significant==False in your own code, filter the returned
    DataFrame yourself, at the right point of your pipeline.

    Parameters:
        results            : DataFrame returned by run_pipeline / build_results
        causal_penalty_fn  : callable (pd.Series) -> float
        inplace_effect     : replaces the original columns with the penalized values

    Returns:
        DataFrame with added columns (same number of input rows),
        ordered by |penalized_effect|.
    """
    import pandas as pd

    if results is None or results.empty:
        return results

    required = {'source_entity', 'source_property', 'target_entity', 'target_property'}
    missing  = required - set(results.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    out = results.copy()

    # ── Preserve the original DML significant before any overwrite ─────────
    out['old_sig'] = out['significant']

    # ── Compute weights row by row ────────────────────────────────────────────
    weights = out.apply(causal_penalty_fn, axis=1).astype(float)

    # ── Redefine significant: True if weight > 0.0 ──────────────────────────
    out['significant'] = weights > 0.0

    # ── Penalty columns ───────────────────────────────────────────────────────
    out['causal_penalty_score'] = weights.round(6)

    if 'causal_effect' in out.columns:
        out['penalized_effect'] = (out['causal_effect'] * weights).round(4)
    if 'ci_95_lower' in out.columns:
        out['ci_lower_pen'] = (out['ci_95_lower'] * weights).round(4)
    if 'ci_95_upper' in out.columns:
        out['ci_upper_pen'] = (out['ci_95_upper'] * weights).round(4)

    if inplace_effect and 'causal_effect' in out.columns:
        out['causal_effect_raw'] = out['causal_effect']
        out['causal_effect']     = out['penalized_effect']
        if 'ci_95_lower' in out.columns:
            out['ci_95_lower'] = out['ci_lower_pen']
        if 'ci_95_upper' in out.columns:
            out['ci_95_upper'] = out['ci_upper_pen']

    # ── Reorder by penalized strength ─────────────────────────────────────────
    sort_col = 'penalized_effect' if 'penalized_effect' in out.columns else 'causal_effect'
    if sort_col in out.columns:
        out = out.assign(_sort_key=out[sort_col].abs().fillna(0))
        out = out.sort_values('_sort_key', ascending=False).drop(columns='_sort_key')
        out = out.reset_index(drop=True)

    return out


def causal_penalty_table(
    causal_penalty_fn: CausalPenaltyFn,
    pairs: List[tuple],
) -> 'pd.DataFrame':
    """
    Returns a DataFrame with the weight computed for each pair (so, sp, to, tp).
    Useful for inspecting the penalty before applying it to the pipeline.
    """
    import pandas as pd
    import numpy as np

    rows = []
    for (so, sp, to, tp) in pairs:
        row = pd.Series({
            'source_entity':     so,
            'source_property':   sp,
            'target_entity':     to,
            'target_property':   tp,
            'lag':               np.nan,
            'lag_penalty_score': np.nan,
            'transfer_entropy':  np.nan,
            'filter_method':            None,
            'causal_effect':     np.nan,
            'standard_error':    np.nan,
            'direction':           None,
            'ci_95_lower':       np.nan,
            'ci_95_upper':       np.nan,
            'pvalue':            np.nan,
            'significant':     None,
            'old_sig':           None,
            'strength':             np.nan,
            'dml_model_used':   None,
        })
        w = causal_penalty_fn(row)
        rows.append({
            'source_entity':   so,
            'source_property': sp,
            'target_entity':   to,
            'target_property': tp,
            'weight':      round(w, 6),
            'significant': w > 0.0,
            'function':    repr(causal_penalty_fn),
        })
    return pd.DataFrame(rows)


# =============================================================================
# __all__ -- public exports of the module
# =============================================================================

__all__ = [
    # ── Types ────────────────────────────────────────────────────────────────
    'LagPenaltyFn',
    'CausalPenaltyFn',
    # ── Lag penalty ──────────────────────────────────────────────────────────
    'UniformLagPenalty',
    'KneeLagPenalty',
    'ExponentialLagPenalty',
    'InverseLagPenalty',
    'StepLagPenalty',
    # ── Causal penalty ───────────────────────────────────────────────────────
    'UniformCausalPenalty',
    'PropertyTypeCausalPenalty',
    'EntityPairCausalPenalty',
    'ComposedCausalPenalty',
    # ── Utilities ────────────────────────────────────────────────────────────
    'DEFAULT_LAG_PENALTY',
    'DEFAULT_CAUSAL_PENALTY',
    'resolve_lag_penalty',
    'apply_lag_penalty_to_results',
    'apply_penalty_to_results',
    'apply_causal_penalty',
    'penalty_table',
    'causal_penalty_table',
]

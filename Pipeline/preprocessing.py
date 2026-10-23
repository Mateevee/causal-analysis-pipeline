"""
preprocessing.py — Stage 2: limited interpolation + z-score normalization +
structural-zero detection.

Order of the steps:
    1. Structural-zero detection — contiguous runs of exactly
       zero values with length >= zero_run_min_length are flagged as
       valid-flagged. Interpolation (step 2) only acts on NaN, so real
       zeros are never overwritten by it — this step only records/flags
       them.
    2. Linear interpolation limited to gaps of up to `gap_limit`
       consecutive points — longer gaps are left as NaN (no data is
       invented via a linear trend fabricated over an extended void).
    3. Z-score normalization over the remaining valid values.

`gap_limit` and `zero_run_min_length` are generic heuristics (not
calibrated on any specific dataset in this work).
"""

from typing import Optional

import numpy as np
import pandas as pd


def _interpolate_short_gaps(vals: np.ndarray, gap_limit: Optional[int]) -> np.ndarray:
    """
    Linearly interpolates (or, at the edges, propagates the nearest valid
    neighbor) ONLY NaN runs of length <= gap_limit — all or nothing, never
    partial. Longer runs are left entirely as NaN.

    gap_limit=None disables the limit: interpolates ANY gap, no matter
    how long.
    """
    if gap_limit is None:
        return pd.Series(vals).interpolate(
            method='linear', limit_direction='both'
        ).values.astype(float)

    n = len(vals)
    out = vals.copy()
    is_nan = np.isnan(vals)

    i = 0
    while i < n:
        if not is_nan[i]:
            i += 1
            continue
        j = i
        while j < n and is_nan[j]:
            j += 1
        run_len = j - i
        if run_len <= gap_limit:
            left_val = vals[i - 1] if i > 0 else None
            right_val = vals[j] if j < n else None
            if left_val is not None and right_val is not None:
                out[i:j] = np.interp(np.arange(i, j), [i - 1, j], [left_val, right_val])
            elif left_val is not None:
                out[i:j] = left_val
            elif right_val is not None:
                out[i:j] = right_val
            # no valid neighbor at all (entire series is NaN) -> stays NaN
        i = j
    return out


def preprocess_series(s: pd.Series, gap_limit: Optional[int] = 3,
                       zero_run_min_length: Optional[int] = 2) -> tuple:
    """Limited interpolation + z-score normalization + structural-zero flag.

    gap_limit=None      : disables the gap limit (interpolates everything).
    zero_run_min_length=None : disables structural-zero detection (no position
                           is flagged; the values themselves do not
                           change, since zeros are never touched by
                           interpolation anyway).

    Returns (vals, zero_run_flags): two arrays the same size as `s`.
    """
    vals_raw = s.copy().astype(float).values

    # (1) structural-zero detection
    flags_zero_run = np.zeros(len(vals_raw), dtype=bool)
    if zero_run_min_length is not None:
        is_zero = np.isclose(vals_raw, 0.0)
        run_start = None
        for i, z in enumerate(is_zero):
            if z and run_start is None:
                run_start = i
            elif not z and run_start is not None:
                if i - run_start >= zero_run_min_length:
                    flags_zero_run[run_start:i] = True
                run_start = None
        if run_start is not None and len(vals_raw) - run_start >= zero_run_min_length:
            flags_zero_run[run_start:] = True

    # (2) limited interpolation — extended gaps are left as NaN, not invented
    vals = _interpolate_short_gaps(vals_raw, gap_limit)

    # (3) z-score normalization, ignoring remaining NaN
    mask = ~np.isnan(vals)
    if mask.sum() > 1:
        mu, sigma = vals[mask].mean(), vals[mask].std()
        if sigma > 1e-10:
            vals[mask] = (vals[mask] - mu) / sigma

    return vals, flags_zero_run


def preprocess_all(df_wide: pd.DataFrame, gap_limit: Optional[int] = 3,
                    zero_run_min_length: Optional[int] = 2) -> tuple:
    """Preprocesses all series. Returns (processed, zero_run_flags),
    two dicts {column: array} of the same shape."""
    processed, flags = {}, {}
    for col in df_wide.columns:
        processed[col], flags[col] = preprocess_series(
            df_wide[col], gap_limit=gap_limit, zero_run_min_length=zero_run_min_length,
        )
    print(f"  {len(processed)} series preprocessed.")

    if zero_run_min_length is not None:
        n_obs_zero_run = sum(int(f.sum()) for f in flags.values())
        affected_series = sorted(col for col, f in flags.items() if f.any())
        if n_obs_zero_run > 0:
            print(f"  Structural zeros (run >= {zero_run_min_length}): "
                  f"{n_obs_zero_run} observations in {len(affected_series)} of "
                  f"{len(processed)} series — kept as valid, not filled in "
                  f"(see 'zero_run_flags' returned by this function).")
            if len(affected_series) <= 10:
                print(f"    affected series: {affected_series}")
        else:
            print(f"  Structural zeros (run >= {zero_run_min_length}): "
                  f"none detected in this run.")

    return processed, flags

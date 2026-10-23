"""
property_scanner.py — Stage 1 (optional): active-property detection.

Decides, for each (entity, property) pair, whether the property is
characteristic of that entity, and allows reducing the pair space before
the causal filter. Independent of any other stage.
"""

import pandas as pd
from typing import Optional


def scan_entity_properties(
    df: pd.DataFrame,
    date_col: str = "Date",
    entity_col: str = "Name",
    property_cols: Optional[list] = None,
    zero_threshold: float = 1e-3,
    min_active_pct: float = 0.09,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        entity, property, n_total, n_active, pct_active, mean_active,
        std_active, characteristic
    """
    if property_cols is None:
        property_cols = [c for c in df.columns if c not in [date_col, entity_col]]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col])
    for col in property_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    entities = sorted(df[entity_col].dropna().unique().tolist())
    rows  = []

    for entity in entities:
        sub = df[df[entity_col] == entity]
        n_total = len(sub)
        if n_total == 0:
            continue

        for prop in property_cols:
            vals = sub[prop].dropna()
            n_valid   = len(vals)
            n_active  = int((vals.abs() > zero_threshold).sum())
            pct_active = n_active / n_valid if n_valid > 0 else 0.0

            active_vals = vals[vals.abs() > zero_threshold]
            mean_active = float(active_vals.mean()) if len(active_vals) > 0 else 0.0
            std_active  = float(active_vals.std())  if len(active_vals) > 1 else 0.0

            characteristic = pct_active >= min_active_pct

            rows.append({
                'entity':         entity,
                'property':       prop,
                'n_total':        n_total,
                'n_valid':        n_valid,
                'n_active':       n_active,
                'pct_active':     round(pct_active, 4),
                'mean_active':    round(mean_active, 4),
                'std_active':     round(std_active, 4),
                'characteristic': characteristic,
            })

    result = pd.DataFrame(rows)

    if verbose and not result.empty:
        print("\n  Characteristic properties by entity:")
        print(f"  {'ENTITY':<20} {'PROPERTY':<10} {'% ACTIVE':>8}  STATUS")
        print("  " + "-" * 52)
        for _, row in result.iterrows():
            status = "ACTIVE" if row['characteristic'] else "inactive"
            marker = "+" if row['characteristic'] else " "
            print(f"  {marker} {row['entity']:<19} {row['property']:<10} "
                  f"{row['pct_active']*100:>7.1f}%  {status}")

        n_characteristic = result['characteristic'].sum()
        n_total_pairs = len(result)
        print(f"\n  {n_characteristic} of {n_total_pairs} (entity, property) pairs are characteristic "
              f"({n_characteristic/n_total_pairs*100:.1f}%)")

    return result


def get_active_pairs(scan_result: pd.DataFrame) -> list:
    """From scan_entity_properties, returns the active 'entity__property' pairs."""
    active = scan_result[scan_result['characteristic']]
    return [f"{row['entity']}__{row['property']}" for _, row in active.iterrows()]

"""
pivot.py — Stage 0: structuring (loading and pivoting).

Transforms the long DataFrame (Date, Name, property1, property2, ...) into a
wide matrix where each column is an "ENTITY__property" pair. Does not depend
on any other pipeline stage.
"""

import pandas as pd
from typing import Optional


def load_and_pivot(
    df: pd.DataFrame,
    date_col: str = "Date",
    entity_col: str = "Name",
    property_cols: Optional[list] = None,
) -> tuple:
    """
    Returns:
        df_wide    : DataFrame (timesteps x n_series), columns = "entity__property"
        pairs      : list of "entity__property" strings
        entities   : list of entity names
        properties : list of properties
        dates      : DatetimeIndex
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values([date_col, entity_col])

    if property_cols is None:
        property_cols = [c for c in df.columns if c not in [date_col, entity_col]]

    for col in property_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    entities   = sorted(df[entity_col].dropna().unique().tolist())
    properties = property_cols

    pivoted = df.pivot_table(
        index=date_col,
        columns=entity_col,
        values=property_cols,
        aggfunc='mean',
    )
    # Flatten MultiIndex: (property, entity) -> "entity__property"
    pivoted.columns = [f"{w}__{f}" for f, w in pivoted.columns]
    pivoted = pivoted.sort_index()

    pairs = list(pivoted.columns)

    print(f"  Entities: {len(entities)}")
    print(f"  Properties: {properties}")
    print(f"  Series (entity x property): {len(pairs)}")
    print(f"  Timesteps: {len(pivoted)}")
    print(f"  Possible directed pairs: {len(pairs) * (len(pairs) - 1):,}")

    return pivoted, pairs, entities, properties, pivoted.index

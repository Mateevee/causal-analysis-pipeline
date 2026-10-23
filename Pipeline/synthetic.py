"""
synthetic.py — utility: synthetic DataFrame generator (for testing the
pipeline without real data), without pre-classification of entities.
"""

import numpy as np
import pandas as pd
from typing import Optional


def make_synthetic_df(n_entities: int = 6, T: int = 250,
                       properties: Optional[list] = None, seed: int = 42):
    """Generates synthetic (df, ground_truth)."""
    if properties is None:
        properties = ['a', 'b', 'c', 'd']

    rng   = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-01', periods=T, freq='W')
    entity_names = [f"ENTITY_{i+1:02d}" for i in range(n_entities)]

    connections = []
    for _ in range(max(4, n_entities // 2)):
        i = rng.integers(0, n_entities)
        j = rng.integers(0, n_entities)
        if i == j:
            continue
        fi = properties[rng.integers(0, len(properties))]
        fj = properties[rng.integers(0, len(properties))]
        theta = rng.uniform(0.3, 0.7) * rng.choice([-1, 1])
        tau   = int(rng.integers(1, min(8, T // 10)))
        connections.append((entity_names[i], fi, entity_names[j], fj, theta, tau))

    series = {}
    for w in entity_names:
        for f in properties:
            base = rng.normal(50, 15, T)
            if rng.random() > 0.5:
                s = int(rng.integers(T // 4, 3 * T // 4))
                base[s:s + int(rng.integers(3, 12))] = 0.0
            series[(w, f)] = np.maximum(base, 0.0)

    for (wi, fi, wj, fj, theta, tau) in connections:
        src = series[(wi, fi)].copy()
        for t in range(tau, T):
            series[(wj, fj)][t] += theta * src[t - tau]

    rows = []
    for i, date in enumerate(dates):
        for w in entity_names:
            row = {'Date': date, 'Name': w}
            for f in properties:
                row[f] = round(float(series[(w, f)][i]), 3)
            rows.append(row)

    gt = pd.DataFrame(connections,
                      columns=['source_entity', 'source_property', 'target_entity',
                                'target_property', 'theta_true', 'lag_true'])
    return pd.DataFrame(rows), gt


# Alias
_make_synthetic_df = make_synthetic_df

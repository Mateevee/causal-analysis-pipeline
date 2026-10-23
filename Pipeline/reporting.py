"""
reporting.py — utility: formatted console report.
"""

import pandas as pd


def print_report(results: pd.DataFrame, only_significant: bool = True):
    """Prints a formatted report."""
    if results.empty:
        print("No results.")
        return

    df = results[results['significant']] if only_significant else results

    print("\n" + "=" * 65)
    label = "SIGNIFICANT CAUSAL EFFECTS" if only_significant else "ALL EFFECTS"
    print(f"{label}  ({len(df)} pairs)")
    print("=" * 65)

    base_cols = ['source_entity', 'source_property', 'target_entity', 'target_property',
                 'lag', 'transfer_entropy', 'causal_effect', 'standard_error', 'direction',
                 'ci_95_lower', 'ci_95_upper', 'pvalue', 'significant']
    dist_cols  = [c for c in ['distance_m', 'penalty_factor', 'penalized_effect']
                  if c in df.columns]
    cols = base_cols + dist_cols
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_rows', 200)
    print(df[cols].to_string(index=False))

    sig = df[df['significant']]
    pos = len(sig[sig['direction'] == 'positive'])
    neg = len(sig[sig['direction'] == 'negative'])
    print(f"\n  Positive: {pos} | Negative: {neg} | Total significant: {len(sig)}")

    if len(sig) > 0:
        print("\n--- Top 10 strongest effects ---")
        strength_col = 'strength_pen' if 'strength_pen' in sig.columns else 'strength'
        extra = ['distance_m', 'penalty_factor', 'penalized_effect']
        extra = [c for c in extra if c in sig.columns]
        top_cols = ['source_entity', 'source_property', 'target_entity', 'target_property',
                    'causal_effect', 'direction', 'lag', 'pvalue'] + extra
        top = sig.nlargest(10, strength_col)[top_cols]
        print(top.to_string(index=False))

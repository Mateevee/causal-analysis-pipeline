"""
consolidate_summaries.py
=========================
Reads every validation_summary.csv inside a results directory and prints a
consolidated report per scenario, including soft metrics.

main.py and the other entry points call consolidate(OUTPUT_DIR) on their own
run; RESULTS_DIR below is only used when this file is run directly.

Usage (from the repository root):
    python Auxiliary/consolidate_summaries.py
"""

import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION  <- edit here
# =============================================================================

RESULTS_DIR = "./Results/Run_20261011_132346"

# Hard columns displayed
HARD_COLUMNS = ["precision", "recall", "f1", "TP", "FP", "FN", "total"]

# Soft columns displayed (present only if validation_summary has them)
SOFT_COLUMNS = ["precision_soft", "recall_soft", "f1_soft",
                "TP_hard", "TP_soft", "FP_hard", "FN"]

# =============================================================================

HARD_COL_WIDTHS = {
    "precision": 9, "recall": 7, "f1": 7,
    "TP": 5, "FP": 5, "FN": 5, "total": 7,
}

SOFT_COL_WIDTHS = {
    "precision_soft": 14, "recall_soft": 11, "f1_soft": 8,
    "TP_hard": 7, "TP_soft": 7, "FP_hard": 7, "FN": 5,
}


def _fmt(col, val, col_widths, average=False):
    w = col_widths[col]
    int_columns = {"TP", "FP", "FN", "total", "TP_hard", "TP_soft", "FP_hard"}
    if col in int_columns:
        return f"{val:>{w}.1f}" if average else f"{int(val):>{w}}"
    return f"{val:>{w}.4f}"


def _header(model_width, columns, col_widths):
    parts = [f"  {'MODEL':<{model_width}}"]
    for col in columns:
        parts.append(f"| {col.upper():^{col_widths[col]}} ")
    return "".join(parts)


def _sep(model_width, columns, col_widths):
    parts = ["-" * (model_width + 2)]
    for col in columns:
        parts.append("+" + "-" * (col_widths[col] + 2))
    return "".join(parts)


def _row(model_width, name, row, columns, col_widths, average=False):
    parts = [f"  {name:<{model_width}}"]
    for col in columns:
        val = row.get(col, float("nan")) if not average else row[col]
        parts.append(f"| {_fmt(col, val, col_widths, average)} ")
    return "".join(parts)


def consolidate(results_dir: str):
    root = Path(results_dir)

    scenario_dirs = sorted([
        d for d in root.iterdir()
        if d.is_dir() and (d / "validation_summary.csv").exists()
    ])

    if not scenario_dirs:
        print(f"[WARNING] No scenario with validation_summary.csv found in '{root}'.")
        return

    SEP = "=" * 72

    print(f"\n{SEP}")
    print(f"  CONSOLIDATED — {root.name}  ({len(scenario_dirs)} scenario(s))")
    print(f"  HARD metrics: strict GT  |  SOFT metrics: FP with consensus = TP_soft")
    print(f"  total = TP + FP + FN  (unique pairs evaluated by each model)")
    print(SEP)

    # Accumulated separately per block (hard / soft)
    hard_summary: list[pd.DataFrame] = []
    soft_summary: list[pd.DataFrame] = []

    for folder in scenario_dirs:
        name = folder.name
        df = pd.read_csv(folder / "validation_summary.csv")
        df_ok = df[df["status"] == "ok"].copy()

        print(f"\n  ▸ {name}")

        if df_ok.empty:
            print("    (no model with a valid result)")
            continue

        df_ok["total"] = df_ok["TP"] + df_ok["FP"] + df_ok["FN"]

        # ---- HARD ----
        hard_columns_present = [c for c in HARD_COLUMNS if c in df_ok.columns]
        if hard_columns_present:
            w = max(df_ok["model"].str.len().max(), 6)
            print(f"\n  [HARD — strict ground truth]")
            print("  " + _sep(w, hard_columns_present, HARD_COL_WIDTHS))
            print(_header(w, hard_columns_present, HARD_COL_WIDTHS))
            print("  " + _sep(w, hard_columns_present, HARD_COL_WIDTHS))
            for _, row in df_ok.iterrows():
                print(_row(w, row["model"], row, hard_columns_present, HARD_COL_WIDTHS))
            print("  " + _sep(w, hard_columns_present, HARD_COL_WIDTHS))
            if "f1" in df_ok.columns:
                best = df_ok.loc[df_ok["f1"].idxmax()]
                print(f"  ★ Best F1 (hard): {best['model']}  "
                      f"(F1={best['f1']:.4f}  Prec={best['precision']:.4f}  "
                      f"Rec={best['recall']:.4f})")
            tmp = df_ok.copy(); tmp["scenario"] = name
            hard_summary.append(tmp)

        # ---- SOFT ----
        soft_columns_present = [c for c in SOFT_COLUMNS if c in df_ok.columns]
        if soft_columns_present:
            # find out which threshold was used (may vary by scenario)
            threshold_label = ""
            if "soft_threshold" in df_ok.columns:
                threshold_values = df_ok["soft_threshold"].dropna().unique()
                if len(threshold_values) == 1:
                    threshold_label = f" — threshold={threshold_values[0]*100:.0f}%"

            w = max(df_ok["model"].str.len().max(), 6)
            print(f"\n  [SOFT{threshold_label}]")
            print("  " + _sep(w, soft_columns_present, SOFT_COL_WIDTHS))
            print(_header(w, soft_columns_present, SOFT_COL_WIDTHS))
            print("  " + _sep(w, soft_columns_present, SOFT_COL_WIDTHS))
            for _, row in df_ok.iterrows():
                print(_row(w, row["model"], row, soft_columns_present, SOFT_COL_WIDTHS))
            print("  " + _sep(w, soft_columns_present, SOFT_COL_WIDTHS))
            if "f1_soft" in df_ok.columns:
                best = df_ok.loc[df_ok["f1_soft"].idxmax()]
                print(f"  ★ Best F1 (soft): {best['model']}  "
                      f"(F1_soft={best['f1_soft']:.4f}  "
                      f"Prec_soft={best['precision_soft']:.4f}  "
                      f"Rec_soft={best['recall_soft']:.4f})")
            tmp = df_ok.copy(); tmp["scenario"] = name
            soft_summary.append(tmp)
        else:
            print("    (soft columns missing — run the updated validation script to generate them)")

    # ------------------------------------------------------------------
    # GLOBAL AVERAGES
    # ------------------------------------------------------------------
    def _print_global_average(title, frames, columns, col_widths, n_scenarios):
        if not frames:
            return
        df_all = pd.concat(frames, ignore_index=True)
        numeric_columns = [c for c in columns if c in df_all.columns]
        if not numeric_columns:
            return
        averages = df_all.groupby("model")[numeric_columns].mean()

        print(f"\n{SEP}")
        print(f"  GLOBAL AVERAGE [{title}]  (all {n_scenarios} scenarios)")
        print(SEP)

        w = max(averages.index.str.len().max(), 6)
        print("  " + _sep(w, numeric_columns, col_widths))
        print(_header(w, numeric_columns, col_widths))
        print("  " + _sep(w, numeric_columns, col_widths))
        for model, row in averages.iterrows():
            print(_row(w, model, row, numeric_columns, col_widths, average=True))
        print("  " + _sep(w, numeric_columns, col_widths))

        f1_col = "f1" if title == "HARD" else "f1_soft"
        if f1_col in averages.columns:
            print(f"\n  FINAL RANKING [{title}] (average F1)")
            print(f"  {'-' * 40}")
            ranking = averages[f1_col].sort_values(ascending=False)
            for i, (model, f1) in enumerate(ranking.items(), 1):
                print(f"  {i:>2}. {model:<{w}}  {f1_col}={f1:.4f}")

    _print_global_average("HARD", hard_summary, HARD_COLUMNS, HARD_COL_WIDTHS, len(scenario_dirs))
    _print_global_average("SOFT", soft_summary, SOFT_COLUMNS, SOFT_COL_WIDTHS, len(scenario_dirs))

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    consolidate(RESULTS_DIR)

"""
FNChecker.py
==========================
For each scenario inside RESULTS_DIR:
  - Reads the scenario's ground_truth.csv
  - For each model (subfolder with results.csv), identifies the FNs
  - Generates a fn_consolidated.csv at the same level as ground_truth.csv

Column absent_in:
    "ALL"             — the FN pair is absent from every model in the scenario
    "model_a, ..."    — list of models where the pair was absent (FN)

Column fn_severity:
    "CRITICAL"        — absent in >= FN_CRITICAL_THRESHOLD of the models
                        (no model, or almost none, detected it — a real miss)
    "PARTIAL"         — absent in some models but detected by others

Column n_models_detected / fraction_detected:
    How many models GOT this pair right (complement of the FN),
    useful for understanding whether it's a hard or easy case.

main.py and the other entry points call export_fn(OUTPUT_DIR) on their own
run; RESULTS_DIR below is only used when this file is run directly.

Usage (from the repository root):
    python Validation/FNChecker.py
"""

import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION — edit here
# =============================================================================

RESULTS_DIR = "./Results/Run_20261011_132346"

# Minimum fraction of models where the pair is ABSENT for it to be "CRITICAL".
# 1.0 = absent in ALL (no model detected it)
# 0.5 = absent in half or more of the models
FN_CRITICAL_THRESHOLD: float = 1.0

# =============================================================================


def _canonical_pair(a: str, b: str):
    if a == b:
        return None
    return frozenset([a, b])


def _gt_pairs(gt: pd.DataFrame) -> dict:
    """Returns dict: frozenset -> GT row."""
    pairs = {}
    causal_col = "causal_gt" if "causal_gt" in gt.columns else None
    rows = gt[gt[causal_col].astype(bool)] if causal_col else gt
    for _, row in rows.iterrows():
        p = _canonical_pair(str(row["source_entity"]).strip(), str(row["target_entity"]).strip())
        if p:
            pairs[p] = row
    return pairs


def process_scenario(folder: Path):
    name = folder.name

    gt = pd.read_csv(folder / "ground_truth.csv", skipinitialspace=True)
    gt_causal_pairs = _gt_pairs(gt)

    model_dirs = sorted([
        d for d in folder.iterdir()
        if d.is_dir() and (d / "results.csv").exists()
    ])

    if not model_dirs:
        print(f"  [WARNING] {name}: no model with results.csv — skipping.")
        return

    all_models = {d.name for d in model_dirs}
    n_models = len(all_models)

    # For each model, collect the pairs it predicted
    predicted_by_model: dict[str, set] = {}
    for model_folder in model_dirs:
        model_name = model_folder.name
        df = pd.read_csv(model_folder / "results.csv", skipinitialspace=True)
        predicted = set()
        if not df.empty:
            for _, row in df.iterrows():
                key = _canonical_pair(
                    str(row["source_entity"]).strip(),
                    str(row["target_entity"]).strip()
                )
                if key is not None:
                    predicted.add(key)
        predicted_by_model[model_name] = predicted

    # FN: pair is in the GT but absent from at least one model's prediction
    record: dict = {}

    for key, row_gt in gt_causal_pairs.items():
        for model_name, predicted in predicted_by_model.items():
            if key not in predicted:
                a, b = sorted(key)
                row_norm = row_gt.copy()
                row_norm["source_entity"] = a
                row_norm["target_entity"] = b

                if key not in record:
                    record[key] = {"row": row_norm, "models_absent": set()}

                record[key]["models_absent"].add(model_name)

    if not record:
        print(f"  {name}: no FN found.")
        return

    min_absent = FN_CRITICAL_THRESHOLD * n_models

    rows = []
    for key, info in sorted(record.items(), key=lambda x: sorted(x[0])):
        row = info["row"].copy()
        absent    = info["models_absent"]
        n_absent  = len(absent)
        n_detected = n_models - n_absent

        row["absent_in"]           = "ALL" if absent >= all_models else ", ".join(sorted(absent))
        row["n_models_absent"]     = n_absent
        row["n_models_detected"]   = n_detected
        row["fraction_absent"]     = round(n_absent   / n_models, 4)
        row["fraction_detected"]   = round(n_detected / n_models, 4)
        row["fn_severity"]         = "CRITICAL" if n_absent >= min_absent else "PARTIAL"

        rows.append(row)

    df_out = pd.DataFrame(rows).reset_index(drop=True)

    extra_cols = ["absent_in", "n_models_absent", "n_models_detected",
                  "fraction_absent", "fraction_detected", "fn_severity"]
    base_cols = [c for c in df_out.columns if c not in extra_cols]
    df_out = df_out[base_cols + extra_cols]

    output_path = folder / "fn_consolidated.csv"
    df_out.to_csv(output_path, index=False)

    total      = len(df_out)
    in_all     = (df_out["absent_in"] == "ALL").sum()
    in_one     = ((df_out["absent_in"].str.count(",") == 0) & (df_out["absent_in"] != "ALL")).sum()
    n_critical = (df_out["fn_severity"] == "CRITICAL").sum()
    n_partial  = (df_out["fn_severity"] == "PARTIAL").sum()

    threshold_label = f"{FN_CRITICAL_THRESHOLD*100:.0f}%"
    print(f"  {name}: {total} unique FNs  |  {in_all} in ALL  |  {in_one} in only 1 model")
    print(f"    → Severity ({threshold_label}): {n_critical} CRITICAL  |  {n_partial} PARTIAL  →  {output_path}")


def export_fn(results_dir: str):
    root = Path(results_dir)

    scenarios = sorted([
        d for d in root.iterdir()
        if d.is_dir() and (d / "ground_truth.csv").exists()
    ])

    if not scenarios:
        print(f"[WARNING] No scenario with ground_truth.csv found in '{root}'.")
        return

    threshold_label = f"{FN_CRITICAL_THRESHOLD*100:.0f}%"
    print(f"\nProcessing {len(scenarios)} scenario(s) in '{root}'...")
    print(f"FN critical threshold: {threshold_label} absence\n")
    for folder in scenarios:
        process_scenario(folder)
    print("\nDone.")


if __name__ == "__main__":
    export_fn(RESULTS_DIR)

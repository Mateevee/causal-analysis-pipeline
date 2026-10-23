"""
FPChecker.py
==========================
For each scenario inside RESULTS_DIR:
  - Reads the scenario's ground_truth.csv
  - For each model (subfolder with results.csv), identifies the FPs
  - Generates a fp_consolidated.csv at the same level as ground_truth.csv

Column present_in:
    "ALL"             — the FP pair appears in every model in the scenario
    "model_a, ..."    — list of models where the pair appeared as an FP

Column soft_classification:
    "TP_soft"         — pair present in >= SOFT_THRESHOLD of the models
                        (enough evidence to consider it a true positive)
    "FP_hard"         — pair below the threshold, considered a real false positive

main.py and the other entry points call export_fp(OUTPUT_DIR) on their own
run; RESULTS_DIR below is only used when this file is run directly.

Usage (from the repository root):
    python Validation/FPChecker.py
"""

import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION — edit here
# =============================================================================

RESULTS_DIR = "./Results/Run_20261011_132346"

# Minimum fraction of models that must predict a pair (outside the GT)
# for it to be classified as TP_soft.
# 1.0 = every model agrees
# 0.5 = simple majority
SOFT_THRESHOLD: float = 1.0

# =============================================================================


def _canonical_pair(a: str, b: str):
    if a == b:
        return None
    return frozenset([a, b])


def _gt_pairs(gt: pd.DataFrame) -> set:
    pairs = set()
    causal_col = "causal_gt" if "causal_gt" in gt.columns else None
    rows = gt[gt[causal_col].astype(bool)] if causal_col else gt
    for _, row in rows.iterrows():
        p = _canonical_pair(str(row["source_entity"]).strip(), str(row["target_entity"]).strip())
        if p:
            pairs.add(p)
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

    # key -> {"row": Series, "models": set()}
    record: dict = {}

    for model_folder in model_dirs:
        model_name = model_folder.name
        df = pd.read_csv(model_folder / "results.csv", skipinitialspace=True)

        if df.empty:
            continue

        for _, row in df.iterrows():
            key = _canonical_pair(
                str(row["source_entity"]).strip(),
                str(row["target_entity"]).strip()
            )
            if key is None:
                continue

            if key in gt_causal_pairs:
                continue

            a, b = sorted(key)
            row_norm = row.copy()
            row_norm["source_entity"] = a
            row_norm["target_entity"] = b

            if key not in record:
                record[key] = {"row": row_norm, "models": set()}

            record[key]["models"].add(model_name)

    if not record:
        print(f"  {name}: no FP found.")
        return

    min_models = SOFT_THRESHOLD * n_models

    rows = []
    for key, info in sorted(record.items(), key=lambda x: sorted(x[0])):
        row = info["row"].copy()
        present = info["models"]
        n_present = len(present)

        row["present_in"] = "ALL" if present >= all_models else ", ".join(sorted(present))
        row["n_models_present"] = n_present
        row["model_fraction"] = round(n_present / n_models, 4)
        row["soft_classification"] = "TP_soft" if n_present >= min_models else "FP_hard"

        rows.append(row)

    df_out = pd.DataFrame(rows).reset_index(drop=True)

    # Ensure the extra columns come last
    extra_cols = ["present_in", "n_models_present", "model_fraction", "soft_classification"]
    base_cols = [c for c in df_out.columns if c not in extra_cols]
    df_out = df_out[base_cols + extra_cols]

    output_path = folder / "fp_consolidated.csv"
    df_out.to_csv(output_path, index=False)

    total      = len(df_out)
    in_all     = (df_out["present_in"] == "ALL").sum()
    in_one     = ((df_out["present_in"].str.count(",") == 0) & (df_out["present_in"] != "ALL")).sum()
    n_tp_soft  = (df_out["soft_classification"] == "TP_soft").sum()
    n_fp_hard  = (df_out["soft_classification"] == "FP_hard").sum()

    threshold_label = f"{SOFT_THRESHOLD*100:.0f}%"
    print(f"  {name}: {total} unique FPs  |  {in_all} in ALL  |  {in_one} in only 1 model")
    print(f"    → Soft ({threshold_label}): {n_tp_soft} TP_soft  |  {n_fp_hard} FP_hard  →  {output_path}")


def export_fp(results_dir: str):
    root = Path(results_dir)

    scenarios = sorted([
        d for d in root.iterdir()
        if d.is_dir() and (d / "ground_truth.csv").exists()
    ])

    if not scenarios:
        print(f"[WARNING] No scenario with ground_truth.csv found in '{root}'.")
        return

    threshold_label = f"{SOFT_THRESHOLD*100:.0f}%"
    print(f"\nProcessing {len(scenarios)} scenario(s) in '{root}'...")
    print(f"Soft threshold: {threshold_label} of the models\n")
    for folder in scenarios:
        process_scenario(folder)
    print("\nDone.")


if __name__ == "__main__":
    export_fp(RESULTS_DIR)

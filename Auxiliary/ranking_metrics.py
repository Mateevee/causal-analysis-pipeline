"""
ranking_metrics.py
==================
AUROC, AUPRC and SHD of every model of an EXISTING run, without re-running
any model.

PAIRS. Same pair identity and comparison universe as main.validate_dataset():
entity level, directional pairs (A->B != B->A), intra-entity pairs counted.
The scored set of a scenario is every pair of ground_truth.csv, plus any
predicted pair outside it (a negative here, as it is an FP there). The
TP/FP/FN/TN recomputed by this script are checked against
validation_summary.csv and the number of mismatches is reported.

SCORE. AUROC and AUPRC need a score per pair, and it depends on the model
(column 'score_used' of the per-scenario file):

    penalized_effect  models whose original.csv has the PENALIZED_COL column
                      (the patch): max |penalized_effect| over the rows of the
                      pair (every lag and property); a pair the model never
                      scored gets 0.
    two_tier          every other model. They save a thresholded graph, not a
                      score matrix, so the ranking is built in tiers:
                        tier 2  pair is in results.csv (the final graph)
                        tier 1  pair is in original.csv but was not kept
                        tier 0  pair never scored (tied at the bottom)
                      and, inside a tier, by 1 - min(pvalue) over the rows of
                      the pair in original.csv.

Two sensitivity checks of that choice are written next to auroc/auprc:
    *_two_tier      the two-tier score for EVERY model (patch included)
    *_pvalue_only   1 - min(pvalue) alone, ignoring the final graph

SHD. Structural Hamming Distance between the final graph and the causal GT:
    shd         a reversed edge counts as ONE edit
    shd_double  a reversed edge counts as TWO edits (= FP + FN)
    shd_norm    shd_double / number of scored pairs

SCOPE. The scenarios named in <run>/evaluation_scope.txt are the held-out
datasets and are averaged apart from the development suite; without that file
every scenario is in the development suite. The `scope` column holds
'development' or 'held_out' (files written before 2026-10-16 hold 'test'
in place of 'development').

Written at the run root:
    auroc_auprc_shd_per_scenario.csv   one line per scenario x model
    auroc_auprc_shd_summary.txt        definitions, mean per model (development
                                       suite and held-out) and per dataset family

Re-running is safe: both files are replaced.

Usage:
    python Auxiliary/ranking_metrics.py [RUN_DIR]
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# =============================================================================
# CONFIGURATION  <- edit here
# =============================================================================

RUN_DIR = "./Results/Run_20261011_132346"

DETAIL_FILE  = "auroc_auprc_shd_per_scenario.csv"
SUMMARY_FILE = "auroc_auprc_shd_summary.txt"

# Lists the held-out scenarios of the run, one '<scenario> : <origin>' per line.
SCOPE_FILE = "evaluation_scope.txt"

# A model whose original.csv has this column is ranked by its absolute value.
PENALIZED_COL = "penalized_effect"

# Dataset families of the per-family tables: the first prefix that matches the
# scenario name (without 'benchmark_' / 'real_') wins, so a longer prefix must
# come before a shorter one.
FAMILIES = ["causaltime", "logistic_maps", "lorenz96", "netsim", "synthetic_var",
            "tigramite_climate", "tigramite", "alarm", "beeline", "dream3", "dream4",
            "rcaeval", "tep", "easyvista", "pems_bay", "petshop", "sachs",
            "causal_rivers"]

# =============================================================================

TRUE_STRINGS = ["true", "1", "1.0", "yes", "t", "y"]

METRICS = ["f1", "auroc", "auprc", "auroc_two_tier", "auprc_two_tier",
           "auroc_pvalue_only", "auprc_pvalue_only", "shd", "shd_double", "shd_norm"]
FAMILY_COLS = ["n", "f1", "auroc", "auprc", "auroc_two_tier", "auprc_two_tier",
               "shd", "shd_double", "shd_norm"]


def _family(scenario: str) -> str:
    base = re.sub(r"^(benchmark|real)_", "", scenario)
    for fam in FAMILIES:
        if base.startswith(fam):
            return fam
    return base.split("_")[0]


def _status(model_dir: Path) -> str:
    time_path = model_dir / "time.txt"
    if not time_path.exists():
        return "N/A"
    for line in time_path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "status":
            return value.strip()
    return "N/A"


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, skipinitialspace=True, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _pairs(df: pd.DataFrame) -> pd.Series:
    """Directional entity pair of each row (main._pair_key, use_canonical_pair=False)."""
    return pd.Series(list(zip(df["source_entity"].astype(str).str.strip(),
                              df["target_entity"].astype(str).str.strip())), index=df.index)


def _held_out(root: Path) -> set:
    path = root / SCOPE_FILE
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        name = line.split(":", 1)[0].strip()
        if name and (root / name).is_dir():
            names.add(name)
    return names


# =============================================================================
# SCORES
# =============================================================================

def _confidence(original: pd.DataFrame) -> pd.Series:
    """1 - min(pvalue) per directional pair; a pair without a finite pvalue gets 0."""
    if original.empty or "source_entity" not in original.columns:
        return pd.Series(dtype=float)
    p = (pd.to_numeric(original["pvalue"], errors="coerce")
         if "pvalue" in original.columns else pd.Series(np.nan, index=original.index))
    conf = (1.0 - p.clip(0.0, 1.0)).fillna(0.0)
    return conf.groupby(_pairs(original)).max()


def _penalized(original: pd.DataFrame):
    """max |PENALIZED_COL| per directional pair, or None if the model has no such column."""
    if original.empty or PENALIZED_COL not in original.columns:
        return None
    pen = pd.to_numeric(original[PENALIZED_COL], errors="coerce").abs().fillna(0.0)
    return pen.groupby(_pairs(original)).max()


def _shd(predicted: set, causal: set) -> int:
    """Structural Hamming Distance, a reversed edge counting as ONE edit."""
    return len({frozenset(p) for p in predicted ^ causal})


def _auc(y: np.ndarray, score: np.ndarray, defined: bool) -> tuple:
    if not defined:
        return float("nan"), float("nan")
    return round(roc_auc_score(y, score), 4), round(average_precision_score(y, score), 4)


# =============================================================================
# SCENARIO LEVEL
# =============================================================================

def process_scenario(folder: Path, scope: str, mismatches: list) -> list:
    gt = _read_csv(folder / "ground_truth.csv")
    if gt.empty:
        return []
    gt_pairs = _pairs(gt)
    if "causal_gt" in gt.columns:
        is_causal = gt["causal_gt"].astype(str).str.strip().str.lower().isin(TRUE_STRINGS)
        causal    = set(gt_pairs[is_causal])
        universe  = set(gt_pairs)
    else:
        causal   = set(gt_pairs)
        universe = set(causal)

    summary = _read_csv(folder / "validation_summary.csv")
    summary = summary.set_index("model") if "model" in summary.columns else pd.DataFrame()

    rows = []
    for model_dir in sorted(d for d in folder.iterdir()
                            if d.is_dir() and (d / "results.csv").exists()):
        if _status(model_dir).upper() == "ERROR":
            continue
        results   = _read_csv(model_dir / "results.csv")
        original  = _read_csv(model_dir / "original.csv")
        predicted = set(_pairs(results)) if not results.empty else set()
        conf      = _confidence(original)

        scored_pairs = sorted(universe | predicted)
        y        = np.array([p in causal for p in scored_pairs], dtype=int)
        in_graph = np.array([p in predicted for p in scored_pairs], dtype=bool)
        tested   = np.array([p in conf.index for p in scored_pairs], dtype=bool)
        pvalue_only = conf.reindex(scored_pairs).fillna(0.0).to_numpy()

        # 0.999 keeps the within-tier order strictly below the next tier.
        two_tier = in_graph * 2.0 + (~in_graph & tested) * 1.0 + pvalue_only * 0.999

        pen = _penalized(original)
        if pen is not None:
            main, score_used = pen.reindex(scored_pairs).fillna(0.0).to_numpy(), PENALIZED_COL
        else:
            main, score_used = two_tier, "two_tier"

        TP = len(causal & predicted)
        FP = len(predicted - causal)
        FN = len(causal - predicted)
        TN = len(universe - causal - predicted)
        n_pos   = int(y.sum())
        defined = 0 < n_pos < len(y)   # AUROC needs both classes

        auroc, auprc       = _auc(y, main, defined)
        auroc_tt, auprc_tt = _auc(y, two_tier, defined)
        auroc_pv, auprc_pv = _auc(y, pvalue_only, defined)

        row = {
            "scope": scope, "family": _family(folder.name),
            "scenario": folder.name, "model": model_dir.name,
            "n_universe": len(universe), "n_causal": len(causal),
            "prevalence": round(n_pos / len(y), 4) if len(y) else float("nan"),
            "n_predicted": len(predicted), "n_scored": int(tested.sum()),
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "score_used": score_used,
            "auroc": auroc, "auprc": auprc,
            "auroc_two_tier": auroc_tt, "auprc_two_tier": auprc_tt,
            "auroc_pvalue_only": auroc_pv, "auprc_pvalue_only": auprc_pv,
            "shd": _shd(predicted, causal),
            "shd_double": FP + FN,
            "shd_norm": round((FP + FN) / len(y), 4) if len(y) else float("nan"),
            "f1": float("nan"),
        }
        if model_dir.name in summary.index:
            ref = summary.loc[model_dir.name]
            row["f1"] = ref.get("f1", float("nan"))
            diff = [k for k in ("TP", "FP", "FN", "TN")
                    if pd.notna(ref.get(k)) and int(ref[k]) != row[k]]
            if diff:
                mismatches.append(f"{folder.name}/{model_dir.name}: {', '.join(diff)}")
        rows.append(row)
    return rows


# =============================================================================
# RUN LEVEL
# =============================================================================

def _table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("model", sort=True)
    out = g[METRICS].mean().round(4)
    out.insert(0, "n", g.size())
    return out


def _definitions(n_checked: int, mismatches: list) -> str:
    return f"""
  HOW EACH METRIC IS COMPUTED
  ---------------------------
  Pairs      Entity level, DIRECTIONAL (A->B != B->A), as in validation_summary.csv.
             Scored set = every pair of ground_truth.csv, plus any predicted pair
             outside it (counted as a negative, as main.py counts it as FP).
             The TP/FP/FN/TN recomputed here were checked against
             validation_summary.csv: {n_checked} rows checked, {len(mismatches)} mismatches.

  Score      The ranking score depends on the model (column "score_used" of the
             per-scenario file):

             patch* models -> PENALIZED EFFECT
               score = max |{PENALIZED_COL}| over the pair's rows (all lags /
               properties) in original.csv; a pair the model never scored gets 0.

             all other models -> TWO TIERS (they save a thresholded graph, not a
             score matrix, and have no penalized effect)
               tier 2  pair is in results.csv (the model's final graph)
               tier 1  pair is in original.csv but was not kept
               tier 0  pair never scored by the model (tied at the bottom)
               Inside a tier, pairs are ordered by 1 - min(pvalue) over the
               pair's rows in original.csv.

  auroc      Area under the ROC curve of that score (ties get half credit).
  auprc      Average precision of that score. Its chance level is the
             prevalence (n_causal / n_scored pairs), which differs per scenario:
             compare AUPRC across models, not across scenarios.
  *_two_tier Same, with the two-tier score for EVERY model (patch included).
             Equal to auroc/auprc for the non-patch models.
  *_pvalue_only
             Same, ranking by 1 - min(pvalue) alone, ignoring the final graph.
             Both are sensitivity checks of the score choice.

  shd        Structural Hamming Distance; a reversed edge counts as ONE edit.
  shd_double Same, a reversed edge counting as TWO edits (= FP + FN).
  shd_norm   shd_double / number of scored pairs (comparable across graph sizes).
             Lower is better for all three.

  CAVEATS
  -------
  - cuts_plus and dynotears save only the edges they keep (original.csv ==
    results.csv), so they have no tier 1: every rejected pair is tied. Their
    AUROC/AUPRC are those of the binary graph plus the ordering inside it.
  - The patch models and the baselines are ranked by different scores, so
    auroc/auprc compare a continuous effect size against a thresholded graph;
    the *_two_tier columns put every model on the same score.
  - |{PENALIZED_COL}| is not scale-free: it is in the units of the variables
    of each pair. Rejected pairs with weight 0 get score 0; pairs rejected
    after the penalty stage keep their full effect.
  - Means are taken over the scenarios each model ran on ("n"). Models with a
    smaller n (varlingam*, patch_rank*) are not averaged over the same scenarios
    as the others.
"""


def run(run_dir: str):
    root = Path(run_dir)
    scenarios = sorted(d for d in root.iterdir()
                       if d.is_dir() and (d / "ground_truth.csv").exists())
    if not scenarios:
        print(f"[WARNING] No scenario with ground_truth.csv found in '{root}'.")
        return

    held_out = _held_out(root)
    rows, mismatches = [], []
    for folder in scenarios:
        rows += process_scenario(folder, "held_out" if folder.name in held_out else "development",
                                 mismatches)
    if not rows:
        print(f"[WARNING] No model with results.csv found in '{root}'.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(root / DETAIL_FILE, index=False)

    SEP = "=" * 100
    n_dev = df.loc[df["scope"] == "development", "scenario"].nunique()
    n_held = df.loc[df["scope"] == "held_out", "scenario"].nunique()
    lines = [SEP,
             f"  AUROC / AUPRC / SHD SUMMARY -- {root.name}",
             f"  Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"  Scenarios: {df['scenario'].nunique()}  (development suite: {n_dev}, held-out: {n_held})"
             f"   Models: {df['model'].nunique()}",
             f"  Per-scenario values: {DETAIL_FILE}",
             SEP,
             _definitions(int(df["f1"].notna().sum()), mismatches)]

    with pd.option_context("display.width", 250, "display.max_rows", 500):
        for scope, label in (("development", "DEVELOPMENT SUITE"), ("held_out", "HELD-OUT DATASETS")):
            sub = df[df["scope"] == scope]
            if sub.empty:
                continue
            lines += [SEP,
                      f"  {label} -- mean per model over {sub['scenario'].nunique()} scenarios",
                      SEP,
                      _table(sub).sort_values("auprc", ascending=False).to_string(),
                      ""]
        lines += [SEP, "  PER DATASET FAMILY -- mean per model", SEP]
        for (scope, fam), sub in df.groupby(["scope", "family"], sort=True):
            lines += [f"\n  [{scope}] {fam}  ({sub['scenario'].nunique()} scenario(s), "
                      f"prevalence {sub['prevalence'].mean():.3f})",
                      _table(sub)[FAMILY_COLS].to_string()]

    lines += ["", f"  Sanity check against the existing validation summaries: "
                  f"{'OK' if not mismatches else str(len(mismatches)) + ' MISMATCH(ES)'}"]
    lines += [f"    {m}" for m in mismatches]

    text = "\n".join(lines) + "\n"
    (root / SUMMARY_FILE).write_text(text, encoding="utf-8")
    print(text)
    print(f"  {len(scenarios)} scenario(s), {len(df)} row(s) -> {root / DETAIL_FILE}")
    print(f"  Summary -> {root / SUMMARY_FILE}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else RUN_DIR)

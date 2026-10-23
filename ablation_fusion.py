"""
ablation_fusion.py
==================
Ablation of the EVIDENCE FUSION stage (Dempster-Shafer, causal_penalty_score).

Replays alternative acceptance rules over the patch's UNFILTERED output
(patch/original.csv) of an existing run, without re-running anything, and
scores each rule against ground_truth.csv with the same hard metric used in
validation_summary.csv (entity level, directional, see main._compute_metrics).

Every arm is evaluated BEFORE structural Phase 2: the DS arm is
causal_penalty_score == 1, not the final `significant` column (which is
DS minus the edges later removed by rejected_phase2_structural). The final
pipeline output is reported as a reference row ("Final (DS + structural)")
and doubles as a sanity check: its F1 must match validation_summary.csv.

Arms:
    1  DS (reference)            causal_penalty_score == 1
    2  No filter                 every line of original.csv
    3  Cascade confidence        cascade_confidence in {high, medium}
    4  p-value BH                pvalue_bh < 0.05
    5  t-stat                    |t_stat| > 1.96
    6  p-value oracle            best F1 per scenario over PV_GRID
    7  t-stat oracle             best F1 per scenario over T_GRID
    8  |t| at matched recall     per scenario, |t| threshold with recall >= DS
                                 recall and maximum precision (exact sweep)
    9  p + CI                    pvalue_bh < 0.05 and CI95 does not cross zero
    10 p + asymmetry             pvalue_bh < 0.05 and causal_asymmetry > 0
                                 (causal_asymmetry = |theta_fwd| - |theta_rev|,
                                  positive = more evidence in the tested
                                  direction; Pipeline/diagnostics_aux.py)

Oracle ties are broken towards the most restrictive threshold. Missing
values (NaN p-value / t_stat / CI / asymmetry) always reject the pair.

Usage:
    python ablation_fusion.py [RUN_DIR] [OUT_DIR]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover
    wilcoxon = None

# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_DIR = "Results/Run_20260921_223813"
MODEL   = "patch"

PV_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
T_GRID  = list(np.round(np.arange(0.5, 4.0 + 1e-9, 0.25), 2))

CONF_ACCEPT = {"high", "medium"}

ARM_ORDER = [
    "1 DS (reference)", "2 No filter", "3 Cascade confidence",
    "4 p-value BH < 0.05", "5 |t| > 1.96", "6 p-value oracle",
    "7 t-stat oracle", "8 |t| at matched recall", "9 p + CI",
    "10 p + asymmetry", "Final (DS + structural)",
]

# =============================================================================
# Metric (mirror of main._predicted_pairs / _causal_gt_pairs / _compute_metrics,
# directional, entity level)
# =============================================================================

def _pairs(df: pd.DataFrame) -> set:
    return set(zip(df["source_entity"].astype(str).str.strip(),
                   df["target_entity"].astype(str).str.strip()))


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def _metrics(pred: set, gt: set, universe: set) -> dict:
    tp = len(gt & pred); fp = len(pred - gt); fn = len(gt - pred)
    tn = len(universe - gt - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "n_pred": len(pred)}


# =============================================================================
# Arms
# =============================================================================

def _num(o: pd.DataFrame, col: str) -> pd.Series:
    if col not in o.columns:
        return pd.Series(np.nan, index=o.index)
    return pd.to_numeric(o[col], errors="coerce")


def _fixed_masks(o: pd.DataFrame) -> dict:
    pbh  = _num(o, "pvalue_bh")
    abst = _num(o, "t_stat").abs()
    lo, hi = _num(o, "ci_95_lower"), _num(o, "ci_95_upper")
    asym = _num(o, "causal_asymmetry")
    conf = o.get("cascade_confidence", pd.Series("", index=o.index))
    conf = conf.astype(str).str.strip().str.lower()

    final = _truthy(o["significant"]) if "significant" in o.columns \
        else pd.Series(False, index=o.index)

    return {
        "1 DS (reference)":       _num(o, "causal_penalty_score") == 1,
        "2 No filter":            pd.Series(True, index=o.index),
        "3 Cascade confidence":  conf.isin(CONF_ACCEPT),
        "4 p-value BH < 0.05":     pbh < 0.05,
        "5 |t| > 1.96":            abst > 1.96,
        "9 p + CI":                (pbh < 0.05) & ((lo > 0) | (hi < 0)),
        "10 p + asymmetry":       (pbh < 0.05) & (asym > 0),
        "Final (DS + structural)": final,
    }


def _best_over_grid(o, values, grid, gt, universe, keep):
    """keep(values, thr) -> mask. Grid must go from most to least restrictive
    so that ties (strict '>') keep the most restrictive threshold."""
    best = None
    for thr in grid:
        m = _metrics(_pairs(o[keep(values, thr)]), gt, universe)
        if best is None or m["f1"] > best[1]["f1"]:
            best = (thr, m)
    return best


def _precision_at_matched_recall(o, gt, universe, target_recall):
    abst = _num(o, "t_stat").abs()
    cands = sorted(abst.dropna().unique(), reverse=True)  # restrictive first
    best = None
    for thr in cands:
        m = _metrics(_pairs(o[abst >= thr]), gt, universe)
        if m["recall"] + 1e-12 < target_recall:
            continue
        if best is None or m["precision"] > best[1]["precision"]:
            best = (thr, m)
    if best is None:  # DS recall unreachable by thresholding |t| (NaN t_stat)
        best = (np.nan, _metrics(_pairs(o[abst.notna()]), gt, universe))
    return best


# =============================================================================
# Driver
# =============================================================================

def _family(name: str) -> str:
    for sep in ("_seed", "_sample"):
        if sep in name:
            return name.split(sep)[0]
    return name


def _load(scn: Path):
    orig_path = scn / MODEL / "original.csv"
    gt_path   = scn / "ground_truth.csv"
    if not orig_path.exists() or not gt_path.exists():
        return None
    o  = pd.read_csv(orig_path, skipinitialspace=True)
    gt = pd.read_csv(gt_path, skipinitialspace=True)
    universe = _pairs(gt)
    gt_causal = _pairs(gt[_truthy(gt["causal_gt"])]) if "causal_gt" in gt.columns else universe
    return o, gt_causal, universe


def run(run_dir: Path) -> tuple:
    rows, sanity = [], []
    for scn in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        loaded = _load(scn)
        if loaded is None:
            continue
        o, gt, universe = loaded
        base = {"scenario": scn.name, "family": _family(scn.name),
                "n_tested": len(o), "n_causal_gt": len(gt), "n_universe": len(universe)}

        for arm, mask in _fixed_masks(o).items():
            rows.append({**base, "arm": arm, "threshold": np.nan,
                         **_metrics(_pairs(o[mask.fillna(False)]), gt, universe)})

        thr, m = _best_over_grid(o, _num(o, "pvalue_bh"), PV_GRID, gt, universe,
                                 lambda v, t: v < t)
        rows.append({**base, "arm": "6 p-value oracle", "threshold": thr, **m})

        thr, m = _best_over_grid(o, _num(o, "t_stat").abs(), T_GRID[::-1], gt, universe,
                                 lambda v, t: v > t)
        rows.append({**base, "arm": "7 t-stat oracle", "threshold": thr, **m})

        ds = _metrics(_pairs(o[_num(o, "causal_penalty_score") == 1]), gt, universe)
        thr, m = _precision_at_matched_recall(o, gt, universe, ds["recall"])
        rows.append({**base, "arm": "8 |t| at matched recall", "threshold": thr, **m})

        # Sanity: the final output must reproduce validation_summary.csv
        vs = scn / "validation_summary.csv"
        if vs.exists():
            s = pd.read_csv(vs)
            s = s[s["model"] == MODEL]
            if len(s) and str(s["status"].iloc[0]) == "ok":
                mine = [r for r in rows if r["scenario"] == scn.name
                        and r["arm"] == "Final (DS + structural)"][0]
                sanity.append((scn.name, float(s["f1"].iloc[0]), round(mine["f1"], 4)))

    return pd.DataFrame(rows), sanity


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ds = df[df.arm == "1 DS (reference)"].set_index("scenario")
    out = []
    for arm in ARM_ORDER:
        g = df[df.arm == arm]
        if g.empty:
            continue
        a = g.set_index("scenario").loc[ds.index]
        fam = g.groupby("family")[["precision", "recall", "f1"]].mean()
        d_f1 = a["f1"] - ds["f1"]
        row = {
            "arm": arm,
            "mean_precision": g["precision"].mean(),
            "mean_recall": g["recall"].mean(),
            "mean_f1": g["f1"].mean(),
            "median_f1": g["f1"].median(),
            "family_mean_f1": fam["f1"].mean(),
            "mean_n_pred": g["n_pred"].mean(),
            "delta_f1_vs_DS": d_f1.mean(),
            "wins_vs_DS": int((d_f1 > 1e-9).sum()),
            "ties_vs_DS": int((d_f1.abs() <= 1e-9).sum()),
            "losses_vs_DS": int((d_f1 < -1e-9).sum()),
            "wilcoxon_p_f1": np.nan,
        }
        if wilcoxon is not None and arm != "1 DS (reference)" and (d_f1.abs() > 1e-9).any():
            row["wilcoxon_p_f1"] = wilcoxon(a["f1"], ds["f1"], zero_method="wilcox").pvalue
        out.append(row)
    return pd.DataFrame(out)


def precision_matched(df: pd.DataFrame) -> pd.DataFrame:
    ds = df[df.arm == "1 DS (reference)"].set_index("scenario")
    a8 = df[df.arm == "8 |t| at matched recall"].set_index("scenario").loc[ds.index]
    t = pd.DataFrame({
        "family": ds["family"],
        "DS_recall": ds["recall"], "DS_precision": ds["precision"],
        "t_threshold": a8["threshold"],
        "t_recall": a8["recall"], "t_precision": a8["precision"],
    })
    t["delta_precision"] = t["DS_precision"] - t["t_precision"]
    return t.reset_index()


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else RUN_DIR)
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2
                   else Path("Results_Ablation") / f"fusion_{run_dir.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df, sanity = run(run_dir)
    if df.empty:
        print(f"No scenario with {MODEL}/original.csv + ground_truth.csv in {run_dir}")
        return

    bad = [s for s in sanity if abs(s[1] - s[2]) > 1e-3]
    print(f"Sanity check (Final vs validation_summary.csv): "
          f"{len(sanity) - len(bad)}/{len(sanity)} scenarios match")
    for name, ref, mine in bad:
        print(f"   MISMATCH {name}: summary f1={ref}  recomputed={mine}")

    summ = summarize(df)
    pm   = precision_matched(df)
    fam  = (df.groupby(["family", "arm"])["f1"].mean().unstack("arm")
              .reindex(columns=[a for a in ARM_ORDER if a in df.arm.unique()]))

    df.to_csv(out_dir / "per_scenario.csv", index=False)
    summ.to_csv(out_dir / "summary.csv", index=False)
    pm.to_csv(out_dir / "precision_at_matched_recall.csv", index=False)
    fam.to_csv(out_dir / "f1_by_family.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"\nRun: {run_dir}  |  scenarios: {df.scenario.nunique()}  "
          f"|  families: {df.family.nunique()}\n")
    print(summ.round(4).to_string(index=False))
    print("\nArm 8 -- precision at matched recall (DS minus |t|):")
    d = pm["delta_precision"]
    print(f"   mean delta = {d.mean():+.4f}   DS better in {(d > 1e-9).sum()}, "
          f"tie in {(d.abs() <= 1e-9).sum()}, |t| better in {(d < -1e-9).sum()} "
          f"of {len(d)} scenarios")
    print("\nF1 by family:")
    print(fam.round(3).to_string())
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()

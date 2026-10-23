"""
seed_replication_analysis.py
============================
Analysis of an E1a run (main_seed_replication.py): new realizations (seeds)
of the four synthetic generators of the development suite. Reads the run's outputs
only; nothing is re-run.

Scope: new realizations of DEVELOPMENT generators, not held-out data (see
main_seed_replication.py). Each scenario is identified through
<run>/seeds_manifest.csv (family, configuration, seed).

LAG WINDOWS. Models whose name ends in '_tau10' form the tau_max = 10 window,
the others the tau_max = 24 window. Rankings and paired differences are
computed within one window; the pipeline of each window is PIPELINE_MODEL.

PART 1 -- F1 per family (pair level, validation_summary.csv: directional
pairs, as in the main run)
    mean and sd over the scenarios of the family; 95% CI by bootstrap over
    the SEEDS: each seed's F1 is the mean over the family's configurations
    (5 VAR sizes, 4 logistic topologies, 2 Lorenz-96 K, 5 tigramite types),
    and the 10 seed values are resampled with replacement.
    Paired difference pipeline - baseline, same bootstrap over seeds.
    A model that failed on a scenario (status ERROR) has no F1 there: it is
    left out of that scenario's means (column n_failed counts those cases)
    and is never first in it (part 2).

PART 2 -- Ranking stability
    Per family and seed, the seed's F1 of every model of the window; a model
    is FIRST when its F1 is within TIE_TOL of the best (ties all count as
    first). Reported: number of seeds (out of n_seeds) and of scenarios where
    each model is first or tied first.

PART 3 -- Tables 4a / 4b on the VAR scenarios (synthetic_var family)
    Ground truth per link (source, target, lag, beta) from scenario_params.txt.
    Every method estimates on z-scored series, so the reference value of an
    estimate is the STANDARDIZED coefficient
        theta* = beta * sd(source) / sd(target)
    (sd of the raw series in source.csv, ddof=0, as Pipeline/preprocessing.py
    and the baselines' z-scoring). The raw beta is kept as a sensitivity
    column (*_raw).

    4a, effect estimation -- over the true links detected AT THE TRUE LAG:
        sign_correct    share with sign(estimate) == sign(beta)
        median_rel_err  median |estimate - theta*| / |theta*|
        coverage        share with theta* inside the method's 95% CI
    What each method reports (EFFECT_SPECS):
        patch*          causal_effect, CI [ci_95_lower, ci_95_upper];
                        also the penalized effect ('<model> [penalized]'),
                        CI [ci_lower_pen, ci_upper_pen]
        varlingam*      B coefficient (no CI)
        pcmci*          partial correlation: sign only (not a coefficient)
        dynotears*, cuts_plus*  F statistic: no signed effect, not in 4a
    These metrics are CONDITIONAL on detection: each method is scored on the
    links it found, so methods with different recall are scored on different
    link sets (n_links tells how many).

    4b, link recovery -- a link is (source, target, lag):
        f1_link         per-scenario F1 over links (mean over scenarios), a
                        lag-0 or wrong-lag prediction counts as FP
        f1_link_micro   F1 over the links pooled across scenarios
        lag_exact       among the true PAIRS detected (any lag), share whose
                        predicted lags include the true lag
        f1_pair         pair-level F1 (validation_summary.csv), for reference

    CIs: 95% percentile bootstrap over SCENARIOS, stratified by VAR size
    (each resample keeps 10 scenarios per size); the pooled statistic is
    recomputed on every resample. Also reported per size.

Written at the run root:
    seed_f1_per_scenario.csv     one line per scenario x model
    seed_f1_family.csv           part 1, per family
    seed_f1_config.csv           part 1, per family x configuration
    seed_f1_delta.csv            part 1, pipeline - baseline
    seed_ranking.csv             part 2
    var_effect_links.csv         part 3, one line per true link x model
    var_table4a.csv              part 3, effect estimation (all sizes + per size)
    var_table4b.csv              part 3, link recovery (all sizes + per size)
    seed_replication_summary.txt definitions and every table above

Re-running is safe: every file is replaced.

Usage:
    python Auxiliary/seed_replication_analysis.py [RUN_DIR]
"""

import ast
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION  <- edit here
# =============================================================================

RUN_DIR = "./Results_Seeds/Run_YYYYMMDD_HHMMSS"

PIPELINE_MODEL = {"tau10": "patch_tau10", "tau24": "patch"}

N_BOOT    = 10000
BOOT_SEED = 20261014
CI_LEVEL  = 0.95
TIE_TOL   = 1e-6

VAR_FAMILY = "synthetic_var"

# base model name (without '_tau10') -> how its results.csv carries an effect.
#   kind 'coefficient'          comparable with theta*: sign, error, coverage
#   kind 'partial_correlation'  sign only
EFFECT_SPECS = {
    "patch": [
        dict(label="", col="causal_effect", lo="ci_95_lower", hi="ci_95_upper",
             kind="coefficient"),
        dict(label=" [penalized]", col="penalized_effect", lo="ci_lower_pen",
             hi="ci_upper_pen", kind="coefficient"),
    ],
    "varlingam":       [dict(label="", col="value", kind="coefficient")],
    "pcmci_tigramite": [dict(label="", col="value", kind="partial_correlation")],
    "pcmciplus":       [dict(label="", col="value", kind="partial_correlation")],
}

# =============================================================================


def _window(model: str) -> str:
    return "tau10" if model.endswith("_tau10") else "tau24"


def _base(model: str) -> str:
    return model[:-len("_tau10")] if model.endswith("_tau10") else model


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, skipinitialspace=True, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _status(model_dir: Path) -> str:
    time_path = model_dir / "time.txt"
    if not time_path.exists():
        return "missing"
    for line in time_path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "status":
            return value.strip().lower()
    return "missing"


def _ci(values: np.ndarray) -> tuple:
    a = (1.0 - CI_LEVEL) / 2.0
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(values, a)), float(np.quantile(values, 1.0 - a))


def _boot_mean(x: np.ndarray, rng: np.random.Generator) -> tuple:
    """Percentile CI of the mean of x (resampling its entries)."""
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, x.size, size=(N_BOOT, x.size))
    return _ci(x[idx].mean(axis=1))


def _fmt_ci(mean, lo, hi, digits=3) -> str:
    if not np.isfinite(mean):
        return "n/a"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# =============================================================================
# PART 1 / 2 -- F1 PER FAMILY, RANKING
# =============================================================================

def load_f1(root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in manifest.itertuples(index=False):
        folder = root / m.scenario
        summary = _read_csv(folder / "validation_summary.csv")
        model_dirs = sorted(d for d in folder.iterdir() if d.is_dir()) if folder.is_dir() else []
        by_model = summary.set_index("model") if "model" in summary.columns else pd.DataFrame()
        for d in model_dirs:
            if not (d / "time.txt").exists():
                continue
            row = {"family": m.family, "config": str(m.config), "seed": int(m.seed),
                   "scenario": m.scenario, "model": d.name, "window": _window(d.name),
                   "status": _status(d), "f1": np.nan, "precision": np.nan,
                   "recall": np.nan, "TP": np.nan, "FP": np.nan, "FN": np.nan}
            if row["status"] == "ok" and d.name in by_model.index:
                ref = by_model.loc[d.name]
                for k in ("f1", "precision", "recall", "TP", "FP", "FN"):
                    row[k] = pd.to_numeric(ref.get(k), errors="coerce")
            rows.append(row)
    return pd.DataFrame(rows)


def _seed_means(sub: pd.DataFrame) -> pd.DataFrame:
    """seed x model table of the mean F1 over the family's configurations."""
    return sub.pivot_table(index="seed", columns="model", values="f1", aggfunc="mean")


def f1_tables(f1: pd.DataFrame, rng: np.random.Generator) -> tuple:
    fam_rows, cfg_rows, delta_rows = [], [], []
    for (window, family), sub in f1.groupby(["window", "family"], sort=True):
        seed_tab = _seed_means(sub)
        for model, msub in sub.groupby("model", sort=True):
            x = msub["f1"].to_numpy(float)
            sm = seed_tab[model].to_numpy(float)
            lo, hi = _boot_mean(sm, rng)
            fam_rows.append({
                "window": window, "family": family, "model": model,
                "n_scenarios": int(np.isfinite(x).sum()),
                "n_failed": int((msub["status"] != "ok").sum()),
                "n_seeds": int(np.isfinite(sm).sum()),
                "f1_mean": np.nanmean(x) if np.isfinite(x).any() else np.nan,
                "f1_sd": np.nanstd(x, ddof=1) if np.isfinite(x).sum() > 1 else np.nan,
                "f1_sd_seed_means": np.nanstd(sm, ddof=1) if np.isfinite(sm).sum() > 1 else np.nan,
                "ci_low": lo, "ci_high": hi})
            for config, csub in msub.groupby("config", sort=True):
                cx = csub["f1"].to_numpy(float)
                clo, chi = _boot_mean(cx, rng)
                cfg_rows.append({
                    "window": window, "family": family, "config": config, "model": model,
                    "n_scenarios": int(np.isfinite(cx).sum()),
                    "f1_mean": np.nanmean(cx) if np.isfinite(cx).any() else np.nan,
                    "f1_sd": np.nanstd(cx, ddof=1) if np.isfinite(cx).sum() > 1 else np.nan,
                    "ci_low": clo, "ci_high": chi})

        pipe = PIPELINE_MODEL.get(window)
        if pipe not in seed_tab.columns:
            continue
        for model in seed_tab.columns:
            if model == pipe:
                continue
            d = (seed_tab[pipe] - seed_tab[model]).to_numpy(float)
            lo, hi = _boot_mean(d, rng)
            delta_rows.append({
                "window": window, "family": family, "pipeline": pipe, "baseline": model,
                "n_seeds": int(np.isfinite(d).sum()),
                "delta_f1_mean": np.nanmean(d) if np.isfinite(d).any() else np.nan,
                "ci_low": lo, "ci_high": hi,
                "seeds_pipeline_better": int((d > TIE_TOL).sum()),
                "seeds_tied": int((np.abs(d) <= TIE_TOL).sum()),
                "seeds_pipeline_worse": int((d < -TIE_TOL).sum())})
    return pd.DataFrame(fam_rows), pd.DataFrame(cfg_rows), pd.DataFrame(delta_rows)


def _first(table: pd.DataFrame) -> pd.DataFrame:
    """Boolean table: entry is within TIE_TOL of its row's best. A model without F1
    in a row (it failed) is not first there; rows where every model failed are dropped."""
    table = table.dropna(how="all")
    return table.ge(table.max(axis=1) - TIE_TOL, axis=0)


def ranking(f1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window, family), sub in f1.groupby(["window", "family"], sort=True):
        by_seed = _first(_seed_means(sub))
        by_scen = _first(sub.pivot_table(index="scenario", columns="model",
                                         values="f1", aggfunc="mean"))
        for model in sorted(sub["model"].unique()):
            rows.append({
                "window": window, "family": family, "model": model,
                "is_pipeline": model == PIPELINE_MODEL.get(window),
                "seeds_first_or_tied": int(by_seed[model].sum()) if model in by_seed else 0,
                "n_seeds": len(by_seed),
                "seeds_sole_first": int((by_seed[model] & (by_seed.sum(axis=1) == 1)).sum())
                                    if model in by_seed else 0,
                "scenarios_first_or_tied": int(by_scen[model].sum()) if model in by_scen else 0,
                "n_scenarios": len(by_scen)})
    return pd.DataFrame(rows)


# =============================================================================
# PART 3 -- VAR: EFFECT ESTIMATION (4a) AND LINK RECOVERY (4b)
# =============================================================================

_EDGES_RE = re.compile(r"^causal_edges\s*:\s*(\[.*\])\s*$", re.M)


def _true_links(folder: Path) -> list:
    text = (folder / "scenario_params.txt").read_text(encoding="utf-8")
    match = _EDGES_RE.search(text)
    if not match:
        raise ValueError(f"{folder.name}: no causal_edges in scenario_params.txt")
    return ast.literal_eval(match.group(1))


def _series_sd(folder: Path) -> pd.Series:
    src = _read_csv(folder / "source.csv")
    value_col = [c for c in src.columns if c not in ("Date", "Name")][0]
    wide = src.pivot_table(index="Date", columns="Name", values=value_col, aggfunc="first")
    return wide.std(ddof=0)


def _links_of(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["source_entity", "target_entity", "lag"])
    out = results.copy()
    out["source_entity"] = out["source_entity"].astype(str).str.strip()
    out["target_entity"] = out["target_entity"].astype(str).str.strip()
    out["lag"] = pd.to_numeric(out["lag"], errors="coerce").round().astype("Int64")
    return out


def var_links(root: Path, manifest: pd.DataFrame) -> tuple:
    """(per true link x model rows, per scenario x model link counts)."""
    link_rows, count_rows = [], []
    var = manifest[manifest["family"] == VAR_FAMILY]
    for m in var.itertuples(index=False):
        folder = root / m.scenario
        if not folder.is_dir():
            continue
        truth = _true_links(folder)
        sd = _series_sd(folder)
        gt_links = {(e["source"], e["target"], int(e["lag"])) for e in truth}
        gt_pairs = {(s, t) for s, t, _ in gt_links}

        for d in sorted(x for x in folder.iterdir() if x.is_dir()):
            if _status(d) != "ok" or not (d / "results.csv").exists():
                continue
            res = _links_of(_read_csv(d / "results.csv"))
            pred_links = {(s, t, int(l)) for s, t, l in
                          zip(res["source_entity"], res["target_entity"], res["lag"])
                          if pd.notna(l)}
            pred_pairs = {(s, t) for s, t, _ in pred_links}
            pair_hits = gt_pairs & pred_pairs
            lag_hits = {(s, t) for s, t, l in gt_links
                        if (s, t) in pred_pairs and (s, t, l) in pred_links}
            count_rows.append({
                "config": str(m.config), "seed": int(m.seed), "scenario": m.scenario,
                "model": d.name, "window": _window(d.name),
                "TP_link": len(gt_links & pred_links),
                "FP_link": len(pred_links - gt_links),
                "FN_link": len(gt_links - pred_links),
                "pairs_detected": len(pair_hits), "pairs_lag_exact": len(lag_hits)})

            specs = EFFECT_SPECS.get(_base(d.name), [])
            for e in truth:
                s, t, lag, beta = e["source"], e["target"], int(e["lag"]), float(e["coef"])
                theta_star = beta * sd[s] / sd[t]
                at_lag = res[(res["source_entity"] == s) & (res["target_entity"] == t)
                             & (res["lag"] == lag)]
                base = {"config": str(m.config), "seed": int(m.seed), "scenario": m.scenario,
                        "window": _window(d.name), "source": s, "target": t,
                        "lag": lag, "beta": beta, "theta_star": theta_star,
                        "pair_detected": (s, t) in pred_pairs,
                        "link_detected": not at_lag.empty,
                        "predicted_lags": sorted({int(l) for (ps, pt, l) in pred_links
                                                  if (ps, pt) == (s, t)})}
                for spec in (specs or [dict(label="", col=None, kind=None)]):
                    row = {**base, "model": d.name + spec["label"], "kind": spec["kind"],
                           "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan}
                    if not at_lag.empty and spec["col"] in at_lag.columns:
                        first = at_lag.iloc[0]
                        row["estimate"] = pd.to_numeric(first[spec["col"]], errors="coerce")
                        if spec.get("lo") in at_lag.columns and spec.get("hi") in at_lag.columns:
                            row["ci_low"] = pd.to_numeric(first[spec["lo"]], errors="coerce")
                            row["ci_high"] = pd.to_numeric(first[spec["hi"]], errors="coerce")
                    link_rows.append(row)

    links = pd.DataFrame(link_rows)
    if not links.empty:
        est = links["estimate"]
        coef = links["kind"] == "coefficient"
        links["sign_correct"] = np.where(est.notna() & links["kind"].notna(),
                                         np.sign(est) == np.sign(links["beta"]), np.nan)
        links["rel_err"] = np.where(coef, (est - links["theta_star"]).abs()
                                    / links["theta_star"].abs(), np.nan)
        links["rel_err_raw"] = np.where(coef, (est - links["beta"]).abs()
                                        / links["beta"].abs(), np.nan)
        has_ci = coef & links["ci_low"].notna() & links["ci_high"].notna() & est.notna()
        links["covered"] = np.where(has_ci, (links["ci_low"] <= links["theta_star"])
                                    & (links["theta_star"] <= links["ci_high"]), np.nan)
        links["covered_raw"] = np.where(has_ci, (links["ci_low"] <= links["beta"])
                                        & (links["beta"] <= links["ci_high"]), np.nan)
    return links, pd.DataFrame(count_rows)


def _f1(tp, fp, fn):
    tp, fp, fn = (np.asarray(v, float) for v in (tp, fp, fn))
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)


def _boot_counts(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(N_BOOT, n) multiplicity of each scenario in each stratified resample."""
    n = len(strata)
    counts = np.zeros((N_BOOT, n))
    rows = np.arange(N_BOOT)[:, None]
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        draws = members[rng.integers(0, members.size, size=(N_BOOT, members.size))]
        np.add.at(counts, (rows, draws), 1.0)
    return counts


def _ratio(w: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Pooled num/den over the scenarios, scenario weights w (k, n)."""
    d = w @ den
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, (w @ num) / np.where(d > 0, d, 1), np.nan)


def _weighted_median(w_links: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Median of `values` with integer weights (k, m) per row; equals np.median for unit weights."""
    if values.size == 0:
        return np.full(w_links.shape[0], np.nan)
    order = np.argsort(values)
    v, cum = values[order], w_links[:, order].cumsum(axis=1)
    half = cum[:, -1:] / 2.0
    lo = np.minimum((cum >= half).argmax(axis=1), v.size - 1)
    hi = np.minimum((cum > half).argmax(axis=1), v.size - 1)
    return np.where(cum[:, -1] > 0, (v[lo] + v[hi]) / 2.0, np.nan)


def _with_ci(point: dict, draws: dict) -> dict:
    out = dict(point)
    for k, v in draws.items():
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = _ci(v)
    return out


def _table_4a(effect: pd.DataFrame, scenarios: list, strata: np.ndarray,
              rng: np.random.Generator) -> dict:
    """Effect metrics of one model over `scenarios` (resampling unit), with CIs."""
    pos = {s: i for i, s in enumerate(scenarios)}
    n = len(scenarios)
    ok = effect[effect["estimate"].notna() & effect["scenario"].isin(pos)]
    idx = ok["scenario"].map(pos).to_numpy(int)

    def per_scenario(col):
        x = ok[col].to_numpy(float)
        fin = np.isfinite(x)
        return (np.bincount(idx[fin], weights=x[fin], minlength=n),
                np.bincount(idx[fin], minlength=n).astype(float))

    sign, cov, cov_raw = (per_scenario(c) for c in ("sign_correct", "covered", "covered_raw"))
    err, err_raw = ok["rel_err"].to_numpy(float), ok["rel_err_raw"].to_numpy(float)
    fin, fin_raw = np.isfinite(err), np.isfinite(err_raw)

    def stats(w):
        return {"sign_correct":   _ratio(w, *sign),
                "median_rel_err": _weighted_median(w[:, idx[fin]], err[fin]),
                "coverage":       _ratio(w, *cov)}

    point = {k: float(v[0]) for k, v in stats(np.ones((1, n))).items()}
    point["n_links"] = len(ok)
    point["median_rel_err_raw"] = float(np.median(err_raw[fin_raw])) if fin_raw.any() else np.nan
    point["coverage_raw"] = float(_ratio(np.ones((1, n)), *cov_raw)[0])
    return _with_ci(point, stats(_boot_counts(strata, rng)))


def _table_4b(counts: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Link-recovery metrics of one model, one line per scenario in `counts`, with CIs."""
    tp, fp, fn = (counts[c].to_numpy(float) for c in ("TP_link", "FP_link", "FN_link"))
    f1_link = _f1(tp, fp, fn)
    f1_pair = counts["f1_pair"].to_numpy(float)
    det, exact = counts["pairs_detected"].to_numpy(float), counts["pairs_lag_exact"].to_numpy(float)

    def mean(w, x):
        fin = np.isfinite(x)
        return _ratio(w, np.where(fin, x, 0.0), fin.astype(float))

    def stats(w):
        return {"f1_link":       mean(w, f1_link),
                "f1_link_micro": _f1(w @ tp, w @ fp, w @ fn),
                "lag_exact":     _ratio(w, exact, det),
                "f1_pair":       mean(w, f1_pair)}

    n = len(counts)
    point = {k: float(v[0]) for k, v in stats(np.ones((1, n))).items()}
    point["n_scenarios"] = n
    return _with_ci(point, stats(_boot_counts(counts["config"].to_numpy(), rng)))


def _by_size(df: pd.DataFrame) -> list:
    return [("all", df)] + list(df.groupby("config", sort=True))


def var_tables(links: pd.DataFrame, counts: pd.DataFrame, f1: pd.DataFrame,
               rng: np.random.Generator) -> tuple:
    pair_f1 = f1[f1["family"] == VAR_FAMILY][["scenario", "model", "f1"]]
    counts = counts.merge(pair_f1.rename(columns={"f1": "f1_pair"}),
                          on=["scenario", "model"], how="left")

    rows_a, rows_b = [], []
    effect = links[links["kind"].notna()] if not links.empty else links
    for model, sub in (effect.groupby("model", sort=True) if not effect.empty else []):
        # resampling unit = every scenario the model ran on, with or without detected links
        ran = counts[counts["model"] == model.split(" ")[0]][["scenario", "config"]]
        for size, part in _by_size(ran):
            r = _table_4a(sub, list(part["scenario"]), part["config"].to_numpy(), rng)
            rows_a.append({"window": _window(model.split(" ")[0]), "model": model,
                           "kind": sub["kind"].iloc[0], "size": size, **r})

    for model, sub in counts.groupby("model", sort=True):
        for size, part in _by_size(sub):
            rows_b.append({"window": _window(model), "model": model, "size": size,
                           **_table_4b(part, rng)})
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)

# =============================================================================
# REPORT
# =============================================================================

def _size_order(s):
    return (0, 0) if s == "all" else (1, int(re.sub(r"\D", "", s) or 0))


def _report(root, manifest, f1, fam, cfg, delta, rank, t4a, t4b) -> str:
    SEP = "=" * 100
    seeds = sorted(int(x) for x in manifest["seed"].unique())
    L = [SEP, f"  E1a -- NEW REALIZATIONS OF THE SYNTHETIC GENERATORS -- {root.name}",
         f"  Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}",
         f"  Scenarios: {manifest['scenario'].nunique()}   Seeds: {seeds}",
         f"  Models: {sorted(f1['model'].unique())}",
         SEP,
         "  SCOPE: new realizations of DEVELOPMENT generators, NOT held-out data. If the",
         "  pipeline had been tuned to the realizations of the main run rather than to the",
         "  type of data, its performance would drop on these seeds.",
         f"  CIs: {int(CI_LEVEL * 100)}% percentile bootstrap, {N_BOOT} resamples "
         f"(seed {BOOT_SEED}).", ""]

    for window in sorted(fam["window"].unique()):
        L += [SEP, f"  PART 1 -- F1 PER FAMILY ({window}); CI over seeds", SEP]
        for family, sub in fam[fam["window"] == window].groupby("family", sort=True):
            n_cfg = cfg[(cfg["window"] == window) & (cfg["family"] == family)]["config"].nunique()
            L.append(f"\n  {family}  ({n_cfg} configuration(s) x seeds)")
            for r in sub.sort_values("f1_mean", ascending=False).itertuples():
                fail = f"  [{r.n_failed} failed]" if r.n_failed else ""
                L.append(f"    {r.model:<26} F1 {r.f1_mean:.3f} ± {r.f1_sd:.3f}   "
                         f"CI95 [{r.ci_low:.3f}, {r.ci_high:.3f}]   "
                         f"n={r.n_scenarios} ({r.n_seeds} seeds){fail}")
        d = delta[delta["window"] == window]
        if not d.empty:
            L += ["", f"  Paired difference {d['pipeline'].iloc[0]} - baseline "
                      f"(per seed; better / tied / worse seeds)"]
            for r in d.itertuples():
                L.append(f"    {r.family:<15} vs {r.baseline:<24} "
                         f"{_fmt_ci(r.delta_f1_mean, r.ci_low, r.ci_high)}   "
                         f"{r.seeds_pipeline_better}/{r.seeds_tied}/{r.seeds_pipeline_worse}")

        L += ["", SEP, f"  PART 2 -- RANKING STABILITY ({window}): seeds where the model is "
                       f"first or tied first", SEP]
        rk = rank[rank["window"] == window]
        for family, sub in rk.groupby("family", sort=True):
            L.append(f"\n  {family}")
            for r in sub.sort_values("seeds_first_or_tied", ascending=False).itertuples():
                star = "  <- pipeline" if r.is_pipeline else ""
                L.append(f"    {r.model:<26} {r.seeds_first_or_tied:>2}/{r.n_seeds} seeds "
                         f"({r.seeds_sole_first} alone)   "
                         f"{r.scenarios_first_or_tied:>3}/{r.n_scenarios} scenarios{star}")

    if not t4a.empty or not t4b.empty:
        n_var = manifest[manifest["family"] == VAR_FAMILY]["scenario"].nunique()
        L += ["", SEP, f"  PART 3 -- VAR SCENARIOS ({n_var}): TABLES 4a / 4b", SEP,
              "  Reference value theta* = beta * sd(source)/sd(target) (estimates are on",
              "  z-scored series). 4a is conditional on detecting the link at the true lag.",
              "  CIs: bootstrap over scenarios, stratified by size.", ""]
        L.append("  TABLE 4a -- effect estimation (all sizes)")
        L.append(f"    {'model':<34}{'n':>5}  {'sign correct':<26}{'median rel. error':<26}"
                 f"{'coverage (95% CI)':<26}")
        for r in t4a[t4a["size"] == "all"].itertuples():
            cov = (_fmt_ci(r.coverage, r.coverage_ci_low, r.coverage_ci_high)
                   if np.isfinite(r.coverage) else "n/a")
            err = (_fmt_ci(r.median_rel_err, r.median_rel_err_ci_low, r.median_rel_err_ci_high)
                   if r.kind == "coefficient" else "n/a (partial corr.)")
            L.append(f"    {r.model:<34}{r.n_links:>5}  "
                     f"{_fmt_ci(r.sign_correct, r.sign_correct_ci_low, r.sign_correct_ci_high):<26}"
                     f"{err:<26}{cov:<26}")
        L.append("    sensitivity, against the RAW beta: "
                 + "; ".join(f"{r.model}: rel.err {r.median_rel_err_raw:.3f}, cov "
                             f"{r.coverage_raw:.3f}" if np.isfinite(r.coverage_raw)
                             else f"{r.model}: rel.err {r.median_rel_err_raw:.3f}"
                             for r in t4a[(t4a["size"] == "all")
                                          & (t4a["kind"] == "coefficient")].itertuples()))

        L += ["", "  TABLE 4b -- link recovery (all sizes)",
              f"    {'model':<26}{'F1 per link':<26}{'F1 per link (pooled)':<26}"
              f"{'lag exact':<26}{'F1 per pair':<26}"]
        for r in t4b[t4b["size"] == "all"].itertuples():
            L.append(f"    {r.model:<26}"
                     f"{_fmt_ci(r.f1_link, r.f1_link_ci_low, r.f1_link_ci_high):<26}"
                     f"{_fmt_ci(r.f1_link_micro, r.f1_link_micro_ci_low, r.f1_link_micro_ci_high):<26}"
                     f"{_fmt_ci(r.lag_exact, r.lag_exact_ci_low, r.lag_exact_ci_high):<26}"
                     f"{_fmt_ci(r.f1_pair, r.f1_pair_ci_low, r.f1_pair_ci_high):<26}")

        L += ["", "  Per VAR size: var_table4a.csv / var_table4b.csv (column 'size')."]
        with pd.option_context("display.width", 250, "display.max_rows", 500):
            b = t4b[t4b["size"] != "all"].copy()
            b["_o"] = b["size"].map(_size_order)
            L += ["", b.sort_values(["model", "_o"])[
                ["model", "size", "n_scenarios", "f1_link", "lag_exact", "f1_pair"]
            ].round(3).to_string(index=False)]
    return "\n".join(L) + "\n"


def run(run_dir: str):
    root = Path(run_dir)
    manifest = _read_csv(root / "seeds_manifest.csv")
    if manifest.empty:
        print(f"[WARNING] No seeds_manifest.csv in '{root}' -- is it a "
              f"main_seed_replication.py run?")
        return
    rng = np.random.default_rng(BOOT_SEED)

    f1 = load_f1(root, manifest)
    if f1.empty:
        print(f"[WARNING] No model results found in '{root}'.")
        return
    fam, cfg, delta = f1_tables(f1, rng)
    rank = ranking(f1)

    links, counts = var_links(root, manifest)
    t4a = t4b = pd.DataFrame()
    if not counts.empty:
        t4a, t4b = var_tables(links, counts, f1, rng)

    f1.to_csv(root / "seed_f1_per_scenario.csv", index=False)
    fam.round(4).to_csv(root / "seed_f1_family.csv", index=False)
    cfg.round(4).to_csv(root / "seed_f1_config.csv", index=False)
    delta.round(4).to_csv(root / "seed_f1_delta.csv", index=False)
    rank.to_csv(root / "seed_ranking.csv", index=False)
    links.to_csv(root / "var_effect_links.csv", index=False)
    t4a.round(4).to_csv(root / "var_table4a.csv", index=False)
    t4b.round(4).to_csv(root / "var_table4b.csv", index=False)

    text = _report(root, manifest, f1, fam, cfg, delta, rank, t4a, t4b)
    (root / "seed_replication_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"  Summary -> {root / 'seed_replication_summary.txt'}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else RUN_DIR)

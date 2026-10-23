"""
discovery.py -- Stages 3-5, the correlation pre-filter: orchestrates
Stage 3 (xcorr) + Stage 4 (TE) + Stage 5 (rescues: Granger, Pearson
bootstrap), applying multiple-comparison correction
(FDR-BH by default, via causal_common._correct_pvalues) to the p-value
of ALL pairs tested in each stage, BEFORE deciding who advances.

Does not contain the math of each test -- that lives in xcorr.py,
transfer_entropy.py and rescue.py. This file only orchestrates: it
builds the tasks, dispatches them in parallel (joblib if available),
applies the correction, and decides who survives/gets rescued, plus
optional logging via WatchPairs.

TWO LAYERS OF MULTIPLICITY CORRECTION:
  1. ACROSS PAIRS (always on by default): FDR-BH/Bonferroni over the
     p-value of every pair tested in each stage -- xcorr_te_correction.
  2. WITHIN each pair, over the up-to-max_lag lags scanned by the
     argmax (correct_lag_multiplicity, off by default -- extra
     permutation cost): xcorr.max_stat_permutation_pvalue /
     transfer_entropy.max_stat_permutation_pvalue replace the winning
     lag's p-value with a permutation p-value of the MAXIMUM score
     over all lags, before layer 1 is applied to the batch.
"""

from typing import Optional
import numpy as np
import pandas as pd

from .penalties import LagPenaltyFn, resolve_lag_penalty
from Auxiliary.causal_common import _correct_pvalues
from .xcorr import xcorr_filter_pair
from .transfer_entropy import te_filter_pair
from .rescue import granger_rescue_pair, pearson_rescue_pair

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def filter_causal_candidates(
    processed: dict,
    pairs: list,
    max_lag: int = 10,
    alpha: float = 0.10,
    min_corr: float = 0.05,
    use_xcorr: bool = True,
    min_te: float = 0.005,
    use_te: bool = True,
    te_method: str = 'histogram',
    te_bins: int = 6,
    te_k_knn: int = 4,
    te_n_perms: int = 50,
    n_jobs: int = 1,
    block_same_property: bool = True,
    focus_entity: Optional[object] = None,
    include_self: bool = False,
    target_entity: Optional[object] = None,
    # ── Granger rescue ─────────────────────────────────────────────────
    use_granger_rescue: bool = True,
    alpha_granger: Optional[float] = None,
    # ── Pearson bootstrap rescue ──────────────────────────────────────
    use_pearson_rescue: bool = True,
    alpha_pearson: Optional[float] = None,
    min_corr_rescue: float = 0.20,
    n_boot: int = 200,
    boot_block_size: Optional[int] = None,
    # ── Generic lag penalty ──────────────────────────────────────────
    lag_penalty_fn: Optional[LagPenaltyFn] = None,
    # ── Knee lag-penalty parameters ──────────────────────────────────
    lag_knee1: int = 4,
    lag_knee2: int = 6,
    lag_rate_zone1: float = 0.02,
    lag_rate_zone2: float = 0.15,
    lag_rate_zone3: float = 0.50,
    use_lag_penalty: bool = True,
    # ── Tracking of pairs of interest ────────────────────────────────
    watch_pairs: Optional['object'] = None,
    # ── Multiple-comparison correction in Stages 3-4 ─────────────────
    xcorr_te_correction: str = 'fdr_bh',
    # ── LAG multiplicity correction within each pair (optional) ──────
    correct_lag_multiplicity: bool = False,
    n_perms_lag_correction: Optional[int] = None,
    correct_lag_multiplicity_rescue: bool = False,
) -> list:
    """
    xcorr_te_correction : 'none' | 'bonferroni' | 'fdr_bh' (default).
        Applied to the raw p-value of ALL pairs tested in Stage 3 and,
        separately, in Stage 4 (over the subset that survived the
        corrected Stage 3) -- the same pattern used by PCMCI pipelines
        over their exhaustive p_matrix.

    correct_lag_multiplicity : bool (default False).
        If True, replaces the p-value of each pair (BEFORE the batch
        correction above) with a max-statistic permutation p-value over
        ALL up to max_lag lags scanned by the lag argmax, not just the
        winning lag -- see xcorr.max_stat_permutation_pvalue and
        transfer_entropy.max_stat_permutation_pvalue. Extra cost: up to
        n_perms_lag_correction x max_lag evaluations per pair (only for
        pairs that would pass the min_corr/min_te gate).
    n_perms_lag_correction : optional int.
        Number of permutations for the correction above; uses te_n_perms
        if None.
    """
    _pen_fn = resolve_lag_penalty(
        lag_penalty_fn, use_lag_penalty,
        lag_knee1, lag_knee2,
        lag_rate_zone1, lag_rate_zone2, lag_rate_zone3,
    )
    _n_perms_lag = n_perms_lag_correction if n_perms_lag_correction is not None else te_n_perms
    print(f"  [lag_penalty] {_pen_fn}")
    print(f"  [multiple-comparison correction — Stages 3-4] {xcorr_te_correction}")
    print(f"  [LAG multiplicity correction — within each pair] "
          f"{'on (n_perms=' + str(_n_perms_lag) + ')' if correct_lag_multiplicity else 'off'}")

    valid_pairs = _build_valid_pairs(
        pairs, focus_entity, target_entity, include_self, block_same_property
    )
    print(f"  Initial candidate pairs: {len(valid_pairs):,}")

    survivors = valid_pairs

    # ── Stage 3: xcorr ────────────────────────────────────────────────
    if use_xcorr:
        passed_xcorr, xcorr_results, _xcorr_removed, _watch_xcorr_removed = _run_stage3_xcorr(
            processed, survivors, max_lag, alpha, min_corr, _pen_fn,
            n_jobs, xcorr_te_correction, watch_pairs, valid_pairs,
            correct_lag_multiplicity, _n_perms_lag,
        )
        survivors = [(r['source'], r['target']) for r in passed_xcorr]
    else:
        passed_xcorr = [
            {'source': src, 'target': tgt, 'lag': 1,
             'corr': 0.0, 'pvalue_discovery': 1.0, 'pvalue_corrected': 1.0,
             'filter_method': 'none'}
            for src, tgt in survivors
        ]
        print(f"  [Stage 3] disabled — {len(survivors)} pairs move on to stage 4")
        _xcorr_removed = set()
        _watch_xcorr_removed = set()

    if not valid_pairs:
        return []

    # ── Stage 4: Transfer Entropy + Stage 5: rescues ──────────────────────────
    # Does NOT return early just because `survivors` (post-xcorr) is
    # empty: if Stage 3 rejects ALL pairs, Stage 4 (with an empty task
    # list) is a harmless no-op, but the Granger/Pearson rescues still
    # need to run over `_xcorr_removed` -- this is exactly the case they
    # exist for. Aborting here would skip the entire rescue step
    # whenever 100% of the pairs failed xcorr.
    if use_te:
        candidates = _run_stage4_te(
            processed, survivors, passed_xcorr, max_lag, alpha, min_te,
            te_method, te_bins, te_k_knn, te_n_perms, use_xcorr,
            _pen_fn, n_jobs, xcorr_te_correction, watch_pairs,
            correct_lag_multiplicity, _n_perms_lag,
        )
        # 'auto': applies the lag multiplicity correction to the rescues
        # only when the series have a genuine time axis (see
        # _detect_time_axis). True/False force the behavior.
        if correct_lag_multiplicity_rescue == 'auto':
            _correct_rescue_lag = _detect_time_axis(processed)
        else:
            _correct_rescue_lag = bool(correct_lag_multiplicity_rescue)

        candidates = _run_stage5_rescues(
            processed, survivors, candidates, _xcorr_removed,
            max_lag, alpha, alpha_granger, alpha_pearson,
            min_corr_rescue, n_boot, boot_block_size,
            use_granger_rescue, use_pearson_rescue,
            _pen_fn, n_jobs, watch_pairs,
            correct_lag_multiplicity_rescue=_correct_rescue_lag,
        )
    else:
        candidates = passed_xcorr
        print(f"  [Stage 4] disabled — using {len(candidates)} pairs from stage 3")

    if watch_pairs is not None and len(watch_pairs) > 0:
        _log_unrescued_watch_pairs(watch_pairs, candidates, _watch_xcorr_removed,
                                    use_te, use_granger_rescue, use_pearson_rescue)

    return candidates


# =============================================================================
# Internal helpers (kept private: the public interface is just
# filter_causal_candidates)
# =============================================================================

def _build_valid_pairs(pairs, focus_entity, target_entity, include_self, block_same_property):
    if focus_entity is None:
        focus_set = None
    elif isinstance(focus_entity, str):
        focus_set = {focus_entity}
    else:
        focus_set = set(focus_entity)

    if target_entity is None:
        target_set = None
    elif isinstance(target_entity, str):
        target_set = {target_entity}
    else:
        target_set = set(target_entity)

    available_entities = {p.split('__', 1)[0] for p in pairs}

    if focus_set is not None:
        missing = focus_set - available_entities
        if missing:
            raise ValueError(
                f"focus_entity: entity(ies) not found: {sorted(missing)}. "
                f"Available: {sorted(available_entities)}"
            )
        print(f"  [focus_entity]  {sorted(focus_set)} | "
              f"include_self={'YES' if include_self else 'NO'}")

    if target_set is not None:
        missing_t = target_set - available_entities
        if missing_t:
            raise ValueError(
                f"target_entity: entity(ies) not found: {sorted(missing_t)}. "
                f"Available: {sorted(available_entities)}"
            )
        print(f"  [target_entity] target restricted to {sorted(target_set)}")

    blocked_same_property = 0
    valid_pairs = []
    for src in pairs:
        src_entity, src_property = src.split('__', 1)
        for tgt in pairs:
            if src == tgt:
                continue
            tgt_entity, tgt_property = tgt.split('__', 1)
            is_self_pair = (src_entity == tgt_entity)

            if is_self_pair:
                if not include_self:
                    continue
                if focus_set is not None and src_entity not in focus_set:
                    continue
                if target_set is not None and tgt_entity not in target_set:
                    continue
                valid_pairs.append((src, tgt))
                continue

            if focus_set is not None:
                if src_entity not in focus_set and tgt_entity not in focus_set:
                    continue
            if target_set is not None and tgt_entity not in target_set:
                continue
            if (focus_set is not None and target_set is not None
                    and focus_set == target_set):
                if src_entity not in focus_set or tgt_entity not in target_set:
                    continue
            if block_same_property and src_property == tgt_property:
                blocked_same_property += 1
                continue

            valid_pairs.append((src, tgt))

    if block_same_property and blocked_same_property > 0:
        print(f"  Same-property pairs ignored: {blocked_same_property} "
              f"(block_same_property=True)")

    return valid_pairs


def _run_stage3_xcorr(processed, survivors, max_lag, alpha, min_corr, pen_fn,
                       n_jobs, xcorr_te_correction, watch_pairs, valid_pairs,
                       correct_lag_multiplicity=False, n_perms_lag=50):
    tasks_xcorr = [
        (src, tgt, processed[src], processed[tgt], max_lag, alpha, min_corr, pen_fn,
         correct_lag_multiplicity, n_perms_lag)
        for src, tgt in survivors
    ]
    print(f"\n  [Stage 3] Lagged cross-correlation "
          f"(alpha={alpha}, min_corr={min_corr}, correction={xcorr_te_correction})...")

    if HAS_JOBLIB and n_jobs != 1:
        xcorr_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(xcorr_filter_pair)(t) for t in tasks_xcorr
        )
    else:
        xcorr_results = [xcorr_filter_pair(t) for t in tasks_xcorr]

    raw_pvalues = [r['pvalue_discovery'] for r in xcorr_results]
    corrected_pvalues = _correct_pvalues(raw_pvalues, xcorr_te_correction)
    for r, p_corr in zip(xcorr_results, corrected_pvalues):
        r['pvalue_corrected'] = round(float(p_corr), 4)

    passed_xcorr = [
        r for r in xcorr_results
        if r['pvalue_corrected'] < alpha and abs(r['corr']) > min_corr
    ]
    pct = len(passed_xcorr) / max(len(tasks_xcorr), 1) * 100
    print(f"  [Stage 3] {len(passed_xcorr)} pairs passed ({pct:.1f}%) "
          f"[{len(xcorr_results)} tested, correction={xcorr_te_correction}]")

    if watch_pairs is not None and len(watch_pairs) > 0:
        _log_watch_stage3(watch_pairs, tasks_xcorr, xcorr_results, passed_xcorr, alpha, min_corr, xcorr_te_correction)

    passed_keys_xcorr = {(r['source'], r['target']) for r in passed_xcorr}
    _xcorr_removed = {
        (src, tgt) for src, tgt in valid_pairs if (src, tgt) not in passed_keys_xcorr
    }
    _watch_xcorr_removed = set()
    if watch_pairs is not None and len(watch_pairs) > 0:
        for src, tgt in _xcorr_removed:
            if watch_pairs.is_watched_key(src, tgt):
                _watch_xcorr_removed.add((src, tgt))

    return passed_xcorr, xcorr_results, _xcorr_removed, _watch_xcorr_removed


def _log_watch_stage3(watch_pairs, tasks_xcorr, xcorr_results, passed_xcorr, alpha, min_corr, xcorr_te_correction):
    passed_keys = {(r['source'], r['target']) for r in passed_xcorr}
    for task_args in tasks_xcorr:
        src, tgt = task_args[0], task_args[1]
        if not watch_pairs.is_watched_key(src, tgt):
            continue
        if (src, tgt) in passed_keys:
            res_wp = next((r for r in passed_xcorr if r['source'] == src and r['target'] == tgt), None)
            detail = ''
            if res_wp:
                detail = (f"corr={res_wp.get('corr', '?'):.4f}, lag={res_wp.get('lag', '?')}, "
                          f"pval={res_wp.get('pvalue_discovery', '?'):.4f}, "
                          f"pval_corrected={res_wp.get('pvalue_corrected', '?'):.4f}, "
                          f"pen_score={res_wp.get('lag_penalty_score', '?'):.4f}")
            watch_pairs.record_by_keys(src, tgt, 'xcorr', 'passed', detail)
            print(f"\n  *** [WATCH] xcorr ✔ PASSED: {src} → {tgt}")
            if detail:
                print(f"             {detail}")
        else:
            all_res_wp = next((r for r in xcorr_results if r['source'] == src and r['target'] == tgt), None)
            reasons = []
            if all_res_wp is not None:
                if all_res_wp['pvalue_corrected'] >= alpha:
                    reasons.append(
                        f"pval_corrected={all_res_wp['pvalue_corrected']:.4f} >= alpha={alpha} "
                        f"(raw pval={all_res_wp['pvalue_discovery']:.4f}, correction={xcorr_te_correction})"
                    )
                if abs(all_res_wp['corr']) <= min_corr:
                    reasons.append(f"|corr|={abs(all_res_wp['corr']):.4f} <= min_corr={min_corr}")
            if not reasons:
                reasons.append("did not reach the penalized threshold after correction")
            reason_str = '; '.join(reasons)
            watch_pairs.record_by_keys(src, tgt, 'xcorr', 'removed', reason_str)
            print(f"\n  *** [WATCH] xcorr ✘ REMOVED: {src} → {tgt}")
            print(f"             Reason: {reason_str}")


def _run_stage4_te(processed, survivors, passed_xcorr, max_lag, alpha, min_te,
                    te_method, te_bins, te_k_knn, te_n_perms, use_xcorr,
                    pen_fn, n_jobs, xcorr_te_correction, watch_pairs,
                    correct_lag_multiplicity=False, n_perms_lag=50):
    lag_hint = {(r['source'], r['target']): r['lag'] for r in passed_xcorr if 'lag' in r}

    tasks_te = [
        (src, tgt, processed[src], processed[tgt],
         max_lag, alpha, min_te, te_method, te_bins, te_k_knn, te_n_perms, pen_fn,
         correct_lag_multiplicity, n_perms_lag)
        for src, tgt in survivors
    ]
    print(f"\n  [Stage 4] Transfer Entropy ({te_method}) "
          f"(alpha={alpha}, min_te={min_te}, perms={te_n_perms}, correction={xcorr_te_correction})...")

    if HAS_JOBLIB and n_jobs != 1:
        te_results_raw = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(te_filter_pair)(t) for t in tasks_te
        )
    else:
        te_results_raw = [te_filter_pair(t) for t in tasks_te]

    raw_pvalues = [r['pvalue_discovery'] for r in te_results_raw]
    corrected_pvalues = _correct_pvalues(raw_pvalues, xcorr_te_correction)
    for r, p_corr in zip(te_results_raw, corrected_pvalues):
        r['pvalue_corrected'] = round(float(p_corr), 4)

    candidates = []
    for r in te_results_raw:
        if r['pvalue_corrected'] >= alpha:
            continue
        key = (r['source'], r['target'])
        if key in lag_hint and use_xcorr:
            r['lag'] = lag_hint[key]
        candidates.append(r)

    pct = len(candidates) / max(len(tasks_te), 1) * 100
    print(f"  [Stage 4] {len(candidates)} pairs passed ({pct:.1f}%) "
          f"[{len(te_results_raw)} tested, correction={xcorr_te_correction}]")

    if watch_pairs is not None and len(watch_pairs) > 0:
        _log_watch_stage4(watch_pairs, tasks_te, te_results_raw, candidates, min_te, alpha, xcorr_te_correction)

    return candidates


def _log_watch_stage4(watch_pairs, tasks_te, te_results_raw, candidates, min_te, alpha, xcorr_te_correction):
    passed_te_keys = {(r['source'], r['target']) for r in candidates}
    for task_args in tasks_te:
        src, tgt = task_args[0], task_args[1]
        if not watch_pairs.is_watched_key(src, tgt):
            continue
        if (src, tgt) in passed_te_keys:
            res_wp = next((r for r in candidates if r['source'] == src and r['target'] == tgt), None)
            detail = ''
            if res_wp:
                detail = (f"te={res_wp.get('te', '?'):.6f}, lag={res_wp.get('lag', '?')}, "
                          f"pval={res_wp.get('pvalue_discovery', '?'):.4f}, "
                          f"pval_corrected={res_wp.get('pvalue_corrected', '?'):.4f}, "
                          f"method={res_wp.get('filter_method', '?')}")
            watch_pairs.record_by_keys(src, tgt, 'TE', 'passed', detail)
            print(f"\n  *** [WATCH] TE ✔ PASSED: {src} → {tgt}")
            if detail:
                print(f"             {detail}")
        else:
            all_res_wp = next((r for r in te_results_raw if r['source'] == src and r['target'] == tgt), None)
            reasons = []
            if all_res_wp is not None:
                if all_res_wp.get('te', 0.0) < min_te:
                    reasons.append(f"TE={all_res_wp.get('te', 0.0):.6f} < min_te={min_te}")
                else:
                    reasons.append(
                        f"TE={all_res_wp.get('te', 0.0):.6f}, "
                        f"pval_corrected={all_res_wp['pvalue_corrected']:.4f} >= alpha={alpha} "
                        f"(raw pval={all_res_wp['pvalue_discovery']:.4f}, correction={xcorr_te_correction})"
                    )
            if not reasons:
                reasons.append("did not reach the penalized threshold after correction")
            reason_str = '; '.join(reasons)
            watch_pairs.record_by_keys(src, tgt, 'TE', 'removed', reason_str)
            print(f"\n  *** [WATCH] TE ✘ REMOVED: {src} → {tgt}")
            print(f"             Reason: {reason_str}")


def _detect_time_axis(processed, alpha_white_noise: float = 2.0,
                       verbose: bool = True) -> bool:
    """
    Decides, using ONLY the data (no ground truth and no dataset name),
    whether the series have genuine temporal structure -- that is,
    whether it makes sense to treat a lag search as a temporal search.

    Statistic: the median of |ACF(1)| over all series, compared against
    the confidence interval of the ACF under the null hypothesis of
    white noise (+- alpha_white_noise/sqrt(n), with alpha=2 ~ 95%). If
    the lag-1 autocorrelation is indistinguishable from white noise, the
    observations are effectively i.i.d. and there is no time axis --
    the case, for example, of datasets that sample a Bayesian network
    row by row (each row is an independent unit, not a time instant).

    Why this matters: the rescues' lag multiplicity correction (see
    rescue.py) charges the price of having scanned up to max_lag lags.
    That price assumes the lag search is a genuine temporal search. On
    i.i.d. data there is nothing to search for -- the pipeline is
    operating outside the domain the step was designed for, and
    applying a temporal penalty there does not make the result more
    valid, only more conservative about a dimension that does not exist
    in that data.

    A dataset with no genuine time axis (e.g. i.i.d. sampling from a
    Bayesian network) is expected to fall
    below the white-noise threshold; a dataset with genuine AR structure,
    even weak, should stay comfortably above it.
    """
    acf1_values, sizes = [], []
    for series in processed.values():
        v = np.asarray(series, dtype=float)
        v = v[~np.isnan(v)]
        if len(v) < 30:
            continue
        v = v - v.mean()
        den = float(np.dot(v, v))
        if den <= 0:
            continue
        acf1_values.append(abs(float(np.dot(v[:-1], v[1:]) / den)))
        sizes.append(len(v))

    if not acf1_values:
        if verbose:
            print("  [time axis] not enough series to decide -> assuming temporal")
        return True

    acf1_median = float(np.median(acf1_values))
    n_median = int(np.median(sizes))
    threshold = alpha_white_noise / np.sqrt(max(n_median, 1))
    has_time_axis = bool(acf1_median >= threshold)
    if verbose:
        print(f"  [time axis] median |ACF(1)|={acf1_median:.4f} "
              f"vs white-noise threshold={threshold:.4f} (n={n_median}) -> "
              f"{'TEMPORAL' if has_time_axis else 'i.i.d. (no time axis)'}")
    return has_time_axis


def _run_stage5_rescues(processed, survivors, candidates, _xcorr_removed,
                  max_lag, alpha, alpha_granger, alpha_pearson,
                  min_corr_rescue, n_boot, boot_block_size,
                  use_granger_rescue, use_pearson_rescue,
                  pen_fn, n_jobs, watch_pairs,
                  correct_lag_multiplicity_rescue=False):
    approved_keys = {(r['source'], r['target']) for r in candidates}

    def _pairs_to_rescue():
        base = [(s, t) for s, t in survivors if (s, t) not in approved_keys]
        extra = [(s, t) for s, t in _xcorr_removed
                 if (s, t) not in approved_keys and s in processed and t in processed]
        return base + extra

    if use_granger_rescue:
        _alpha_g = alpha_granger if alpha_granger is not None else alpha
        rescue_pairs = _pairs_to_rescue()
        rescue_tasks = [
            (src, tgt, processed[src], processed[tgt], max_lag, _alpha_g, pen_fn,
             correct_lag_multiplicity_rescue)
            for src, tgt in rescue_pairs
        ]
        if rescue_tasks:
            print(f"\n  [Stage 5 Granger rescue] Testing {len(rescue_tasks)} pairs "
                  f"discarded by TE (alpha={_alpha_g}, "
                  f"lag multiplicity correction="
                  f"{'Sidak' if correct_lag_multiplicity_rescue else 'none'})...")
            if HAS_JOBLIB and n_jobs != 1:
                rescue_results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(granger_rescue_pair)(t) for t in rescue_tasks
                )
            else:
                rescue_results = [granger_rescue_pair(t) for t in rescue_tasks]
            rescued = [r for r in rescue_results if r is not None]
            print(f"  [Stage 5 Granger rescue] {len(rescued)} pairs rescued (filter_method='Granger_rescue')")
            candidates.extend(rescued)
            approved_keys = {(r['source'], r['target']) for r in candidates}
        else:
            print("  [Stage 5 Granger rescue] No pairs to rescue.")

    if use_pearson_rescue:
        _alpha_p = alpha_pearson if alpha_pearson is not None else alpha
        pearson_pairs = _pairs_to_rescue()
        pearson_tasks = [
            (src, tgt, processed[src], processed[tgt],
             max_lag, _alpha_p, min_corr_rescue, n_boot, boot_block_size, pen_fn,
             correct_lag_multiplicity_rescue)
            for src, tgt in pearson_pairs
        ]
        if pearson_tasks:
            print(f"\n  [Stage 5 Pearson rescue] Testing {len(pearson_tasks)} pairs still discarded "
                  f"(bootstrap, alpha={_alpha_p}, min_corr={min_corr_rescue}, n_boot={n_boot}, "
                  f"lag multiplicity correction="
                  f"{'Sidak' if correct_lag_multiplicity_rescue else 'none'})...")
            if HAS_JOBLIB and n_jobs != 1:
                pearson_results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(pearson_rescue_pair)(t) for t in pearson_tasks
                )
            else:
                pearson_results = [pearson_rescue_pair(t) for t in pearson_tasks]
            rescued_p = [r for r in pearson_results if r is not None]
            print(f"  [Stage 5 Pearson rescue] {len(rescued_p)} pairs rescued (filter_method='Pearson_rescue')")
            candidates.extend(rescued_p)
        else:
            print("  [Stage 5 Pearson rescue] No remaining pairs to rescue.")

    return candidates


def _log_unrescued_watch_pairs(watch_pairs, candidates, _watch_xcorr_removed,
                                use_te, use_granger_rescue, use_pearson_rescue):
    approved_final = {(r['source'], r['target']) for r in candidates}
    for src, tgt in _watch_xcorr_removed:
        if not watch_pairs.is_watched_key(src, tgt):
            continue
        k = watch_pairs.get_key_from_keys(src, tgt)
        if k is None:
            continue
        events = watch_pairs._log.get(k, [])
        rescue_stages = {e['stage'] for e in events
                          if e['stage'] in ('Granger_rescue', 'Pearson_rescue', 'TE')}
        if not rescue_stages:
            reason = "rescues disabled (use_te=False or use_granger_rescue=False and use_pearson_rescue=False)"
            watch_pairs.record_by_keys(src, tgt, 'rescue', 'not_rescued', reason)
            print(f"\n  *** [WATCH] ✘ NOT RESCUED (no rescue stage): {src} → {tgt}")
            print(f"             Reason: {reason}")

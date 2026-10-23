"""
coherence.py — evidence-coherence guard (assessed at the end of Stage 6,
applied by the Stage 8 regime guards).

Stages 7 (causal penalty) and 8a (structural pruning) remove pairs by
RANKING them on the evidence Stages 3-6 produced. That only removes false
pairs preferentially if the evidence carries structure; when it does not,
the same stages remove true and false pairs at the same rate and every
removal costs recall for nothing.

The pipeline has two evidence channels that are computed independently of
each other for every candidate pair:

    - directional, model-free:   transfer entropy            (Stages 3-5)
    - conditional, model-based:  |t| of the DML effect       (Stage 6)

If the pairs carry a real signal the two channels agree on which pairs are
strong, so their rank correlation rho over the candidates is positive. The
guard classifies the dataset from rho and its confidence interval (Fisher
z-transform of Spearman's rho, one-sided, level `alpha`):

    INCOHERENT    no candidate was confirmed by transfer entropy (every pair
                  came in through a rescue, so the directional channel is
                  empty), or the UPPER confidence bound of rho is below
                  `min_rho`: with confidence 1 - alpha the two channels
                  agree less than a medium correlation.
    COHERENT      the LOWER confidence bound of rho is above zero.
    UNDETERMINED  anything else (few candidates, wide interval): the guard
                  stays out and the pipeline runs as usual.

`min_rho` = 0.3 is the conventional "medium" correlation (Cohen, 1988), not
a value fitted to any dataset. The guard never looks at ground truth, at the
dataset's name or at its size: the decision is a property of the candidate
table alone, and with few candidates the interval is wide, so a small graph
is never declared incoherent.

In the incoherent regime the pipeline keeps every pair with a significant
Stage 6 effect and does not apply the Stage 7 / 8a rejections (see
Pipeline/pipeline.py and keep_stage6_pairs below). The pairs are still
tested: only the ranking-based pruning, which needs coherent evidence to be
more than a random cut, is suspended. The regime is written to the output
('evidence_regime', 'evidence_coherence_rho') so those edges can be reported
as low-confidence.

EXCEPTION (pipeline option evidence_coherence_fallback='auto'). The fallback
trusts the lagged Stage 6 tests alone, which presumes that the series are
driven by their past. When the past of all series explains little of each
of them (median out-of-sample R2 at or below the Stage 8b gate,
contemporaneous_max_past_r2), the dependence between the series is mostly
contemporaneous -- what temporal aggregation does to lagged causality
(Breitung & Swanson, 2002; Gong et al., 2017) -- and a large set of
Stage-6-significant lagged pairs is not credible. The fallback is then not
applied: the Stage 7/8a decision stands and Stage 8b adds the lag-0 links.
The R2 is the larger of the all-series and the own-past estimates
(evidence_fallback_own_past_floor), so that a short series is not mistaken
for one without dynamics. See _stage8_regime_guards in Pipeline/pipeline.py
and past_explained_variance / own_past_explained_variance in
Pipeline/contemporaneous.py; 'evidence_fallback' and 'evidence_past_r2' in
the output record which path was taken.

WHY THIS FALLBACK
    The Stage 6 decision stands on its own. Double/debiased machine learning
    gives a valid test of the conditional effect (Neyman-orthogonal score,
    cross-fitting) without any of the later stages (Chernozhukov et al.,
    2018), so keeping exactly the pairs it accepts is a complete decision
    rule, not a relaxation of one.

    Stage 7 should not be applied to conflicting sources. It combines the
    two channels by Dempster-Shafer fusion, and Dempster's rule is known to
    give unreliable, counter-intuitive results when the sources conflict
    (Zadeh, 1984); the remedies in that literature are to discount or leave
    out the sources that cannot be trusted rather than to combine them
    (Shafer, 1976; Murphy, 2000; Smets, 2007). Suspending the fusion
    when its two main sources disagree is that remedy applied at the level
    of the whole dataset. Stage 8a ranks competing parents on the same
    evidence and is suspended with it.

    Agreement between methods with unrelated sources of bias as a
    reliability signal, and its absence as a reason for caution, is the
    triangulation argument (Lawlor, Tilling & Davey Smith, 2016); using the
    compatibility of a method's own outputs to assess causal discovery
    without ground truth follows Faller et al. (2024).

    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
        Newey, W. & Robins, J. (2018). Double/debiased machine learning for
        treatment and structural parameters. The Econometrics Journal,
        21(1), C1-C68.
    Faller, P. M., Vankadara, L. C., Mastakouri, A. A., Locatello, F. &
        Janzing, D. (2024). Self-compatibility: evaluating causal discovery
        without ground truth. AISTATS 2024, PMLR 238, 4132-4140.
    Lawlor, D. A., Tilling, K. & Davey Smith, G. (2016). Triangulation in
        aetiological epidemiology. International Journal of Epidemiology,
        45(6), 1866-1886.
    Murphy, C. K. (2000). Combining belief functions when evidence
        conflicts. Decision Support Systems, 29(1), 1-9.
    Shafer, G. (1976). A Mathematical Theory of Evidence. Princeton
        University Press.
    Smets, P. (2007). Analyzing the combination of conflicting belief
        functions. Information Fusion, 8(4), 387-412.
    Zadeh, L. A. (1984). Review of "A Mathematical Theory of Evidence".
        AI Magazine, 5(3), 81-83.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def assess_evidence_coherence(
    results: pd.DataFrame,
    min_rho: float = 0.3,
    alpha: float = 0.05,
    min_pairs: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Returns {'regime': 'coherent' | 'incoherent' | 'undetermined',
             'rho', 'rho_lower', 'rho_upper': float (NaN when not
             computable), 'n': int, 'reason': str}.
    """
    out = {'regime': 'undetermined', 'rho': float('nan'),
           'rho_lower': float('nan'), 'rho_upper': float('nan'),
           'n': 0, 'reason': ''}
    if results is None or results.empty or \
            not {'transfer_entropy', 't_stat'}.issubset(results.columns):
        out['reason'] = 'no candidate table'
        return _report(out, verbose)

    te = pd.to_numeric(results['transfer_entropy'], errors='coerce')
    t = pd.to_numeric(results['t_stat'], errors='coerce').abs()
    ok = te.notna() & t.notna() & np.isfinite(t)
    te, t = te[ok].to_numpy(), t[ok].to_numpy()
    out['n'] = int(len(t))

    if len(t) < min_pairs:
        out['reason'] = f'{len(t)} candidates < {min_pairs}: rank agreement not assessable'
        return _report(out, verbose)

    if np.ptp(te) <= 0:
        out['regime'] = 'incoherent'
        out['reason'] = 'no candidate carries transfer-entropy evidence'
        return _report(out, verbose)
    if np.ptp(t) <= 0:
        out['reason'] = 'constant effect evidence'
        return _report(out, verbose)

    rho = float(stats.spearmanr(te, t)[0])
    out['rho'] = rho
    if not np.isfinite(rho):
        out['reason'] = 'rank correlation not computable'
        return _report(out, verbose)

    # Fisher z interval; 1.06 / (n - 3) is the variance of atanh(rho) for
    # Spearman's coefficient (Fieller, Hartley & Pearson, 1957).
    half = stats.norm.ppf(1.0 - alpha) * np.sqrt(1.06 / (len(t) - 3))
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    lower, upper = float(np.tanh(z - half)), float(np.tanh(z + half))
    out['rho_lower'], out['rho_upper'] = lower, upper

    if upper < min_rho:
        out['regime'] = 'incoherent'
        out['reason'] = (f'transfer entropy and DML |t| do not agree on the ranking '
                         f'(upper bound {upper:+.2f} < {min_rho})')
    elif lower > 0:
        out['regime'] = 'coherent'
        out['reason'] = 'transfer entropy and DML |t| agree on the ranking'
    else:
        out['reason'] = 'agreement neither established nor excluded'
    return _report(out, verbose)


def _report(out: dict, verbose: bool) -> dict:
    if verbose:
        rho = 'n/a' if not np.isfinite(out['rho']) else (
            f"{out['rho']:+.3f} [{out['rho_lower']:+.2f}, {out['rho_upper']:+.2f}]")
        print(f"  [evidence coherence] n={out['n']}, rho(TE, |t|)={rho} "
              f"-> {out['regime'].upper()} ({out['reason']})")
    return out


def keep_stage6_pairs(results: pd.DataFrame) -> tuple:
    """
    Incoherent regime: the Stage 7 fusion and the Stage 8a competition rank
    the candidates on evidence whose two channels do not agree, so their
    rejections are not applied. The decision falls back to Stage 6: a pair
    is kept if and only if the DML cascade accepted its effect
    ('significant_stage6'). What Stages 7-8a had decided stays in
    'significant_before_guard'.

    Returns (results, number of pairs Stages 7-8a had kept, number of pairs
    kept now).
    """
    results['significant_before_guard'] = results['significant']
    n_before = int((results['significant'] == True).sum())  # noqa: E712
    results['significant'] = results['significant_stage6'].fillna(False).astype(bool)
    return results, n_before, int(results['significant'].sum())

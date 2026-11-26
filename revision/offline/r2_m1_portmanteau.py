"""
R2-m1, R2-m1b and R3-1 (part with the paper data) -- Time-axis diagnostic
against portmanteau tests; non-stationary series.

Request
  R2: "It is unclear how robust this single summary statistic is in
       heterogeneous multivariate networks, where some series are strongly
       non-stationary and others behave as white noise. The authors should
       clarify whether pair-specific portmanteau tests were considered".
  R3: "individual series can have zero autocorrelation while exhibiting strong
       cross-series lagged dependence. Would be great to have revisions or
       targeted validation against such cases."

Design (fixed before running; series from each scenario's source.csv, standardized)
  Current rule: "no time axis" when the median of |ACF(1)| over the
    series with at least 30 values falls below 2/sqrt(n)
    (Pipeline/discovery._detect_time_axis).
  Hosking multivariate portmanteau (m = 5): Q = n^2 sum_k tr(C_k' C_0^-1
    C_k C_0^-1)/(n - k), chi-squared with d^2 m degrees of freedom; "no
    temporal dependence" if p > 0.05. With d^2 m > n the test is flagged as
    unreliable.
  Pairwise test (Haugh): each series is prewhitened by an AR(p), p <= 5 by
    BIC; for each ordered pair (X, Y), n sum_{k=1..m} r_XY(k)^2 ~ chi-squared(m),
    with X leading by k steps; BH over the pairs. Measures R3's case: lagged cross
    dependence that the ACF(1) of each series does not see.
  Per series (R2-m1b): Ljung-Box (lag 10) for white noise; ADF (AIC) and KPSS
    (level) for unit root: "unit root" = ADF does not reject and KPSS rejects
    at 5%; "stationary" = ADF rejects and KPSS does not reject; the rest, ambiguous.
  Reporting per scenario and per family: decision of the rule and of Hosking; fraction of
    pairs with lagged dependence (Haugh, BH 5%); fractions of white and
    unit-root series; heterogeneous scenarios (at least one white series and one
    with a unit root); and the scenarios in which the rule says "no axis" but the pairwise
    test finds lagged dependence.

Usage:  python revision/offline/r2_m1_portmanteau.py [--verificar]
Output: Results_Revisao/r2_m1_portmanteau/
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c  # noqa: E402

NAME = "r2_m1_portmanteau"
M = 5


def series(scen) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(c.folder(scen), "source.csv"))
    props = [k for k in d.columns if k not in ("Date", "Name")]
    w = d.pivot_table(index="Date", columns="Name", values=props, aggfunc="first")
    w.columns = [f"{e}.{p}" for p, e in w.columns]
    w = w.sort_index()
    w = w.loc[:, w.notna().sum() >= 10]
    w = w.loc[:, w.std(skipna=True) > 1e-12]
    return w


def current_rule(w):
    acf, n = [], []
    for col in w.columns:
        v = w[col].to_numpy(float)
        v = v[~np.isnan(v)]
        if len(v) < 30:
            continue
        v = v - v.mean()
        den = float(v @ v)
        if den > 0:
            acf.append(abs(float(v[:-1] @ v[1:]) / den))
            n.append(len(v))
    if not acf:
        return None, None, None
    nm = int(np.median(n))
    med = float(np.median(acf))
    return med, 2 / np.sqrt(nm), med < 2 / np.sqrt(nm)


def hosking(X, m=M):
    n, d = X.shape
    X = X - X.mean(axis=0)
    C0 = X.T @ X / n
    C0i = np.linalg.pinv(C0)
    q = 0.0
    for k in range(1, m + 1):
        Ck = X[k:].T @ X[:-k] / n
        q += np.trace(Ck.T @ C0i @ Ck @ C0i) / (n - k)
    q *= n * n
    dof = d * d * m
    return float(q), float(chi2.sf(q, dof)), dof > n


def prewhiten(v, pmax=5):
    v = v - v.mean()
    best = (np.inf, v)
    n = len(v)
    for p in range(0, pmax + 1):
        if p == 0:
            res = v[pmax:]
        else:
            Y = v[pmax:]
            Xl = np.column_stack([v[pmax - j:n - j] for j in range(1, p + 1)])
            beta, *_ = np.linalg.lstsq(Xl, Y, rcond=None)
            res = Y - Xl @ beta
        bic = len(res) * np.log(np.mean(res ** 2) + 1e-300) + p * np.log(len(res))
        if bic < best[0]:
            best = (bic, res)
    r = best[1]
    return (r - r.mean()) / (r.std() + 1e-12)


def haugh(X, m=M):
    """p-values of the lagged-dependence test X_i -> X_j (i leading), per ordered pair."""
    R = np.column_stack([prewhiten(X[:, j]) for j in range(X.shape[1])])
    n, d = R.shape
    ps = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            s = sum((float(R[:-k, i] @ R[k:, j]) / n) ** 2 for k in range(1, m + 1))
            ps.append(chi2.sf(n * s, m))
    return np.array(ps)


def per_series(w):
    white = root = stat = 0
    tot = 0
    for col in w.columns:
        v = w[col].dropna().to_numpy(float)
        if len(v) < 30:
            continue
        tot += 1
        try:
            if acorr_ljungbox(v, lags=[10], return_df=True)["lb_pvalue"].iloc[0] > 0.05:
                white += 1
            p_adf = adfuller(v, autolag="AIC")[1]
            p_kpss = kpss(v, regression="c", nlags="auto")[1]
            if p_adf > 0.05 and p_kpss < 0.05:
                root += 1
            elif p_adf <= 0.05 and p_kpss >= 0.05:
                stat += 1
        except Exception:
            pass
    return tot, white, root, stat


def main():
    check = "--verificar" in sys.argv
    rel = c.Report(NAME + ("_verificacao" if check else ""),
                   "R2-m1 / R2-m1b / R3-1 — Time axis, portmanteau and non-stationary series")
    scens = c.scenarios()
    if check:
        scens = [s for s in scens if c.family(s) in ("ALARM", "Sachs", "VAR synthetic", "Walker Circulation")]
    rows = []
    for i, s in enumerate(scens, 1):
        print(f"  [{i}/{len(scens)}] {s}")
        w = series(s)
        med, lim, no_axis = current_rule(w)
        X = w.interpolate(limit_direction="both").to_numpy(float)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        q, p_h, unstable = hosking(X)
        ph = haugh(X)
        sig = multipletests(ph, method="fdr_bh")[0] if len(ph) else np.array([], bool)
        tot, white, root, stat = per_series(w)
        rows.append({"cenario": s, "familia": c.family(s), "heldout": c.is_heldout(s), "n": X.shape[0],
                     "d": X.shape[1], "acf1_mediana": med, "limiar": lim, "regra_sem_eixo": no_axis,
                     "hosking_p": p_h, "hosking_sem_dependencia": p_h > 0.05, "hosking_instavel": unstable,
                     "pares_defasados_frac": float(sig.mean()) if len(sig) else np.nan,
                     "series": tot, "brancas_frac": white / tot if tot else np.nan,
                     "raiz_unitaria_frac": root / tot if tot else np.nan,
                     "estacionarias_frac": stat / tot if tot else np.nan,
                     "heterogeneo": bool(white > 0 and root > 0)})
    D = pd.DataFrame(rows)
    rel.csv("por_cenario.csv", D)
    rel.p(f"{len(D)} scenarios. Current rule: no axis in {int(D['regra_sem_eixo'].sum())} "
          f"({', '.join(D.loc[D['regra_sem_eixo'] == True, 'cenario'])}). "  # noqa: E712
          f"Hosking (m = {M}): no temporal dependence in {int(D['hosking_sem_dependencia'].sum())}; "
          f"unstable test (d²m > n) in {int(D['hosking_instavel'].sum())}.")
    disc = D[D["regra_sem_eixo"] != D["hosking_sem_dependencia"]]
    rel.p(f"Different decisions between the rule and Hosking: {len(disc)} scenarios.")
    target = D[(D["regra_sem_eixo"] == True) & (D["pares_defasados_frac"] > 0)]  # noqa: E712
    rel.p(f"Rule says 'no axis' but the pairwise test finds lagged dependence: {len(target)} scenarios "
          f"({', '.join(target['cenario']) or 'none'}).")
    rel.h("## Per family")
    g = D.groupby("familia")
    body = []
    for f, d in g:
        body.append([f, len(d), int(d["regra_sem_eixo"].sum()), int(d["hosking_sem_dependencia"].sum()),
                     c.fmt(d["pares_defasados_frac"].mean()), c.fmt(d["brancas_frac"].mean()),
                     c.fmt(d["raiz_unitaria_frac"].mean()), int(d["heterogeneo"].sum())])
    rel.table(["Family", "Scenarios", "Rule: no axis", "Hosking: no dependence",
               "Pairs with lagged dependence (fraction)", "White series (fraction)",
               "Unit-root series (fraction)", "Heterogeneous scenarios"], body)
    rel.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        main()

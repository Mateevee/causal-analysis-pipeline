"""
R2-m1c / R3-1c -- Comparison of the time-axis rules: the original and the two
proposals, the decision only (without running the pipeline, without F1).

Request
  R2: "It is unclear how robust this single summary statistic is in
       heterogeneous multivariate networks, where some series are strongly
       non-stationary and others behave as white noise. The authors should
       clarify whether pair-specific portmanteau tests were considered".
  R3: "individual series can have zero autocorrelation while exhibiting strong
       cross-series lagged dependence. Would be great to have revisions or
       targeted validation against such cases."
  Protocol: revision/PROTOCOL_time_axis_rule.md (draft at the time); this script
  produces the comparison the authors asked for before choosing the rule.

Design (fixed before running, 2026-11-09)
  Input of each rule: the series the pipeline hands to Stage 3, obtained
    with the pipeline's own functions and the parameters of the paper (defaults):
    Stage 0 (pivot.load_and_pivot), Stage 1 (property_scanner, min_active_pct
    0.09, zero_threshold 1e-3), Stage 2 (preprocessing.preprocess_all,
    gap_limit 3, zero_run_min_length 2). On the scenarios of the paper, from the
    source.csv of the paper's run.
  Rules ("no time axis" = True):
    R0 original: not discovery._detect_time_axis(processed) -- the pipeline's
       function, not a copy (median of |ACF(1)| < 2/sqrt(n)).
    R1 proposal (A)+(B): R0 and no ordered pair (i, j), i != j, with
       lagged dependence by Haugh's test: residuals of AR(p), p <= 5 by
       BIC, standardized; Q_ij = n sum_{k=1..m} r_ij(k)^2 ~ chi-square(m),
       i leading by k steps; BH at q over the d(d-1) pairs.
    R2 blockwise Hosking: none of the d^2 blocks rejects, in a single BH family at q:
       off the diagonal, the same Q_ij as R1; on the diagonal, the Ljung-Box test of the series itself
       (not prewhitened), Q_ii = n(n+2) sum_{k=1..m} r_ii(k)^2/(n-k) ~
       chi-square(m).
    Tests of R1 and R2 on the series with at least 30 values and standard deviation > 0 (the
    same set R0 uses); NaN left by Stage 2 (gaps > 3) are
    linearly interpolated for the test only; the interpolated fraction is reported.
  Parameters: main m = 5, q = 0.05; sensitivity m in {3, 5, 10} and q in
    {0.01; 0.05; 0.10}.
  Simulated scenarios (own numpy generator, seeds 4101-4120, d in {5, 12,
    40}, T in {100, 300}; 20 per cell), passed through the same Stages 0-2:
    S1 iid: d independent white series.                     Right: no axis.
    S2 iid_contemporaneous: white with contemporaneous correlation (Sigma =
       correlation of L L' + I, L d x 2 normal) -- the case of ALARM.  Right: no axis.
    S5 heterogeneous: no cross influence at all; 70% white, 15% AR(1) with
       phi ~ U(0.5; 0.9), 15% random walk (at least one of each) -- the case
       of R2.                                                Right: with axis.
    S3 and S4: the 40 scenarios of r3_1_axis_counterexample (the repository's VAR
       generator, same seeds): S3 without AR (the case of R3), S4 with AR.
                                                             Right: with axis.
    "Right" comes from the definition (no axis = multivariate white noise), not from F1;
    in S5 there is no cross link, so the decision does not change F1 -- it matters for the
    claim that there is no axis and for the Šidák correction of Stage 5.
  Criteria fixed beforehand:
    - A rule "passes" if it is right on at least 90% of the scenarios of each cell
      (arm x d x T) with m = 5, q = 0.05.
    - Expected, not assumed: R0 fails on S3 and S5; R1 fails on S5; R2 passes on
      all.
    - On the 90 scenarios of the paper: R1 should change 5 scenarios and keep "no axis" on
      ALARM and lt_actuators_white (earlier count, on source.csv); for R2 there is
      no earlier count. Mandatory check: if R2 takes ALARM out of "no
      axis", this is reported as a cost of the rule.
    - The choice between R1 and R2 uses only the simulated scenarios (definition) and the
      check of ALARM; no F1 is computed here.
  Also reported: R0 on source.csv (count of r2_m1_portmanteau) against R0
    on the series of Stage 2, for each scenario of the paper.

Declared context: the comparison is made after the held-out datasets had been
seen and after the r3_1 simulation, which motivated the change.

Usage:  python revision/offline/r2_m1c_axis_rules.py [--verificar]
Output: Results_Revisao/r2_m1c_regras_eixo/ (relatorio.md, artigo.csv, simulado.csv; file and column names kept in
Portuguese, see revision/README.md, Notes)
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REVISION = os.path.dirname(HERE)
ROOT = os.path.dirname(REVISION)
for p in (ROOT, REVISION, HERE, os.path.join(REVISION, "execution")):
    if p not in sys.path:
        sys.path.insert(0, p)
import common as c  # noqa: E402
import r2_m1_portmanteau as base  # noqa: E402
from Pipeline import pivot, preprocessing, property_scanner  # noqa: E402
from Pipeline.discovery import _detect_time_axis  # noqa: E402

NAME = "r2_m1c_regras_eixo"
MS = (3, 5, 10)
QS = (0.01, 0.05, 0.10)
M0, Q0 = 5, 0.05
SEEDS = list(range(4101, 4121))
DS = (5, 12, 40)
TS = (100, 300)
P_MAX = 5


# ---------------------------------------------------------------- input

def dates(df: pd.DataFrame) -> pd.DataFrame:
    """Converts Date to datetime before Stage 0.

    In the paper, the pipeline receives Date already as datetime (from the loaders). Read back from source.csv, Date
    is ISO text, and pivot.load_and_pivot (dayfirst=True) reads it as year-day-month: days > 12 become NaT and the
    rows vanish. The first run of this script (2026-11-09 15:59, id 545fb7bbee14) fell into this (authors' lab notes).
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        conv = pd.to_datetime(df["Date"], format="ISO8601", errors="coerce")
        if conv.isna().any():
            raise ValueError(f"dates outside ISO 8601 in {int(conv.isna().sum())} rows")
        df["Date"] = conv
    return df


def stages_0_2(df: pd.DataFrame) -> dict:
    """Series the pipeline hands to Stage 3 (default parameters = paper)."""
    df = dates(df)
    with contextlib.redirect_stdout(io.StringIO()):
        df_wide, pairs, *_ = pivot.load_and_pivot(df, "Date", "Name", None)
        scan = property_scanner.scan_entity_properties(df, "Date", "Name", None, zero_threshold=1e-3,
                                                       min_active_pct=0.09, verbose=False)
        active = set(property_scanner.get_active_pairs(scan))
        cols = [k for k in df_wide.columns if k in active]
        processed, _ = preprocessing.preprocess_all(df_wide[cols], gap_limit=3, zero_run_min_length=2)
    return processed


def matrix(processed: dict):
    cols, vals = [], []
    for k, v in processed.items():
        v = np.asarray(v, float)
        ok = ~np.isnan(v)
        if ok.sum() < 30 or np.nanstd(v) <= 1e-12:
            continue
        cols.append(k)
        vals.append(v)
    if not cols:
        return None, 0.0
    W = pd.DataFrame(np.column_stack(vals))
    frac_nan = float(W.isna().to_numpy().mean())
    W = W.interpolate(limit_direction="both")
    X = W.to_numpy(float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    return X, frac_nan


# ---------------------------------------------------------------- tests

def statistics(X: np.ndarray, mmax: int = max(MS)):
    """Cumulative sums, per m, of the off-diagonal (Haugh) and diagonal (Ljung-Box) statistics."""
    n, d = X.shape
    R = np.column_stack([base.prewhiten(X[:, j], P_MAX) for j in range(d)])
    nr = R.shape[0]
    off = np.zeros((mmax, d, d))
    for k in range(1, mmax + 1):
        Ck = R[:-k].T @ R[k:] / nr          # (i, j): i at t-k, j at t
        off[k - 1] = nr * Ck ** 2
    off = np.cumsum(off, axis=0)
    Z = X - X.mean(axis=0)
    den = (Z ** 2).sum(axis=0)
    lb = np.zeros((mmax, d))
    for k in range(1, mmax + 1):
        rk = (Z[:-k] * Z[k:]).sum(axis=0) / den
        lb[k - 1] = n * (n + 2) * rk ** 2 / (n - k)
    lb = np.cumsum(lb, axis=0)
    return off, lb


def decisions(processed: dict):
    """Decisions of R0, R1 and R2 in every (m, q) combination, plus counts."""
    r0 = not _detect_time_axis(processed, verbose=False)
    X, frac_nan = matrix(processed)
    out = {"R0": r0, "frac_interpolada": frac_nan, "d": 0 if X is None else X.shape[1],
           "n": 0 if X is None else X.shape[0]}
    if X is None:
        for m in MS:
            for q in QS:
                out[f"R1_m{m}_q{q}"] = r0
                out[f"R2_m{m}_q{q}"] = False
        return out
    off, lb = statistics(X)
    d = X.shape[1]
    mask = ~np.eye(d, dtype=bool)
    for m in MS:
        p_off = chi2.sf(off[m - 1][mask], m)
        p_diag = chi2.sf(lb[m - 1], m)
        for q in QS:
            rej_off = multipletests(p_off, alpha=q, method="fdr_bh")[0] if len(p_off) else np.array([], bool)
            out[f"R1_m{m}_q{q}"] = bool(r0 and not rej_off.any())
            p_all = np.concatenate([p_diag, p_off])
            rej = multipletests(p_all, alpha=q, method="fdr_bh")[0]
            out[f"R2_m{m}_q{q}"] = not bool(rej.any())
            if m == M0 and q == Q0:
                out["pares_sig_R1"] = int(rej_off.sum())
                out["diag_sig_R2"] = int(rej[:d].sum())
                out["pares_sig_R2"] = int(rej[d:].sum())
    return out


# ---------------------------------------------------------------- simulation

def long_format(X: np.ndarray) -> pd.DataFrame:
    T, d = X.shape
    days = pd.date_range("2015-01-01", periods=T, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({"Date": np.repeat(days, d), "Name": np.tile([f"P{i + 1}" for i in range(d)], T),
                         "VAL": X.reshape(-1)})


def generate(arm: str, d: int, T: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed * 1000 + d * 10 + (T // 100))
    if arm == "S1":
        return rng.standard_normal((T, d))
    if arm == "S2":
        L = rng.standard_normal((d, 2))
        S = L @ L.T + np.eye(d)
        s = np.sqrt(np.diag(S))
        S = S / np.outer(s, s)
        return rng.multivariate_normal(np.zeros(d), S, size=T)
    if arm == "S5":
        n_ar = max(1, round(0.15 * d))
        n_rw = max(1, round(0.15 * d))
        X = rng.standard_normal((T + 100, d))
        for j in range(n_ar):
            phi = rng.uniform(0.5, 0.9)
            for t in range(1, T + 100):
                X[t, j] = phi * X[t - 1, j] + X[t, j]
        for j in range(n_ar, n_ar + n_rw):
            X[:, j] = np.cumsum(X[:, j])
        X = X[100:]
        return X[:, rng.permutation(d)]
    raise ValueError(arm)


def simulated_scenarios(check: bool):
    seeds = SEEDS[:2] if check else SEEDS
    ds = DS[:1] if check else DS
    for arm in ("S1", "S2", "S5"):
        for d in ds:
            for T in TS:
                for s in seeds:
                    yield arm, d, T, s, long_format(generate(arm, d, T, s))
    import r3_1_axis_counterexample as r31
    with contextlib.redirect_stdout(io.StringIO()):
        cs = r31.scenarios(check)
    for scen in cs:
        arm = "S3" if "semAR" in scen["name"] else "S4"
        d = scen["df"]["Name"].nunique()
        T = scen["df"]["Date"].nunique()
        yield arm, d, T, scen["name"], scen["df"]


RIGHT_NO_AXIS = {"S1": True, "S2": True, "S3": False, "S4": False, "S5": False}
DESCR = {"S1": "independent i.i.d.", "S2": "i.i.d. with contemporaneous correlation (ALARM-like)",
         "S3": "no AR, with lagged cross-dependence (the case of R3)", "S4": "with AR and cross-dependence",
         "S5": "heterogeneous: white + AR + random walk, no cross-dependence (the case of R2)"}


# ---------------------------------------------------------------- report

def main():
    check = "--verificar" in sys.argv
    rel = c.Report(NAME + ("_verificacao" if check else ""),
                   "R2-m1c / R3-1c — Time-axis rules: original (R0), (A)+(B) (R1) and blockwise Hosking (R2)")
    rel.p("Only the decision of each rule; no F1 is computed. Design and criteria in the docstring of the script "
          "(fixed before running). 'no axis' = True.")
    k0 = f"m{M0}_q{Q0}"

    # ---- simulation
    rows = []
    for i, (arm, d, T, s, df) in enumerate(simulated_scenarios(check), 1):
        if i % 40 == 0:
            print(f"  simulated {i}")
        dec = decisions(stages_0_2(df))
        rows.append({"braco": arm, "d": d, "T": T, "semente": s, "n_datas_entrada": df["Date"].nunique(), **dec})
    S = pd.DataFrame(rows)
    rel.csv("simulado.csv", S)
    cut = S[S["n"] != S["n_datas_entrada"]]
    rel.p(f"Check: simulated scenarios in which Stage 2 does not have every time point of the input: {len(cut)}.")
    if len(cut):
        raise RuntimeError("series cut at the input; invalid result")
    rel.h(f"## Simulation: fraction right (m = {M0}, q = {Q0})")
    rel.p("Right = 'no axis' in S1 and S2; 'with axis' in S3, S4 and S5. Criterion: >= 0.90 in each cell.")
    body, failures = [], {r: [] for r in ("R0", "R1", "R2")}
    for (b, d, T), g in S.groupby(["braco", "d", "T"]):
        right = RIGHT_NO_AXIS[b]
        acc = {r: float((g[r if r == "R0" else f"{r}_{k0}"] == right).mean()) for r in ("R0", "R1", "R2")}
        for r, a in acc.items():
            if a < 0.90:
                failures[r].append(f"{b} d={d} T={T} ({a:.2f})")
        body.append([b, DESCR[b], d, T, len(g), c.fmt(acc["R0"], 2), c.fmt(acc["R1"], 2), c.fmt(acc["R2"], 2),
                     c.fmt(g["diag_sig_R2"].mean(), 1) if "diag_sig_R2" in g else "—",
                     c.fmt(g["pares_sig_R2"].mean(), 1) if "pares_sig_R2" in g else "—"])
    rel.table(["Arm", "Case", "d", "T", "Scenarios", "R0", "R1", "R2",
               "R2: series rejecting (mean)", "R2: pairs rejecting (mean)"], body)
    rel.p("")
    for r in ("R0", "R1", "R2"):
        rel.p(f"- {r}: " + ("passes in every cell." if not failures[r]
                            else f"below 0.90 in {len(failures[r])} cells: " + "; ".join(failures[r]) + "."))

    rel.h("## Simulation: sensitivity of R1 and R2 to m and q (fraction right, all cells together)")
    body = []
    for b in sorted(S["braco"].unique()):
        g = S[S["braco"] == b]
        right = RIGHT_NO_AXIS[b]
        row = [b]
        for r in ("R1", "R2"):
            for m in MS:
                for q in QS:
                    row.append(c.fmt(float((g[f"{r}_m{m}_q{q}"] == right).mean()), 2))
        body.append(row)
    rel.table(["Arm"] + [f"{r} m={m} q={q}" for r in ("R1", "R2") for m in MS for q in QS], body)

    # ---- scenarios of the paper
    scens = c.scenarios()
    if check:
        scens = [next(s for s in scens if c.family(s) == f) for f in ("ALARM", "VAR synthetic", "BEELINE")]
    rows = []
    for i, s in enumerate(scens, 1):
        print(f"  [{i}/{len(scens)}] {s}")
        src = pd.read_csv(os.path.join(c.folder(s), "source.csv"))
        dec = decisions(stages_0_2(src))
        r0_src = base.current_rule(base.series(s))[2]
        rows.append({"cenario": s, "familia": c.family(s), "heldout": c.is_heldout(s),
                     "n_datas_entrada": src["Date"].nunique(), "R0_source_csv": r0_src, **dec})
    A = pd.DataFrame(rows)
    rel.csv("artigo.csv", A)
    rel.h(f"## Scenarios of the paper ({len(A)}), m = {M0}, q = {Q0}")
    cut = A[(A["d"] > 0) & (A["n"] != A["n_datas_entrada"])]
    rel.p(f"Check: scenarios in which Stage 2 does not have every time point of source.csv: {len(cut)} "
          f"({', '.join(cut['cenario']) or 'none'}).")
    for r in ("R0", f"R1_{k0}", f"R2_{k0}"):
        no_axis = A.loc[A[r] == True, "cenario"].tolist()  # noqa: E712
        rel.p(f"- {r.split('_')[0]}: no axis on {len(no_axis)} ({', '.join(no_axis) or 'none'}).")
    dif_src = A[A["R0"] != A["R0_source_csv"]]
    rel.p(f"- R0 on source.csv x R0 on the series of Stage 2: they differ on {len(dif_src)} "
          f"({', '.join(dif_src['cenario']) or 'none'}).")
    for r in ("R1", "R2"):
        changed = A[A["R0"] != A[f"{r}_{k0}"]]
        rel.p(f"- {r} changes the decision of R0 on {len(changed)} ({', '.join(changed['cenario']) or 'none'}).")
    alarm = A[A["familia"] == "ALARM"]
    if len(alarm):
        rel.p(f"- Check of ALARM: R0 = {bool(alarm['R0'].iloc[0])}, R1 = {bool(alarm[f'R1_{k0}'].iloc[0])}, "
              f"R2 = {bool(alarm[f'R2_{k0}'].iloc[0])}; R2 series rejecting = {alarm['diag_sig_R2'].iloc[0]}, "
              f"pairs = {alarm['pares_sig_R2'].iloc[0]}.")
    rel.p(f"- Fraction interpolated for the test only (gaps > 3 left by Stage 2): maximum "
          f"{c.fmt(A['frac_interpolada'].max())}; scenarios with a fraction > 0: {int((A['frac_interpolada'] > 0).sum())}.")
    det = A[(A["R0"] == True) | (A[f"R1_{k0}"] == True) | (A[f"R2_{k0}"] == True)]  # noqa: E712
    rel.h("### Scenarios on which some rule says 'no axis'")
    rel.table(["Scenario", "Family", "Held-out", "d", "n", "R0", "R1", "R2", "R1: pairs", "R2: series", "R2: pairs"],
              [[r["cenario"], r["familia"], r["heldout"], r["d"], r["n"], r["R0"], r[f"R1_{k0}"], r[f"R2_{k0}"],
                r["pares_sig_R1"], r["diag_sig_R2"], r["pares_sig_R2"]] for _, r in det.iterrows()])
    rel.h("### Sensitivity on the scenarios of the paper: number of scenarios with 'no axis'")
    body = []
    for r in ("R1", "R2"):
        for m in MS:
            body.append([r, m] + [int(A[f"{r}_m{m}_q{q}"].sum()) for q in QS])
    rel.table(["Rule", "m"] + [f"q = {q}" for q in QS], body)
    rel.data = {"falhas_simulacao": failures, "artigo_sem_eixo": {r: A.loc[A[r] == True, "cenario"].tolist()  # noqa: E712
                                                                  for r in ("R0", f"R1_{k0}", f"R2_{k0}")}}
    rel.save()


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        main()

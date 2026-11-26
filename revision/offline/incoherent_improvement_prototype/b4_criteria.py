# Provenance: 4th analysis of 2026-11-15 (section 11 of ACHADOS_melhoria_incoherent_2026-11-15.md). Read-only. POST HOC.
# Criteria WITHOUT the ground truth to decide when to apply CA (Stage 6 ∩ VAR + adaptive lasso; section 10.4).
# Defined BEFORE looking at the result, with the natural threshold 0 (not tuned):
#   K1  predictive gain of the joint selection: does the sparse VAR chosen by the adaptive lasso predict out of sample
#       better than the own AR? statistic = median, over the targets, of 1 - MSE_AL / MSE_own. Applies if > 0.
#   K2  the pruning costs no prediction: predict each target with its lags + the parents of the current output (A) or
#       only the parents of CA (B), same predictor (ridge). statistic = median of (MSE_A - MSE_B) / MSE_A. Applies if >= 0.
#   K12 K1 and K2.
# Cross-validation in 5 contiguous blocks, with a gap of L rows on each side of the test block; standardization with the
# training mean and standard deviation; the lasso selection is redone within each training set (no leakage).
# A and B differ only in the pairs with lag >= 1; those with lag 0 are the same in both.
# Output: the CSV named on the command line, the input of b4_evaluate.py (column names kept in Portuguese; see
# revision/README.md, Notes). Usage: python revision/offline/incoherent_improvement_prototype/b4_criteria.py res_b4_criterios.csv
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC, RidgeCV

warnings.filterwarnings("ignore")
F2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repository root
L = 10
K = 5
SETS = [("val12", os.path.join(F2, "Results_Revisao", "r1_3b_previa_validacao", "run"), "patch_tau10_final"),
        ("art90", os.path.join(F2, "Results", "Run_20261011_132346"), "patch_tau10"),
        ("art90", os.path.join(F2, "Results_Revisao", "r1_5d_auprc_90", "cc"), "patch_tau10"),
        ("cc9_visto", os.path.join(F2, "Results_Revisao", "r1_2c_benchmark_novo", "run"), "patch_tau10")]


def tb(s):
    return s.astype(str) == "True"


def pairs(df):
    return {(a, b) for a, b in zip(df.source_entity, df.target_entity)}


def lagmat(X):
    n, k = X.shape
    return np.column_stack([X[L - l:n - l, j] for j in range(k) for l in range(1, L + 1)])


def al_coef(Z, Y, own):
    # adaptive lasso (OLS weights, gamma = 1, BIC), as in lingam.utils.predict_adaptive_lasso.
    # When the training set has fewer rows than regressors + 2 (it only happens within the folds), the BIC does not
    # estimate the noise variance; the variance of the residual of the own AR is used, the rule of configuration (iii) of rule C.
    w = np.abs(np.linalg.lstsq(np.column_stack([np.ones(len(Z)), Z]), Y, rcond=None)[0][1:])
    nv = None
    if len(Z) <= Z.shape[1] + 1:
        Zo = np.column_stack([np.ones(len(Z)), Z[:, own]])
        nv = float((Y - Zo @ np.linalg.lstsq(Zo, Y, rcond=None)[0]).var()) + 1e-6
    m = LassoLarsIC(criterion="bic", noise_variance=nv).fit(Z * w, Y)
    return m.coef_ * w, m.intercept_


def al_select(X, names):
    Z = lagmat(X)
    k = X.shape[1]
    sel = set()
    for t in range(k):
        c, _ = al_coef(Z, X[L:, t], slice(t * L, (t + 1) * L))
        c = np.abs(c).reshape(k, L).sum(1)
        sel |= {(names[j], names[t]) for j in range(k) if j != t and c[j] > 0}
    return sel


def folds(m):
    b = m // K
    for f in range(K):
        t0, t1 = f * b, ((f + 1) * b if f < K - 1 else m)
        te = np.arange(t0, t1)
        tr = np.concatenate([np.arange(0, max(t0 - L, 0)), np.arange(min(t1 + L, m), m)])
        yield tr, te


def criteria(Xraw, names, A, B):
    k = Xraw.shape[1]
    idx = {nm: i for i, nm in enumerate(names)}
    m = Xraw.shape[0] - L
    sse = {key: np.zeros(k) for key in ["own", "al", "A", "B"]}
    for tr, te in folds(m):
        rows_tr = tr + L
        mu, sd = Xraw[rows_tr].mean(0), Xraw[rows_tr].std(0) + 1e-12
        X = (Xraw - mu) / sd
        Z = lagmat(X)
        for t in range(k):
            Y = X[L:, t]
            own = slice(t * L, (t + 1) * L)
            b = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), Z[tr, own]]), Y[tr], rcond=None)[0]
            sse["own"][t] += ((Y[te] - np.column_stack([np.ones(len(te)), Z[te, own]]) @ b) ** 2).sum()
            c, i0 = al_coef(Z[tr], Y[tr], own)
            sse["al"][t] += ((Y[te] - (Z[te] @ c + i0)) ** 2).sum()
            for key, G in [("A", A), ("B", B)]:
                par = sorted(idx[s] for s, d in G if d == names[t] and s in idx and s != names[t])
                cols = list(range(t * L, (t + 1) * L)) + [j * L + l for j in par for l in range(L)]
                r = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z[tr][:, cols], Y[tr])
                sse[key][t] += ((Y[te] - r.predict(Z[te][:, cols])) ** 2).sum()
    g1 = 1 - sse["al"] / sse["own"]
    g2 = (sse["A"] - sse["B"]) / sse["A"]
    return dict(K1=float(np.median(g1)), K1_frac=float((g1 > 0).mean()), K2=float(np.median(g2)), K2_frac=float((g2 >= 0).mean()))


def met(decl, gt, N):
    tp = len(decl & gt)
    fp = len(decl) - tp
    fn = len(gt) - tp
    tn = N - tp - fp - fn
    p = tp / len(decl) if decl else 0.0
    r = tp / len(gt) if gt else 0.0
    f = 2 * p * r / (p + r) if tp else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return f, ((tp * tn - fp * fn) / den if den > 0 else 0.0)


def main(output):
    rows = []
    for set_name, root, mod in SETS:
        for sc in sorted(glob.glob(os.path.join(root, "*", ""))):
            need = ["ground_truth.csv", "source.csv", mod + "/original.csv", "validation_summary.csv"]
            if not all(os.path.exists(sc + f) for f in need):
                continue
            name = os.path.basename(sc[:-1])
            src = pd.read_csv(sc + "source.csv")
            wide = src.pivot_table(index="Date", columns="Name", values=src.columns[2]).sort_index()
            T, n = wide.shape
            if not (T - L) > n * L + 5:
                continue
            o = pd.read_csv(sc + mod + "/original.csv", low_memory=False)
            if "significant_stage6" not in o:
                continue
            g = pd.read_csv(sc + "ground_truth.csv")
            gt = {(a, b) for a, b, c in zip(g.source_entity, g.target_entity, g.causal_gt) if c}
            N = len(g)
            reg = o.evidence_regime.iloc[0]
            fb = o.evidence_fallback.iloc[0] if "evidence_fallback" in o else ("stage6" if reg == "incoherent" else "-")
            names = list(wide.columns)
            Xraw = np.nan_to_num(wide.to_numpy(float) - np.nanmean(wide.to_numpy(float), 0))
            f_ = o[tb(o.significant)]
            lag0 = pairs(f_[f_.lag == 0])
            A = pairs(f_[f_.lag > 0])
            s6 = pairs(o[tb(o.significant_stage6) & (o.lag > 0)])
            Xs = (Xraw - Xraw.mean(0)) / (Xraw.std(0) + 1e-12)
            B = s6 & al_select(Xs, names)
            crit = criteria(Xraw, names, A, B)
            v = pd.read_csv(sc + "validation_summary.csv").set_index("model")
            fA, mA = met(A | lag0, gt, N)
            fB, mB = met(B | lag0, gt, N)
            r = dict(conj=set_name, cenario=name, grupo=(reg if reg != "incoherent" else "incoh/" + str(fb)), n=n, T=T,
                     nA=len(A), nB=len(B), f1_atual=fA, f1_CA=fB, mcc_atual=mA, mcc_CA=mB,
                     f1_varlingam=v.loc["varlingam_tau10", "f1"] if "varlingam_tau10" in v.index else np.nan, **crit)
            rows.append(r)
            print(set_name, name[-34:], r["grupo"], round(fA, 3), round(fB, 3), {k_: round(v_, 3) for k_, v_ in crit.items()}, flush=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main(sys.argv[1])

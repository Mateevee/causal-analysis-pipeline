# Provenance: 2nd search for an improvement in the incoherent regime, 2026-11-15 (section 9 of ACHADOS_melhoria_incoherent_2026-11-15.md).
# Only reads outputs already saved; does not run the pipeline or any baseline.
# Prototype of 2026-11-15 (2nd session): rule C (Stage 6 ∩ lasso-BIC, configuration (iii) of ACHADOS) with and without a
# sample-adequacy gate (T - L >= n*L), with F1 and chance-corrected measures (MCC; precision / base rate).
# Only reads outputs already saved; does not run the pipeline. Sets: 12 of the validation, 90 of the paper and, APART, the 9
# Causal Chambers scenarios of R1-2c already seen in the interim look (contaminated: they gave rise to the diagnosis).
# Output: the CSV named on the command line (res_b2_portao.csv, the input of b3_heldout.py; column names kept in Portuguese,
# see revision/README.md, Notes).
import os
import pandas as pd, numpy as np, glob, os, sys, warnings
from sklearn.linear_model import LassoLarsIC
warnings.filterwarnings("ignore")
F2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repository root
OUTPUT = sys.argv[1]; L = 10
SETS = [("val12", os.path.join(F2, "Results_Revisao", "r1_3b_previa_validacao", "run"), "patch_tau10_final"),
        ("art90", os.path.join(F2, "Results", "Run_20261011_132346"), "patch_tau10"),
        ("art90", os.path.join(F2, "Results_Revisao", "r1_5d_auprc_90", "cc"), "patch_tau10"),
        ("cc9_visto", os.path.join(F2, "Results_Revisao", "r1_2c_benchmark_novo", "run"), "patch_tau10")]
BL = ["pcmci_tigramite_tau10", "pcmciplus_tau10", "dynotears_tau10", "cuts_plus_tau10", "varlingam_tau10"]
def fam(c):
    for k in ["synthetic", "dream3_size10", "dream3_size50", "beeline", "causaltime", "causal_chamber", "netsim", "logistic", "lorenz", "tigramite", "rcaeval", "easyvista", "petshop", "tep", "pems", "alarm", "sachs"]:
        if k in c: return k
    return c
def pairs(df): return {(a, b) for a, b in zip(df.source_entity, df.target_entity)}
def tb(s): return s.astype(str) == "True"
def met(decl, gt, N):
    tp = len(decl & gt); fp = len(decl) - tp; fn = len(gt) - tp; tn = N - tp - fp - fn
    p = tp / len(decl) if decl else 0.0; r = tp / len(gt) if gt else 0.0; f = 2*p*r/(p+r) if tp else 0.0
    den = np.sqrt(float(tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)); return len(decl), p, r, f, ((tp*tn - fp*fn) / den if den > 0 else 0.0)
def lasso_var(wide):
    X = wide.to_numpy(float); X = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-12); X = np.nan_to_num(X)
    n, k = X.shape; names = list(wide.columns)
    lagm = np.column_stack([X[L-l:n-l, j] for j in range(k) for l in range(1, L+1)]); sel = set()
    for t in range(k):
        Y = X[L:, t]; own = lagm[:, t*L:(t+1)*L]; e = Y - own @ np.linalg.lstsq(own, Y, rcond=None)[0]
        m = LassoLarsIC(criterion="bic", noise_variance=float(e.var()) + 1e-6).fit(lagm, Y)
        c = np.abs(m.coef_).reshape(k, L).sum(1); sel |= {(names[j], names[t]) for j in range(k) if j != t and c[j] > 0}
    return sel
rows = []
for set_name, root, mod in SETS:
    for sc in sorted(glob.glob(os.path.join(root, "*", ""))):
        if not (os.path.exists(sc + "ground_truth.csv") and os.path.exists(sc + mod + "/original.csv") and os.path.exists(sc + "source.csv")): continue
        name = os.path.basename(sc[:-1])
        g = pd.read_csv(sc + "ground_truth.csv"); gt = {(a, b) for a, b, c in zip(g.source_entity, g.target_entity, g.causal_gt) if c}; N = len(g)
        o = pd.read_csv(sc + mod + "/original.csv", low_memory=False)
        reg = o.evidence_regime.iloc[0] if "evidence_regime" in o else "-"
        fb = o.evidence_fallback.iloc[0] if "evidence_fallback" in o else ("stage6" if reg == "incoherent" else "-")
        src = pd.read_csv(sc + "source.csv"); wide = src.pivot_table(index="Date", columns="Name", values=src.columns[2]).sort_index()
        T, n = wide.shape; fin = pairs(o[tb(o.significant)]); acts = (reg == "incoherent" and fb == "stage6"); gate = (T - L) >= n * L
        C = fin
        if acts:
            f_ = o[tb(o.significant)]; C = (pairs(o[tb(o.significant_stage6) & (o.lag > 0)]) & lasso_var(wide)) | pairs(f_[f_.lag == 0])
        Cg = C if gate else fin
        v = pd.read_csv(sc + "validation_summary.csv").set_index("model")
        r = dict(conj=set_name, cenario=name, fam=fam(name), regime=reg, fallback=fb, n=n, T=T, obs_por_regressor=(T - L) / (n * L), atua=acts, portao=gate,
                 n_gt=len(gt), universo=N, taxa_base=len(gt) / N, piso=2 * len(gt) / N / (1 + len(gt) / N), f1_oficial=v.loc[mod, "f1"])
        for k_, d in [("atual", fin), ("C", C), ("Cg", Cg)]:
            nd, p, rc, f, mcc = met(d, gt, N); r.update({f"{k_}_n": nd, f"{k_}_prec": p, f"{k_}_rev": rc, f"{k_}_f1": f, f"{k_}_mcc": mcc})
        for b in BL: r[b] = v.loc[b, "f1"] if b in v.index else np.nan
        rows.append(r); print(set_name, name[-34:], reg, fb, n, T, round(r["atual_f1"], 3), round(r["C_f1"], 3), round(r["Cg_f1"], 3), flush=True)
pd.DataFrame(rows).to_csv(OUTPUT, index=False)

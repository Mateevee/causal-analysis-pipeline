# Provenance: 3rd analysis of 2026-11-15 (section 10 of ACHADOS_melhoria_incoherent_2026-11-15.md): cause of the loss of detection.
# Only reads outputs and series already saved; does not run the pipeline or any baseline. POST HOC.
# What does VARLiNGAM have that the pipeline does not? OFFLINE test: its lagged part (VAR by OLS + pruning by adaptive lasso
# with BIC, per target, lags 1..10 of all series), without the LiNGAM part (lag 0). Only where T - L > n*L (OLS estimable).
# Compares with: current output, simple lasso (rule C), saved VARLiNGAM. Only reads outputs and series already saved.
import os
import pandas as pd, numpy as np, glob, os, sys, warnings
from sklearn.linear_model import LassoLarsIC, LinearRegression
warnings.filterwarnings("ignore")
F2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repository root; L = 10
SETS = [("val12", os.path.join(F2, "Results_Revisao", "r1_3b_previa_validacao", "run"), "patch_tau10_final"),
        ("art90", os.path.join(F2, "Results", "Run_20261011_132346"), "patch_tau10"),
        ("art90", os.path.join(F2, "Results_Revisao", "r1_5d_auprc_90", "cc"), "patch_tau10"),
        ("cc9_visto", os.path.join(F2, "Results_Revisao", "r1_2c_benchmark_novo", "run"), "patch_tau10")]
def pairs(df): return {(a, b) for a, b in zip(df.source_entity, df.target_entity)}
def tb(s): return s.astype(str) == "True"
def met(decl, gt, N):
    tp = len(decl & gt); fp = len(decl) - tp; fn = len(gt) - tp; tn = N - tp - fp - fn
    p = tp / len(decl) if decl else 0.0; r = tp / len(gt) if gt else 0.0; f = 2*p*r/(p+r) if tp else 0.0
    den = np.sqrt(float(tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)); return len(decl), tp, p, r, f, ((tp*tn - fp*fn) / den if den > 0 else 0.0)
def select(wide):
    X = wide.to_numpy(float); X = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-12); X = np.nan_to_num(X)
    n, k = X.shape; names = list(wide.columns)
    lagm = np.column_stack([X[L-l:n-l, j] for j in range(k) for l in range(1, L+1)]); la, al = set(), set()
    for t in range(k):
        Y = X[L:, t]; own = lagm[:, t*L:(t+1)*L]; e = Y - own @ np.linalg.lstsq(own, Y, rcond=None)[0]
        m = LassoLarsIC(criterion="bic", noise_variance=float(e.var()) + 1e-6).fit(lagm, Y)
        c = np.abs(m.coef_).reshape(k, L).sum(1); la |= {(names[j], names[t]) for j in range(k) if j != t and c[j] > 0}
        w = np.abs(LinearRegression().fit(lagm, Y).coef_)                      # adaptive lasso, as in lingam.utils.predict_adaptive_lasso (gamma = 1)
        m2 = LassoLarsIC(criterion="bic").fit(lagm * w, Y)
        c2 = np.abs(m2.coef_ * w).reshape(k, L).sum(1); al |= {(names[j], names[t]) for j in range(k) if j != t and c2[j] > 0}
    return la, al
rows = []
for set_name, root, mod in SETS:
    for sc in sorted(glob.glob(os.path.join(root, "*", ""))):
        if not (os.path.exists(sc + "ground_truth.csv") and os.path.exists(sc + mod + "/original.csv") and os.path.exists(sc + "source.csv")): continue
        name = os.path.basename(sc[:-1])
        src = pd.read_csv(sc + "source.csv"); wide = src.pivot_table(index="Date", columns="Name", values=src.columns[2]).sort_index(); T, n = wide.shape
        if not (T - L) > n * L + 5: continue
        g = pd.read_csv(sc + "ground_truth.csv"); gt = {(a, b) for a, b, c in zip(g.source_entity, g.target_entity, g.causal_gt) if c}; N = len(g)
        o = pd.read_csv(sc + mod + "/original.csv", low_memory=False)
        if "significant_stage6" not in o: continue
        reg = o.evidence_regime.iloc[0]; fb = o.evidence_fallback.iloc[0] if "evidence_fallback" in o else ("stage6" if reg == "incoherent" else "-")
        f_ = o[tb(o.significant)]; fin = pairs(f_); lag0 = pairs(f_[f_.lag == 0]); s6 = pairs(o[tb(o.significant_stage6) & (o.lag > 0)]); cand = pairs(o[o.lag > 0])
        la, al = select(wide)
        v = pd.read_csv(sc + "validation_summary.csv").set_index("model")
        r = dict(conj=set_name, cenario=name, grupo=(reg if reg != "incoherent" else "incoh/" + str(fb)), n=n, T=T, n_gt=len(gt), taxa_base=len(gt) / N, f1_varlingam=v.loc["varlingam_tau10", "f1"] if "varlingam_tau10" in v.index else np.nan,
                 cand_n=len(cand), cand_tp=len(cand & gt), al_fora_dos_candidatos=len(al - cand), al_tp_fora_dos_candidatos=len((al - cand) & gt))
        for k_, d in [("atual", fin), ("s6", s6), ("lasso", la | lag0), ("C", (s6 & la) | lag0), ("AL", al | lag0), ("CA", (s6 & al) | lag0)]:
            nd, tp, p, rc, f, mcc = met(d, gt, N); r.update({f"{k_}_n": nd, f"{k_}_tp": tp, f"{k_}_prec": p, f"{k_}_rev": rc, f"{k_}_f1": f, f"{k_}_mcc": mcc})
        rows.append(r); print(set_name, name[-34:], r["grupo"], n, T, round(r["atual_f1"], 3), round(r["C_f1"], 3), round(r["AL_f1"], 3), round(r["CA_f1"], 3), r["f1_varlingam"], flush=True)
D = pd.DataFrame(rows); D.to_csv(sys.argv[1], index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
D["fam"] = D.cenario.str.replace("benchmark_|real_", "", regex=True).str.split("_").str[:2].str.join("_"); D.loc[D.cenario.str.contains("causal_chamber"), "fam"] = "causal_chamber"
C = D[D.fam == "causal_chamber"]
print("\n== Causal Chambers:", len(C), "scenarios"); cols = ["atual_f1", "C_f1", "AL_f1", "CA_f1", "f1_varlingam", "atual_n", "AL_n", "AL_prec", "AL_rev", "CA_prec", "CA_rev", "atual_prec", "atual_rev"]
print(C.groupby(["conj", "grupo"]).agg(cen=("n", "size"), **{c: (c, "mean") for c in cols}).round(3).to_string())
print(C[cols].mean().round(3).to_string())
print("   AL >= VARLiNGAM - 0.03 in", int((C.AL_f1 >= C.f1_varlingam - 0.03).sum()), "of", len(C), "| corr(AL, VARLiNGAM) =", round(np.corrcoef(C.AL_f1, C.f1_varlingam)[0, 1], 2))
print("   adaptive lasso hits that were NOT pipeline candidates:", int(C.al_tp_fora_dos_candidatos.sum()), "of", int(C.AL_tp.sum()))
print("\n== Other families with estimable OLS (AL = adaptive lasso alone; CA = Stage 6 intersect AL)")
R = D[(D.fam != "causal_chamber") & (D.conj != "cc9_visto")]
print(R.groupby("fam").agg(cen=("n", "size"), atual=("atual_f1", "mean"), C=("C_f1", "mean"), AL=("AL_f1", "mean"), CA=("CA_f1", "mean"), varl=("f1_varlingam", "mean"), atual_mcc=("atual_mcc", "mean"), AL_mcc=("AL_mcc", "mean")).round(3).to_string())

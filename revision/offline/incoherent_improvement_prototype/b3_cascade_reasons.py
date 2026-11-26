# Provenance: 3rd analysis of 2026-11-15 (section 10 of ACHADOS_melhoria_incoherent_2026-11-15.md): cause of the loss of detection.
# Only reads outputs and series already saved; does not run the pipeline or any baseline. POST HOC.
# Why does the best-fit model reject? Rejection reasons of the linear model (bayesian_ridge) and of the best-fit model,
# for true and false pairs, and what the conditional test alone (only the p-value of that model) would deliver.
import os
import pandas as pd, numpy as np, glob, os, sys, re
F2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repository root
SETS = [("val12", os.path.join(F2, "Results_Revisao", "r1_3b_previa_validacao", "run"), "patch_tau10_final"),
        ("art90", os.path.join(F2, "Results", "Run_20261011_132346"), "patch_tau10"),
        ("art90", os.path.join(F2, "Results_Revisao", "r1_5d_auprc_90", "cc"), "patch_tau10"),
        ("cc9_visto", os.path.join(F2, "Results_Revisao", "r1_2c_benchmark_novo", "run"), "patch_tau10")]
NS = {"np": np, "nan": float("nan"), "inf": float("inf"), "__builtins__": {}}
def tb(s): return s.astype(str) == "True"
def fl(x):
    try: return float(x)
    except Exception: return np.nan
def reasons(m):
    out = []
    for r in m.get("reasons_fail", []):
        k = re.search(r"\[(.*?)\]", r); out.append(k.group(1) if k else r[:20])
    return out
rows = []
for set_name, root, mod in SETS:
    for sc in sorted(glob.glob(os.path.join(root, "*", ""))):
        if not (os.path.exists(sc + "ground_truth.csv") and os.path.exists(sc + mod + "/original.csv")): continue
        name = os.path.basename(sc[:-1])
        g = pd.read_csv(sc + "ground_truth.csv"); gt = {(a, b) for a, b, c in zip(g.source_entity, g.target_entity, g.causal_gt) if c}
        o = pd.read_csv(sc + mod + "/original.csv", low_memory=False)
        if "cascade_log" not in o or "significant_stage6" not in o: continue
        o = o[o.lag > 0]
        reg = o.evidence_regime.iloc[0] if len(o) else "-"; fb = o.evidence_fallback.iloc[0] if "evidence_fallback" in o and len(o) else ("stage6" if reg == "incoherent" else "-")
        for a, b, lag, s6, x in zip(o.source_entity, o.target_entity, o.lag, tb(o.significant_stage6), o.cascade_log):
            if not (isinstance(x, str) and x.startswith("[")): continue
            log = eval(x, NS); ev = [m for m in log if m.get("t_stat") is not None]
            if not ev: continue
            loss = lambda m: (1 - fl(m.get("r2_nuisance_T"))) * (1 - fl(m.get("r2_nuisance_O")))
            ok = [m for m in ev if np.isfinite(loss(m))]; best = min(ok, key=loss) if ok else ev[0]
            lin = next((m for m in ev if m["model"] == "bayesian_ridge"), None)
            r = dict(conj=set_name, cenario=name, grupo=(reg if reg != "incoherent" else "incoh/" + str(fb)), src=a, tgt=b, lag=lag, ehv=(a, b) in gt, s6=bool(s6),
                     best=best["model"], best_acc=bool(best["accepted"]), best_p=fl(best.get("pvalue")), best_pmax=fl(best.get("pval_max_used")), best_t=fl(best.get("t_stat")), best_varT=fl(best.get("var_res_T")),
                     best_r2T=fl(best.get("r2_nuisance_T")), best_r2O=fl(best.get("r2_nuisance_O")), best_motivos="|".join(reasons(best)))
            if lin is not None:
                r.update(lin_acc=bool(lin["accepted"]), lin_p=fl(lin.get("pvalue")), lin_pmax=fl(lin.get("pval_max_used")), lin_varT=fl(lin.get("var_res_T")), lin_r2T=fl(lin.get("r2_nuisance_T")), lin_r2O=fl(lin.get("r2_nuisance_O")), lin_motivos="|".join(reasons(lin)))
            rows.append(r)
        print(set_name, name[-30:], flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)

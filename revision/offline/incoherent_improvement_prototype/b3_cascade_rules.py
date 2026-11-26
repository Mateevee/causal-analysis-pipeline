# Provenance: 3rd analysis of 2026-11-15 (section 10 of ACHADOS_melhoria_incoherent_2026-11-15.md): cause of the loss of detection.
# Only reads outputs and series already saved; does not run the pipeline or any baseline. POST HOC.
# OFFLINE re-evaluation of the Stage 6 cascade decision from the saved cascade_log (no model is run).
# Rules: 'any' (current), 'best_nuisance' and 'majority' (options that already exist in CascadeConfig.cascade_decision) and
# 'r2min0' (acceptance only counts if the out-of-sample R2 of both nuisances is >= 0; this is r2_nuisance_min = 0, which already exists).
# Limitation: only the Stage 6 set is redone; Stages 7-8 and the guard regime are not recomputed.
import os
import pandas as pd, numpy as np, glob, os, sys, re
F2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repository root; OUTPUT = sys.argv[1]
SETS = [("val12", os.path.join(F2, "Results_Revisao", "r1_3b_previa_validacao", "run"), "patch_tau10_final"),
        ("art90", os.path.join(F2, "Results", "Run_20261011_132346"), "patch_tau10"),
        ("art90", os.path.join(F2, "Results_Revisao", "r1_5d_auprc_90", "cc"), "patch_tau10"),
        ("cc9_visto", os.path.join(F2, "Results_Revisao", "r1_2c_benchmark_novo", "run"), "patch_tau10")]
NS = {"np": np, "nan": float("nan"), "inf": float("inf"), "__builtins__": {}}
def tb(s): return s.astype(str) == "True"
def met(decl, gt, N):
    tp = len(decl & gt); fp = len(decl) - tp; fn = len(gt) - tp; tn = N - tp - fp - fn
    p = tp / len(decl) if decl else 0.0; r = tp / len(gt) if gt else 0.0; f = 2*p*r/(p+r) if tp else 0.0
    den = np.sqrt(float(tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)); return len(decl), tp, p, r, f, ((tp*tn - fp*fn) / den if den > 0 else 0.0)
def fl(x):
    try: return float(x)
    except Exception: return np.nan
def rules(log):
    ev = [m for m in log if m.get("t_stat") is not None]
    acc = [m for m in ev if m.get("accepted")]
    loss = lambda m: (1 - fl(m.get("r2_nuisance_T"))) * (1 - fl(m.get("r2_nuisance_O")))
    with_r2 = [m for m in ev if np.isfinite(loss(m))]
    best = min(with_r2, key=loss) if with_r2 else None
    return dict(n_ev=len(ev), n_acc=len(acc), best=(best["model"] if best else ""), best_acc=bool(best["accepted"]) if best else True,
                best_r2T=fl(best.get("r2_nuisance_T")) if best else np.nan, best_r2O=fl(best.get("r2_nuisance_O")) if best else np.nan,
                maj=(2 * len(acc) >= len(ev)) if ev else True,
                acc_r2pos=any(fl(m.get("r2_nuisance_T")) >= 0 and fl(m.get("r2_nuisance_O")) >= 0 for m in acc),
                lin_acc=any(m["model"] == "bayesian_ridge" for m in acc),
                max_r2O_acc=max([fl(m.get("r2_nuisance_O")) for m in acc], default=np.nan), max_r2T_acc=max([fl(m.get("r2_nuisance_T")) for m in acc], default=np.nan))
rows, pair_rows = [], []
for set_name, root, mod in SETS:
    for sc in sorted(glob.glob(os.path.join(root, "*", ""))):
        if not (os.path.exists(sc + "ground_truth.csv") and os.path.exists(sc + mod + "/original.csv")): continue
        name = os.path.basename(sc[:-1])
        g = pd.read_csv(sc + "ground_truth.csv"); gt = {(a, b) for a, b, c in zip(g.source_entity, g.target_entity, g.causal_gt) if c}; N = len(g)
        o = pd.read_csv(sc + mod + "/original.csv", low_memory=False)
        if "cascade_log" not in o or "significant_stage6" not in o: continue
        o = o[o.lag > 0].copy()
        if o.empty: continue
        R = pd.DataFrame([rules(eval(x, NS)) if isinstance(x, str) and x.startswith("[") else rules([]) for x in o.cascade_log], index=o.index)
        o = pd.concat([o, R], axis=1); o["s6"] = tb(o.significant_stage6); o["ehv"] = [(a, b) in gt for a, b in zip(o.source_entity, o.target_entity)]
        reg = o.evidence_regime.iloc[0]; fb = o.evidence_fallback.iloc[0] if "evidence_fallback" in o else ("stage6" if reg == "incoherent" else "-")
        fin_full = pd.read_csv(sc + mod + "/original.csv", low_memory=False, usecols=["source_entity", "target_entity", "lag", "significant"])
        fin = fin_full[tb(fin_full.significant)]; FIN = {(a, b) for a, b in zip(fin.source_entity, fin.target_entity)}; LAG0 = {(a, b) for a, b, l in zip(fin.source_entity, fin.target_entity, fin.lag) if l == 0}
        P = lambda m: {(a, b) for a, b in zip(o.source_entity[m], o.target_entity[m])}
        sets = {"any": P(o.s6), "best": P(o.s6 & o.best_acc), "maj": P(o.s6 & o.maj), "r2min0": P(o.s6 & o.acc_r2pos)}
        r = dict(conj=set_name, cenario=name, regime=reg, fallback=fb, grupo=(reg if reg != "incoherent" else "incoh/" + str(fb)), n_gt=len(gt), universo=N, taxa_base=len(gt) / N, piso=2 * len(gt) / N / (1 + len(gt) / N),
                 n_cand=len(P(o.s6 | ~o.s6)), cand_tp=len(P(o.s6 | ~o.s6) & gt), final_f1=met(FIN, gt, N)[4], final_n=len(FIN))
        for k, S in sets.items():
            nd, tp, p, rc, f, mcc = met(S, gt, N); r.update({f"{k}_n": nd, f"{k}_tp": tp, f"{k}_prec": p, f"{k}_rev": rc, f"{k}_f1": f, f"{k}_mcc": mcc})
            if reg == "incoherent" and fb == "stage6": r[f"saida_{k}_f1"] = met(S | LAG0, gt, N)[4]
            else: r[f"saida_{k}_f1"] = np.nan
        a = o[o.s6]
        r.update(frac_s6_melhor_rejeita=float((~a.best_acc).mean()) if len(a) else np.nan, frac_s6_so_r2neg=float((~a.acc_r2pos).mean()) if len(a) else np.nan, frac_s6_linear_aceita=float(a.lin_acc.mean()) if len(a) else np.nan,
                 mediana_r2O_melhor=float(a.best_r2O.median()) if len(a) else np.nan, mediana_max_r2O_aceitos=float(a.max_r2O_acc.median()) if len(a) else np.nan)
        rows.append(r)
        pair_rows.append(a.assign(conj=set_name, cenario=name, grupo=r["grupo"])[["conj", "cenario", "grupo", "source_entity", "target_entity", "lag", "ehv", "dml_model_used", "n_acc", "n_ev", "best", "best_acc", "best_r2T", "best_r2O", "acc_r2pos", "lin_acc", "max_r2O_acc", "max_r2T_acc"]])
        print(set_name, name[-34:], r["grupo"], r["any_n"], r["best_n"], r["r2min0_n"], flush=True)
pd.DataFrame(rows).to_csv(OUTPUT, index=False); pd.concat(pair_rows).to_csv(OUTPUT.replace(".csv", "_pares.csv"), index=False)

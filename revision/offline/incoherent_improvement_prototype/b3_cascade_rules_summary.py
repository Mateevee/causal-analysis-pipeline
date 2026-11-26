# Provenance: 3rd analysis of 2026-11-15 (section 10 of ACHADOS_melhoria_incoherent_2026-11-15.md): cause of the loss of detection.
# Only reads outputs and series already saved; does not run the pipeline or any baseline. POST HOC.
import pandas as pd, numpy as np, sys
D = pd.read_csv(sys.argv[1]); Pp = pd.read_csv(sys.argv[1].replace(".csv", "_pares.csv"))
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
D["fam"] = D.cenario.str.replace("benchmark_|real_", "", regex=True).str.split("_").str[:2].str.join("_")
D.loc[D.cenario.str.contains("causal_chamber"), "fam"] = "causal_chamber"
for k in ["any", "best", "maj", "r2min0"]: D[f"{k}_lift"] = D[f"{k}_prec"] / D.taxa_base
print("== 1. Who accepts the Stage 6 pairs (accepted pairs, all sets)")
Pp["fam"] = Pp.cenario.str.replace("benchmark_|real_", "", regex=True).str.split("_").str[:2].str.join("_"); Pp.loc[Pp.cenario.str.contains("causal_chamber"), "fam"] = "causal_chamber"
t = Pp.groupby("fam").agg(pares=("ehv", "size"), verd=("ehv", "mean"), melhor_rejeita=("best_acc", lambda x: 1 - x.mean()), so_r2_neg=("acc_r2pos", lambda x: 1 - x.mean()), linear_aceita=("lin_acc", "mean"),
                          r2O_melhor=("best_r2O", "median"), r2O_max_aceito=("max_r2O_acc", "median"), r2T_max_aceito=("max_r2T_acc", "median")).round(3)
print(t.to_string())
print("\n   precision of the accepted pairs, depending on whether the best-fit model accepts or rejects:")
print(Pp.groupby(["fam", "best_acc"]).ehv.agg(["size", "mean"]).round(3).unstack().to_string())
print("\n== 2. Stage 6 set per rule (means per scenario)")
cols = []
for k in ["any", "best", "maj", "r2min0"]: cols += [f"{k}_n", f"{k}_rev", f"{k}_lift", f"{k}_f1", f"{k}_mcc"]
print(D.groupby(["conj", "grupo"]).agg(cen=("n_gt", "size"), **{c: (c, "mean") for c in cols}).round(3).to_string())
print("\n== 3. Per family (all development sets; cc9 separately)")
for set_sel, name in [(D.conj != "cc9_visto", "val12 + art90"), (D.conj == "cc9_visto", "cc9_visto")]:
    S = D[set_sel]; print("--", name); print(S.groupby("fam").agg(cen=("n_gt", "size"), final_f1=("final_f1", "mean"), any_n=("any_n", "mean"), best_n=("best_n", "mean"), any_rev=("any_rev", "mean"), best_rev=("best_rev", "mean"), any_lift=("any_lift", "mean"), best_lift=("best_lift", "mean"),
          any_f1=("any_f1", "mean"), best_f1=("best_f1", "mean"), maj_f1=("maj_f1", "mean"), r2min0_f1=("r2min0_f1", "mean"), any_mcc=("any_mcc", "mean"), best_mcc=("best_mcc", "mean")).round(3).to_string())
print("\n== 4. Emulated output on the incoherent/stage6 scenarios (Stage 6 of the rule + lag 0; assumes the same regime)")
S = D[D.grupo == "incoh/stage6"]
print(S.groupby(["conj", "fam"]).agg(cen=("n_gt", "size"), atual=("final_f1", "mean"), any=("saida_any_f1", "mean"), best=("saida_best_f1", "mean"), maj=("saida_maj_f1", "mean"), r2min0=("saida_r2min0_f1", "mean"), piso=("piso", "mean")).round(3).to_string())
print(S.groupby("conj")[["final_f1", "saida_any_f1", "saida_best_f1", "saida_maj_f1", "saida_r2min0_f1"]].mean().round(4).to_string())

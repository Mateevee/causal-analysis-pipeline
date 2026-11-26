"""Descriptive disaggregation of R1-3b (decision of 2026-11-14), read-only, from the extracted zip."""
import json
import os
import sys

import numpy as np
import pandas as pd

F2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
EXT = F2 + "/Results_Revisao/_outra_maquina/r1_3b_2026-11-20/r1_3b_transicoes_graduais"
sys.path[:0] = [F2 + "/revision", F2 + "/revision/execution", F2]
os.chdir(F2)
import common as c
import r1_3b_graded_transitions as r
from variants import r1_3b_graded as g

RUN = EXT + "/run"
diag = r._read_diagnostics(RUN + "/padrao/diag.jsonl")
scens = sorted({k[0] for k in diag})
econ = pd.read_csv(EXT + "/economia.csv")
was_run = {(a.cenario, a.versao, a.config): bool(a.rodar) for a in econ.itertuples()}


def family_of(s):
    p = s.replace("benchmark_", "").replace("real_", "")
    for f in ("beeline", "rcaeval", "causaltime", "dream3", "logistic_maps", "synthetic_var", "tigramite",
              "easyvista", "causal_chamber", "netsim", "finance", "tep"):
        if f in p:
            return f
    return p.split("_")[0]


def f1(cfg, s, v):
    mod = r.VERSIONS[v]
    if cfg != "padrao" and not was_run.get((s, v, cfg), False):
        cfg = "padrao"
    k = r._counts(os.path.join(RUN, cfg), s, mod)
    return None if k is None else r._f1(k)


recs = []
for s in scens:
    d = diag[(s, "graduada", "padrao")]
    coh, dm = d["coerencia"], d["dml"]
    r2 = np.asarray(dm.get("r2_pares") or [], dtype=float)
    w_inc = g.weight_inc(coh.get("rho"), coh.get("n"), coh.get("regime"), 0.3, 0.05)
    w_lean = g.weight_lean(int((r2 < 0.5).sum()), len(r2)) if len(r2) else float("nan")
    in_band = lambda w: bool(np.isfinite(w) and 0.001 < w < 0.999)
    row = {"cenario": s, "familia": family_of(s), "held_out": bool(c.is_heldout(c.scenario_key(s))),
           "w_inc": w_inc, "w_lean": w_lean, "inc_faixa": in_band(w_inc), "lean_faixa": in_band(w_lean)}
    row["faixa"] = row["inc_faixa"] or row["lean_faixa"]
    for v in r.VERSIONS:
        for cfg in r.CONFIGS:
            row[f"{v}|{cfg}"] = f1(cfg, s, v)
    recs.append(row)
T = pd.DataFrame(recs)
T["dif_padrao"] = T["graduada|padrao"] - T["final|padrao"]
out = F2 + "/Results_Revisao/r1_3b_desagregacao"
T.to_csv(out + "/desagregacao_por_cenario.csv", index=False)

print("== Default: graded − final, by weight inside the band")
for k, G in T.groupby("faixa"):
    dd = G.dif_padrao
    print(f"in_band={k}: n={len(G)} (HO {G.held_out.sum()}), changed={int((dd.abs() > 1e-9).sum())}, "
          f"up={int((dd > 1e-9).sum())}, down={int((dd < -1e-9).sum())}, mean={dd.mean():+.4f}")
print("inc inside the band:", int(T.inc_faixa.sum()), " lean inside the band:", int(T.lean_faixa.sum()))
print("\n== Default: scenarios where graded ≠ final")
m = T[T.dif_padrao.abs() > 1e-9].sort_values("dif_padrao")
print(m[["cenario", "held_out", "w_inc", "w_lean", "final|padrao", "graduada|padrao", "dif_padrao"]].to_string(index=False))
print("\n== Default: by family")
print(T.groupby("familia").agg(n=("cenario", "size"), faixa=("faixa", "sum"),
                               mudou=("dif_padrao", lambda x: int((x.abs() > 1e-9).sum())),
                               dif_media=("dif_padrao", "mean")).to_string())

print("\n== Perturbation: scenarios whose F1 changes relative to the default of the same version")
res = []
for cfg in [k for k in r.CONFIGS if k != "padrao"]:
    row = {"config": cfg}
    for v in r.VERSIONS:
        dd = T[f"{v}|{cfg}"] - T[f"{v}|padrao"]
        row[f"{v}_mudam"] = int((dd.abs() > 1e-9).sum())
        row[f"{v}_max_abs"] = round(float(dd.abs().max()), 4)
        row[f"{v}_soma_quedas"] = round(float(dd[dd < 0].sum()), 4)
        row[f"{v}_HO_media_dif"] = round(float(dd[T.held_out].mean()), 4)
    res.append(row)
R = pd.DataFrame(res)
print(R.to_string(index=False))
R.to_csv(out + "/perturbacao_mudancas.csv", index=False)

print("\n== Per scenario: number of configurations in which it changes and largest |Δ| (only scenarios that change in some)")
cfgs = [k for k in r.CONFIGS if k != "padrao"]
rows = []
for _, a in T.iterrows():
    for v in r.VERSIONS:
        ds = [a[f"{v}|{k}"] - a[f"{v}|padrao"] for k in cfgs]
        nm = sum(abs(x) > 1e-9 for x in ds)
        if nm:
            rows.append({"cenario": a.cenario, "versao": v, "HO": a.held_out, "n_cfg": nm,
                         "max_abs": round(max(abs(x) for x in ds), 4)})
P = pd.DataFrame(rows)
print(P.groupby("versao").agg(cenarios=("cenario", "size"), n_cfg_total=("n_cfg", "sum"),
                              max_abs=("max_abs", "max"), med_max=("max_abs", "median")).to_string())
print(P.sort_values("max_abs", ascending=False).head(12).to_string(index=False))

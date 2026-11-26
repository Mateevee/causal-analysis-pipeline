"""
Computations done in the terminal on 2026-11-22 that became published numbers, now under version control.

Why: the reproducibility index promises that every number comes out of a named script. These aggregates are not
printed by the reports of the source scripts; they were computed from the same output files, in the terminal, while
assembling tables S28-S44. This script recomputes them from the same files and checks them against the published
value; it changes no number. It exits with code 1 if any value does not match.

Computations (source file -> where the number is published):
  A. r1_m7a_tempos_i7: time per method with ok runs only (main-text Table XII; S43.1) and fractions of time per
     stage on the 90 scenarios and on the scaling set (main-text III-I and V-H; S43.2; ST7).
  B. r2_2a_varredura_T (round 5): pipeline counts at T = 40 and distance to the best method (V-B; S32).
  C. r1_4c_divisao_amostra (round 3): classification of the 1,089 links of half A (V-C; S34).
  D. r1_4e_intervencoes: reading by the published graph (27 of 43; the 9 pairs of the angle sensors) (V-C; S35).
  E. r2_m2_jaccard (round 5): fraction of stable edges and Jaccard per part; Walker (V-G; S42.2).
  F. r1_2c_benchmark_novo_88: paired means pipeline x VARLiNGAM per block and best method per family (V-A; S30.3).

Usage:  python revision/offline/supplement_checks_2026-11-22.py
Output: Results_Revisao/contas_suplemento_2026-11-22/relatorio.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path[:0] = [REV, os.path.join(REV, "execution")]
RES = os.path.join(os.path.dirname(REV), "Results_Revisao")
OM = os.path.join(RES, "_outra_maquina")
OUTPUT = os.path.join(RES, "contas_suplemento_2026-11-22")

lines, failures = [], []


def check(label, value, published, tol=0.0005):
    ok = abs(float(value) - float(published)) <= tol
    lines.append(f"| {label} | {value:.4g} | {published} | {'ok' if ok else 'DIFFERENT'} |")
    if not ok:
        failures.append(label)


def section(title):
    lines.extend(["", f"## {title}", "", "| Computation | Recomputed | Published | Check |", "|---|---|---|---|"])


def main():
    # A. times on the i7
    T = os.path.join(OM, "execucao_2026-11-19", "r1_m7a_tempos_i7")
    d = pd.read_csv(os.path.join(T, "tempo_por_metodo_por_cenario.csv"))
    d = d[d.status == "ok"]
    section("A. Times on the i7 (Table XII; S43)")
    pub = {"pipeline": (22.6, 81.1, 104.5, 893.8, 90), "PCMCI": (2.5, 4.7, 21.3, 105.7, 90),
           "PCMCI+": (1.0, 2.8, 7.0, 36.6, 90), "DYNOTEARS": (1.8, 3.1, 39.6, 240.3, 90),
           "CUTS+": (5.4, 14.0, 19.7, 69.3, 90), "VARLiNGAM": (0.2, 0.6, 3.8, 10.1, 77)}
    for m, (med, ho, f2140, mx, n) in pub.items():
        g = d[d.metodo == m]
        check(f"{m}: scenarios", g.cenario.nunique(), n)
        check(f"{m}: median", round(g.total_s.median(), 1), med, 0.05)
        check(f"{m}: held-out median", round(g[g.heldout == True].total_s.median(), 1), ho, 0.05)
        check(f"{m}: median 21-40 series", round(g[g.faixa == "21–40"].total_s.median(), 1), f2140, 0.05)
        check(f"{m}: maximum", round(g.total_s.max(), 1), mx, 0.05)
    t = pd.read_csv(os.path.join(T, "tempos_por_execucao.csv"))
    stages = ["_stage3to5_prefilter", "_stage6_dml", "_stage8a_structural_pruning", "_stage8b_contemporaneous"]
    a = t[(t.parte == "A") & (t.modelo == "patch_tau10")]
    sa = a[stages + ["total_s"]].sum()
    for e, p in zip(stages, (21, 32, 41, 4)):
        check(f"90 scenarios: fraction {e} (%)", round(100 * sa[e] / sa.total_s), p, 0.5)
    c = t[(t.parte == "C") & (t.modelo == "patch_tau10")].copy()
    c["n"] = c.cenario.str.extract(r"_n(\d+)_").astype(int)
    gm = c.groupby("n")[stages + ["total_s"]].mean()
    for n, (pf, dm, s8a, tot) in {10: (25, 20, 47, 14.9), 80: (58, 22, 18, 359.5), 160: (60, 21, 19, 1377.7)}.items():
        check(f"scaling n={n}: prefilter (%)", round(100 * gm.loc[n, stages[0]] / gm.loc[n, "total_s"]), pf, 0.5)
        check(f"scaling n={n}: Stage 6 (%)", round(100 * gm.loc[n, stages[1]] / gm.loc[n, "total_s"]), dm, 0.5)
        check(f"scaling n={n}: Stage 8a (%)", round(100 * gm.loc[n, stages[2]] / gm.loc[n, "total_s"]), s8a, 0.5)
        check(f"scaling n={n}: total (s)", round(gm.loc[n, "total_s"], 1), tot, 0.05)

    # B. short series
    r = pd.read_csv(os.path.join(OM, "rodada5_R2", "r2_2a_varredura_T", "resumo.csv"))
    p40 = r[(r.model == "patch_tau10") & (r["T"] == 40)]
    section("B. Short series (S32)")
    check("T=40: scenarios with no hit", int((p40.TP == 0).sum()), 17)
    check("T=40: summed false positives", int(p40.FP.sum()), 18)
    check("T=40: aggregate precision", round(p40.TP.sum() / (p40.TP.sum() + p40.FP.sum()), 2), 0.40, 0.005)
    best = r.groupby(["T", "model"]).f1.mean().unstack()
    for TT, gap in {40: -0.222, 84: -0.184, 200: -0.105, 300: -0.001, 1000: -0.077}.items():
        b = best.loc[TT].drop("patch_tau10_final", errors="ignore")
        check(f"T={TT}: pipeline − best", round(b["patch_tau10"] - b.max(), 3), gap, 0.0015)

    # C. sample split
    L = pd.read_csv(os.path.join(OM, "rodada3_2026-11-11", "r1_4c_divisao_amostra", "links.csv"))
    section("C. Sample split (S34)")
    check("reported links", len(L), 1089)
    check("true at the true lag", int((L.verdadeiro & L.lag_correto).sum()), 965)
    check("causal pair at another lag", int((L.verdadeiro & ~L.lag_correto).sum()), 25)
    f = L[~L.verdadeiro]
    check("non-causal pair", len(f), 99)
    check("non-causal pair with CI in B excluding zero", round(((f.lo_B > 0) | (f.hi_B < 0)).mean(), 3), 0.212, 0.0005)

    # D. interventions
    P = pd.read_csv(os.path.join(RES, "r1_4e_intervencoes", "por_par.csv"))
    ok = P.sinal_ok.astype(bool)
    ang = P.alvo.astype(str).str.startswith("angle")
    section("D. Interventions, reading by the published graph (S35)")
    check("pairs", len(P), 52)
    check("same sign, all", int(ok.sum()), 33)
    check("pairs of the angle sensors", int(ang.sum()), 9)
    check("same sign outside the angle sensors", int(ok[~ang].sum()), 27)
    disc = ang & (np.sign(P.beta) != np.sign(P.contraste_mid_strong))
    check("angle: beta and contrast with opposite signs", int(disc.sum()), 4)
    check("  of these, counted as hits", int((disc & ok).sum()), 4)
    check("reported by the pipeline / with same sign", int(P.reportado.astype(bool).sum()), 7)

    # E. stability
    J = pd.read_csv(os.path.join(OM, "rodada5_R2", "r2_m2_jaccard", "por_cenario.csv"))
    section("E. Stability under subsampling (S42.2)")
    for h, (stable_, jac) in {False: (0.838, 0.818), True: (0.561, 0.629)}.items():
        g = J[J.heldout == h]
        check(f"{'held-out' if h else 'development'}: stable edges", round(g.arestas_estaveis_frac.mean(), 3), stable_)
        check(f"{'held-out' if h else 'development'}: Jaccard between subsamples",
              round(g.jaccard_entre_subamostras.mean(), 3), jac)
    w = J[J.familia.astype(str).str.contains("Walker")]
    check("Walker: mean F1 of the subsamples", round(w.f1_subamostras.mean(), 3), 0.617)

    # F. new benchmark with 88
    B = pd.read_csv(os.path.join(RES, "r1_2c_benchmark_novo_88", "T1_por_cenario.csv"))
    F = B.pivot_table(index="cenario", columns="model", values="f1")
    bl = B.drop_duplicates("cenario").set_index("cenario")["bloco"]
    section("F. New benchmark, 88 scenarios (V-A; S30.3)")
    s = bl[bl == "disco"].index
    x = F.loc[s, ["patch_tau10", "varlingam_tau10"]].dropna()
    check("disco: pairs with VARLiNGAM", len(x), 23)
    check("disco: pipeline on the pairs", round(x.patch_tau10.mean(), 3), 0.211)
    check("disco: VARLiNGAM on the pairs", round(x.varlingam_tau10.mean(), 3), 0.325)
    check("disco: VARLiNGAM above", int((x.varlingam_tau10 > x.patch_tau10).sum()), 21)
    fam = B.drop_duplicates("cenario").set_index("cenario")["familia"]
    M = F.groupby(fam).mean()
    main_models = ["patch_tau10", "pcmci_tigramite_tau10", "pcmciplus_tau10", "dynotears_tau10", "cuts_plus_tau10",
                   "varlingam_tau10"]
    leader = M[main_models].idxmax(axis=1)
    check("families where the pipeline leads", int((leader == "patch_tau10").sum()), 1)
    d3 = [k for k in M.index if "DREAM3" in k][0]
    check("DREAM3: pipeline − VARLiNGAM", round(M.loc[d3, "patch_tau10"] - M.loc[d3, "varlingam_tau10"], 3), 0.001,
          0.0006)

    os.makedirs(OUTPUT, exist_ok=True)
    header = ["# Supplement computations under version control (2026-11-22)", "",
              f"Checks: {len([l for l in lines if l.startswith('| ') and l.endswith(' |') and ('ok' in l or 'DIFFERENT' in l)])}; "
              f"different: {len(failures)}."]
    open(os.path.join(OUTPUT, "relatorio.md"), "w", encoding="utf-8", newline="\r\n").write("\n".join(header + lines) + "\n")
    print("\n".join(header + lines))
    if failures:
        raise SystemExit(f"values different from the published ones: {failures}")


if __name__ == "__main__":
    from run_log import execution
    with execution("contas_suplemento_2026-11-22", __file__):
        main()

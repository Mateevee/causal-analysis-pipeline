"""
R1-4e — pipeline effects against real interventions in the light tunnel.

Follows revision/PROTOCOL_R1-4e_interventions.md, frozen on 2026-11-21 20:37
(CRLF 2b7ac0c2...27be8b9; LF 7ea51d43...2b9dc3). This script was written after the freeze. If it diverges from the
protocol, the protocol prevails.

Usage (from the repository root):  python revision/offline/r1_4e_interventions.py
Output: Results_Revisao/r1_4e_intervencoes/
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "execution")]
import common as c  # noqa: E402

NAME = "r1_4e_intervencoes"
INT = os.path.join(os.path.dirname(c.OUTPUT_ROOT), "external_data", "causal_chamber", "lt_interventions_standard_v1")
R12C = os.path.join(c.OUTPUT_ROOT, "_outra_maquina", "r1_2c_completo_2026-11-18", "r1_2c_benchmark_novo",
                    "r1_2c_benchmark_novo", "run", "real_causal_chamber_lt_color_mix")
COLORS = ["red", "green", "blue"]
ACTUATORS = {"red", "green", "blue", "osr_c", "v_c", "pol_1", "pol_2", "osr_angle_1", "osr_angle_2", "v_angle_1",
             "v_angle_2", "l_11", "l_12", "l_21", "l_22", "l_31", "l_32", "diode_ir_1", "diode_vis_1", "diode_ir_2",
             "diode_vis_2", "diode_ir_3", "diode_vis_3", "t_ir_1", "t_vis_1", "t_ir_2", "t_vis_2", "t_ir_3", "t_vis_3"}
NON_VARIABLES = {"timestamp", "config", "counter", "flag", "intervention"}


def scenes():
    out = {"lt_color_mix": R12C}
    for s in ("real_causal_chamber_lt_actuators_white", "real_causal_chamber_lt_smooth_polarizers"):
        out[s.replace("real_causal_chamber_", "")] = c.folder(c.scenario_key(s))
    return out


def truth():
    """{(color, sensor): dict(beta, se, z, contrast)} — section 3 of the protocol."""
    ref = pd.read_csv(os.path.join(INT, "uniform_reference.csv"))
    sens = [x for x in ref.columns if x not in ACTUATORS and x not in NON_VARIABLES]
    out = {}
    for color in COLORS:
        mid = pd.read_csv(os.path.join(INT, f"uniform_{color}_mid.csv"))
        st = pd.read_csv(os.path.join(INT, f"uniform_{color}_strong.csv"))
        E = pd.concat([ref, mid, st], ignore_index=True)
        reg = [a for a in sorted(ACTUATORS) if a in E.columns and E[a].nunique() > 1]
        X = sm.add_constant(E[reg].astype(float))
        for y in sens:
            if E[y].nunique() <= 1 or not np.issubdtype(E[y].dtype, np.number):
                continue
            m = sm.OLS(E[y].astype(float), X, missing="drop").fit(cov_type="HC3")
            b, se = float(m.params[color]), float(m.bse[color])
            dc = st[color].mean() - mid[color].mean()
            contrast = (st[y].mean() - mid[y].mean()) / dc if dc else np.nan
            out[(color, y)] = {"beta": b, "ep_beta": se, "z_beta": b / se if se else np.nan,
                               "contraste_mid_strong": contrast, "regressores": ",".join(reg)}
    return out, sens


def main():
    rel = c.Report(NAME, "R1-4e — pipeline effects against real interventions (light tunnel)")
    rel.p("Protocol: `revision/PROTOCOL_R1-4e_interventions.md` (frozen 2026-11-21 20:37; CRLF 2b7ac0c2…27be8b9).")
    V, sens = truth()
    rows_ = []
    for scene, folder in scenes().items():
        src = pd.read_csv(os.path.join(folder, "source.csv"))
        sd = src.groupby("Name").VAL.std()
        o = pd.read_csv(os.path.join(folder, "patch_tau10", "original.csv"), skipinitialspace=True)
        o["source_entity"] = o.source_entity.astype(str).str.strip()
        o["target_entity"] = o.target_entity.astype(str).str.strip()
        rs = pd.read_csv(os.path.join(folder, "patch_tau10", "results.csv"))
        rep = set(zip(rs.source_entity.astype(str).str.strip(), rs.target_entity.astype(str).str.strip()))
        o = o[(o.lag >= 1) & o.source_entity.isin(COLORS) & o.target_entity.isin(sens)]
        for r in o.itertuples():
            v = V.get((r.source_entity, r.target_entity))
            if v is None or r.source_entity not in sd or r.target_entity not in sd or not sd[r.source_entity]:
                continue
            k = sd[r.target_entity] / sd[r.source_entity]
            th = float(r.causal_effect) * k if pd.notna(r.causal_effect) else np.nan
            lo = float(r.ci_95_lower) * k if pd.notna(r.ci_95_lower) else np.nan
            hi = float(r.ci_95_upper) * k if pd.notna(r.ci_95_upper) else np.nan
            rows_.append({"cena": scene, "fonte": r.source_entity, "alvo": r.target_entity, "lag": int(r.lag),
                          "reportado": (r.source_entity, r.target_entity) in rep, "theta_z": r.causal_effect,
                          "theta_bruto": th, "ic_lo_bruto": lo, "ic_hi_bruto": hi, **v})
    D = pd.DataFrame(rows_)
    D["beta_sig"] = D.z_beta.abs() > 1.96
    D["sinal_ok"] = np.sign(D.theta_bruto) == np.sign(D.beta)
    D["razao"] = D.theta_bruto / D.beta
    D["ic_contem_beta"] = (D.ic_lo_bruto <= D.beta) & (D.beta <= D.ic_hi_bruto)
    D["ic_exclui_zero"] = (D.ic_lo_bruto > 0) | (D.ic_hi_bruto < 0)
    D["sinal_beta_contraste_ok"] = np.sign(D.beta) == np.sign(D.contraste_mid_strong)
    rel.csv("por_par.csv", D)
    rel.p(f"Pairs evaluated (color → sensor, row with lag ≥ 1): {len(D)}; with significant β: {int(D.beta_sig.sum())}; "
          f"without θ̂ (NaN): {int(D.theta_bruto.isna().sum())}.")
    rel.p(f"Check of the ground truth: sign of β equal to that of the mid → strong contrast in "
          f"{int(D[D.beta_sig].sinal_beta_contraste_ok.sum())} of {int(D.beta_sig.sum())} pairs with significant β.")
    S = D[D.beta_sig & D.theta_bruto.notna()]
    rows = []
    for name, g in list(S.groupby("cena")) + [("total", S)]:
        rho = spearmanr(g.theta_bruto, g.beta).statistic if len(g) > 2 else np.nan
        rows.append([name, len(g), f"{int(g.sinal_ok.sum())} of {len(g)} ({g.sinal_ok.mean():.0%})", c.fmt(rho),
                     c.fmt(g.razao.median()), f"{c.fmt(g.razao.quantile(0.25))} to {c.fmt(g.razao.quantile(0.75))}",
                     f"{int(g.ic_contem_beta.sum())} of {len(g)}"])
    rel.h("## E1 to E4 — pairs with significant β")
    rel.table(["Scene", "Pairs", "E1 same sign", "E2 Spearman θ̂×β", "E3 ratio θ̂/β (median)", "E3 quartiles",
               "E4 CI contains β"], rows)
    N = D[~D.beta_sig & D.theta_bruto.notna()]
    rel.h("## E5 — pairs without a significant interventional effect")
    rel.p(f"{len(N)} pairs; 95% CI of θ̂ excludes zero in {int(N.ic_exclui_zero.sum())}.")
    R = S[S.reportado]
    rel.h("## E6 — only the pairs reported by the pipeline")
    rel.p(f"{len(R)} reported pairs with significant β; same sign in {int(R.sinal_ok.sum())}; median θ̂/β ratio "
          f"{c.fmt(R.razao.median()) if len(R) else '—'}.")
    rel.h("## All pairs")
    cols = ["cena", "fonte", "alvo", "lag", "reportado", "theta_bruto", "ic_lo_bruto", "ic_hi_bruto", "beta", "ep_beta",
            "contraste_mid_strong", "sinal_ok"]
    rel.table(cols, [[(f"{x:.4g}" if isinstance(x, float) else x) for x in row] for row in D[cols].values.tolist()])
    e1 = S.sinal_ok.mean() if len(S) else np.nan
    if np.isnan(e1):
        reading = "No pairs with significant β: no reading."
    elif e1 >= 0.90:
        reading = (f"E1 = {e1:.0%} (≥ 90%): the sign of the estimated effect agrees with that of the real intervention in "
                   f"{int(S.sinal_ok.sum())} of {len(S)} pairs; supports 'directional predictive influence' on real data.")
    elif e1 >= 0.70:
        reading = f"E1 = {e1:.0%} (70–90%): partial agreement; see the discordant pairs."
    else:
        reading = f"E1 = {e1:.0%} (< 70%): the direction does not hold on real data; report as a limitation."
    rel.h("## Reading (fixed beforehand)")
    rel.p("**" + reading + "** Magnitude and interval: descriptive only (section 5 of the protocol); the text does not "
          "claim quantitative validity.")
    rel.data.update({"pares": len(D), "beta_sig": int(D.beta_sig.sum()), "E1": None if np.isnan(e1) else float(e1),
                     "E3_razao_mediana": float(S.razao.median()) if len(S) else None, "leitura": reading})
    rel.save()
    print(open(os.path.join(rel.dir, "relatorio.md"), encoding="utf-8").read())


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        main()

# BLIND CHECK OF THE DIAGNOSIS (VARLiNGAM x pipeline) ON THE 79 REMAINING SCENARIOS OF R1-2c.
# Design fixed on 2026-11-15, before any result of these 79 was seen.
# See section 9 of revision/ACHADOS_melhoria_incoherent_2026-11-15.md.
# NOT part of R1-2c: it is a post hoc analysis, declared, reported separately. Runs no method: only reads the
# saved outputs of the single run (patch_tau10, varlingam_tau10, source.csv, ground_truth.csv).
# The 9 scenarios already seen in the intermediate look are left out.
# Usage: python blind_check_79.py <output_folder> <run_root_1> [<run_root_2> ...]
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LassoLarsIC

warnings.filterwarnings("ignore")
L = 10
MOD = "patch_tau10"
VL = "varlingam_tau10"
SEEN = {
    "real_causal_chamber_lt_color_mix",
    "real_causal_chamber_wt_actuators_random_walk_11",
    "real_causal_chamber_wt_actuators_random_walk_13",
    "real_causal_chamber_wt_actuators_random_walk_15",
    "real_causal_chamber_wt_loads_hatch_mix_fast_run_1",
    "real_causal_chamber_wt_loads_hatch_mix_fast_run_3",
    "real_causal_chamber_wt_loads_hatch_mix_fast_run_5",
    "real_causal_chamber_wt_loads_hatch_mix_slow_run_2",
    "real_causal_chamber_wt_loads_hatch_mix_slow_run_4",
}


def pairs(df):
    return {(a, b) for a, b in zip(df.source_entity, df.target_entity)}


def tb(s):
    return s.astype(str) == "True"


def met(decl, gt, N):
    tp = len(decl & gt)
    fp = len(decl) - tp
    fn = len(gt) - tp
    tn = N - tp - fp - fn
    p = tp / len(decl) if decl else 0.0
    r = tp / len(gt) if gt else 0.0
    f = 2 * p * r / (p + r) if tp else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return len(decl), tp, f, ((tp * tn - fp * fn) / den if den > 0 else 0.0)


def lasso_var(wide):
    # Rule C, configuration (iii) of ACHADOS: lasso-BIC per target on a VAR(10) of all series, standardized, in
    # levels; noise variance = variance of the residual of the target's own AR(10).
    X = wide.to_numpy(float)
    X = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-12)
    X = np.nan_to_num(X)
    n, k = X.shape
    names = list(wide.columns)
    lagm = np.column_stack([X[L - l:n - l, j] for j in range(k) for l in range(1, L + 1)])
    sel = set()
    for t in range(k):
        Y = X[L:, t]
        own = lagm[:, t * L:(t + 1) * L]
        e = Y - own @ np.linalg.lstsq(own, Y, rcond=None)[0]
        m = LassoLarsIC(criterion="bic", noise_variance=float(e.var()) + 1e-6).fit(lagm, Y)
        c = np.abs(m.coef_).reshape(k, L).sum(1)
        sel |= {(names[j], names[t]) for j in range(k) if j != t and c[j] > 0}
    return sel


def verdict(ok, n):
    return "NO SCENARIOS" if n == 0 else ("CONFIRMED" if ok else "REFUTED")


def table_row(name, S):
    cells = []
    for k in ["C", "Cg"]:
        d = S[k + "_f1"] - S.atual_f1
        pv = wilcoxon(S[k + "_f1"], S.atual_f1).pvalue if (d.abs() > 1e-12).any() else 1.0
        cells.append("%.4f (%d/%d; %.3g)" % (S[k + "_f1"].mean(), int((d > 1e-9).sum()), int((d < -1e-9).sum()), pv))
    return "| %s | %d | %d (gate: %d) | %.4f | %s | %s | %.4f | %.4f | %.4f |" % (
        name, len(S), int(S.atua.sum()), int((S.atua & S.portao).sum()), S.atual_f1.mean(), cells[0], cells[1],
        S.atual_mcc.mean(), S.C_mcc.mean(), S.Cg_mcc.mean())


def main(output, roots, include_seen=False):
    os.makedirs(output, exist_ok=True)
    rows = []
    done = set()
    for root in roots:
        for sc in sorted(glob.glob(os.path.join(root, "*", ""))):
            name = os.path.basename(sc[:-1])
            if name in done or (name in SEEN and not include_seen):
                continue
            required = ["ground_truth.csv", "source.csv", MOD + "/original.csv", "validation_summary.csv"]
            if not all(os.path.exists(sc + f) for f in required):
                continue
            done.add(name)
            g = pd.read_csv(sc + "ground_truth.csv")
            gt = {(a, b) for a, b, c in zip(g.source_entity, g.target_entity, g.causal_gt) if c}
            N = len(g)
            o = pd.read_csv(sc + MOD + "/original.csv", low_memory=False)
            reg = o.evidence_regime.iloc[0] if "evidence_regime" in o else "-"
            if "evidence_fallback" in o:
                fb = o.evidence_fallback.iloc[0]
            else:
                fb = "stage6" if reg == "incoherent" else "-"
            src = pd.read_csv(sc + "source.csv")
            wide = src.pivot_table(index="Date", columns="Name", values=src.columns[2]).sort_index()
            T, n = wide.shape
            f_ = o[tb(o.significant)]
            fin = pairs(f_)
            applies = (reg == "incoherent" and fb == "stage6")
            gate = (T - L) >= n * L
            s6 = pairs(o[tb(o.significant_stage6) & (o.lag > 0)]) if "significant_stage6" in o else set()
            C = ((s6 & lasso_var(wide)) | pairs(f_[f_.lag == 0])) if applies else fin
            Cg = C if gate else fin
            v = pd.read_csv(sc + "validation_summary.csv").set_index("model")
            r = dict(cenario=name, cc="causal_chamber" in name, passeio="random_walk" in name, regime=reg, fallback=fb,
                     atua=applies, portao=gate, n=n, T=T, n_gt=len(gt), universo=N,
                     piso=2 * len(gt) / N / (1 + len(gt) / N), s6_tp=len(s6 & gt), f1_oficial=v.loc[MOD, "f1"],
                     f1_varlingam=v.loc[VL, "f1"] if VL in v.index else np.nan)
            for k, d in [("atual", fin), ("C", C), ("Cg", Cg)]:
                nd, tp, f, mcc = met(d, gt, N)
                r.update({k + "_n": nd, k + "_tp": tp, k + "_f1": f, k + "_mcc": mcc})
            r["vl_tp"] = r["vl_tp_so_lag0"] = np.nan
            if os.path.exists(sc + VL + "/original.csv"):
                w = pd.read_csv(sc + VL + "/original.csv")
                w = w[tb(w.significant)]
                if len(w):
                    lagmax = w.groupby(["source_entity", "target_entity"]).lag.max()
                    vtp = set(lagmax.index) & gt
                    r["vl_tp"] = len(vtp)
                    r["vl_tp_so_lag0"] = sum(lagmax[q] == 0 for q in vtp)
                else:
                    r["vl_tp"] = r["vl_tp_so_lag0"] = 0
            rows.append(r)
            print(name, reg, fb, n, T, round(r["atual_f1"], 3), round(r["C_f1"], 3), flush=True)
    D = pd.DataFrame(rows)
    if D.empty:
        print("no new scenario found")
        return
    D.to_csv(os.path.join(output, "conferencia_cega_por_cenario.csv"), index=False)
    cc = D[D.cc]
    ncc, ntot = len(cc), len(D)
    out = []
    warning = "" if (ntot == 79 and ncc == 9) else "  **INCOMPLETE** (expected: 79 and 9)"
    out.append("# Blind check of the diagnosis — %d new scenarios (%d from Causal Chambers)%s" % (ntot, ncc, warning))
    out.append("")
    out.append("Checks the F1 of the current method against the official one: max|diff| = %.5f" % (D.atual_f1 - D.f1_oficial).abs().max())
    out.append("")
    out.append("## Part A — predictions (remaining Causal Chambers)")
    out.append("")
    a = cc[cc.atua]
    m = int((a.C_f1 > a.atual_f1 + 1e-9).sum())
    out.append("- **P1** (C improves the F1 in at least 2/3 of the incoherent/stage6 scenarios): %d of %d -> **%s**" % (
        m, len(a), verdict(len(a) > 0 and m >= 2 * len(a) / 3, len(a))))
    p = a[a.passeio & a.f1_varlingam.notna()]
    diff = (p.f1_varlingam - p.atual_f1).mean()
    recovered = (p.C_f1 - p.atual_f1).mean()
    note = " (there was no difference to recover)" if len(p) and diff <= 0 else ""
    out.append("- **P2** (on the incoherent random walks, C recovers on average at least half of the VARLiNGAM − current difference): "
               "difference %+.3f, recovered %+.3f, %d scenarios -> **%s**%s" % (
                   diff, recovered, len(p), verdict(len(p) > 0 and (diff <= 0 or recovered >= 0.5 * diff), len(p)), note))
    c = cc[cc.regime == "coherent"]
    s6tp, fintp = int(c.s6_tp.sum()), int(c.atual_tp.sum())
    out.append("- **P3** (on the coherent ones, the final output retains less than half of the true edges accepted at Stage 6, "
               "in total): %d of %d, %d scenarios -> **%s**" % (
                   fintp, s6tp, len(c), verdict(s6tp > 0 and fintp < 0.5 * s6tp, len(c) if s6tp > 0 else 0)))
    vt, v0 = cc.vl_tp.sum(), cc.vl_tp_so_lag0.sum()
    out.append("- **P4** (at most 15%% of the VARLiNGAM hits come only with lag 0): %d of %d -> **%s**" % (
        int(v0), int(vt), verdict(vt > 0 and v0 <= 0.15 * vt, int(vt))))
    group = np.where(cc.regime == "coherent", "coherent",
                     np.where(~cc.atua, "other", np.where(cc.passeio, "incoherent, random walk", "incoherent, others")))
    cc2 = cc.assign(grupo=group)
    out.append("")
    out.append("Descriptive by group (mean F1):")
    out.append("")
    out.append("| Group | Scenarios | Current | C | VARLiNGAM |")
    out.append("|---|---|---|---|---|")
    for gname, G in cc2.groupby("grupo"):
        out.append("| %s | %d | %.3f | %.3f | %.3f |" % (gname, len(G), G.atual_f1.mean(), G.C_f1.mean(), G.f1_varlingam.mean()))
    out.append("| all | %d | %.3f | %.3f | %.3f |" % (ncc, cc.atual_f1.mean(), cc.C_f1.mean(), cc.f1_varlingam.mean()))
    out.append("")
    out.append("## Part B — rule C and C with gate (T − L ≥ n·L) on the new scenarios, exploratory")
    out.append("")
    out.append("| Set | Scenarios | Rule applies to | Current F1 | C F1 (+/−; p) | Cg F1 (+/−; p) | Current MCC | C MCC | Cg MCC |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    out.append(table_row("all", D))
    if len(cc):
        out.append(table_row("Causal Chambers", cc))
    if len(D[~D.cc]):
        out.append(table_row("others", D[~D.cc]))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(output, "conferencia_cega.md"), "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write(txt)
    sys.stdout.reconfigure(encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    SMOKE = "--fumaca-incluir-vistos"
    args = [x for x in sys.argv[1:] if x != SMOKE]
    main(args[0], args[1:], include_seen=SMOKE in sys.argv)

"""
R1-m6b -- POST HOC diagnosis of PCMCI + CMIknn on the 36 scenarios (2026-11-19). Only reads files; no method is run.

Decided AFTER seeing the result (mean F1 0.007). It changes no number of the official report; it serves to
explain the result and to check the floor diagnosis made on the runner.

(A) Floor and BH, from the MCI logs (verbosity 2): each block lists the links with p < 0.05 WITHOUT correction,
    including the self-edges, which original.csv does not contain. Counts the links at the floor (1/501) and redoes
    tigramite's BH at alpha 0.05 over m = N^2 * tau_max (tau_max = 10; contemporaneous links excluded). Since
    k*alpha/m < alpha, only links with p < 0.05 can pass, and the redone BH is exact (save for the log's rounding to 5
    decimal places). Checks against the significant links of original.csv (which excludes the self-edges).
(B) Signal on the true pairs: smallest raw p (without correction) among the 10 lags, per directed pair of
    ground_truth.csv, split into true and false pairs. Descriptive: the minimum over 10 lags inflates the p < 0.05 of
    the false ones (under H0, ~40% expected).

Usage: python -I revision/offline/r1_m6b_posthoc_floor.py <r1_m6b_cmiknn folder with logs_36 and run_36_padrao> <output>
"""
import glob
import math
import os
import re
import sys

import pandas as pd

FLOOR, ALPHA, TAU = 1.0 / 501, 0.05, 10
AT_FLOOR = 0.00200 + 1e-9  # the log prints 1/501 as 0.00200; the next value (2/501) is 0.00399
MODEL = "pcmci_cmiknn_tau10_dividido"
RE_VAR = re.compile(r"^\s*Variable (\S+) has (\d+) link")
RE_LNK = re.compile(r"^\s*\((\S+) -(\d+)\): pval = ([0-9.]+) \| val =\s*(-?[0-9.]+)")
STR_COLS = {c: str for c in ("source_entity", "source_property", "target_entity", "target_property")}


def group(c):
    if "tigramite_large" in c:
        return "tigramite N12"
    if c.startswith("benchmark_tigramite"):
        return "tigramite N6"
    if "logistic" in c:
        return "logisticos"
    if "lorenz" in c:
        return "lorenz"
    return "reais"


def floor_bh(logs, run, scen):
    pv = {}
    fs = glob.glob(os.path.join(logs, f"{scen}_mci_bloco*.log"))
    for f in fs:
        target, sig = None, False
        for ln in open(f, encoding="utf-8", errors="replace"):
            if ln.startswith("## Significant links"):
                sig = True
                continue
            if not sig:
                continue
            m = RE_VAR.match(ln)
            if m:
                target = m.group(1)
                continue
            m = RE_LNK.match(ln)
            if m and target:
                pv[(m.group(1), int(m.group(2)), target)] = float(m.group(3))
    n = len(glob.glob(os.path.join(logs, f"{scen}_pc1_alvo*.log")))
    m_fam = n * n * TAU
    ps = sorted(FLOOR if p <= AT_FLOOR else p for p in pv.values())
    k_bh = max([k for k, p in enumerate(ps, 1) if p <= k * ALPHA / m_fam], default=0)
    o = pd.read_csv(os.path.join(run, scen, MODEL, "original.csv"))
    return dict(cenario=scen, grupo=group(scen), N=n, m=m_fam, blocos_mci=len(fs), p_menor_005=len(ps),
                no_piso=sum(p <= FLOOR for p in ps),
                no_piso_auto=sum(1 for (s, _, a), p in pv.items() if p <= AT_FLOOR and s == a),
                k_estrela=math.ceil(FLOOR * m_fam / ALPHA), bh_refeito=k_bh,
                sig_original_csv=int(o.significant.astype(bool).sum()))


def signal(run, scen):
    gt = pd.read_csv(os.path.join(run, scen, "ground_truth.csv"), dtype=str)
    gt["v"] = gt.causal_gt.str.strip().str.lower() == "true"
    o = pd.read_csv(os.path.join(run, scen, MODEL, "original.csv"), dtype=STR_COLS)
    pmin = o.groupby([o.source_entity + "|" + o.source_property, o.target_entity + "|" + o.target_property]).pvalue.min()
    gt["pmin"] = [pmin.get((a, b), float("nan")) for a, b in
                  zip(gt.source_entity + "|" + gt.source_property, gt.target_entity + "|" + gt.target_property)]
    v, f = gt[gt.v], gt[~gt.v]
    return dict(cenario=scen, grupo=group(scen), verd=len(v), verd_piso=int((v.pmin <= AT_FLOOR).sum()),
                verd_p005=int((v.pmin < 0.05).sum()), falsos=len(f), falsos_piso=int((f.pmin <= AT_FLOOR).sum()),
                falsos_p005=int((f.pmin < 0.05).sum()), sem_p=int(gt.pmin.isna().sum()))


def main(root, output):
    logs, run = os.path.join(root, "logs_36"), os.path.join(root, "run_36_padrao")
    scens = sorted(os.listdir(run))
    if len(scens) != 36:
        raise SystemExit(f"expected 36 scenarios, found {len(scens)}")
    os.makedirs(output, exist_ok=True)
    a = pd.DataFrame([floor_bh(logs, run, c) for c in scens])
    b = pd.DataFrame([signal(run, c) for c in scens])
    a.to_csv(os.path.join(output, "piso_bh_por_cenario.csv"), index=False)
    b.to_csv(os.path.join(output, "sinal_verdadeiras_por_cenario.csv"), index=False)
    gb = b.groupby("grupo")[["verd", "verd_piso", "verd_p005", "falsos", "falsos_piso", "falsos_p005"]].sum()
    gb.loc["TOTAL"] = gb.sum()
    gb.to_csv(os.path.join(output, "sinal_verdadeiras_por_grupo.csv"))
    pd.set_option("display.width", 250)
    print(a.to_string(index=False))
    print("Redone BH == significant links of original.csv + self-edges? (cases with BH > 0):",
          a.loc[a.bh_refeito > 0, ["cenario", "bh_refeito", "sig_original_csv"]].to_dict("records"))
    print("scenarios with no_piso >= k*:", int((a.no_piso >= a.k_estrela).sum()))
    print(gb.to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

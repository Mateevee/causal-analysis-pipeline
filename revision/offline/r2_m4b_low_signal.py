"""
R2-m4b -- Why RCAEval and EasyVista have near-zero precision for all
methods.

Request
  R2: "discuss why certain domain families, such as RCAEval and EasyVista,
       yield near-zero precision across all evaluated causal discovery methods."

Design (descriptive; no cause is demonstrated with an approximate ground truth)
  Per scenario of RCAEval and EasyVista, and per family as a contrast:
    prevalence and trivial floor; precision and F1 of each method;
    AUROC and AUPRC of the pipeline scores over the universe of pairs --
      penalized_effect (the score of Table S7), maximum Transfer Entropy and |t|;
    recorded evidence regime and rho(TE, |t|);
    structure of the ground truth: concentration of the out-degree (fraction of the
      edges leaving the node with the largest out-degree), which in RCAEval reflects the
      "fan out from the faulty service" approximation.
  The fractions of white and unit-root series come from r2_m1_portmanteau.py
  (read, if its report already exists).

Usage:  python revision/offline/r2_m4b_low_signal.py
Output: Results_Revisao/r2_m4b_baixo_sinal/
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c  # noqa: E402

NAME = "r2_m4b_baixo_sinal"
TARGET = ("RCAEval", "EasyVista")


def precision(scen, m):
    r = c.rows()[scen].get(m)
    if not r:
        return None
    return r["tp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else 0.0


def main():
    rel = c.Report(NAME, "R2-m4b — RCAEval and EasyVista: low-signal diagnostic")
    port = os.path.join(c.OUTPUT_ROOT, "r2_m1_portmanteau", "por_cenario.csv")
    P = pd.read_csv(port).set_index("cenario") if os.path.exists(port) else None
    rows = []
    for s in c.scenarios():
        T, U = c.truth(s)
        d = c.original(s)
        g = d.groupby(["source_entity", "target_entity"]) if not d.empty else None
        U2 = sorted(U)
        y = [int(p in T) for p in U2]
        row = {"cenario": s, "familia": c.family(s), "alvo": c.family(s) in TARGET,
               "prevalencia": len(T) / len(U), "piso": c.trivial_floor(s)}
        for name, col in (("penalized_effect", "penalized_effect"), ("TE", "transfer_entropy"), ("abs_t", "t_stat")):
            if g is not None and col in d.columns:
                v = pd.to_numeric(d[col], errors="coerce").abs()
                sc = v.groupby(c.pairs(d)).max()
                xs = [float(sc.get(p, 0.0)) if pd.notna(sc.get(p, 0.0)) else 0.0 for p in U2]
                row[f"auroc_{name}"], row[f"auprc_{name}"] = c.auroc(y, xs), c.auprc(y, xs)
        for m in [c.PIPE] + c.BASELINES:
            row[f"precisao_{c.LABEL[m]}"] = precision(s, m)
            row[f"f1_{c.LABEL[m]}"] = c.f1_model(s, m)
        if not d.empty:
            row["regime"] = d["evidence_regime"].dropna().astype(str).mode().iloc[0] if "evidence_regime" in d and d["evidence_regime"].notna().any() else None
            row["rho_te_t"] = pd.to_numeric(d.get("evidence_coherence_rho"), errors="coerce").dropna().median() \
                if "evidence_coherence_rho" in d else None
        degree = Counter(a for a, _ in T)
        row["concentracao_saida"] = max(degree.values()) / len(T) if T else np.nan
        if P is not None and s in P.index:
            row["brancas_frac"] = P.loc[s, "brancas_frac"]
            row["raiz_unitaria_frac"] = P.loc[s, "raiz_unitaria_frac"]
        rows.append(row)
    D = pd.DataFrame(rows)
    rel.csv("por_cenario.csv", D)
    cols = ["prevalencia", "piso", "auroc_penalized_effect", "auprc_penalized_effect", "auroc_TE", "auroc_abs_t",
            "concentracao_saida"] + [k for k in ("brancas_frac", "raiz_unitaria_frac") if k in D.columns]
    G = D.groupby("familia")[cols + [f"precisao_{c.LABEL[m]}" for m in [c.PIPE] + c.BASELINES]].mean()
    G = G.loc[[f for f in c.FAMILIES_DEV + c.FAMILIES_HO if f in G.index]]
    rel.table(["Family"] + cols, [[f] + [c.fmt(G.loc[f, k]) for k in cols] for f in G.index])
    rel.h("## Mean precision per method")
    pm = [f"precisao_{c.LABEL[m]}" for m in [c.PIPE] + c.BASELINES]
    rel.table(["Family"] + [k.split("_", 1)[1] for k in pm], [[f] + [c.fmt(G.loc[f, k]) for k in pm] for f in G.index])
    rel.p("")
    rel.p("Reading: an AUROC close to 0.5 for the pipeline scores indicates that the ranking of the pairs does not separate "
          "positives from negatives of the ground truth; with an approximate ground truth this is compatible both with "
          "little signal and with a ground truth that does not describe the dependencies present in the data.")
    rel.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        main()

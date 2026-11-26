"""
R1-3c / R2-4c -- Evidence-coherence guard at scale: decision of the current guard and of
a minimal variant, the decision only (no F1). Input for the authors' decision.

Finding that motivates it (NOTAS.md, R2-4b): at n = 160 the guard declared "incoherent"
a clean linear VAR in 2 of 3 seeds (F1 0.37), because rho(TE, |t|) falls with the
number of candidates and the upper bound of the interval crosses 0.3.

Reading of the code (Pipeline/coherence.assess_evidence_coherence): "incoherent"
= upper bound of rho < min_rho (0.3); "coherent" = lower bound > 0. The
two conditions can hold together (narrow interval between 0 and 0.3: established but weak
agreement); the code tests "incoherent" first.

Design (fixed before running, 2026-11-09)
  G0 current: the pipeline's function, not a copy.
  G1 minimal variant: the same, but when both conditions hold (lower bound
     > 0 and upper bound < 0.3) the regime is "coherent" -- the guard only acts when the
     agreement between the channels is not established.
  Input: the stored candidate table (original.csv of patch_tau10: columns
    transfer_entropy and t_stat), which is the table the guard sees at the end of Stage 6.
  Mandatory check: recomputed G0 = stored evidence_regime, scenario by
    scenario (if it does not reproduce, the result is invalid).
  Scenarios: the 90 of the paper and the 15 runs of R2-4b (n = 10 to 160, 3 seeds).
  Report: regime of G0 and G1, rho and its interval, number of candidates; list of the scenarios
    on which G1 changes the decision. No F1 is computed here: the consequence for F1
    only enters after the authors choose the design (its own protocol, revision/PROTOCOL_coherence_guard.md).
  G1 is the rule evidence_coherence_overlap='coherent' of the corrected variant (Sections III-G and V-A of the paper).

Usage:  python revision/offline/r1_3c_guard_rho_decision.py
Output: Results_Revisao/r1_3c_guarda_rho_decisao/ (columns of por_cenario.csv kept in Portuguese; see
revision/README.md, Notes)
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REVISION = os.path.dirname(HERE)
ROOT = os.path.dirname(REVISION)
for p in (ROOT, REVISION):
    if p not in sys.path:
        sys.path.insert(0, p)
import common as c  # noqa: E402
from Pipeline.coherence import assess_evidence_coherence  # noqa: E402

NAME = "r1_3c_guarda_rho_decisao"


def regimes(tab: pd.DataFrame):
    with contextlib.redirect_stdout(io.StringIO()):
        g0 = assess_evidence_coherence(tab, verbose=False)
    g1 = dict(g0)
    if g0["regime"] == "incoherent" and np.isfinite(g0["rho_lower"]) and g0["rho_lower"] > 0:
        g1["regime"] = "coherent"
    return g0, g1


def row(name, group, tab):
    g0, g1 = regimes(tab)
    stored = tab["evidence_regime"].dropna().astype(str).iloc[0] if "evidence_regime" in tab and \
        tab["evidence_regime"].notna().any() else None
    return {"cenario": name, "grupo": group, "candidatos": g0["n"], "rho": g0["rho"], "rho_inf": g0["rho_lower"],
            "rho_sup": g0["rho_upper"], "G0": g0["regime"], "G1": g1["regime"], "gravado": stored,
            "reproduz": (stored is None) or (stored == g0["regime"])}


def main():
    rel = c.Report(NAME, "R1-3c — Evidence-coherence guard: current decision (G0) and minimal variant (G1)")
    rows = []
    for s in c.scenarios():
        tab = c.original(s)
        if not tab.empty:
            rows.append(row(s, "paper, held-out" if c.is_heldout(s) else "paper, development", tab))
    scale = os.path.join(c.OUTPUT_ROOT, "r2_4b_escala", "run")
    if os.path.isdir(scale):
        for s in sorted(os.listdir(scale)):
            p = os.path.join(scale, s, "patch_tau10", "original.csv")
            if os.path.exists(p):
                rows.append(row(s, "scale R2-4b", pd.read_csv(p)))
    D = pd.DataFrame(rows)
    rel.csv("por_cenario.csv", D)
    nrep = int((~D["reproduz"]).sum())
    rel.p(f"Check: recomputed G0 = stored regime in {len(D) - nrep} of {len(D)} "
          f"({'ok' if nrep == 0 else 'DOES NOT REPRODUCE: ' + ', '.join(D.loc[~D['reproduz'], 'cenario'])}).")
    if nrep:
        rel.save()
        sys.exit(1)
    rel.h("## Regimes per group")
    body = []
    for g, d in D.groupby("grupo"):
        for r in ("incoherent", "coherent", "undetermined"):
            body.append([g, r, int((d["G0"] == r).sum()), int((d["G1"] == r).sum())])
    rel.table(["Group", "Regime", "G0 current", "G1 variant"], body)
    changed = D[D["G0"] != D["G1"]]
    rel.h(f"## Scenarios on which G1 changes the decision ({len(changed)})")
    rel.table(["Scenario", "Group", "Candidates", "rho", "Interval", "G0", "G1"],
              [[r.cenario, r.grupo, r.candidatos, c.fmt(r.rho), f"[{r.rho_inf:+.2f}, {r.rho_sup:+.2f}]", r.G0, r.G1]
               for r in changed.itertuples()])
    inc = D[D["G0"] == "incoherent"]
    rel.h(f"## All scenarios incoherent under the current guard ({len(inc)})")
    rel.table(["Scenario", "Group", "Candidates", "rho", "Interval", "G1"],
              [[r.cenario, r.grupo, r.candidatos, c.fmt(r.rho), f"[{r.rho_inf:+.2f}, {r.rho_sup:+.2f}]", r.G1]
               for r in inc.itertuples()])
    rel.save()


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        main()

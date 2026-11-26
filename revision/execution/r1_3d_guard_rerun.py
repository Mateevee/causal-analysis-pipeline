"""
R1-3d / R2-4d -- Evidence-coherence guard G1: test of the implementation (C1) and
rerun (C2/C3) of revision/PROTOCOL_coherence_guard.md (frozen; SHA-256 in the authors' lab notes).

Design (fixed before running, 2026-11-09)
  testar : C1 -- with the default, the regime recomputed by assess_evidence_coherence
           equals the stored one on the 105 scenarios (90 of the paper + 15 of R2-4b); with
           overlap='coherent', it equals the G1 of revision/offline/r1_3c_guard_rho_decision.py. Stops if
           there is any disagreement.
  rodar  : C2 -- patch_tau10 and patch_tau10_final (time_axis_rule='block_portmanteau',
           evidence_coherence_overlap='coherent'), same code, same run, on the
           3 scenarios of the paper on which G1 changes the decision and on seeds 6101 and 6103
           of R2-4b with n = 160 (generator and parameters of revision/execution/r2_4b_scale.py).
  avaliar: C3 -- F1, TP, FP, FN and regime per scenario and version; mean over the 90 with the
           3 swapped (approximation: the paper's F1 on the unchanged ones), whatever it is.

Usage
  python revision/execution/r1_3d_guard_rerun.py testar
  python revision/execution/r1_3d_guard_rerun.py rodar     (~1 h; n = 160 ~15 min per run)
  python revision/execution/r1_3d_guard_rerun.py avaliar
Output: Results_Revisao/r1_3d_guarda_reexecucao/ (file and column names kept in Portuguese; see revision/README.md,
Notes)
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common_exec as ce  # noqa: E402

NAME = "r1_3d_guarda_reexecucao"
FINAL = "patch_tau10_final"
PAPER_3 = ["benchmark_causaltime_medical_sample0", "benchmark_rcaeval_RE1-OB_checkoutservice_disk_1",
           "real_petshop_temporal_traffic1"]
SCALE = ["revisao_escala_n160_seed6101", "revisao_escala_n160_seed6103"]


def models():
    return [ce.model("patch_tau10"),
            ce.model("patch_tau10", FINAL, time_axis_rule="block_portmanteau", evidence_coherence_overlap="coherent")]


def test():
    import common as c
    from Pipeline.coherence import assess_evidence_coherence
    ref = pd.read_csv(os.path.join(ce.OUTPUT_ROOT, "r1_3c_guarda_rho_decisao", "por_cenario.csv")).set_index("cenario")
    scale = os.path.join(ce.OUTPUT_ROOT, "r2_4b_escala", "run")
    disc = []
    for s in ref.index:
        p = os.path.join(scale, s, "patch_tau10", "original.csv")
        tab = pd.read_csv(p) if s.startswith("revisao_escala") else c.original(s)
        with contextlib.redirect_stdout(io.StringIO()):
            g0 = assess_evidence_coherence(tab, verbose=False)["regime"]
            g1 = assess_evidence_coherence(tab, verbose=False, overlap="coherent")["regime"]
        if g0 != ref.loc[s, "G0"] or (isinstance(ref.loc[s, "gravado"], str) and g0 != ref.loc[s, "gravado"]):
            disc.append(f"{s} (default)")
        if g1 != ref.loc[s, "G1"]:
            disc.append(f"{s} (G1)")
    out = ce.output_dir(NAME)
    txt = [f"# C1 — guard G1: {len(ref) * 2 - len(disc)}/{len(ref) * 2} decisions match",
           "", "Disagreements: " + (", ".join(disc) or "none") + "."]
    open(os.path.join(out, "teste_C1.md"), "w", encoding="utf-8").write("\n".join(txt) + "\n")
    print("\n".join(txt))
    if disc:
        sys.exit(1)


def run():
    folder = ce.output_dir(os.path.join(NAME, "run"))
    dfs = [d for d in ce.paper_scenarios() if d["name"] in PAPER_3]
    missing = set(PAPER_3) - {d["name"] for d in dfs}
    if missing:
        raise RuntimeError(f"scenarios not found: {sorted(missing)}")
    sc = ce.generate_var([6101, 6102, 6103], 300, 160, min(0.3, 2 / 159), 3, 3, 2, name_prefix="revisao_escala_n160")
    dfs += [d for d in sc if d["name"] in SCALE]
    ce.run_resuming(dfs, models(), folder)
    evaluate(folder)


def evaluate(folder=None):
    import common as c
    folder = folder or os.path.join(ce.OUTPUT_ROOT, NAME, "run")
    R = ce.summary(folder)
    rows = []
    for s in sorted(R["cenario"].unique()):
        row = {"cenario": s, "grupo": "artigo" if s in PAPER_3 else "escala n = 160"}
        for m in R[R["cenario"] == s].itertuples():
            tag = "final" if m.model == FINAL else "antigo"
            row[f"f1_{tag}"] = ce.f1(m.TP, m.FP, m.FN) if m.status == "ok" else np.nan
            row[f"tp_{tag}"], row[f"fp_{tag}"], row[f"fn_{tag}"] = m.TP, m.FP, m.FN
            o = os.path.join(folder, s, m.model, "original.csv")
            if os.path.exists(o):
                d = pd.read_csv(o, usecols=lambda k: k == "evidence_regime")
                row[f"regime_{tag}"] = d["evidence_regime"].dropna().iloc[0] if len(d.dropna()) else None
        if s in PAPER_3:
            row["f1_gravado_artigo"] = c.f1_model(c.scenario_key(s), c.PIPE)
            row["heldout"] = c.is_heldout(s)
        rows.append(row)
    D = pd.DataFrame(rows)
    out = os.path.dirname(folder)
    D.to_csv(os.path.join(out, "por_cenario.csv"), index=False)
    cols = ["cenario", "grupo", "f1_gravado_artigo", "regime_antigo", "f1_antigo", "regime_final", "f1_final",
            "tp_antigo", "fp_antigo", "fn_antigo", "tp_final", "fp_final", "fn_final"]
    txt = ["# R1-3d — Evidence-coherence guard G1: rerun", "",
           "antigo (old) = patch_tau10; final = patch_tau10_final (new time-axis rule + G1). Same code, same run.", "",
           D[[k for k in cols if k in D]].round(3).to_markdown(index=False), ""]
    base = {s: c.f1_model(s, c.PIPE) for s in c.scenarios()}
    swapped = dict(base)
    for r in D[D["grupo"] == "artigo"].itertuples():
        swapped[c.scenario_key(r.cenario)] = r.f1_final
    txt += ["## Mean over the 90 (approximation: the paper's F1 on the 87 unchanged)", "",
            "| Part | F1 of the paper | With the 3 swapped |", "|---|---|---|"]
    for part in ("todos", "dev", "ho"):
        sc = c.scenarios(part)
        txt.append(f"| {part} | {np.mean([base[s] for s in sc]):.3f} | {np.mean([swapped[s] for s in sc]):.3f} |")
    open(os.path.join(out, "relatorio.md"), "w", encoding="utf-8").write("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        action = sys.argv[1] if len(sys.argv) > 1 else "avaliar"
        {"testar": test, "rodar": run, "avaliar": evaluate}[action]()

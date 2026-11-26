"""
R1-3b -- checks of the separate implementation (NOTAS.md, "R1-3b -- DEVIATION FROM THE PROTOCOL", 2026-11-14).

(a) method_code_hash equal to d0473069ebef1978 (Pipeline/ was not touched).
(b) identity: benchmark_synthetic_var_1_seed11 with patch_tau10_final, without and with the swap of
    _STAGES installed, in the same process; original.csv and results.csv identical byte by byte.
(b2) identity of the logging module (diagnostics for the economy rule): the same scenario with
    patch_tau10_final through the logging module (final_model(register=True)), with the swap
    installed; identical byte by byte to those of (b) without the swap. Also checks that the diagnostics were
    written.
    Run with a fixed PYTHONHASHSEED: PYTHONHASHSEED=0 python revision/variants/check_r1_3b.py
(c) smoke: the graded variant runs end to end on the first scenario, in alphabetical order,
    with w_inc within (0.001; 0.999) by the tables of the paper (benchmark_beeline_HSC_HSC-2000-10),
    and writes the weight columns. Only checks that it runs and what it wrote; it is NOT an evaluation (F1 is not
    read here).
Output: Results_Revisao/r1_3b_transicoes_graduais/checagem_implementacao/ (folder names and the keys of checagem.json
kept in Portuguese, as in the stored results; see revision/README.md, Notes). In this release, (a) reports the hash of
the released method code, 5c10222886dcc287 (revision/README.md, Notes).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, REV)
sys.path.insert(0, os.path.join(REV, "execution"))
sys.path.insert(0, os.path.dirname(REV))

import pandas as pd  # noqa: E402

FROZEN_HASH = "d0473069ebef1978"
SCEN_IDENT = "benchmark_synthetic_var_1_seed11"
SCEN_SMOKE = "benchmark_beeline_HSC_HSC-2000-10"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def main():
    import common_exec as ce
    import run_log
    from variants import r1_3b_graded as g
    base = ce.output_dir(os.path.join("r1_3b_transicoes_graduais", "checagem_implementacao"))
    out = {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}

    h = run_log.method_code_hash() if hasattr(run_log, "method_code_hash") else None
    out["hash_codigo_metodo"] = h
    out["a_hash_igual"] = (h == FROZEN_HASH) if h else None

    dfs = {d["name"]: d for d in ce.paper_scenarios([SCEN_IDENT, SCEN_SMOKE])}
    final = g.final_model(ce)

    p_without = os.path.join(base, "sem_troca")
    p_with = os.path.join(base, "com_troca")
    ce.main_mod.run_all(dataframes=[dfs[SCEN_IDENT]], models=[final], config=ce.cfg_exec.PIPELINE_CONFIG,
                        output_dir=p_without)
    orig = g.install()
    try:
        ce.main_mod.run_all(dataframes=[dfs[SCEN_IDENT]], models=[final], config=ce.cfg_exec.PIPELINE_CONFIG,
                            output_dir=p_with)
        p_log = os.path.join(base, "com_registro")
        ce.main_mod.run_all(dataframes=[dfs[SCEN_IDENT]], models=[g.final_model(ce, register=True)],
                            config=ce.cfg_exec.PIPELINE_CONFIG, output_dir=p_log)
        diag_log = {"coerencia": dict(g.LAST_COHERENCE),
                    "dml": {k: v for k, v in g.logging_dml_module().ultimo.items() if k != "r2_pares"},
                    "n_r2_pares": len(g.logging_dml_module().ultimo.get("r2_pares", []))}
        p_smoke = os.path.join(base, "fumaca")
        ce.main_mod.run_all(dataframes=[dfs[SCEN_SMOKE]], models=[g.graded_model(ce)],
                            config=ce.cfg_exec.PIPELINE_CONFIG, output_dir=p_smoke)
    finally:
        g.uninstall(orig)
    for f in ("original.csv", "results.csv"):
        a = sha(os.path.join(p_without, SCEN_IDENT, final["name"], f))
        b = sha(os.path.join(p_with, SCEN_IDENT, final["name"], f))
        out[f"b_{f}_sem"], out[f"b_{f}_com"], out[f"b_{f}_igual"] = a, b, (a is not None and a == b)
        r = sha(os.path.join(p_log, SCEN_IDENT, final["name"], f))
        out[f"b2_{f}_registro"], out[f"b2_{f}_igual"] = r, (a is not None and a == r)
    out["b2_diagnosticos"] = diag_log

    pf = os.path.join(p_smoke, SCEN_SMOKE, g.MODEL_NAME, "original.csv")
    out["c_rodou"] = os.path.exists(pf)
    if out["c_rodou"]:
        d = pd.read_csv(pf)
        for col in ("regime_weight_stage6", "regime_weight_stage6_raw", "regime_weight_lean", "regime_lean_path",
                    "evidence_regime", "evidence_coherence_rho"):
            out[f"c_{col}"] = (d[col].dropna().unique().tolist()[:3] if col in d else "AUSENTE")
        if {"significant", "significant_before_guard", "significant_stage6"}.issubset(d.columns):
            out["c_n_significant"] = int((d["significant"].astype(str) == "True").sum())
            out["c_n_antes_guarda"] = int((d["significant_before_guard"].astype(str) == "True").sum())
            out["c_n_stage6"] = int((d["significant_stage6"].astype(str) == "True").sum())
    json.dump(out, open(os.path.join(base, "checagem.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False,
              default=str)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    from run_log import execution
    with execution("r1_3b_checagem_implementacao", __file__):
        main()

"""
R1-6b -- D-S masses stored per pair, without changing the method's behaviour.

Request (R1, point 6): "The 17 specialists have hand-set mass functions; masses are not calibrated and not stored".
Until now the answer was the R1-6a recomputation, which reproduces the stored decision for 99.5% of the 12,655 candidates.
Decision (Claude, delegated by the authors, 2026-11-21; NOTAS.md): store the masses in the run itself, with the
behaviour checked to be unchanged, so as to answer "they are stored, here" instead of "we recomputed them".

How it stores them, without touching Pipeline/ (method_code_hash remains d0473069ebef1978):
  The Stage 7 selector enters through the configuration (causal_penalty_fn). patch_tau10 uses
  Config.execution.ADAPTIVE_CAUSAL_PENALTY (AdaptiveCausalPenalty). MassRecorder is a subclass with the same
  parameters: it inherits calibrate() unchanged; in __call__ it returns EXACTLY the value of the original __call__
  (super().__call__, the same causal_pen) and, separately, calls Pipeline.SelectorFunc.causal_pen_diagnostics(row, ...)
  with the same calibrated floor, which returns the full state of the pair (the masses of each specialist, the
  combined mass, the conflict, the decision and the 17+ variables of the FeatureSet). causal_pen_diagnostics is a pure
  function of the row; the value returned to the pipeline does not depend on it. The rows go to a module buffer and,
  after each scenario, to <folder>/<scenario>/patch_tau10_massas/massas.jsonl.

Design (fixed before running):
  - Method: patch_tau10 (the submitted one, that of Table III and of R1-6a). Scenarios: the paper's 90.
  - inst: model patch_tau10_massas (= patch_tau10 with the MassRecorder). ctrl: patch_tau10 unchanged.
  - Both run with PYTHONHASHSEED=0 (the queue sets it), in separate processes; with the fixed seed the pipeline
    repeats byte for byte across processes (round 4: D6a B1 = B2 on the 12 scenarios, D6b B1 = B2 at n = 160).
  - The 90 scenarios in two halves balanced by the times measured on the i7 (R1-m7a, 19/11; HALF_1 below, 44
    scenarios, ~43 min; half 2 is the other 46, ~43 min). Each (version, half) writes to its own folder:
    run/inst_1, run/inst_2, run/ctrl_1, run/ctrl_2. Validation (validate_all) runs at the end of each step, only on
    its folder.
  - Progress: one line per finished scenario in run/<step>_progresso.jsonl (the queue counts the lines).

avaliar (read-only; fixed beforehand):
  (a) Identity: per scenario, original.csv and results.csv of inst and ctrl equal (content, in LF) and F1 equal.
  (b) Consistency of the masses: (b1) one mass line per selector call -- number of lines = number of
      rows of original.csv that went through Stage 7 (those with causal_penalty_score); (b2) on every line,
      the value returned to the pipeline equals 1.0 if the diagnostics say accepted and 0.0 otherwise (CONTINUOUS_RETURN =
      False); (b3) matching on (source, property, target, property, lag), the stored returned value equals the
      causal_penalty_score of the original.csv of the same run.
  (c) F1 of ctrl against that of the paper (the paper's runs, run without PYTHONHASHSEED): informative only; small
      differences are expected from the hash seed (round 4) and are reported per scenario.
  (d) Descriptive summary of the masses: number of pairs, of specialists per pair, calibrated floor per scenario.
Reading (fixed beforehand):
  - (a) 90 of 90 and (b) with no failure at all: the masses are stored for every pair evaluated in Stage 7 of the 90
    scenarios, with the method's behaviour checked to be unchanged. [text] R1-6b: "masses are now stored per
    pair (file, columns)", with the path of the files in the reproducibility index.
  - Any failure in (a) or (b): the file does not count as a record of the method; the answer goes back to the
    R1-6a recomputation (99.5%), and the failure is reported with the scenarios. Nothing is re-run because of a number.

Usage (from the repository root, with PYTHONHASHSEED=0 in the execution steps)
  python revision/execution/r1_6b_masses.py fumaca                  (E0a: 1 scenario, inst and ctrl in the same process,
                                                                   in r1_6b_massas_verificacao/; exit code 1 on failure)
  python revision/execution/r1_6b_masses.py rodar inst|ctrl 1|2
  python revision/execution/r1_6b_masses.py avaliar [--base <r1_6b_massas folder>]
Output: Results_Revisao/r1_6b_massas/
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

NAME = "r1_6b_massas"
MODEL_INST = "patch_tau10_massas"
MODEL_CTRL = "patch_tau10"
FROZEN_HASH = "d0473069ebef1978"
SMOKE_SCEN = "benchmark_synthetic_var_1_seed11"
HALF_1 = [
    "benchmark_beeline_HSC_HSC-2000-1", "benchmark_beeline_HSC_HSC-2000-10", "benchmark_beeline_VSC_VSC-2000-1",
    "benchmark_beeline_VSC_VSC-2000-10", "benchmark_causaltime_medical_sample0", "benchmark_causaltime_pm25_sample0",
    "benchmark_causaltime_pm25_sample2", "benchmark_causaltime_traffic_sample1", "benchmark_dream3_size10_Ecoli1",
    "benchmark_dream3_size10_Ecoli2", "benchmark_dream3_size10_Yeast2", "benchmark_logistic_maps_bifurcation_seed2",
    "benchmark_logistic_maps_bifurcation_seed3", "benchmark_logistic_maps_convergence_seed3",
    "benchmark_logistic_maps_with_isolated_seed1", "benchmark_logistic_maps_with_isolated_seed3",
    "benchmark_lorenz96_K10_seed2", "benchmark_lorenz96_K8_seed2", "benchmark_lorenz96_K8_seed3",
    "benchmark_netsim_sim3_subj0", "benchmark_netsim_sim3_subj1", "benchmark_rcaeval_RE1-OB_adservice_cpu_5",
    "benchmark_rcaeval_RE1-OB_adservice_mem_2", "benchmark_rcaeval_RE1-OB_cartservice_delay_3",
    "benchmark_rcaeval_RE1-OB_checkoutservice_disk_1", "benchmark_rcaeval_RE1-OB_checkoutservice_mem_5",
    "benchmark_rcaeval_RE1-OB_productcatalogservice_cpu_2", "benchmark_synthetic_var_1_seed11",
    "benchmark_synthetic_var_2_seed22", "benchmark_synthetic_var_4_seed44", "benchmark_tep_normal",
    "benchmark_tigramite_baseline_linear_seed11", "benchmark_tigramite_large_mixed_seed11",
    "benchmark_tigramite_large_mixed_seed33", "benchmark_tigramite_quadratic_seed22",
    "benchmark_tigramite_saturation_seed22", "benchmark_tigramite_saturation_seed33",
    "benchmark_tigramite_threshold_seed33", "real_causal_chamber_lt_actuators_white",
    "real_causal_chamber_wt_actuators_random_walk_3", "real_easyvista_antivirus_1", "real_easyvista_mom_1",
    "real_easyvista_web_1", "real_sachs_proteins",
]
PAIR_KEY = ("source_entity", "source_property", "target_entity", "target_property", "lag")

# ----------------------------------------------------------------------------- the recorder
_BUFFER: list = []


def _num(x):
    """Converts to something json can write (NaN/inf become text, so as not to break strict readers)."""
    if x is None or isinstance(x, (bool, str)):
        return x
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(f) or math.isinf(f):
        return str(f)
    return int(f) if isinstance(x, int) else f


def _mass(pair):
    """(name, {m: value}) -> {"especialista": name, "massas": {...}}."""
    name, m = pair
    return {"especialista": name, "massas": {k: _num(v) for k, v in dict(m).items()}}


def _state_json(st: dict) -> dict:
    out = {}
    for k, v in st.items():
        if k == "feature_set":
            out[k] = None if v is None else {a: _num(b) for a, b in dataclasses.asdict(v).items()}
        elif k in ("primary_experts", "active_meta_experts", "active_interaction_experts"):
            out[k] = [_mass(p) for p in v]
        elif k in ("full_meta_experts", "full_interaction_experts"):
            out[k] = {n: _mass(p) for n, p in v.items()}
        elif k == "conflicts_by_expert":
            out[k] = [[n, _num(c)] for n, c in v]
        else:
            out[k] = _num(v)
    return out


def recorder_class():
    from Pipeline.SelectorFunc import AdaptiveCausalPenalty, causal_pen_diagnostics

    class MassRecorder(AdaptiveCausalPenalty):
        """AdaptiveCausalPenalty that, besides returning the same value, stores the full state of the pair."""

        def __call__(self, row, target_fn=None):
            w = super().__call__(row, target_fn=target_fn)
            st = causal_pen_diagnostics(row, domain_gate_fn=self.domain_gate_fn, fusion_rule=self.fusion_rule,
                                        min_support_threshold=self.calibrated_threshold_)
            rec = {c: _num(row.get(c)) if hasattr(row, "get") else None for c in PAIR_KEY}
            rec.update({"indice": _num(getattr(row, "name", None)), "valor_devolvido": _num(w),
                        "piso_calibrado": _num(self.calibrated_threshold_),
                        "n_candidatos_calibracao": _num(self.n_calibration_candidates_),
                        "estado": _state_json(st)})
            _BUFFER.append(rec)
            return w

    return MassRecorder


def recorder(ce):
    """New MassRecorder with exactly the parameters of the ADAPTIVE_CAUSAL_PENALTY in Config/."""
    ref = ce.cfg_exec.ADAPTIVE_CAUSAL_PENALTY
    G = recorder_class()
    return G(percentile=ref.percentile, min_absolute_floor=ref.min_absolute_floor,
             max_absolute_floor=ref.max_absolute_floor, min_n_candidates=ref.min_n_candidates,
             domain_gate_fn=ref.domain_gate_fn, fusion_rule=ref.fusion_rule)


def model_inst(ce):
    return ce.model(MODEL_CTRL, MODEL_INST, causal_penalty_fn=recorder(ce))


# ----------------------------------------------------------------------------- execution
def _check_hash():
    import run_log
    h = run_log.method_code_hash()
    if h != FROZEN_HASH:
        print(f"WARNING: method_code_hash {h} != {FROZEN_HASH}, the hash of the frozen method code (the released files are English translations of the frozen ones; see revision/README.md)", file=sys.stderr)


def _check_seed():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(f"PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED')!r}; this script requires 0 (the queue sets it).")


def _scenarios(ce, half):
    dfs = sorted(ce.paper_scenarios(), key=lambda d: d["name"])
    if len(dfs) != 90:
        raise SystemExit(f"{len(dfs)} paper scenarios loaded (expected 90)")
    names = {d["name"] for d in dfs}
    absent = [s for s in HALF_1 if s not in names]
    if absent:
        raise SystemExit(f"HALF_1 scenarios missing: {absent}")
    if half == "1":
        return [d for d in dfs if d["name"] in HALF_1]
    if half == "2":
        return [d for d in dfs if d["name"] not in HALF_1]
    raise SystemExit(f"half {half!r}: use 1 or 2")


def _done(folder, scen, mod):
    import common_exec as ce
    if not ce.already_run(folder, scen, mod):
        return False
    return mod != MODEL_INST or os.path.exists(os.path.join(folder, scen, mod, "massas.jsonl"))


def _flush(folder, scen):
    dest = os.path.join(folder, scen, MODEL_INST, "massas.jsonl")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in _BUFFER:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, dest)
    n = len(_BUFFER)
    _BUFFER.clear()
    return n


def _run_list(ce, dfs, version, folder, progress):
    import datetime
    import time
    for d in dfs:
        m = model_inst(ce) if version == "inst" else ce.model(MODEL_CTRL)
        if _done(folder, d["name"], m["name"]):
            continue
        _BUFFER.clear()
        t0 = time.perf_counter()
        ce.main_mod.run_all(dataframes=[d], models=[m], config=ce.cfg_exec.PIPELINE_CONFIG, output_dir=folder)
        n = _flush(folder, d["name"]) if version == "inst" else None
        with open(progress, "a", encoding="utf-8") as f:
            f.write(json.dumps({"cenario": d["name"], "versao": version, "modelo": m["name"],
                                "linhas_massas": n, "fim": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                                "duracao_s": round(time.perf_counter() - t0, 1),
                                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}) + "\n")
    ce.main_mod.validate_all(output_dir=folder, show_pairs=False, soft_threshold=ce.cfg_exec.SOFT_THRESHOLD,
                             ignore_intra_entity=not ce.cfg_exec.DETECT_INTRA_ENTITY,
                             use_canonical_pair=ce.cfg_exec.USE_CANONICAL_PAIR)


def run():
    import common_exec as ce
    _check_hash()
    _check_seed()
    version, half = sys.argv[2], sys.argv[3]
    if version not in ("inst", "ctrl"):
        raise SystemExit(f"version {version!r}: use inst or ctrl")
    dfs = _scenarios(ce, half)
    base = ce.output_dir(NAME)
    folder = os.path.join(base, "run", f"{version}_{half}")
    os.makedirs(folder, exist_ok=True)
    print(f"[{NAME}] {version} half {half}: {len(dfs)} scenarios -> {folder}", flush=True)
    _run_list(ce, dfs, version, folder, os.path.join(base, "run", f"{version}_{half}_progresso.jsonl"))


def smoke():
    """E0a: 1 scenario, inst and ctrl in sequence in the same process; identity and masses. Exit code 1 on failure."""
    import common_exec as ce
    _check_hash()
    _check_seed()
    dfs = ce.paper_scenarios([SMOKE_SCEN])
    dfs = [d for d in dfs if d["name"] == SMOKE_SCEN]
    base = ce.output_dir(NAME + "_verificacao")
    folder = os.path.join(base, "run")
    for v in ("inst", "ctrl"):
        _run_list(ce, dfs, v, folder, os.path.join(base, "progresso.jsonl"))
    r = _compare_scenario(folder, folder, SMOKE_SCEN)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    ok = r["original_igual"] and r["results_igual"] and r["massas_linhas"] and not r["falhas_b"]
    print("SMOKE R1-6b: " + ("OK" if ok else "FAILED"))
    if not ok:
        sys.exit(1)


# ----------------------------------------------------------------------------- evaluate
def _lf(p):
    return open(p, "rb").read().replace(b"\r\n", b"\n") if os.path.exists(p) else None


def _sha(b):
    return hashlib.sha256(b).hexdigest() if b is not None else None


def _f1(folder, scen, mod):
    import pandas as pd
    p = os.path.join(folder, scen, "validation_summary.csv")
    if not os.path.exists(p):
        return None
    v = pd.read_csv(p)
    r = v[(v["model"] == mod) & (v["status"] == "ok")]
    if r.empty:
        return None
    tp, fp, fn = int(r.TP.iloc[0]), int(r.FP.iloc[0]), int(r.FN.iloc[0])
    return 0.0 if tp == 0 else round(2 * tp / (2 * tp + fp + fn), 4)


def _compare_scenario(p_inst, p_ctrl, scen):
    import pandas as pd
    oi, oc = (_lf(os.path.join(p, scen, m, "original.csv")) for p, m in ((p_inst, MODEL_INST), (p_ctrl, MODEL_CTRL)))
    ri, rc = (_lf(os.path.join(p, scen, m, "results.csv")) for p, m in ((p_inst, MODEL_INST), (p_ctrl, MODEL_CTRL)))
    out = {"cenario": scen, "original_igual": oi is not None and oi == oc, "results_igual": ri is not None and ri == rc,
           "sha_original_inst": _sha(oi), "sha_original_ctrl": _sha(oc),
           "f1_inst": _f1(p_inst, scen, MODEL_INST), "f1_ctrl": _f1(p_ctrl, scen, MODEL_CTRL)}
    pm = os.path.join(p_inst, scen, MODEL_INST, "massas.jsonl")
    lines = [json.loads(x) for x in open(pm, encoding="utf-8")] if os.path.exists(pm) else []
    out["massas_linhas"] = len(lines)
    failures = []
    orig = pd.read_csv(os.path.join(p_inst, scen, MODEL_INST, "original.csv")) if oi is not None else pd.DataFrame()
    n7 = int(orig["causal_penalty_score"].notna().sum()) if "causal_penalty_score" in orig.columns else 0
    out["linhas_stage7_original"] = n7
    if len(lines) != n7:
        failures.append(f"b1: {len(lines)} mass lines x {n7} rows with causal_penalty_score")
    nb2 = sum(1 for r in lines
              if float(r["valor_devolvido"]) != (1.0 if r["estado"].get("accepted") else 0.0))
    if nb2:
        failures.append(f"b2: {nb2} lines with a returned value different from the diagnostics decision")
    if n7 and lines:
        def k(d):
            return tuple(str(d[c]) if c != "lag" else str(int(float(d[c]))) for c in PAIR_KEY)
        score = {}
        for _, r in orig[orig["causal_penalty_score"].notna()].iterrows():
            score.setdefault(k(r), []).append(round(float(r["causal_penalty_score"]), 6))
        nb3 = 0
        for r in lines:
            if round(float(r["valor_devolvido"]), 6) not in score.get(k(r), []):
                nb3 += 1
        if nb3:
            failures.append(f"b3: {nb3} lines without an equal causal_penalty_score in original.csv")
    out["falhas_b"] = failures
    out["especialistas_por_par"] = sorted({len(r["estado"].get("primary_experts", [])) +
                                           len(r["estado"].get("active_meta_experts", [])) +
                                           len(r["estado"].get("active_interaction_experts", []))
                                           for r in lines})
    out["piso_calibrado"] = sorted({r["piso_calibrado"] for r in lines}, key=str)
    out["aceitos"] = sum(1 for r in lines if r["estado"].get("accepted"))
    return out


def evaluate():
    import pandas as pd
    import common as c
    base = c.OUTPUT_ROOT + os.sep + NAME
    if "--base" in sys.argv:
        base = sys.argv[sys.argv.index("--base") + 1]
    rep = c.Report(NAME, "R1-6b — D-S masses stored per pair (patch_tau10, 90 scenarios)")
    rep.p(f"Source: `{base}`. Inst = patch_tau10 with the MassRecorder; ctrl = patch_tau10. Both with "
          "PYTHONHASHSEED=0, in separate processes.")
    records = []
    for half in ("1", "2"):
        pi, pc = os.path.join(base, "run", f"inst_{half}"), os.path.join(base, "run", f"ctrl_{half}")
        scens = sorted(set(os.listdir(pi) if os.path.isdir(pi) else []) | set(os.listdir(pc) if os.path.isdir(pc) else []))
        for scen in scens:
            if os.path.isdir(os.path.join(pi, scen)) or os.path.isdir(os.path.join(pc, scen)):
                r = _compare_scenario(pi, pc, scen)
                r["metade"] = half
                records.append(r)
    T = pd.DataFrame(records)
    n = len(T)
    rep.p(f"Scenarios with output: {n} (90 expected){'' if n == 90 else ' — **PARTIAL**'}.")
    if n == 0:
        rep.save()
        return
    ok_a = T.original_igual & T.results_igual & (T.f1_inst == T.f1_ctrl)
    rep.h("## (a) Identity inst × ctrl")
    rep.table(["Measure", "Scenarios"], [
        ["original.csv equal (LF)", int(T.original_igual.sum())], ["results.csv equal (LF)", int(T.results_igual.sum())],
        ["F1 equal", int((T.f1_inst == T.f1_ctrl).sum())], ["All three", int(ok_a.sum())]])
    if (~ok_a).any():
        rep.table(["Scenario", "original", "results", "F1 inst", "F1 ctrl"],
                  [[r.cenario, r.original_igual, r.results_igual, r.f1_inst, r.f1_ctrl] for r in T[~ok_a].itertuples()])
    rep.h("## (b) Consistency of the masses")
    fb = T[T.falhas_b.map(len) > 0]
    rep.p(f"Mass lines: {int(T.massas_linhas.sum())} (rows with causal_penalty_score in the original.csv files: "
          f"{int(T.linhas_stage7_original.sum())}); pairs accepted by the selector: {int(T.aceitos.sum())}. "
          f"Scenarios with some failure in b1, b2 or b3: {len(fb)}.")
    if len(fb):
        rep.table(["Scenario", "Failures"], [[r.cenario, "; ".join(r.falhas_b)] for r in fb.itertuples()])
    rep.h("## (c) F1 of ctrl against the paper (informative only; the paper ran without PYTHONHASHSEED)")
    rows = c.rows()
    diff = []
    for r in T.itertuples():
        k = c.scenario_key(r.cenario)
        if k and MODEL_CTRL in rows.get(k, {}):
            a = rows[k][MODEL_CTRL]
            fa = 0.0 if a["tp"] == 0 else round(2 * a["tp"] / (2 * a["tp"] + a["fp"] + a["fn"]), 4)
            if r.f1_ctrl is not None and abs(fa - r.f1_ctrl) > 1e-9:
                diff.append([r.cenario, fa, r.f1_ctrl, round(r.f1_ctrl - fa, 4)])
    rep.p(f"Scenarios with F1 different from the paper: {len(diff)}.")
    if diff:
        rep.table(["Scenario", "F1 paper", "F1 ctrl", "Difference"], diff)
    rep.h("## (d) Descriptive")
    spec = sorted({e for l in T.especialistas_por_par for e in l})
    rep.p(f"Specialists in the fusion per pair (distinct values): {spec}.")
    verdict = bool(ok_a.all() and n == 90 and len(fb) == 0)
    rep.h("## Reading (fixed beforehand)")
    rep.p("**(a) 90 of 90 and (b) without failure: masses stored, behaviour unchanged.**" if verdict else
          "**Failure in (a) or (b), or partial run: the file does not count as a record of the method; the answer goes "
          "back to the R1-6a recomputation (99.5%).**")
    rep.data.update({"cenarios": n, "identidade_ok": int(ok_a.sum()), "falhas_b": len(fb),
                     "dif_artigo": len(diff), "veredito_massas_gravadas": verdict})
    T.drop(columns=["falhas_b"]).assign(falhas_b=T.falhas_b.map("; ".join)).to_csv(
        os.path.join(rep.dir, "por_cenario.csv"), index=False)
    rep.save()


if __name__ == "__main__":
    from run_log import execution
    mode = sys.argv[1] if len(sys.argv) > 1 else "avaliar"
    with execution(NAME + ("_fumaca" if mode == "fumaca" else ""), __file__):
        if mode == "rodar":
            run()
        elif mode == "fumaca":
            smoke()
        else:
            evaluate()

"""
R1-2c -- the two skipped scenarios (DREAM3 size 100, Yeast2 and Yeast3), run to complete the registered design
(88 of 88). Decision of the author on 2026-11-21: "note that they will be run, and if all goes well we will include
them; put running them tonight as a priority in the queue, without limiters, only the 60-min one".

Design (fixed before running; the R1-2c protocol still applies; if they differ, the protocol prevails):
  - The same 9 models and the same order as `r1_2c_new_benchmark.py rodar`, with the same frozen code
    (method_code_hash d0473069ebef1978):
      1. per scenario: patch_tau10 and the 5 baselines;
      2. per scenario: patch_tau10_final and patch_tau10_graded (the R1-3b guard swap installed; diag.jsonl);
      3. per scenario: patch_semS7_tau10 (the exploratory one, without Stage 7).
  - The SAME limit as R1-2c: 60 min per run (scenario x model), equal for all methods. Past the limit, the
    process is terminated (whole tree) and time.txt receives status ERROR and "DEMORA EXCESSIVA (timeout)", as
    in the other DREAM3 runs. Each run is in its own process; the limit counts from the start of the process, with 30 s
    of slack to import the libraries and load the scenario. Validation (main.validate_all) runs once at the end,
    outside the limit.
  - No other limit. The stage runs ALONE on the machine (the queue opens no other front while it runs), so that the
    60-min limit has the same meaning it had in the other 86 scenarios, which ran in sequence.
  - PYTHONHASHSEED = 0 (the queue sets it), as in the reproduction package (decision D15).
  - Output: the SAME run/ folder as R1-2c (Results_Revisao/r1_2c_benchmark_novo/run), so that `avaliar` joins the 88.
    Before that, the markers of the skip of 2026-11-17 (a time.txt with ERROR "DEMORA EXCESSIVA (prevista)" in each
    model) are MOVED, without deleting anything, to Results_Revisao/r1_2c_benchmark_novo/pulados_2026-11-17/. The
    script only moves them if every time.txt of the scenario folder is a skip marker.
  - Inclusion rule, fixed before running: the two scenarios enter the report with whatever comes out, including a time
    cutoff of any method, the pipeline included (treated as in the other scenarios with a cutoff). Declare that they ran
    after the other 86, with the same code and the same limit.

Modes
  rodar               the run (orchestrator; each run in a child process).
  um <scenario> <mod> a child process (internal use).
  fumaca              test of the mechanism on one VALIDATION scenario (walk 1) with a 20 s limit, in
                      Results_Revisao/r1_2c_yeast_verificacao/ (never the test).

Usage (from the repository root, with PYTHONHASHSEED=0):  python revision/execution/r1_2c_yeast.py rodar
The evaluation with the 88 scenarios is revision/offline/r1_2c_evaluate_88.py (Supplementary Table S30).
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
F2 = os.path.dirname(REV)
sys.path[:0] = [HERE, REV, F2]

NAME = "r1_2c_yeast"
FROZEN_HASH = "d0473069ebef1978"
SCENARIOS = ["benchmark_dream3_size100_Yeast2", "benchmark_dream3_size100_Yeast3"]
LIMIT_S = 3600
SLACK_S = 30
SKIP_MARK = "DEMORA EXCESSIVA (prevista)"


def _check():
    import run_log
    h = run_log.method_code_hash()
    if h != FROZEN_HASH:
        print(f"WARNING: method_code_hash {h} != {FROZEN_HASH}, the hash of the frozen method code (the released files are English translations of the frozen ones; see revision/README.md)", file=sys.stderr)
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(f"PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED')!r}; this script requires 0 (the reproduction settings).")


def _plan():
    import r1_2c_new_benchmark as B
    import r1_3b_graded_transitions as R3
    steps = []
    for c in SCENARIOS:
        steps += [(c, m) for m in [B.SUBMITTED] + B.BASELINES]
    for c in SCENARIOS:
        steps += [(c, v) for v in R3.VERSIONS.values()]
    for c in SCENARIOS:
        steps.append((c, B.NO_S7))
    return steps


def _scenario(name):
    import r1_2c_new_benchmark as B
    blocks = ["fumaca"] if name.endswith("actuators_random_walk_1") else ["dream3"]
    for b in blocks:
        for d in B.block_scenarios(b):
            if d["name"] == name:
                return d
    raise SystemExit(f"scenario {name} not found")


def _folders(smoke):
    import common_exec as ce
    if smoke:
        base = ce.output_dir(NAME + "_verificacao")
        return base, os.path.join(base, "run")
    base = os.path.join(ce.OUTPUT_ROOT, "r1_2c_benchmark_novo")
    return base, os.path.join(base, "run")


def one(scen, mod, folder):
    """Child process: runs ONE model on ONE scenario (without validate_all)."""
    import common_exec as ce
    import r1_2c_new_benchmark as B
    d = _scenario(scen)
    if mod in (B.SUBMITTED, *B.BASELINES):
        ce.main_mod.run_all(dataframes=[d], models=ce.models([mod]), config=ce.cfg_exec.PIPELINE_CONFIG,
                            output_dir=folder)
    elif mod == B.NO_S7:
        from Pipeline.penalties import UniformCausalPenalty
        m = ce.model("patch_tau10", B.NO_S7, causal_penalty_fn=UniformCausalPenalty())
        ce.main_mod.run_all(dataframes=[d], models=[m], config=ce.cfg_exec.PIPELINE_CONFIG, output_dir=folder)
    else:
        import r1_3b_graded_transitions as R3
        from variants import r1_3b_graded as g
        version = {v: k for k, v in R3.VERSIONS.items()}[mod]
        orig = g.install()
        try:
            m, modu = R3._model(ce, g, version, "padrao")
            modu.ultimo = {}
            g.LAST_COHERENCE.clear()
            ce.main_mod.run_all(dataframes=[d], models=[m], config=ce.cfg_exec.PIPELINE_CONFIG, output_dir=folder)
            record = {"cenario": scen, "versao": version, "modelo": m["name"], "config": "padrao",
                      "coerencia": dict(g.LAST_COHERENCE), "dml": dict(modu.ultimo)}
            with open(os.path.join(folder, "diag.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=float) + "\n")
        finally:
            g.uninstall(orig)


def _time_txt(model_folder, mod, start, end, limit_s):
    # The text written to time.txt is kept in Portuguese, as in the stored runs (revision/README.md, Notes).
    sec = (end - start).total_seconds()
    h, r = divmod(sec, 3600)
    mi, s = divmod(r, 60)
    dur = (f"{int(h)}h " if h else "") + (f"{int(mi)}min " if mi or h else "") + f"{s:.2f}s"
    txt = (f"model      : {mod}\n"
           f"module     : desconhecido (interrompido pelo limite)\n"
           f"start      : {start:%Y-%m-%d %H:%M:%S}\n"
           f"end        : {end:%Y-%m-%d %H:%M:%S}\n"
           f"duration    : {dur}\n"
           f"duration_s  : {sec:.4f}\n"
           f"status     : ERROR\n"
           f"error      : DEMORA EXCESSIVA (timeout): a execução passou do limite de {str(limit_s // 60) + " min" if limit_s >= 60 else str(limit_s) + " s"} (regra do autor de "
           f"2026-11-17, a mesma dos outros cenários do R1-2c) e foi interrompida por revisao/execucao/r1_2c_yeast.py "
           f"(fila de 2026-11-21b); não é falha do método nem dos dados\n")
    os.makedirs(model_folder, exist_ok=True)
    p = os.path.join(model_folder, "time.txt")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)


def _move_markers(base, folder):
    """Moves the folders of the scenarios skipped on 2026-11-17 (only if every time.txt in them is a skip marker)."""
    target_root = os.path.join(base, "pulados_2026-11-17")
    for c in SCENARIOS:
        p = os.path.join(folder, c)
        if not os.path.isdir(p):
            continue
        tts = [os.path.join(p, m, "time.txt") for m in os.listdir(p) if os.path.isfile(os.path.join(p, m, "time.txt"))]
        if not tts:
            continue
        marks = [SKIP_MARK in open(t, encoding="utf-8", errors="replace").read() for t in tts]
        if all(marks):
            os.makedirs(target_root, exist_ok=True)
            target = os.path.join(target_root, c)
            if os.path.exists(target):
                target += "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(p, target)
            print(f"[{NAME}] skip markers moved (nothing deleted): {p} -> {target}", flush=True)
        elif any(marks):
            raise SystemExit(f"[{NAME}] {p} mixes skip markers with new runs; check by hand before running.")
        else:
            print(f"[{NAME}] {p}: no skip markers (resuming this queue)", flush=True)


def run(smoke=False):
    import common_exec as ce
    _check()
    base, folder = _folders(smoke)
    os.makedirs(folder, exist_ok=True)
    limit = 20 if smoke else LIMIT_S
    slack = 0 if smoke else SLACK_S
    prog = os.path.join(base, "yeast_progresso.jsonl" if not smoke else "progresso.jsonl")
    if smoke:
        import r1_2c_new_benchmark as B
        scen = [d["name"] for d in B.block_scenarios("fumaca")][0]
        steps = [(scen, "varlingam_tau10"), (scen, "patch_tau10")]
    else:
        _move_markers(base, folder)
        steps = _plan()
    print(f"[{NAME}] {len(steps)} runs; limit {limit} s (+{slack} s of slack); output {folder}", flush=True)
    for scen, mod in steps:
        if ce.already_run(folder, scen, mod):
            continue
        start = datetime.datetime.now()
        args = [sys.executable, os.path.abspath(__file__), "um", scen, mod, folder]
        p = subprocess.Popen(args, cwd=F2)
        try:
            code = p.wait(timeout=limit + slack)
            cut = False
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
            p.wait()
            code, cut = None, True
        end = datetime.datetime.now()
        pm = os.path.join(folder, scen, mod)
        if cut:
            _time_txt(pm, mod, start, end, limit)
        st = "?"
        tp = os.path.join(pm, "time.txt")
        if os.path.exists(tp):
            st = next((ln.split(":", 1)[1].strip() for ln in open(tp, encoding="utf-8", errors="replace")
                       if ln.strip().startswith("status")), "?")
        rec = {"cenario": scen, "modelo": mod, "inicio": f"{start:%Y-%m-%d %H:%M:%S}", "fim": f"{end:%Y-%m-%d %H:%M:%S}",
               "duracao_s": round((end - start).total_seconds(), 1), "cortado_60min": cut, "codigo": code,
               "status_time_txt": st, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}
        with open(prog, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{NAME}] {scen} / {mod}: {st}{' (CUT at the limit)' if cut else ''} in {rec['duracao_s']} s", flush=True)
    print(f"[{NAME}] validation (outside the limit)", flush=True)
    ce.main_mod.validate_all(output_dir=folder, show_pairs=False, soft_threshold=ce.cfg_exec.SOFT_THRESHOLD,
                             ignore_intra_entity=not ce.cfg_exec.DETECT_INTRA_ENTITY,
                             use_canonical_pair=ce.cfg_exec.USE_CANONICAL_PAIR)
    missing = [(c, m) for c, m in steps if not ce.already_run(folder, c, m)]
    if missing:
        print(f"[{NAME}] NO time.txt: {missing}", flush=True)
        sys.exit(1)
    print(f"[{NAME}] done: {len(steps)} runs with time.txt", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "um":
        one(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        from run_log import execution
        with execution(NAME + ("_fumaca" if mode == "fumaca" else ""), __file__):
            if mode == "rodar":
                run()
            elif mode == "fumaca":
                run(smoke=True)
            else:
                raise SystemExit("use: rodar | fumaca")

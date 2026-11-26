"""
R2-4b / R1-m7b -- Scale: time per stage from 10 to 160 entities.

Request
  R2: "scaling to enterprise cyber-physical systems, such as SCADA networks or
       industrial plants with hundreds of entities ... is likely to create severe
       computational bottlenecks. The paper presents no empirical wall-clock
       latency or hardware profiling".
  R1: "the O(n^2 f^2 T^3) cost matters for the stated industrial use."

Design (fixed before running)
  Internal VAR generator, n in {10, 20, 40, 80, 160} entities, T = 300, up to 3
  parents per target, lag up to 3, edge probability 2/(n - 1) (about two parents
  per target at every n), seeds 6101-6103. Paper pipeline (patch_tau10);
  the baselines only with --baselines.
  Time per stage: the stage functions of Pipeline/pipeline.py (_STAGES) are
  wrapped in a timer only while this script runs -- no pipeline code is
  changed and the result is the same. Also records the number of candidates
  that reach Stage 6 and the F1.
  Order: run n = 80 first (--n 80) and only then decide about 160, as the
  revision protocol requires.
  Reporting: total and per-stage time against n, empirical log-log exponent,
  F1, and the machine (CPUs, n_jobs).

Usage
  python revision/execution/r2_4b_scale.py rodar --n 80
  python revision/execution/r2_4b_scale.py rodar [--n 10 20 40] [--baselines]
  python revision/execution/r2_4b_scale.py avaliar
Output: Results_Revisao/r2_4b_escala/
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_exec as ce  # noqa: E402

NAME = "r2_4b_escala"
NS = [10, 20, 40, 80, 160]
SEEDS = [6101, 6102, 6103]
T = 300


def n_argument():
    if "--n" not in sys.argv:
        return NS
    vals = []
    for a in sys.argv[sys.argv.index("--n") + 1:]:
        if a.startswith("--"):
            break
        vals.append(int(a))
    return vals


def run():
    import Pipeline.pipeline as pp
    folder = ce.output_dir(os.path.join(NAME, "run"))
    times_path = os.path.join(ce.OUTPUT_ROOT, NAME, "tempos_por_estagio.jsonl")
    originals = pp._STAGES          # tuple; run_pipeline reads the module attribute on every call
    stage_times = {}

    def timed(f):
        def g(run):
            t0 = time.perf_counter()
            f(run)
            stage_times[f.__name__] = stage_times.get(f.__name__, 0.0) + time.perf_counter() - t0
            if f.__name__ == "_stage6_dml":
                stage_times["candidatos_stage6"] = int(len(run.results)) if run.results is not None else 0
        g.__name__ = f.__name__
        return g

    pp._STAGES = tuple(timed(f) for f in originals)
    try:
        names = ce.PAPER_MODELS_TAU10 if "--baselines" in sys.argv else ["patch_tau10"]
        for n in n_argument():
            dfs = ce.generate_var(SEEDS, T, n, min(0.3, 2 / (n - 1)), 3, 3, 2, name_prefix=f"revisao_escala_n{n}")
            for d, ms in ce.pending(dfs, ce.models(names), folder):
                stage_times.clear()
                t0 = time.perf_counter()
                ce.main_mod.run_all(dataframes=[d], models=ms, config=ce.cfg_exec.PIPELINE_CONFIG, output_dir=folder)
                with open(times_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"cenario": d["name"], "n": n, "total_s": time.perf_counter() - t0,
                                        "estagios": dict(stage_times), "ambiente": ce.environment()}) + "\n")
        ce.main_mod.validate_all(output_dir=folder, show_pairs=False, soft_threshold=ce.cfg_exec.SOFT_THRESHOLD,
                                 ignore_intra_entity=not ce.cfg_exec.DETECT_INTRA_ENTITY,
                                 use_canonical_pair=ce.cfg_exec.USE_CANONICAL_PAIR)
    finally:
        pp._STAGES = originals
    evaluate()


def evaluate():
    base = os.path.join(ce.OUTPUT_ROOT, NAME)
    tp = os.path.join(base, "tempos_por_estagio.jsonl")
    if not os.path.exists(tp):
        print("No recorded run.")
        return
    L = [json.loads(x) for x in open(tp, encoding="utf-8")]
    rows = []
    for r in L:
        row = {"cenario": r["cenario"], "n": r["n"], "total_s": r["total_s"]}
        row.update({k: v for k, v in r["estagios"].items()})
        rows.append(row)
    D = pd.DataFrame(rows)
    R = ce.summary(os.path.join(base, "run"))
    if len(R):
        R = R[R.model == "patch_tau10"]
        D = D.merge(R.assign(f1=[ce.f1(a, b, c) for a, b, c in zip(R.TP, R.FP, R.FN)])[["cenario", "f1"]],
                    on="cenario", how="left")
    cols = [k for k in D.columns if k.startswith("_stage")] + ["candidatos_stage6", "total_s", "f1"]
    G = D.groupby("n")[[k for k in cols if k in D]].mean()
    txt = ["# R2-4b — Scale", "", "Means per n (seconds; 3 seeds):", "", G.round(2).to_markdown(), ""]
    if len(G) > 1:
        b = np.polyfit(np.log(G.index.to_numpy(float)), np.log(G["total_s"].to_numpy()), 1)[0]
        txt.append(f"Empirical exponent of total time in n: {b:.2f}.")
    txt.append(f"Machine: {L[-1]['ambiente']}")
    D.to_csv(os.path.join(base, "por_cenario.csv"), index=False)
    open(os.path.join(base, "relatorio.md"), "w", encoding="utf-8").write("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        if len(sys.argv) > 1 and sys.argv[1] == "rodar":
            run()
        else:
            evaluate()

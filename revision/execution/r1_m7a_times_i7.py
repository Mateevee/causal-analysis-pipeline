"""
R1-m7a / R2-4 / R1-7b -- Times on the i7 (the runner): pipeline and baselines on the 90 scenarios, and the Stage 7
savings.

Requests
  R1 (minor): "Include the runtime table in the main text; the O(n^2 f^2 T^3) cost matters for the stated
       industrial use."
  R2-4a: "The paper presents no empirical wall-clock latency or hardware profiling".
  R1-7b (support): the pre-filter sentence ("Stage 7 ... cuts that stage's run time by a median factor of X") only
       enters the text with the complete measurement (NOTAS.md, 2026-11-14).

Why measure again (decisions of the author and of Claude, NOTAS.md, 2026-11-16 and 2026-11-19)
  The author decided to cite only times from the i7 (the runner). The machine of the paper's runs is not confirmed, so
  the paper's times do not go into the text (path 2 of R1-m7a). The complete Stage 7 measurement and R1-m7a are done
  together: the full variant of Stage 7 is the same pipeline execution that R1-m7a needs.

Design (fixed before running; 2026-11-19)
  Method code frozen (method_code_hash d0473069ebef1978); the paper's configuration (PIPELINE_CONFIG).
  Machine with no other use during the step; one scenario and one model at a time.
  Part A, the paper's 90 scenarios × {patch_tau10 (full), patch_semS7_tau10 (UniformCausalPenalty: Stage 7
    does not run; Stage 8 receives the output of Stage 6)}. Scenario order: alphabetical. Variant order
    alternated per scenario (even index: full first; odd: without S7 first), against machine drift.
  Part B, the paper's 5 baselines (pcmci_tigramite, pcmciplus, dynotears, cuts_plus, varlingam; _tau10) on the 90
    scenarios, one at a time, in the alphabetical order of the scenarios and in the order above for the models.
  Part C, the R2-4b scale: VAR generator (T = 300, up to 3 parents, lag up to 3, edge probability 2/(n - 1)),
    n in {10, 20, 40, 80, 160}, seeds 6101 to 6103, in both variants, with the order alternated as in A.
  Part D, timing noise: 10 scenarios from A (fixed systematic sample: indices 0, 9, ..., 81 of the alphabetical
    order, the same as in the pilot) run again once in each variant, in a separate folder.
  Order of the parts: A, C, B, D. Resumable: each (part, scenario, model) already stored in tempos.jsonl is skipped.
  Warm-up (added on 2026-11-19 after the smoke test and before the official run): at the start of each call,
    each model that is going to run runs once on benchmark_tigramite_quadratic_seed11, in run_aquecimento/, unmeasured.
    Reason: in the smoke test, the 1st execution of the process took ~2 times the time of the repetition (imports and
    start-up of the process pool), which would contaminate the 1st scenario of each call and part D.
  Measurement: perf_counter around each run_all (total) and each pipeline stage (Pipeline.pipeline._STAGES
    wrapped only while this script runs; no pipeline code changes), plus the number of significant pairs at the
    end of each stage; and the duration stored in the time.txt of each execution.
  Reporting (avaliar):
    R1-m7a: time per method on the 90 (median, mean, maximum; development and held-out), per range of number of
      series, and empirical exponents of the full pipeline (log time ~ log series + log T). Hardware stated.
    Stage 7: time of Stage 8a and total with and without Stage 7 on the 90 (median ratio, median difference, paired
      Wilcoxon, better / equal / worse); pairs entering Stage 8a; F1 of both variants on the same machine.
    Scale: means per n in both variants (replaces the R2-4b scale table).
    Noise: relative difference between the repetition and the first execution (median of the absolute value), per
      variant, in the total and in Stage 8a. Differences between variants smaller than this noise are not read as an effect.
  Every result is reported, including zero or negative savings.

Usage
  python revision/execution/r1_m7a_times_i7.py rodar [--partes A,C,B,D]
  python revision/execution/r1_m7a_times_i7.py avaliar
  python revision/execution/r1_m7a_times_i7.py fumaca        (1 small scenario, unmeasured; folder *_fumaca)
Output: Results_Revisao/r1_m7a_tempos_i7/
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

NAME = "r1_m7a_tempos_i7"
FULL, NO_S7 = "patch_tau10", "patch_semS7_tau10"
BASELINES = ["pcmci_tigramite_tau10", "pcmciplus_tau10", "dynotears_tau10", "cuts_plus_tau10", "varlingam_tau10"]
NS, SEEDS, T = (10, 20, 40, 80, 160), (6101, 6102, 6103), 300
NOISE_INDICES = tuple(range(0, 90, 9))
PARTS = ("A", "C", "B", "D")
WARMUP = "benchmark_tigramite_quadratic_seed11"  # 2 entities, 6 series; warm-up only, not measured
_STATE = {"nome": NAME}


def _folder(sub=""):
    import common_exec as ce
    return ce.output_dir(os.path.join(_STATE["nome"], sub)) if sub else ce.output_dir(_STATE["nome"])


def _paper(ce):
    every = sorted(ce.paper_scenarios(), key=lambda d: d["name"])
    if len(every) != 90:
        raise SystemExit(f"expected 90 paper scenarios, got {len(every)}")
    return every


def _scale(ce):
    out = []
    for n in NS:
        out += ce.generate_var(list(SEEDS), T, n, min(0.3, 2 / (n - 1)), 3, 3, 2, name_prefix=f"tempos_escala_n{n}")
    return out


def _models(ce):
    from Pipeline.penalties import UniformCausalPenalty
    m = {FULL: ce.model(FULL), NO_S7: ce.model(FULL, NO_S7, causal_penalty_fn=UniformCausalPenalty())}
    for b in BASELINES:
        m[b] = ce.model(b)
    return m


def _time_txt_duration(folder, scen, mod):
    p = os.path.join(folder, scen, mod, "time.txt")
    if not os.path.exists(p):
        return None, None
    txt = open(p, encoding="utf-8", errors="replace").read()
    st = re.search(r"status\s*:\s*(\S+)", txt)
    return txt.strip()[:300], (st.group(1) if st else None)


def _plan(ce, parts):
    """List of (part, scenario, model, folder) in the order of the design."""
    plan = []
    art = _paper(ce)
    for p in parts:
        if p == "A":
            for i, d in enumerate(art):
                for m in ([FULL, NO_S7] if i % 2 == 0 else [NO_S7, FULL]):
                    plan.append(("A", d, m, "run"))
        elif p == "C":
            for i, d in enumerate(_scale(ce)):
                for m in ([FULL, NO_S7] if i % 2 == 0 else [NO_S7, FULL]):
                    plan.append(("C", d, m, "run_escala"))
        elif p == "B":
            for d in art:
                for m in BASELINES:
                    plan.append(("B", d, m, "run"))
        elif p == "D":
            for k, i in enumerate(NOISE_INDICES):
                for m in ([FULL, NO_S7] if k % 2 == 0 else [NO_S7, FULL]):
                    plan.append(("D", art[i], m, "run_ruido"))
    return plan


def _log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(_folder(), "progresso.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(parts=PARTS, limit=None, measure=True, name_filter=None):
    import common_exec as ce
    import Pipeline.pipeline as pp
    times_file = os.path.join(_folder(), "tempos.jsonl")
    done = set()
    if os.path.exists(times_file):
        for l in open(times_file, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["parte"], r["cenario"], r["modelo"]))
    mods = _models(ce)
    plan = _plan(ce, parts)
    if name_filter:
        plan = [x for x in plan if x[1]["name"] == name_filter]
    if limit:
        plan = plan[:limit]
    originals = pp._STAGES
    reg = {}

    def timed(f):
        def g(run):
            t0 = time.perf_counter()
            f(run)
            reg[f.__name__] = reg.get(f.__name__, 0.0) + time.perf_counter() - t0
            r = run.results
            if r is not None and "significant" in getattr(r, "columns", []):
                reg["sig_apos" + f.__name__] = int((r["significant"] == True).sum())  # noqa: E712
        g.__name__ = f.__name__
        return g

    pp._STAGES = tuple(timed(f) for f in originals)
    pending_runs = [x for x in plan if (x[0], x[1]["name"], x[2]) not in done]
    _log(f"start: parts {','.join(parts)}; {len(plan)} executions in the plan, {len(pending_runs)} pending")
    if pending_runs and measure:
        # Warm-up (added on 2026-11-19, before the official run; the smoke test showed the 1st execution of each
        # process ~2 times slower, because of imports and the start-up of the process pool): each model of this
        # call runs once on the small warm-up scenario, in run_aquecimento/, unmeasured.
        wu = next(d for d in _paper(ce) if d["name"] == WARMUP)
        for m in sorted({x[2] for x in pending_runs}):
            _log(f"warm-up: {m} (not measured)")
            ce.main_mod.run_all(dataframes=[wu], models=[mods[m]], config=ce.cfg_exec.PIPELINE_CONFIG,
                                output_dir=_folder("run_aquecimento"))
    try:
        for k, (part, d, m, sub) in enumerate(pending_runs, 1):
            folder = _folder(sub)
            reg.clear()
            _log(f"({k}/{len(pending_runs)}) {part} {d['name']} {m}: start")
            t0 = time.perf_counter()
            ce.main_mod.run_all(dataframes=[d], models=[mods[m]], config=ce.cfg_exec.PIPELINE_CONFIG,
                                output_dir=folder)
            total = time.perf_counter() - t0
            dur, status = _time_txt_duration(folder, d["name"], m)
            _log(f"({k}/{len(pending_runs)}) {part} {d['name']} {m}: end in {total:.1f} s, status {status}")
            if measure:
                with open(times_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"parte": part, "cenario": d["name"], "modelo": m, "total_s": total,
                                        "estagios": dict(reg), "time_txt": dur, "status": status,
                                        "quando": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "ambiente": ce.environment()}) + "\n")
    finally:
        pp._STAGES = originals
    for sub in sorted({x[3] for x in plan} - {"run_aquecimento"}):
        ce.main_mod.validate_all(output_dir=_folder(sub), show_pairs=False, soft_threshold=ce.cfg_exec.SOFT_THRESHOLD,
                                 ignore_intra_entity=not ce.cfg_exec.DETECT_INTRA_ENTITY,
                                 use_canonical_pair=ce.cfg_exec.USE_CANONICAL_PAIR)
    _log("end")


# ----------------------------------------------------------------------------- evaluate
def _wilcoxon(a, b):
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.allclose(a, b):
        return float("nan")
    return float(wilcoxon(a, b).pvalue)


def _size(scen_folder):
    d = pd.read_csv(os.path.join(scen_folder, "source.csv"))
    props = [k for k in d.columns if k not in ("Date", "Name")]
    n_series = int(sum(d.groupby("Name")[p].apply(lambda v: v.notna().any()).sum() for p in props))
    return n_series, int(d.groupby("Name").size().median())


def evaluate():
    import common as c
    import common_exec as ce
    base = _folder()
    L = [json.loads(l) for l in open(os.path.join(base, "tempos.jsonl"), encoding="utf-8")]
    rep = c.Report(_STATE["nome"], "R1-m7a / R2-4 / R1-7b — Times on the i7: pipeline, baselines and Stage 7")
    env = L[0]["ambiente"] if L else {}
    rep.p(f"Machine: {env}. Stored executions: {len(L)}. Method code: hash d0473069ebef1978 (log).")
    D = pd.DataFrame([{"parte": r["parte"], "cenario": r["cenario"], "modelo": r["modelo"], "total_s": r["total_s"],
                       "status": r.get("status"), **{k: v for k, v in r["estagios"].items()}} for r in L])
    rep.csv("tempos_por_execucao.csv", D)
    expected = {"A": 180, "B": 450, "C": 30, "D": 20}
    rep.p("Executions per part (stored / expected): " + "; ".join(
        f"{p} {int((D.parte == p).sum())}/{n}" for p, n in expected.items()) + ".")
    not_ok = D[D.status.astype(str).str.lower() != "ok"]
    rep.p(f"Executions without status ok: {len(not_ok)}" + (f": {not_ok[['parte', 'cenario', 'modelo', 'status']].to_dict('records')}"
                                                            if len(not_ok) else "."))
    f1 = {}
    for sub in ("run", "run_escala", "run_ruido"):
        R = ce.summary(os.path.join(base, sub)) if os.path.isdir(os.path.join(base, sub)) else pd.DataFrame()
        for r in R.itertuples():
            f1[(sub, r.cenario, r.model)] = ce.f1(r.TP, r.FP, r.FN) if str(r.status) == "ok" else np.nan

    # --- R1-m7a: time per method on the 90
    A = D[D.parte.isin(["A", "B"])].copy()
    A = A[A.modelo != NO_S7]
    lab = {FULL: "pipeline", "pcmci_tigramite_tau10": "PCMCI", "pcmciplus_tau10": "PCMCI+",
           "dynotears_tau10": "DYNOTEARS", "cuts_plus_tau10": "CUTS+", "varlingam_tau10": "VARLiNGAM"}
    A["metodo"] = A.modelo.map(lab)
    A["heldout"] = [c.is_heldout(c.scenario_key(s)) if c.scenario_key(s) else None for s in A.cenario]
    sizes = {s: _size(os.path.join(base, "run", s)) for s in A.cenario.unique()}
    A["n_series"] = A.cenario.map(lambda s: sizes[s][0])
    A["T"] = A.cenario.map(lambda s: sizes[s][1])
    rep.h("## R1-m7a — time per method (s, perf_counter around each execution)")
    body = []
    for m in lab.values():
        for part, msk in (("all", A.metodo == m), ("dev", (A.metodo == m) & (A.heldout == False)),  # noqa: E712
                          ("held-out", (A.metodo == m) & (A.heldout == True))):  # noqa: E712
            x = A.loc[msk, "total_s"].dropna()
            if len(x):
                body.append([m, part, len(x), f"{x.median():.1f}", f"{x.mean():.1f}", f"{x.max():.1f}"])
    rep.table(["Method", "Part", "Scenarios", "Median (s)", "Mean (s)", "Maximum (s)"], body)
    A["faixa"] = pd.cut(A.n_series, [0, 10, 20, 40, 10 ** 6], labels=["≤10", "11–20", "21–40", ">40"])
    piv = A.pivot_table(index="faixa", columns="metodo", values="total_s", aggfunc="median", observed=False)
    cnt = A[A.metodo == "pipeline"].groupby("faixa", observed=False).size()
    rep.table(["Series", "Scenarios"] + [f"{m}, median (s)" for m in lab.values()],
              [[str(f), int(cnt.get(f, 0))] + [c.fmt(piv.loc[f, m], 1) if m in piv.columns else "—"
                                               for m in lab.values()] for f in piv.index])
    P = A[A.metodo == "pipeline"].dropna(subset=["total_s"])
    if len(P) > 3:
        X = np.column_stack([np.ones(len(P)), np.log(P.n_series), np.log(P["T"])])
        coef, *_ = np.linalg.lstsq(X, np.log(P.total_s), rcond=None)
        rep.p(f"Full pipeline: log(time) ~ {coef[1]:.2f}·log(series) + {coef[2]:.2f}·log(T) ({len(P)} scenarios; "
              "empirical exponents, not the O(n²f²T³) bound).")
        rep.data["expoentes_pipeline"] = {"series": float(coef[1]), "T": float(coef[2])}
    rep.csv("tempo_por_metodo_por_cenario.csv", A)

    # --- Stage 7: savings on the 90, on the scale and the noise
    def paired(part, sub):
        g = D[D.parte == part]
        recs = []
        for s in sorted(g.cenario.unique()):
            a, b = g[(g.cenario == s) & (g.modelo == FULL)], g[(g.cenario == s) & (g.modelo == NO_S7)]
            if not len(a) or not len(b):
                continue
            a, b = a.iloc[0], b.iloc[0]
            recs.append({"cenario": s, "t7": a.get("_stage7_causal_filter", np.nan),
                         "t8a_com": a.get("_stage8a_structural_pruning", np.nan),
                         "t8a_sem": b.get("_stage8a_structural_pruning", np.nan),
                         "total_com": a.total_s, "total_sem": b.total_s,
                         "entrada8a_com": a.get("sig_apos_stage7_causal_filter", np.nan),
                         "entrada8a_sem": b.get("sig_apos_stage7_causal_filter", np.nan),
                         "f1_com": f1.get((sub, s, FULL)), "f1_sem": f1.get((sub, s, NO_S7))})
        return pd.DataFrame(recs)

    for part, sub, title in (("A", "run", "the paper's 90 scenarios"), ("C", "run_escala", "R2-4b scale")):
        S = paired(part, sub)
        if S.empty:
            continue
        S["razao_8a"] = S.t8a_sem / S.t8a_com
        S["razao_total"] = S.total_sem / S.total_com
        rep.csv(f"stage7_{part}.csv", S)
        rep.h(f"## Stage 7 — {title} ({len(S)} scenarios with both variants)")
        w8 = c.wins(S.t8a_com, S.t8a_sem)
        wt = c.wins(S.total_com, S.total_sem)
        rep.p(f"- Stage 7: sum {S.t7.sum():.1f} s; median {S.t7.median():.2f} s per scenario.")
        rep.p(f"- Stage 8a: median ratio (without/with) {S.razao_8a.median():.2f}; median difference "
              f"{(S.t8a_sem - S.t8a_com).median():.1f} s; sum {S.t8a_com.sum():.0f} s with and {S.t8a_sem.sum():.0f} s "
              f"without; with Stage 7 faster / equal / slower: {w8[2]} / {w8[1]} / {w8[0]}; Wilcoxon p "
              f"{c.fmt(_wilcoxon(S.t8a_com, S.t8a_sem))}.")
        rep.p(f"- Total: median ratio {S.razao_total.median():.2f}; sum {S.total_com.sum():.0f} s with and "
              f"{S.total_sem.sum():.0f} s without ({100 * (S.total_sem.sum() / S.total_com.sum() - 1):+.0f}% without Stage 7); "
              f"faster / equal / slower with Stage 7: {wt[2]} / {wt[1]} / {wt[0]}; Wilcoxon p "
              f"{c.fmt(_wilcoxon(S.total_com, S.total_sem))}.")
        rep.p(f"- Pairs entering Stage 8a: median {S.entrada8a_com.median():.0f} with and {S.entrada8a_sem.median():.0f} "
              f"without; median ratio {(S.entrada8a_sem / S.entrada8a_com.replace(0, np.nan)).median():.2f}.")
        ff = S.dropna(subset=["f1_com", "f1_sem"])
        if len(ff):
            wf = c.wins(ff.f1_com, ff.f1_sem)
            rep.p(f"- F1 (same machine): {ff.f1_com.mean():.3f} with and {ff.f1_sem.mean():.3f} without Stage 7; with "
                  f"Stage 7 better / equal / worse: {wf[0]} / {wf[1]} / {wf[2]}.")
        if part == "C":
            S["n"] = S.cenario.str.extract(r"_n(\d+)")[0].astype(float)
            G = S.groupby("n")[["t7", "t8a_com", "t8a_sem", "total_com", "total_sem", "f1_com", "f1_sem"]].mean()
            rep.table(["n"] + list(G.columns), [[int(n)] + [c.fmt(v, 2) for v in G.loc[n]] for n in G.index])

    R = D[D.parte == "D"]
    if len(R):
        A1 = D[D.parte == "A"].set_index(["cenario", "modelo"])
        recs = []
        for _, r in R.iterrows():
            if (r.cenario, r.modelo) in A1.index:
                o = A1.loc[(r.cenario, r.modelo)]
                k8 = "_stage8a_structural_pruning"
                recs.append({"cenario": r.cenario, "modelo": r.modelo,
                             "dif_rel_total": abs(r.total_s - o.total_s) / o.total_s,
                             "dif_rel_8a": abs(r[k8] - o[k8]) / o[k8]})
        N = pd.DataFrame(recs)
        rep.csv("ruido.csv", N)
        rep.h("## Timing noise (repetition × first execution)")
        for m in (FULL, NO_S7):
            g = N[N.modelo == m]
            if len(g):
                rep.p(f"- {m}: {len(g)} scenarios; median |relative difference| {100 * g.dif_rel_total.median():.1f}% in the "
                      f"total and {100 * g.dif_rel_8a.median():.1f}% in Stage 8a (maximum {100 * g.dif_rel_total.max():.1f}% "
                      f"and {100 * g.dif_rel_8a.max():.1f}%).")
    rep.save()
    print(open(os.path.join(base, "relatorio.md"), encoding="utf-8").read())


def smoke():
    """The tigramite scenario of the noise sample (small): both variants of A, 1 baseline of B and D; plus the
    smallest scale of C. Folder *_fumaca; none of this enters the report (it only checks the plan, the storage and avaliar)."""
    import common_exec as ce
    _STATE["nome"] = NAME + "_fumaca"
    art = _paper(ce)
    target = next(art[i]["name"] for i in NOISE_INDICES if "tigramite" in art[i]["name"])
    print(f"smoke: scenario {target}; noise sample: {[art[i]['name'] for i in NOISE_INDICES]}", flush=True)
    run(parts=("A",), name_filter=target)
    run(parts=("B",), limit=1, name_filter=target)
    run(parts=("C",), limit=2)
    run(parts=("D",), name_filter=target)
    evaluate()


if __name__ == "__main__":
    from run_log import execution
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "fumaca":
        with execution(NAME + "_fumaca", __file__):
            smoke()
    elif mode in ("rodar", "avaliar"):
        parts = PARTS
        if "--partes" in sys.argv:
            parts = tuple(sys.argv[sys.argv.index("--partes") + 1].split(","))
            if not set(parts) <= set(PARTS):
                raise SystemExit(f"valid parts: {PARTS}")
        with execution(NAME, __file__):
            run(parts) if mode == "rodar" else evaluate()
    else:
        print(__doc__)

"""
R1-m6b -- PCMCI with CMIknn (library default parameters), SPLIT execution, on the 36 scenarios of the GPDC check.

Request
  R1: "The fully nonparametric CMIknn ... were not evaluated (you say so); I would add them since the claim is
       domain-agnosticism."

Design (fixed before running; version of 2026-11-16)
  Scope (author's decision of 2026-11-16): the same 36 scenarios as the paper's GPDC check
  (main_gpdc.GPDC_SCENARIO_FILTER, Table S21), in the order of SCOPE (from the cheap groups to the expensive ones). PCMCI only.
  Method: the paper's entry pcmci_tigramite_tau10 (tau_max = 10, pc_alpha = 0.20, alpha 0.20, tigramite's FDR-BH
  over N²·tau_max links), with the CMIknn test at the library's DEFAULT PARAMETERS (tigramite 5.2:
  knn = 0.2; 500 permutations; shuffle_neighbors = 5; ranks; seed = 42). Only change: workers = 1 in the copy of the
  test that each process receives (number of threads of the neighbour search; it changes neither CMI nor the p-value,
  checked in the conferir mode). The method code (ModelComparison/, Pipeline/ etc.) is not modified.
  Split: PCMCI.run_pcmci is replaced, only in the main process, by a version that
    (1) runs the PC1 phase (run_pc_stable) of each target series in a separate process (link_assumptions of the target only);
    (2) gathers the parents and runs the MCI phase (run_mci, with all parents) in up to N_MCI blocks of links;
    (3) returns p_matrix and val_matrix to the paper's wrapper, which applies the same correction and the same output.
  Difference from the whole execution: only the drawn permutations (each process creates the test with seed 42; the
  whole execution consumes a single generator test after test). The split is checked with ParCorr (deterministic):
  split == whole, row by row (conferir mode), otherwise it does not run.
  Resumable: a scenario with time.txt is skipped. Progress in progresso_36.log; one log per process in logs_36/.
  Pilot (2026-11-15, main machine): Walker only, N_MCI = 10, in run_dividido_padrao/ (NOTAS.md).

Usage
  python revision/execution/r1_m6b_split_pcmci.py carregar                 (only counts the 36 scenarios)
  python revision/execution/r1_m6b_split_pcmci.py conferir                 (seconds)
  python revision/execution/r1_m6b_split_pcmci.py rodar [--grupos a,b]     (groups: tigramite, logisticos,
                                                                             lorenz, reais; without --grupos, all)
  python revision/execution/r1_m6b_split_pcmci.py avaliar
Output: Results_Revisao/r1_m6b_cmiknn/run_36_padrao/; report in Results_Revisao/r1_m6b_cmiknn_36/.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

NAME = "r1_m6b_cmiknn"
NAME_REPORT = "r1_m6b_cmiknn_36"
SUB = "run_36_padrao"
MODEL = ("pcmci_tigramite_tau10", "pcmci_cmiknn_tau10_dividido", {"pcmci_cond_ind_test": "cmiknn"})
N_MCI = 20
TAU, ALPHA, B_DEFAULT = 10, 0.20, 500
_S = ("seed11", "seed22", "seed33")
_L = ("seed1", "seed2", "seed3")
GROUPS = {
    "tigramite": [f"benchmark_tigramite_{g}_{s}" for g in ("baseline_linear", "quadratic", "threshold", "saturation",
                                                            "large_mixed") for s in _S],
    "logisticos": [f"benchmark_logistic_maps_{g}_{s}" for g in ("chain", "convergence", "bifurcation", "with_isolated")
                   for s in _L],
    "lorenz": [f"benchmark_lorenz96_{g}_{s}" for g in ("K8", "K10") for s in _L],
    "reais": ["real_tigramite_climate_walker", "real_sachs_proteins", "real_pems_bay_traffic"],
}
SCOPE = GROUPS["tigramite"] + GROUPS["logisticos"] + GROUPS["lorenz"] + GROUPS["reais"]
assert len(SCOPE) == 36 and len(set(SCOPE)) == 36
CHECK_SCENARIOS = ["real_tigramite_climate_walker", "benchmark_tigramite_large_mixed_seed11",
                   "benchmark_lorenz96_K10_seed1"]

_STATE = {"dir_logs": None, "progresso": None}


# ------------------------------------------------------------------ child processes
def _open_log(name):
    d = _STATE.get("dir_logs")
    return open(os.path.join(d, name), "a", encoding="utf-8", buffering=1) if d else open(os.devnull, "w")


def _child_pc1(data, var_names, cit, j, tau_min, tau_max, pc_alpha, verbosity, dir_logs, label):
    import contextlib
    import tigramite.data_processing as pp
    from tigramite.pcmci import PCMCI
    _STATE["dir_logs"] = dir_logs
    t0 = time.perf_counter()
    N = data.shape[1]
    la = {k: {} for k in range(N)}
    la[j] = {(i, -tau): "-?>" for i in range(N) for tau in range(max(1, tau_min), tau_max + 1)}
    with _open_log(f"{label}_pc1_alvo{j}.log") as f, contextlib.redirect_stdout(f):
        p = PCMCI(dataframe=pp.DataFrame(data, var_names=var_names), cond_ind_test=cit, verbosity=verbosity)
        par = p.run_pc_stable(link_assumptions=la, tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)
    return j, par[j], time.perf_counter() - t0


def _child_mci(data, var_names, cit, links, parents, tau_min, tau_max, alpha_level, verbosity, k, dir_logs, label):
    import contextlib
    import tigramite.data_processing as pp
    from tigramite.pcmci import PCMCI
    _STATE["dir_logs"] = dir_logs
    t0 = time.perf_counter()
    N = data.shape[1]
    la = {j: {} for j in range(N)}
    for i, j, tau in links:
        la[j][(i, -tau)] = "-?>"
    with _open_log(f"{label}_mci_bloco{k}.log") as f, contextlib.redirect_stdout(f):
        p = PCMCI(dataframe=pp.DataFrame(data, var_names=var_names), cond_ind_test=cit, verbosity=verbosity)
        r = p.run_mci(link_assumptions=la, tau_min=tau_min, tau_max=tau_max, parents=parents, alpha_level=alpha_level)
    return links, r["p_matrix"], r["val_matrix"], time.perf_counter() - t0


# ------------------------------------------------------------------ split run_pcmci
def _log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _STATE.get("progresso"):
        with open(_STATE["progresso"], "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_pcmci_split(self, selected_links=None, link_assumptions=None, tau_min=0, tau_max=1, save_iterations=False,
                    pc_alpha=0.2, max_conds_dim=None, max_combinations=1, max_conds_py=None, max_conds_px=None,
                    alpha_level=0.05, fdr_method="none"):
    """Same interface as PCMCI.run_pcmci, for the call made by the paper's wrapper (no link_assumptions, no
    conditioning limits, fdr_method 'none'). Any other use is refused."""
    import copy
    import multiprocessing as mp
    if (selected_links is not None or link_assumptions is not None or save_iterations or max_conds_dim is not None
            or max_combinations != 1 or max_conds_py is not None or max_conds_px is not None or fdr_method != "none"):
        raise ValueError("run_pcmci_split only covers the call made by the paper's wrapper")
    data = np.asarray(self.dataframe.values[0], dtype=float)
    names = list(self.var_names)
    N = data.shape[1]
    cit = copy.deepcopy(self.cond_ind_test)  # initial state of the test (seed 42) for each process
    if type(cit).__name__ == "CMIknn":
        cit.workers = 1
    verb = 2 if _STATE.get("dir_logs") else 0
    lab = _STATE.get("rotulo", "x")
    dl = _STATE.get("dir_logs")
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    _log(f"  PC1: {N} processes (one per target), test {type(cit).__name__}")
    with ctx.Pool(processes=N) as pool:
        res = [pool.apply_async(_child_pc1, (data, names, cit, j, tau_min, tau_max, pc_alpha, verb, dl, lab))
               for j in range(N)]
        parents = {}
        for r in res:
            j, pj, dur = r.get()
            parents[j] = pj
            _log(f"  PC1 target {j} ({names[j]}) in {dur / 60:.1f} min: {len(pj)} parents")
    links = [(i, j, tau) for j in range(N) for i in range(N) for tau in range(max(1, tau_min), tau_max + 1)]
    n_b = min(N_MCI, len(links))
    blocks = [links[k::n_b] for k in range(n_b)]
    _log(f"  MCI: {len(links)} links in {n_b} processes")
    p_mat = val_mat = None
    with ctx.Pool(processes=n_b) as pool:
        res = [pool.apply_async(_child_mci, (data, names, cit, b, parents, tau_min, tau_max, alpha_level, verb, k,
                                             dl, lab)) for k, b in enumerate(blocks)]
        durs = []
        for r in res:
            lk, pm, vm, dur = r.get()
            durs.append(dur)
            if p_mat is None:
                p_mat, val_mat = pm.copy(), vm.copy()
            for i, j, tau in lk:
                p_mat[i, j, tau] = pm[i, j, tau]
                val_mat[i, j, tau] = vm[i, j, tau]
    _log(f"  MCI finished (blocks from {min(durs) / 60:.1f} to {max(durs) / 60:.1f} min); "
         f"split PCMCI in {(time.perf_counter() - t0) / 60:.1f} min")
    self.all_parents = parents
    self.results = {"p_matrix": p_mat, "val_matrix": val_mat}
    return self.results


# ------------------------------------------------------------------ modes
def _scenarios(names):
    import common_exec as ce
    by_name = {d["name"]: d for d in ce.paper_scenarios(names)}
    absent = [n for n in names if n not in by_name]
    if absent:
        raise SystemExit(f"scenarios not found: {absent}")
    return [by_name[n] for n in names]


def load():
    """Only loads the 36 scenarios and counts variables (columns PCMCI receives) and points; no method runs."""
    import common_exec as ce
    from Auxiliary.causal_common import _pcmci_load_and_pivot
    m = ce.model("pcmci_tigramite_tau10")
    cfg = {**ce.cfg_exec.PIPELINE_CONFIG, **m["config_override"]}
    rows = []
    for d in _scenarios(SCOPE):
        X = _pcmci_load_and_pivot(d["df"], date_col=cfg.get("date_col", "Data"), entity_col=cfg.get("entity_col", "Nome"),
                                  property_cols=cfg.get("property_cols"))
        rows.append((d["name"], X.shape[1], X.shape[0]))
        print(f"{d['name']:48s} variables={X.shape[1]:3d} points={X.shape[0]}", flush=True)
    print(f"total: {len(rows)} scenarios (expected 36)", flush=True)
    if len(rows) != 36:
        raise SystemExit(1)


def check():
    """(1) CMIknn: workers changes neither CMI nor the p-value. (2) ParCorr: the wrapper with the split PCMCI gives
    output identical to that of the whole PCMCI."""
    import common_exec as ce
    from tigramite.independence_tests.cmiknn import CMIknn
    from tigramite.pcmci import PCMCI
    ok = True
    rng = np.random.default_rng(1)
    for effect in (0.0, 0.3):
        for dz in (0, 3, 8):
            arr = rng.standard_normal((2 + dz, 300))
            arr[1] += effect * arr[0]
            xyz = np.array([0, 1] + [2] * dz)
            res = []
            for w in (-1, 1):
                t = CMIknn(workers=w, sig_samples=100)  # 100 only here, so that the check takes seconds
                v = t.get_dependence_measure(arr.copy(), xyz)
                res.append((v, t.get_shuffle_significance(arr.copy(), xyz, v)))
            ok &= res[0] == res[1]
            print(f"CMIknn effect={effect} dimZ={dz}: workers -1 {res[0]} | 1 {res[1]} | identical: {res[0] == res[1]}",
                  flush=True)
    default_test = CMIknn()
    print(f"CMIknn default in this installation: knn={default_test.knn}, sig_samples={default_test.sig_samples}, "
          f"shuffle_neighbors={default_test.shuffle_neighbors}, transform={default_test.transform}", flush=True)
    ok &= default_test.knn == 0.2 and default_test.sig_samples == B_DEFAULT and default_test.shuffle_neighbors == 5
    original = PCMCI.run_pcmci
    m = ce.model("pcmci_tigramite_tau10")
    cfg = {**ce.cfg_exec.PIPELINE_CONFIG, **m["config_override"]}
    for d in _scenarios(CHECK_SCENARIOS):
        PCMCI.run_pcmci = original
        a = m["pipeline"](d["df"], **cfg)
        PCMCI.run_pcmci = run_pcmci_split
        b = m["pipeline"](d["df"], **cfg)
        PCMCI.run_pcmci = original
        same = a.reset_index(drop=True).equals(b.reset_index(drop=True))
        ok &= same
        print(f"ParCorr {d['name']}: {len(a)} rows, {int(a.significant.sum())} significant | split identical: "
              f"{same}", flush=True)
    print("CHECK:", "OK" if ok else "FAILED -- do not run", flush=True)
    if not ok:
        raise SystemExit(1)


def run():
    import common_exec as ce
    from tigramite.pcmci import PCMCI
    groups = list(GROUPS)
    if "--grupos" in sys.argv:
        groups = sys.argv[sys.argv.index("--grupos") + 1].split(",")
        if any(g not in GROUPS for g in groups):
            raise SystemExit(f"valid groups: {list(GROUPS)}")
    names = [n for n in SCOPE if any(n in GROUPS[g] for g in groups)]
    folder = ce.output_dir(os.path.join(NAME, SUB))
    base = os.path.dirname(folder)
    _STATE["progresso"] = os.path.join(base, "progresso_36.log")
    _STATE["dir_logs"] = os.path.join(base, "logs_36")
    os.makedirs(_STATE["dir_logs"], exist_ok=True)
    m = ce.model(*MODEL[:2], **MODEL[2])
    missing = [n for n in names if not ce.already_run(folder, n, m["name"])]
    _log(f"start: groups {groups}; {len(names)} scenarios, {len(names) - len(missing)} already done; {m['name']}; "
         f"CMIknn library default, workers = 1; N_MCI = {N_MCI}")
    PCMCI.run_pcmci = run_pcmci_split
    for k, d in enumerate(_scenarios(missing), 1):
        _STATE["rotulo"] = d["name"]
        t0 = time.time()
        _log(f"({k}/{len(missing)}) {d['name']}: start")
        ce.main_mod.run_all(dataframes=[d], models=[m], config=ce.cfg_exec.PIPELINE_CONFIG, output_dir=folder)
        ok = ce.already_run(folder, d["name"], m["name"])
        _log(f"({k}/{len(missing)}) {d['name']}: {'finished' if ok else 'NO time.txt (error; see log)'} in "
             f"{(time.time() - t0) / 60:.1f} min")
    _log("validating")
    ce.main_mod.validate_all(output_dir=folder, show_pairs=False, soft_threshold=ce.cfg_exec.SOFT_THRESHOLD,
                             ignore_intra_entity=not ce.cfg_exec.DETECT_INTRA_ENTITY,
                             use_canonical_pair=ce.cfg_exec.USE_CANONICAL_PAIR)
    _log("end")


def _floor(model_folder):
    """Diagnostic of the p-value floor (1/(B+1)) against FDR-BH: links at the floor and the k* that BH requires."""
    import pandas as pd
    f = os.path.join(model_folder, "original.csv")
    if not os.path.exists(f):
        return {}
    o = pd.read_csv(f)
    vs = set(zip(o.source_entity, o.source_property)) | set(zip(o.target_entity, o.target_property))
    n = len(vs)
    floor = 1.0 / (B_DEFAULT + 1)
    return {"N variáveis": n, "no piso": int((o.pvalue <= floor + 1e-12).sum()),
            "k* (BH)": math.ceil(floor * n * n * TAU / ALPHA), "significativas": int(o.significant.astype(bool).sum())}


def evaluate():
    import pandas as pd
    import common as c
    import common_exec as ce
    rep = c.Report(NAME_REPORT, "R1-m6b — PCMCI with CMIknn (library default), 36 scenarios of the GPDC check")
    rep.p("CMIknn with tigramite's default parameters (knn 0.2; 500 permutations), workers = 1; split execution "
          "(PC1 per target, MCI in blocks); the drawn permutations differ from those of a whole execution. ParCorr and GPDC: "
          "the paper's results (tau 10).")
    folder = os.path.join(c.OUTPUT_ROOT, NAME, SUB)
    R = ce.summary(folder)
    rows = []
    for s in SCOPE:
        sc = c.scenario_key(s)
        group = next(g for g, v in GROUPS.items() if s in v)
        row = {"grupo": group, "cenario": c.O.canonical(s), "heldout": c.is_heldout(sc) if sc else None,
               "ParCorr": c.f1_model(sc, "pcmci_tigramite") if sc else np.nan,
               "GPDC": c.f1_model(sc, "pcmci_gpdc") if sc else np.nan}
        r = R[(R["cenario"] == s) & (R["model"] == MODEL[1])] if not R.empty else R
        if len(r) and r.status.iloc[0] == "ok":
            row["CMIknn"] = ce.f1(r.TP.iloc[0], r.FP.iloc[0], r.FN.iloc[0])
            row["tempo"] = r.duration.iloc[0]
            row.update(_floor(os.path.join(folder, s, MODEL[1])))
        else:
            row["CMIknn"] = np.nan
            row["status"] = "não concluído"
        rows.append(row)
    D = pd.DataFrame(rows)
    rep.csv("por_cenario.csv", D)
    ok = D[D["CMIknn"].notna()]
    rep.p(f"Finished: {len(ok)} of {len(SCOPE)}. Not finished: {list(D.loc[D['CMIknn'].isna(), 'cenario'])}")
    body = []
    for t in ("ParCorr", "GPDC", "CMIknn"):
        ss = ok[ok[t].notna()]
        w, e, l = c.wins(ss[t], ss["ParCorr"])
        body.append([t, len(ss), c.fmt(ss[t].mean()), "—" if t == "ParCorr" else f"{w} / {e} / {l}",
                     "—" if t == "ParCorr" else c.fmt(c.wilcoxon_p(ss[t], ss["ParCorr"]))])
    rep.table(["Test (PCMCI, tau 10)", "Scenarios", "Mean F1", "Better / equal / worse than ParCorr",
               "Wilcoxon p"], body)
    ss = ok[ok["GPDC"].notna()]
    w, e, l = c.wins(ss["CMIknn"], ss["GPDC"])
    rep.p(f"CMIknn against GPDC: {w} / {e} / {l} (better / equal / worse), Wilcoxon p = "
          f"{c.fmt(c.wilcoxon_p(ss['CMIknn'], ss['GPDC']))}, on {len(ss)} scenarios.")
    body = [[g, len(v), c.fmt(v["ParCorr"].mean()), c.fmt(v["GPDC"].mean()), c.fmt(v["CMIknn"].mean())]
            for g, v in ok.groupby("grupo", sort=False)]
    rep.table(["Group", "Scenarios", "ParCorr", "GPDC", "CMIknn"], body)
    if "no piso" in ok:
        lim = ok[(ok["significativas"] == 0) & (ok["no piso"] < ok["k* (BH)"])]
        rep.p(f"p-value floor (1/{B_DEFAULT + 1}): {len(lim)} scenario(s) with no edge at all and with fewer links "
              f"at the floor than BH's k* (where the floor may have been limiting): {list(lim['cenario'])}.")
    rep.save()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "conferir":
        check()
    elif mode == "carregar":
        load()
    elif mode in ("rodar", "avaliar"):
        from run_log import execution
        with execution(NAME_REPORT, __file__):
            run() if mode == "rodar" else evaluate()
    else:
        print(__doc__)

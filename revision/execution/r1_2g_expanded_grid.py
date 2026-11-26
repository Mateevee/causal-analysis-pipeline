"""
R1-2a / R1-2b / R2-3a -- Widened grid of the baselines, run from scratch.

Request
  R1, point 2: "each baseline runs untuned with library defaults (and PCMCI with a
       non-default BH correction ...). ... Please add (a) per-benchmark tuned
       baselines (or at least a small grid), (b) a leave-one-family-out protocol ..."
  R2, point 3: "the tuning protocol is asymmetric".

Design (fixed before running)
  That of revision/PROTOCOL_expanded_baseline_grid.md (sections 3 to 5), frozen on
  2026-11-10. SHA-256 of the protocol:
    content in git, LF    : e561cb2ab911ee046d8338c100c1f26249760ef34e2a8a89b39a5ea6f1f94103
    file on disk, CRLF    : 3ee76e1694f6eafccef9ed7dc7c9101324067638df1c90c5167186a233b64c48
  If this file and the protocol differ, the protocol prevails.

  Grid rule: scale parameters = the paper's value x {1/4, 1/2, 1, 2, 4};
  CUTS+ threshold from 0.1 to 0.9; categorical = every option of the wrapper;
  full crossing within each method; plus every value of the original grid
  (r1_2a) that the rule does not generate. tau_max = 10 and ParCorr fixed.

  Reruns per scenario (306), in the order of the grid (the order breaks ties):
    PCMCI      pc_alpha {0.05; 0.1; 0.2; 0.4; 0.8}                                   5
    PCMCI+     pc_alpha {0.001; 0.0025; 0.005; 0.01; 0.02; 0.04; 0.05; 0.1}
               x collider rule {majority, conservative, none}                       24
    DYNOTEARS  lambda_W x lambda_A, each in {0.0125; 0.025; 0.05; 0.1; 0.2}
               x thresholds (W; A) {(0.075; 0.025); (0.15; 0.05); (0.3; 0.1);
               (0.6; 0.2); (1.2; 0.4)}; plus the 7 of the original grid outside the
               crossing (lambda_W = lambda_A: 0.01 with (0.3; 0.1), (0.2; 0.05) and
               (0.4; 0.2); 0.05 and 0.1 with (0.2; 0.05) and (0.4; 0.2))           132
    CUTS+      threshold {0.1 ... 0.9} x network {8; 16; 32; 64; 128} x seed {0; 1; 2} 135
    VARLiNGAM  criterion {aic, bic, hqic, fpe, none} x pruning {yes, no}            10
  The default of each method is rerun (it is in the crossing). Nothing is read from
  the paper's run or from the original grid to form the F1 values of this grid.

  Variants computed offline in mode `tabular` (they are not reruns):
    PCMCI: final alpha {0.01; 0.0125; 0.025; 0.05; 0.1; 0.2} x correction {BH,
      none}, on the original.csv of each rerun (with BH, the q-values stored in
      pvalue_corrected; without correction, the pvalue; lag >= 1 only) -- the
      computation of `offline_grid` in r1_2a.
    DYNOTEARS, CUTS+ and VARLiNGAM: native graph, plus the F-test gate, alpha
      {0.01; 0.05} x correction {none, BH}, on the results.csv of each
      rerun -- the computation of r1_2a. Empty native graph: the gate variants
      take the same empty graph (a correction of the r1_2a rule, in which they had
      no value; the nativo_vazio column allows the old rule to be redone).
    CUTS+: the F1 of a configuration is the mean over the 3 seeds; seed 0 (the
      paper's) goes beside it, in f1_semente0.
  Total: 60 + 24 + 660 + 225 + 50 = 1,019 configurations.

  Details fixed here, before any execution (the protocol does not define them):
    - Order of the 1,019 configurations (tie-break): that of the reruns above (in each
      crossing, the first parameter cited varies slowest); within each
      rerun, PCMCI: final alpha (increasing) and, for each alpha, BH before
      "none"; the others: native graph, then gate with alpha 0.01 and 0.05 and, for
      each alpha, "none" before BH. The order is written to configuracoes.csv.
    - CUTS+: a configuration has a value on a scenario only if all 3 seeds ended
      with status ok; otherwise it has no value (empty f1), and f1_semente0 holds seed
      0 if it ended. For CUTS+, nativo_vazio = empty native graph in at
      least one of the 3 seeds.
    - For the methods without a seed, f1_semente0 = f1.
    - TP/FP/FN of each rerun: those of main.validate_all (validation_summary.csv),
      as in the paper; mode `tabular` recounts them from results.csv and
      ground_truth.csv in the same folder and reports every divergence.
    - Order of the scenarios in the execution: decreasing sum of duration_s in the
      original grid on this machine (Results_Revisao/r1_2a_baselines_ajustados/run);
      it changes no result.

Usage
  python revision/execution/r1_2g_expanded_grid.py grade
  python revision/execution/r1_2g_expanded_grid.py rodar --paralelo N [--verificar] [--filtro names...]
      (one process per scenario, one thread each; resumes; --filtro last)
  python revision/execution/r1_2g_expanded_grid.py tabular [--verificar]
Output: Results_Revisao/r1_2g_grade_ampliada/ (run/ with the reruns); with
--verificar, Results_Revisao/r1_2g_grade_ampliada_verificacao/.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

NAME = "r1_2g_grade_ampliada"
ORIGINAL_GRID = "r1_2a_baselines_ajustados"

# ---------------------------------------------------------------- the grid (frozen)
BASE = {"pcmci_tigramite": "pcmci_tigramite_tau10", "pcmciplus": "pcmciplus_tau10",
        "dynotears": "dynotears_tau10", "cuts_plus": "cuts_plus_tau10", "varlingam": "varlingam_tau10"}
PCMCI_PCA = (0.05, 0.1, 0.2, 0.4, 0.8)
PCMCI_ALPHA = (0.01, 0.0125, 0.025, 0.05, 0.1, 0.2)
PCMCI_CORR = ("fdr_bh", "none")
PCMCIPLUS_PCA = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.05, 0.1)
COLLIDER = ("majority", "conservative", "none")
DYN_LAMBDA = (0.0125, 0.025, 0.05, 0.1, 0.2)
DYN_THRESHOLDS = ((0.075, 0.025), (0.15, 0.05), (0.3, 0.1), (0.6, 0.2), (1.2, 0.4))
DYN_EXTRAS = ((0.01, (0.3, 0.1)), (0.01, (0.2, 0.05)), (0.01, (0.4, 0.2)),
              (0.05, (0.2, 0.05)), (0.05, (0.4, 0.2)),
              (0.1, (0.2, 0.05)), (0.1, (0.4, 0.2)))
CUTS_THRESHOLD = tuple(round(0.1 * i, 1) for i in range(1, 10))
CUTS_NETWORK = (8, 16, 32, 64, 128)
CUTS_SEEDS = (0, 1, 2)
VL_CRITERION = ("aic", "bic", "hqic", "fpe", None)
VL_PRUNE = (True, False)
GATE_ALPHA = (0.01, 0.05)
GATE_CORR = ("none", "fdr_bh")
WITH_GATE = ("dynotears", "cuts_plus", "varlingam")

EXPECTED_RERUNS = {"pcmci_tigramite": 5, "pcmciplus": 24, "dynotears": 132, "cuts_plus": 135, "varlingam": 10}
EXPECTED_CONFIGS = {"pcmci_tigramite": 60, "pcmciplus": 24, "dynotears": 660, "cuts_plus": 225, "varlingam": 50}
DEFAULT_RERUN = {"pcmci_tigramite": "pcmci_pca0.2_tau10", "pcmciplus": "pcmciplus_pca0.01_majority_tau10",
                 "dynotears": "dynotears_lw0.05_la0.05_w0.3_a0.1_tau10",
                 "cuts_plus": "cuts_plus_th0.5_h32_s0_tau10", "varlingam": "varlingam_aic_prune_tau10"}
DEFAULT_CONFIG = {"pcmci_tigramite": "pcmci_pca0.2_alpha0.05_fdr_bh_tau10",
                  "pcmciplus": "pcmciplus_pca0.01_majority_tau10",
                  "dynotears": "dynotears_lw0.05_la0.05_w0.3_a0.1_tau10",
                  "cuts_plus": "cuts_plus_th0.5_h32_tau10", "varlingam": "varlingam_aic_prune_tau10"}


def _reruns() -> list:
    g = []

    def add(baseline, root, override, seed=None):
        name = root + (f"_s{seed}" if seed is not None else "") + "_tau10"
        g.append({"baseline": baseline, "base": BASE[baseline], "nome": name, "raiz": root,
                  "semente": seed, "override": override})

    for a in PCMCI_PCA:
        add("pcmci_tigramite", f"pcmci_pca{a}", {"pcmci_pc_alpha": a})
    for a in PCMCIPLUS_PCA:
        for r in COLLIDER:
            add("pcmciplus", f"pcmciplus_pca{a}_{r}",
                {"pcmciplus_pc_alpha": a, "pcmciplus_contemp_collider_rule": r})

    def dyn(lw, la, w, a):
        add("dynotears", f"dynotears_lw{lw}_la{la}_w{w}_a{a}",
            {"dynotears_lambda_w": lw, "dynotears_lambda_a": la, "dynotears_w_threshold": w,
             "dynotears_a_threshold": a})

    for lw in DYN_LAMBDA:
        for la in DYN_LAMBDA:
            for (w, a) in DYN_THRESHOLDS:
                dyn(lw, la, w, a)
    for lam, (w, a) in DYN_EXTRAS:
        dyn(lam, lam, w, a)
    for t in CUTS_THRESHOLD:
        for h in CUTS_NETWORK:
            for s in CUTS_SEEDS:
                add("cuts_plus", f"cuts_plus_th{t}_h{h}",
                    {"cuts_causal_threshold": t, "cuts_hidden": h, "cuts_seed": s}, seed=s)
    for cr in VL_CRITERION:
        for p in VL_PRUNE:
            add("varlingam", f"varlingam_{cr if cr is not None else 'none'}_{'prune' if p else 'noprune'}",
                {"varlingam_criterion": cr, "varlingam_prune": p})
    return g


GRID = _reruns()
SPECS = [(x["base"], x["nome"], x["override"]) for x in GRID]


def configurations() -> list:
    """The 1,019 configurations, in the order of the grid (the order breaks ties)."""
    out, seen = [], set()

    def add(x, config, kind, alpha=None, corr=None):
        out.append({"ordem": len(out) + 1, "baseline": x["baseline"], "config": config, "tipo": kind,
                    "raiz": x["raiz"], "alpha": alpha, "corr": corr,
                    "default": config == DEFAULT_CONFIG[x["baseline"]]})

    for x in GRID:
        b = x["baseline"]
        if b == "pcmci_tigramite":
            for al in PCMCI_ALPHA:
                for corr in PCMCI_CORR:
                    add(x, f"{x['raiz']}_alpha{al}_{corr}_tau10", "pcmci", al, corr)
        elif b == "pcmciplus":
            add(x, f"{x['raiz']}_tau10", "nativo")
        else:
            if (b, x["raiz"]) in seen:      # CUTS+: seeds 1 and 2 of the same configuration
                continue
            seen.add((b, x["raiz"]))
            add(x, f"{x['raiz']}_tau10", "nativo")
            for al in GATE_ALPHA:
                for corr in GATE_CORR:
                    add(x, f"{x['raiz']}_ftest{al}_{corr}_tau10", "portao", al, corr)
    return out


def _effective(ce, base: str, override: dict) -> dict:
    """Parameters the wrapper runs with: the defaults of its signature, the base configuration of the
    paper (only the keys the wrapper receives) and the overrides of the model."""
    m = ce.model(base, "x", **override)
    sig = inspect.signature(m["pipeline"]).parameters
    d = {k: v.default for k, v in sig.items() if v.default is not inspect.Parameter.empty}
    d.update({k: v for k, v in ce.cfg_exec.PIPELINE_CONFIG.items() if k in sig})
    d.update(m["config_override"])
    return d


def check_grid() -> dict:
    """Checks of mode `grade` (fails with an error if any does not hold). Returns
    {rerun configuration of the original grid: equivalent rerun of this grid}."""
    import common_exec as ce
    import r1_2a_tuned_baselines as old
    cnt = Counter(x["baseline"] for x in GRID)
    if dict(cnt) != EXPECTED_RERUNS or len(GRID) != 306:
        raise SystemExit(f"[{NAME}] grid with wrong count: {dict(cnt)} (total {len(GRID)})")
    if len({x["nome"] for x in GRID}) != len(GRID):
        raise SystemExit(f"[{NAME}] repeated rerun names")
    cfgs = configurations()
    cc = Counter(x["baseline"] for x in cfgs)
    if dict(cc) != EXPECTED_CONFIGS or len(cfgs) != 1019 or len({x["config"] for x in cfgs}) != 1019:
        raise SystemExit(f"[{NAME}] configurations with wrong count: {dict(cc)} (total {len(cfgs)})")
    for b in BASE:
        if sum(1 for x in cfgs if x["baseline"] == b and x["default"]) != 1:
            raise SystemExit(f"[{NAME}] default of {b} does not appear exactly once in the configurations")
    ef = {x["nome"]: (x["base"], _effective(ce, x["base"], x["override"])) for x in GRID}

    def matching(base, override):
        target = _effective(ce, base, override)
        return [n for n, (bb, v) in ef.items() if bb == base and v == target]

    mapping = {}
    for base, name, override in old.GRID_RERUN:
        eq = matching(base, override)
        if len(eq) != 1:
            raise SystemExit(f"[{NAME}] original configuration {name} matches {len(eq)} reruns: {eq}")
        mapping[name] = eq[0]
    if len(mapping) != 20 or len(set(mapping.values())) != 20:
        raise SystemExit(f"[{NAME}] the 20 original configurations do not match 20 distinct reruns")
    for b, base in BASE.items():
        eq = matching(base, {})
        if eq != [DEFAULT_RERUN[b]]:
            raise SystemExit(f"[{NAME}] default of {b} matches {eq}, expected {[DEFAULT_RERUN[b]]}")
    return mapping


def write_grid(folder: str, mapping: dict) -> None:
    os.makedirs(folder, exist_ok=True)
    inv = {v: k for k, v in mapping.items()}
    pd.DataFrame([{"ordem": i + 1, "baseline": x["baseline"], "modelo_base": x["base"], "reexecucao": x["nome"],
                   "config": x["raiz"] + "_tau10", "semente": x["semente"],
                   "parametros": json.dumps(x["override"], sort_keys=True),
                   "default_do_artigo": x["nome"] == DEFAULT_RERUN[x["baseline"]],
                   "equivale_na_grade_original": inv.get(x["nome"], "")}
                  for i, x in enumerate(GRID)]).to_csv(os.path.join(folder, "grade.csv"), index=False)
    pd.DataFrame(configurations()).to_csv(os.path.join(folder, "configuracoes.csv"), index=False)


def grid_mode():
    import common_exec as ce
    mapping = check_grid()
    check = "--verificar" in sys.argv
    folder = ce.output_dir(NAME + ("_verificacao" if check else ""))
    write_grid(folder, mapping)
    cnt = Counter(x["baseline"] for x in GRID)
    cc = Counter(x["baseline"] for x in configurations())
    print(f"[{NAME}] grid checked: {len(GRID)} reruns per scenario "
          f"({', '.join(f'{b} {cnt[b]}' for b in BASE)}); {sum(cc.values())} configurations "
          f"({', '.join(f'{b} {cc[b]}' for b in BASE)}).")
    print(f"[{NAME}] the 20 configurations of the original grid and the 5 defaults each match "
          f"exactly one rerun.")
    for k, v in mapping.items():
        print(f"    {k}  ->  {v}")
    print(f"[{NAME}] written: {os.path.join(folder, 'grade.csv')} and configuracoes.csv")


# ---------------------------------------------------------------- run
def _duration(pm: str):
    t = read_time(pm)
    try:
        return float(t["duration_s"]) if t else None
    except (KeyError, ValueError):
        return None


def sort_by_weight(ce, dfs: list):
    """Heaviest scenarios first: sum of duration_s of the original grid on this machine.
    A scenario without a recorded time goes first. It changes no result."""
    ref = os.path.join(ce.OUTPUT_ROOT, ORIGINAL_GRID, "run")
    weight = {}
    for d in dfs:
        ps = os.path.join(ref, d["name"])
        ds = [_duration(os.path.join(ps, m)) for m in os.listdir(ps)] if os.path.isdir(ps) else []
        ds = [x for x in ds if x is not None]
        weight[d["name"]] = sum(ds) if ds else float("inf")
    return sorted(dfs, key=lambda d: -weight[d["name"]]), weight


def run():
    import common_exec as ce
    mapping = check_grid()
    check = "--verificar" in sys.argv
    name_filter = sys.argv[sys.argv.index("--filtro") + 1:] if "--filtro" in sys.argv else None
    base = ce.output_dir(NAME + ("_verificacao" if check else ""))
    folder = ce.output_dir(os.path.join(NAME + ("_verificacao" if check else ""), "run"))
    write_grid(base, mapping)
    dfs, weight = sort_by_weight(ce, ce.paper_scenarios(name_filter))
    pd.DataFrame([{"ordem": i + 1, "cenario": d["name"], "duration_s_grade_original": weight[d["name"]]}
                  for i, d in enumerate(dfs)]).to_csv(os.path.join(base, "ordem_cenarios.csv"), index=False)
    print(f"[{NAME}] {len(dfs)} scenarios x {len(SPECS)} reruns -> {folder}")
    if "--paralelo" in sys.argv:
        ce.run_parallel(dfs, SPECS, folder, int(sys.argv[sys.argv.index("--paralelo") + 1]))
    else:
        ce.run_resuming(dfs, [ce.model(b, n, **o) for b, n, o in SPECS], folder)


# ---------------------------------------------------------------- tabular
def f1c(tp, fp, fn) -> float:
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def f1_set(S: set, T: set) -> float:
    return f1c(len(S & T), len(S - T), len(T - S))


def read_time(pm: str):
    p = os.path.join(pm, "time.txt")
    if not os.path.isfile(p):
        return None
    info = {}
    for ln in open(p, encoding="utf-8", errors="replace"):
        if ":" in ln:
            k, v = ln.split(":", 1)
            info.setdefault(k.strip(), v.strip())
    return info


def folder_truth(ps: str) -> set:
    """Causal pairs (directional, entity level) of the ground_truth.csv in the scenario folder."""
    gt = pd.read_csv(os.path.join(ps, "ground_truth.csv"), skipinitialspace=True)
    for col in ("source_entity", "target_entity"):
        gt[col] = gt[col].astype(str).str.strip()
    pos = gt["causal_gt"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
    return set(map(tuple, gt.loc[pos, ["source_entity", "target_entity"]].values))


def _read(p: str, **kw) -> pd.DataFrame:
    if not os.path.isfile(p):
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, **kw)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    for col in ("source_entity", "target_entity"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def predicted_pairs(pm: str) -> set:
    r = _read(os.path.join(pm, "results.csv"), skipinitialspace=True)
    if r.empty:
        return set()
    return set(zip(r["source_entity"], r["target_entity"]))


def pcmci_offline(pm: str, T: set) -> dict:
    """{(alpha, correction): F1} from the original.csv of a PCMCI execution. The computation of
    r1_2a.offline_grid: with BH, the stored q-values (pvalue_corrected, tigramite's
    family); without correction, the MCI p-value; lag >= 1 only."""
    d = _read(os.path.join(pm, "original.csv"), skipinitialspace=True)
    if d.empty or "pvalue" not in d.columns:
        return {}
    p = pd.to_numeric(d["pvalue"], errors="coerce").fillna(1.0).to_numpy()
    pc = pd.to_numeric(d["pvalue_corrected"], errors="coerce").fillna(1.0).to_numpy()
    lag = pd.to_numeric(d["lag"], errors="coerce").fillna(0).to_numpy()
    pr = list(zip(d["source_entity"], d["target_entity"]))
    out = {}
    for corr in PCMCI_CORR:
        q = pc if corr == "fdr_bh" else p
        for a in PCMCI_ALPHA:
            S = {pr[i] for i in range(len(p)) if q[i] < a and lag[i] >= 1}
            out[(a, corr)] = f1_set(S, T)
    return out


def gate_offline(pm: str, T: set, old_rule: bool = False):
    """({(alpha, correction): F1}, empty native graph?) of the post hoc F-test gate on the
    results.csv (native graph) of an execution. The computation of r1_2a.offline_grid. Empty
    native graph: the variants take the empty graph; with old_rule=True they have no value."""
    r = _read(os.path.join(pm, "results.csv"))
    if r.empty:
        if old_rule:
            return {}, True
        return {(a, corr): f1_set(set(), T) for a in GATE_ALPHA for corr in GATE_CORR}, True
    if "pvalue" not in r.columns:
        return {}, False
    p = pd.to_numeric(r["pvalue"], errors="coerce").fillna(1.0).to_numpy()
    out = {}
    for corr in GATE_CORR:
        q = multipletests(p, method="fdr_bh")[1] if corr == "fdr_bh" and len(p) else p
        for a in GATE_ALPHA:
            S = {(s, t) for s, t, qq in zip(r["source_entity"], r["target_entity"], q) if qq < a}
            out[(a, corr)] = f1_set(S, T)
    return out, False


def tabular():
    import common as c
    check = "--verificar" in sys.argv
    suf = "_verificacao" if check else ""
    mapping = check_grid()
    rel = c.Report(NAME + suf, "R1-2 — Widened grid of the baselines (tabulation)")
    run = os.path.join(c.OUTPUT_ROOT, NAME + suf, "run")
    cens = sorted(d for d in os.listdir(run) if os.path.isdir(os.path.join(run, d)))
    cfgs = configurations()
    nan = float("nan")

    # ---- one row per scenario x rerun
    E, exec_rows, cfg_rows, errors = {}, [], [], []
    for cen in cens:
        ps = os.path.join(run, cen)
        key = c.scenario_key(cen) or cen
        T = folder_truth(ps)
        pv = os.path.join(ps, "validation_summary.csv")
        vs = {r["model"]: r for _, r in pd.read_csv(pv).iterrows()} if os.path.isfile(pv) else {}
        for x in GRID:
            pm = os.path.join(ps, x["nome"])
            t = read_time(pm)
            st = "sem_execucao" if t is None else ("error" if t.get("status", "").upper() == "ERROR"
                                                  else t.get("status", "?"))
            e = {"cenario": key, "baseline": x["baseline"], "reexecucao": x["nome"], "status": st,
                 "TP": nan, "FP": nan, "FN": nan, "f1": nan,
                 "duration_s": float(t["duration_s"]) if t and "duration_s" in t else nan,
                 "confere_recontagem": ""}
            if st == "ok":
                S = predicted_pairs(pm)
                mine = (len(S & T), len(S - T), len(T - S))
                v = vs.get(x["nome"])
                if v is not None and v["status"] == "ok":
                    tp, fp, fn = int(v["TP"]), int(v["FP"]), int(v["FN"])
                    e["confere_recontagem"] = (tp, fp, fn) == mine
                else:
                    tp, fp, fn = mine
                    e["confere_recontagem"] = "sem validation_summary"
                e.update(TP=tp, FP=fp, FN=fn, f1=f1c(tp, fp, fn))
                if x["baseline"] == "pcmci_tigramite":
                    e["_var"] = pcmci_offline(pm, T)
                elif x["baseline"] in WITH_GATE:
                    e["_var"], e["_vazio"] = gate_offline(pm, T)
            else:
                errors.append({"cenario": key, "baseline": x["baseline"], "reexecucao": x["nome"],
                               "status": st, "erro": (t or {}).get("error", "")})
            E[(cen, x["nome"])] = e
            exec_rows.append({k: v for k, v in e.items() if not k.startswith("_")})

        # ---- the 1,019 configurations of the scenario
        for g in cfgs:
            b, root = g["baseline"], g["raiz"]
            names = ([f"{root}_s{s}_tau10" for s in CUTS_SEEDS] if b == "cuts_plus" else [f"{root}_tau10"])
            es = [E[(cen, n)] for n in names]

            def value(e):
                if e["status"] != "ok":
                    return nan
                if g["tipo"] == "nativo":
                    return e["f1"]
                return e["_var"].get((g["alpha"], g["corr"]), nan)

            vals = [value(e) for e in es]
            f = float(np.mean(vals)) if not any(np.isnan(v) for v in vals) else nan
            cfg_rows.append({"cenario": key, "baseline": b, "config": g["config"], "f1": f,
                             "f1_semente0": vals[0],
                             "nativo_vazio": any(e.get("_vazio", False) for e in es)})

    X = pd.DataFrame(exec_rows)
    F = pd.DataFrame(cfg_rows)
    rel.csv("f1_por_execucao.csv", X)
    rel.csv("f1_por_config.csv", F)
    rel.csv("erros.csv", pd.DataFrame(errors, columns=["cenario", "baseline", "reexecucao", "status", "erro"]))
    keys = sorted(X["cenario"].unique())
    dev = [s for s in keys if not c.is_heldout(s)]
    ho = [s for s in keys if c.is_heldout(s)]
    rel.p(f"{len(keys)} scenarios ({len(dev)} development, {len(ho)} held-out) x {len(GRID)} "
          f"reruns = {len(X)} executions; {len(cfgs)} configurations per scenario."
          + (" **CHECK (test on a few scenarios; not a result).**" if check else ""))
    rel.data.update(cenarios=len(keys), dev=len(dev), heldout=len(ho), execucoes=len(X))

    # ---- 1. executions ok and with errors
    rel.h("## 1. Executions ok and with errors")
    body = []
    for b in BASE:
        G = X[X.baseline == b]
        cnt = G["status"].value_counts().to_dict()
        body.append([c.LABEL.get(b, b), len(G), cnt.get("ok", 0), len(G) - cnt.get("ok", 0),
                     "; ".join(f"{k}: {v}" for k, v in sorted(cnt.items()) if k != "ok") or "—"])
    rel.table(["Method", "Executions", "ok", "Not ok", "Status of the not ok"], body)
    nok = X[X.status != "ok"]
    rel.data["nao_ok"] = int(len(nok))
    if len(nok):
        rel.p("")
        rel.p("Not ok per rerun configuration (scenarios):")
        for (b, n), G in nok.groupby(["baseline", "reexecucao"], sort=False):
            rel.p(f"- {n}: {len(G)} — " + ", ".join(sorted(G["cenario"])))
        E_ = pd.DataFrame(errors)
        rel.p("")
        rel.p("Error messages (first line; count):")
        for (b, m), G in E_.groupby(["baseline", E_["erro"].str.slice(0, 160)]):
            rel.p(f"- {c.LABEL.get(b, b)} ({len(G)}): {m}")
    div = X[(X.status == "ok") & (X.confere_recontagem.astype(str) == "False")]
    no_vs = X[(X.status == "ok") & (X.confere_recontagem.astype(str) == "sem validation_summary")]
    rel.p("")
    rel.p(f"Recount of TP/FP/FN from results.csv and ground_truth.csv: {len(div)} divergence(s) from "
          f"validation_summary.csv in {int((X.status == 'ok').sum())} ok executions; {len(no_vs)} without a row in "
          f"validation_summary.csv (for those the recount holds).")
    rel.data.update(divergencias_recontagem=int(len(div)), sem_validation_summary=int(len(no_vs)))
    if len(div):
        rel.csv("divergencias_recontagem.csv", div)

    # ---- 2. rerun default against the paper
    rel.h("## 2. Rerun default against the paper's run (TP/FP/FN)")
    L = c.rows()
    rows = []
    for cen in cens:
        k = c.scenario_key(cen)
        for b in BASE:
            e = E[(cen, DEFAULT_RERUN[b])]
            art = L[k].get(b) if k is not None else None
            row = {"cenario": k or cen, "baseline": b, "no_artigo": art is not None,
                   "tp_artigo": art["tp"] if art else nan, "fp_artigo": art["fp"] if art else nan,
                   "fn_artigo": art["fn"] if art else nan,
                   "f1_artigo": f1c(art["tp"], art["fp"], art["fn"]) if art else nan,
                   "status_reexecucao": e["status"], "tp": e["TP"], "fp": e["FP"], "fn": e["FN"], "f1": e["f1"]}
            if art is None and e["status"] != "ok":
                row["igual"] = "ausente nos dois"
            elif art is None or e["status"] != "ok":
                row["igual"] = "só num dos dois" if k is not None else "cenário fora das runs do artigo desta máquina"
            else:
                row["igual"] = (art["tp"], art["fp"], art["fn"]) == (e["TP"], e["FP"], e["FN"])
            rows.append(row)
    D = pd.DataFrame(rows)
    rel.csv("default_contra_artigo.csv", D)
    body = []
    for b in BASE:
        G = D[D.baseline == b]
        comp = G[G["igual"].isin([True, False])]
        body.append([c.LABEL.get(b, b), len(comp), int((comp["igual"] == True).sum()),  # noqa: E712
                     int((comp["igual"] == False).sum()), int((G["igual"] == "ausente nos dois").sum()),  # noqa: E712
                     int((G["igual"] == "só num dos dois").sum()),
                     int((G["igual"] == "cenário fora das runs do artigo desta máquina").sum())])
    rel.table(["Method", "Scenarios compared", "Equal TP/FP/FN", "Different", "No value in either",
               "Value in only one of the two", "Scenario without a paper run here"], body)
    dif = D[D["igual"].isin([False, "só num dos dois"])]
    rel.data["default_diferente"] = int(len(dif))
    if len(dif):
        rel.p("")
        rel.p("All differences (paper → rerun):")
        rel.table(["Scenario", "Method", "TP/FP/FN paper", "F1 paper", "Status", "TP/FP/FN rerun", "F1"],
                  [[r.cenario, c.LABEL.get(r.baseline, r.baseline),
                    f"{c.fmt(r.tp_artigo, 0)}/{c.fmt(r.fp_artigo, 0)}/{c.fmt(r.fn_artigo, 0)}", c.fmt(r.f1_artigo),
                    r.status_reexecucao, f"{c.fmt(r.tp, 0)}/{c.fmt(r.fp, 0)}/{c.fmt(r.fn, 0)}", c.fmt(r.f1)]
                   for r in dif.itertuples()])

    # ---- 3. the 20 original configurations against the original grid on this machine
    rel.h("## 3. The 20 configurations of the original grid against the original grid on this machine (TP/FP/FN)")
    old_run = os.path.join(c.OUTPUT_ROOT, ORIGINAL_GRID, "run")
    rows = []
    for cen in cens:
        pa = c.folder_in(old_run, cen)
        pva = os.path.join(pa, "validation_summary.csv") if pa else ""
        va = {r["model"]: r for _, r in pd.read_csv(pva).iterrows()} if os.path.isfile(pva) else {}
        for old, new in mapping.items():
            e, a = E[(cen, new)], va.get(old)
            sa = a["status"] if a is not None else "ausente"
            ta = (int(a["TP"]), int(a["FP"]), int(a["FN"])) if sa == "ok" else None
            tn = (int(e["TP"]), int(e["FP"]), int(e["FN"])) if e["status"] == "ok" else None
            rows.append({"cenario": c.scenario_key(cen) or cen, "baseline": e["baseline"], "config_original": old,
                         "reexecucao": new, "status_original": sa, "tp_fp_fn_original": ta,
                         "status_novo": e["status"], "tp_fp_fn_novo": tn,
                         "igual": (ta == tn) if (ta is not None or tn is not None) else "sem valor nos dois"})
    Oz = pd.DataFrame(rows)
    rel.csv("originais_contra_grade_original.csv", Oz)
    body = []
    for b in BASE:
        G = Oz[Oz.baseline == b]
        body.append([c.LABEL.get(b, b), len(G), int((G["igual"] == True).sum()),  # noqa: E712
                     int((G["igual"] == False).sum()), int((G["igual"] == "sem valor nos dois").sum())])  # noqa: E712
    rel.table(["Method", "Comparisons", "Equal", "Different", "No value in either (error in both)"], body)
    dif = Oz[Oz["igual"] == False]  # noqa: E712
    rel.data["originais_diferentes"] = int(len(dif))
    if len(dif):
        rel.p("")
        rel.p("All differences:")
        rel.table(["Scenario", "Original configuration", "Status", "TP/FP/FN original", "New status",
                   "TP/FP/FN new"],
                  [[r.cenario, r.config_original, r.status_original, r.tp_fp_fn_original, r.status_novo,
                    r.tp_fp_fn_novo] for r in dif.itertuples()])

    # ---- 4. CUTS+: spread between the 3 seeds
    rel.h("## 4. CUTS+: spread of F1 between the 3 seeds (native graph)")
    rows = []
    for g in cfgs:
        if g["baseline"] != "cuts_plus" or g["tipo"] != "nativo":
            continue
        dd = {"dev": [], "ho": []}
        for cen in cens:
            fs = [E[(cen, f"{g['raiz']}_s{s}_tau10")] for s in CUTS_SEEDS]
            if all(e["status"] == "ok" for e in fs):
                k = c.scenario_key(cen) or cen
                dd["ho" if c.is_heldout(k) else "dev"].append(float(np.std([e["f1"] for e in fs], ddof=1)))
        rows.append({"config": g["config"], "default": g["default"],
                     "cenarios_dev": len(dd["dev"]), "desvio_medio_dev": np.mean(dd["dev"]) if dd["dev"] else nan,
                     "desvio_max_dev": np.max(dd["dev"]) if dd["dev"] else nan,
                     "cenarios_heldout": len(dd["ho"]), "desvio_medio_heldout": np.mean(dd["ho"]) if dd["ho"] else nan,
                     "desvio_max_heldout": np.max(dd["ho"]) if dd["ho"] else nan})
    Sd = pd.DataFrame(rows)
    rel.csv("cuts_plus_sementes.csv", Sd)
    rel.p("Sample standard deviation (n − 1) of F1 between seeds 0, 1 and 2, per scenario; mean and maximum over the "
          "scenarios. One row per configuration in `cuts_plus_sementes.csv`.")
    d0 = Sd[Sd["default"]].iloc[0]
    rel.table(["Configuration", "Mean SD dev", "Max. SD dev", "Mean SD held-out", "Max. SD held-out"],
              [[d0["config"] + " (default)", c.fmt(d0.desvio_medio_dev), c.fmt(d0.desvio_max_dev),
                c.fmt(d0.desvio_medio_heldout), c.fmt(d0.desvio_max_heldout)],
               ["mean of the 45 configurations", c.fmt(Sd.desvio_medio_dev.mean()), c.fmt(Sd.desvio_max_dev.max()),
                c.fmt(Sd.desvio_medio_heldout.mean()), c.fmt(Sd.desvio_max_heldout.max())]])

    # ---- 5. mean F1 of each configuration
    rel.h("## 5. Mean F1 of each configuration, on development and on held-out")
    order = {g["config"]: g for g in cfgs}
    rows = []
    for (b, cfg), G in F.groupby(["baseline", "config"], sort=False):
        v = G.dropna(subset=["f1"]).set_index("cenario")
        vd, vh = v[v.index.isin(dev)], v[v.index.isin(ho)]
        rows.append({"ordem": order[cfg]["ordem"], "baseline": b, "config": cfg, "default": order[cfg]["default"],
                     "cenarios_dev": len(vd), "f1_dev": vd["f1"].mean() if len(vd) else nan,
                     "f1_semente0_dev": vd["f1_semente0"].mean() if len(vd) else nan,
                     "cenarios_heldout": len(vh), "f1_heldout": vh["f1"].mean() if len(vh) else nan,
                     "f1_semente0_heldout": vh["f1_semente0"].mean() if len(vh) else nan})
    Md = pd.DataFrame(rows).sort_values("ordem")
    rel.csv("f1_medio_por_config.csv", Md)
    rel.p("Mean over the scenarios on which the configuration has a value (the number of scenarios is on each row; "
          "means over different sets are not comparable). Full table in `f1_medio_por_config.csv`. "
          "Below, per method: the default, the 5 with the highest mean F1 on development and the 5 with the highest "
          "mean F1 on held-out (descriptive reading; the selection of analyses A1 to A3 is done in `r1_2f`).")
    for b in BASE:
        G = Md[Md.baseline == b]
        sel = pd.concat([G[G["default"]], G.sort_values("f1_dev", ascending=False, kind="stable").head(5),
                         G.sort_values("f1_heldout", ascending=False, kind="stable").head(5)])
        sel = sel[~sel.index.duplicated()]
        rel.p("")
        rel.p(f"**{c.LABEL.get(b, b)}** ({len(G)} configurations)")
        rel.table(["Configuration", "Dev scenarios", "F1 dev", "Held-out scenarios", "F1 held-out"],
                  [[r.config + (" (default)" if r.default else ""), r.cenarios_dev, c.fmt(r.f1_dev),
                    r.cenarios_heldout, c.fmt(r.f1_heldout)] for r in sel.itertuples()])

    # ---- 6. the offline computation, applied to the paper's run, reproduces the f1_por_config.csv of r1_2a
    rel.h("## 6. Check of the offline computation on the paper's run (against the `f1_por_config.csv` of r1_2a)")
    pa = os.path.join(c.OUTPUT_ROOT, ORIGINAL_GRID, "f1_por_config.csv")
    if not os.path.isfile(pa):
        rel.p("Not done: the `f1_por_config.csv` of r1_2a is missing.")
    else:
        A = pd.read_csv(pa)
        ref = {(r.cenario, r.config): r.f1 for r in A.itertuples()}
        n = n_equal = 0
        difs = []
        for s in sorted(set(A["cenario"]) & set(L)):
            T, _ = c.truth(s)
            mine = {}
            if "pcmci_tigramite" in L[s]:
                for (a, corr), f in pcmci_offline(os.path.join(c.folder(s, "pcmci_tigramite"),
                                                               "pcmci_tigramite_tau10"), T).items():
                    mine[f"pcmci_alpha{a}_{corr}"] = f
            for b in WITH_GATE:
                if b in L[s]:
                    v, _ = gate_offline(os.path.join(c.folder(s, b), f"{b}_tau10"), T, old_rule=True)
                    for (a, corr), f in v.items():
                        mine[f"{b}_ftest{a}_{corr}"] = f
            for cfg, f in mine.items():
                if (s, cfg) in ref:
                    n += 1
                    if abs(ref[(s, cfg)] - f) < 1e-12:
                        n_equal += 1
                    else:
                        difs.append((s, cfg, ref[(s, cfg)], f))
            for (ss, cfg) in ref:
                if ss == s and ("_alpha" in cfg or "_ftest" in cfg) and cfg not in mine:
                    difs.append((s, cfg, ref[(ss, cfg)], nan))
        rel.p(f"Offline variants (PCMCI: final alpha x correction; DYNOTEARS, CUTS+ and VARLiNGAM: F-test "
              f"gate, by the old rule) recomputed on the paper's run with the functions of this script: "
              f"{n_equal} of {n} values equal to those of the `f1_por_config.csv` of r1_2a "
              f"({A['cenario'].nunique()} scenarios in that file); {len(difs)} difference(s).")
        rel.data.update(offline_comparados=n, offline_iguais=n_equal, offline_diferencas=len(difs))
        for d in difs[:40]:
            rel.p(f"- {d[0]} · {d[1]}: r1_2a {c.fmt(d[2], 6)} · here {c.fmt(d[3], 6)}")
    rel.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("grade", "rodar", "tabular"):
        raise SystemExit(__doc__)
    with execution(NAME, __file__):
        {"grade": grid_mode, "rodar": run, "tabular": tabular}[mode]()

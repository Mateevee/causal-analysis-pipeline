"""
R1-2c -- New benchmark after the freeze: the SINGLE RUN.

Design: revision/PROTOCOL_new_benchmark_R1-2c.md, part A frozen on 2026-11-13
  (CRLF c4ed4b4dd13b43fb728b4a05a27acce7da1f364d7ad46a410cbaaf85ff75ed03), and annex B
  (revision/ANNEX_B_R1-2c.md, hash in NOTAS.md). If this script and the protocol differ, the protocol prevails.
  Previous version of this file (2026-11-13, refused to run): versoes_anteriores/r1_2c_benchmark_novo_2026-11-13.py.

Sets (section 3 of the protocol)
  test        block (i), disk: wind tunnel, walks 10 to 16 and loads_hatch_mix (10 runs); light tunnel,
              color_mix; DREAM4 size 100 (5 networks); DREAM3 size 100 (5 networks).
              block (ii), external regime (annex B): CausalRivers East Germany, 30 root_cause_5 + 30 close_5
              (r1_2c_causalrivers.py).
  validation  wind tunnel, walks 1 and 4 to 9; DREAM3 size 50. Only for front B; never enters the R1-2c
              report (already used by the R1-3b preview, Results_Revisao/r1_3b_previa_validacao/).
Methods (section 5; tau_max = 10): patch_tau10 (the submitted one: main comparison), patch_tau10_final, the R1-3b
  variant (patch_tau10_graded; revision/variants/r1_3b_graded.py, hash 90a38422...) and the paper's 5 baselines.
  Plus the EXPLORATORY variant patch_semS7_tau10 (without Stage 7; decision logged in NOTAS before running), which
  runs in a final pass, after all blocks.

Modes
  carregar [--validacao]  only the count allowed before the run (series, points, pairs and ground-truth edges).
  rodar [--blocos B ...] [--sem-exploratoria]
                          single run, sequential, default threads, resumed via time.txt. Block order:
                          chambers -> rivers -> dream4 -> dream3 (the cheap ones first). In each block: the paper's
                          6 methods without the guard swap; then the final and the graded ones with it (R1-3b path,
                          with diagnostics). After all blocks, the exploratory one.
  exploratoria [--blocos B ...]
                          only the variant without Stage 7.
  --fumaca                smoke check on one VALIDATION scenario (walk 1), in
                          Results_Revisao/r1_2c_benchmark_novo_verificacao/ (never the test).
  avaliar [--raizes P ...] T1 to T3, tests and counts of section 7 (plus the secondary tests added on
                          2026-11-14). --raizes merges the run/ folders of more than one machine; the first root takes
                          precedence per scenario, and the report says which root each scenario came from.
                          Since 2026-11-18: the scenarios in DECLARED_EXCLUSIONS leave the expected set and are listed in
                          the report with the reason (author's decision; there is no other change to the analysis).

Usage (from the repository root):  python revision/execution/r1_2c_new_benchmark.py carregar|rodar|exploratoria|avaliar
Output: Results_Revisao/r1_2c_benchmark_novo/
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common_exec as ce  # noqa: E402

NAME = "r1_2c_benchmark_novo" + ("_verificacao" if "--fumaca" in sys.argv else "")
FROZEN_HASH = "d0473069ebef1978"

# ---- section 3 of the protocol ----------------------------------------------------------------------
WT_TEST = [f"actuators_random_walk_{i}" for i in range(10, 17)] + \
          [f"loads_hatch_mix_{v}_run_{i}" for v in ("fast", "slow") for i in range(1, 6)]
LT_TEST = ["color_mix"]
WT_VALIDATION = ["actuators_random_walk_1"] + [f"actuators_random_walk_{i}" for i in range(4, 10)]
DREAM3_TEST, DREAM3_VALIDATION = (100,), (50,)
DREAM4_TEST = dict(sizes=[100], networks=[1, 2, 3, 4, 5], mode="concatenated")
CAUSAL_CHAMBER_N_POINTS = 500   # the one in main_heldout.py
BLOCKS = ("chambers", "rivers", "dream4", "dream3")
EXPECTED = {"chambers": 18, "rivers": 60, "dream4": 5, "dream3": 5}
# Deviation from the protocol, author's decision on the runner (2026-11-17, ~22:58; NOTAS_outra_maquina.md), for time
# only and without seeing results; the agent there recommended not skipping. Handling in avaliar decided by the author
# on 2026-11-18: declared exclusions, instead of an INCOMPLETE report.
DECLARED_EXCLUSIONS = {
    "benchmark_dream3_size100_Yeast2": "skipped for all methods by the author's decision (2026-11-17 ~22:58), for "
                                       "time only: PCMCI+ exceeded 60 min on the 3 DREAM3 networks it ran on and "
                                       "this network has 389 true edges",
    "benchmark_dream3_size100_Yeast3": "skipped for all methods by the author's decision (2026-11-17 ~22:58), for "
                                       "time only: PCMCI+ exceeded 60 min on the 3 DREAM3 networks it ran on and "
                                       "this network has 551 true edges",
}

# ---- section 5 of the protocol ----------------------------------------------------------------------
SUBMITTED = "patch_tau10"
FINAL = "patch_tau10_final"
VARIANT_R1_3B = "patch_tau10_graded"
NO_S7 = "patch_semS7_tau10"
ANNEX_B = "causal_rivers_east_germany"
BASELINES = ["pcmci_tigramite_tau10", "pcmciplus_tau10", "dynotears_tau10", "cuts_plus_tau10", "varlingam_tau10"]
MAIN_METHODS = [SUBMITTED, FINAL, VARIANT_R1_3B] + BASELINES
ALL_METHODS = MAIN_METHODS + [NO_S7]


def arg(name, default=None):
    if name not in sys.argv:
        return default
    out = []
    for a in sys.argv[sys.argv.index(name) + 1:]:
        if a.startswith("--"):
            break
        out.append(a)
    return out


def block_scenarios(b: str) -> list:
    from Data import (generate_causal_chamber_benchmark_scenarios, generate_dream3_benchmark_scenarios,
                      generate_dream4_benchmark_scenarios)
    if b == "chambers":
        exps = [("wt_walks_v1", e) for e in WT_TEST] + [("lt_walks_v1", e) for e in LT_TEST]
        out = generate_causal_chamber_benchmark_scenarios(experiments=exps, base_dir="./external_data/causal_chamber",
                                                          n_points=CAUSAL_CHAMBER_N_POINTS)
    elif b == "dream3":
        out = generate_dream3_benchmark_scenarios(sizes=DREAM3_TEST, base_dir="./external_data/dream3")
    elif b == "dream4":
        out = generate_dream4_benchmark_scenarios(base_dir="./external_data/dream4", **DREAM4_TEST)
    elif b == "rivers":
        import r1_2c_causalrivers as CR
        out = CR.scenarios()
    elif b == "chambers9":
        # start of the single run (2026-11-15, at the author's request, logged in NOTAS before running): 9 of the 18
        # Causal Chambers scenarios, by a fixed rule -- the 18 names in alphabetical order, positions 0, 2, ..., 16.
        # They run in the same run/ folder; the full "chambers" block later skips them (resumed via time.txt).
        return sorted(block_scenarios("chambers"), key=lambda d: d["name"])[0:18:2]
    elif b == "fumaca":
        # smoke check: one VALIDATION scenario (already run in the R1-3b preview); never the test
        out = [d for d in scenarios("validacao") if d["name"].endswith("actuators_random_walk_1")]
        return out
    else:
        raise ValueError(b)
    out = list(out)
    assert len(out) == EXPECTED[b], (b, len(out))
    return out


def scenarios(subset: str = "teste") -> list:
    if subset == "teste":
        return [d for b in BLOCKS for d in block_scenarios(b)]
    if subset == "validacao":
        from Data import generate_causal_chamber_benchmark_scenarios, generate_dream3_benchmark_scenarios
        exps = [("wt_walks_v1", e) for e in WT_VALIDATION]
        out = generate_causal_chamber_benchmark_scenarios(experiments=exps, base_dir="./external_data/causal_chamber",
                                                          n_points=CAUSAL_CHAMBER_N_POINTS)
        out += generate_dream3_benchmark_scenarios(sizes=DREAM3_VALIDATION, base_dir="./external_data/dream3")
        return out
    raise ValueError(subset)


def block(name: str) -> str:
    return "externo" if "causal_rivers" in name else "disco"


def family(name: str) -> str:
    for k, v in (("causal_chamber_wt_loads_hatch_mix", "Causal Chambers, túnel de vento, loads_hatch_mix"),
                 ("causal_chamber_wt", "Causal Chambers, túnel de vento, passeios"),
                 ("causal_chamber_lt", "Causal Chambers, túnel de luz"),
                 ("root_cause_5", "CausalRivers East Germany, root_cause_5"),
                 ("close_5", "CausalRivers East Germany, close_5"),
                 ("dream3", "DREAM3"), ("dream4", "DREAM4")):
        if k in name:
            size = next((t for t in ("size100", "size50", "size10") if t in name), "")
            return f"{v} tamanho {size[4:]}" if size else v
    return "outro"


# ---------------------------------------------------------------------------------------------- carregar
def load():
    subset = "validacao" if "--validacao" in sys.argv else "teste"
    rows = []
    for d in scenarios(subset):
        df = d["df"]
        gt = d.get("gt")
        n_pos = n_pair = None
        if isinstance(gt, pd.DataFrame) and "causal_gt" in gt.columns:
            n_pair = int(len(gt))
            n_pos = int(gt["causal_gt"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).sum())
        rows.append({"cenario": d["name"], "bloco": block(d["name"]), "familia": family(d["name"]),
                     "series": int(df["Name"].nunique()), "pontos": int(df["Date"].nunique()),
                     "pares_verdade": n_pair, "arestas_verdade": n_pos})
    D = pd.DataFrame(rows)
    base = ce.output_dir(NAME)
    D.to_csv(os.path.join(base, f"carregamento_{subset}.csv"), index=False)
    print(D.to_string(index=False))
    print(f"\n{len(D)} scenarios ({subset}). Count only; no model was run.")


# ---------------------------------------------------------------------------------------------- rodar
def _check():
    import run_log
    h = run_log.method_code_hash()
    if h != FROZEN_HASH:
        print(f"WARNING: method_code_hash {h} != {FROZEN_HASH}, the hash of the frozen method code (the released files are English translations of the frozen ones; see revision/README.md)", file=sys.stderr)


def _blocks():
    if "--fumaca" in sys.argv:
        return ["fumaca"]
    return arg("--blocos") or BLOCKS


def _log(msg):
    ce.log_progress(os.path.join(ce.OUTPUT_ROOT, NAME, "progresso.log"), msg)


def run():
    import r1_3b_graded_transitions as R3
    from variants import r1_3b_graded as g
    _check()
    folder = ce.output_dir(os.path.join(NAME, "run"))
    for b in _blocks():
        dfs = block_scenarios(b)
        _log(f"block {b}: {len(dfs)} scenarios; the paper's methods")
        ce.run_resuming(dfs, ce.models([SUBMITTED] + BASELINES), folder)
        _log(f"block {b}: final and graded (guard swap installed)")
        orig = g.install()
        try:
            R3._run_list(ce, g, [(d, v, "padrao") for d in dfs for v in R3.VERSIONS], folder,
                         os.path.join(folder, "diag.jsonl"))
        finally:
            g.uninstall(orig)
        _log(f"block {b}: done")
    if "--sem-exploratoria" not in sys.argv:
        exploratory()


def exploratory():
    from Pipeline.penalties import UniformCausalPenalty
    _check()
    folder = ce.output_dir(os.path.join(NAME, "run"))
    m = ce.model("patch_tau10", NO_S7, causal_penalty_fn=UniformCausalPenalty())
    for b in _blocks():
        _log(f"exploratory (without Stage 7), block {b}")
        ce.run_resuming(block_scenarios(b), [m], folder)
        _log(f"exploratory, block {b}: done")


# ---------------------------------------------------------------------------------------------- avaliar
def _test_table(Fb, c, m, opponents, holm=True):
    p = {}
    for b in opponents:
        if b in Fb and m in Fb:
            x = Fb[[m, b]].dropna()
            p[b] = (c.wilcoxon_p(x[m], x[b]), len(x), x[m].mean() - x[b].mean(), c.wins(x[m], x[b]))
    ph = c.holm({k: v[0] for k, v in p.items()}) if holm else {k: np.nan for k in p}
    lines = ["| Against | Paired scenarios | Mean F1 difference | Better/equal/worse | p | Holm p |",
             "|---|---|---|---|---|---|"]
    for b, (pv, n, dm, v) in p.items():
        lines.append(f"| {b} | {n} | {dm:+.3f} | {v[0]}/{v[1]}/{v[2]} | {pv:.4f} | "
                     f"{ph[b]:.4f} |" if holm else f"| {b} | {n} | {dm:+.3f} | {v[0]}/{v[1]}/{v[2]} | {pv:.4f} | — |")
    return lines, p


def _time_txt(model_folder):
    """Fields of time.txt (format of main._save_time), or None if it does not exist."""
    f = os.path.join(model_folder, "time.txt")
    if not os.path.exists(f):
        return None
    d = {}
    for line in open(f, encoding="utf-8", errors="replace"):
        if ":" in line:
            k, v = line.split(":", 1)
            d.setdefault(k.strip(), v.strip())
    return d


def evaluate(folder=None):
    import common as c
    roots = arg("--raizes") or [folder or os.path.join(ce.OUTPUT_ROOT, NAME, "run")]
    base = os.path.join(ce.OUTPUT_ROOT, NAME)
    parts, where = [], {}
    for r in roots:
        x = ce.summary(r)
        if len(x):
            x = x[~x["cenario"].isin(where)]
            for s in x["cenario"].unique():
                where[s] = os.path.join(r, s)
            parts.append(x)
    R = pd.concat(parts, ignore_index=True)
    R = R[~R["cenario"].isin(DECLARED_EXCLUSIONS)].reset_index(drop=True)
    R["status"] = R["status"].astype(str)
    ok = R["status"] == "ok"
    R["f1"] = [ce.f1(a, b, d) if s else np.nan for a, b, d, s in zip(R.TP, R.FP, R.FN, ok)]
    R["precisao"] = np.where(ok & ((R.TP + R.FP) > 0), R.TP / (R.TP + R.FP).replace(0, np.nan), np.nan)
    R["revocacao"] = np.where(ok & ((R.TP + R.FN) > 0), R.TP / (R.TP + R.FN).replace(0, np.nan), np.nan)
    R["bloco"] = R["cenario"].map(block)
    R["familia"] = R["cenario"].map(family)
    R.to_csv(os.path.join(base, "T1_por_cenario.csv"), index=False)
    F = R.pivot_table(index="cenario", columns="model", values="f1")
    n_scen, n_ok = R["cenario"].nunique(), int(ok.sum())
    expected_scen = sum(EXPECTED.values()) - len(DECLARED_EXCLUSIONS)
    expected = expected_scen * len(ALL_METHODS)
    txt = ["# R1-2c — new benchmark", "",
           "Protocol: `revision/PROTOCOL_new_benchmark_R1-2c.md` (part A) and `revision/ANNEX_B_R1-2c.md`.", "",
           f"Scenarios with output: {n_scen} of {expected_scen} (the protocol's {sum(EXPECTED.values())} minus "
           f"{len(DECLARED_EXCLUSIONS)} declared exclusions); runs with a result: {n_ok} of {expected} "
           f"(8 methods + the exploratory one).", ""]
    if DECLARED_EXCLUSIONS:
        txt += ["**Excluded scenarios (declared deviation from the protocol):**", ""]
        txt += [f"- `{s}`: {reason}." for s, reason in DECLARED_EXCLUSIONS.items()] + [""]
    txt += ["Origin of the scenarios (`--raizes`; the first root takes precedence per scenario):", ""]
    for r in roots:
        n_r = sum(1 for s in R["cenario"].unique() if os.path.dirname(where[s]) == r)
        txt.append(f"- `{r}`: {n_r} scenarios.")
    txt.append("")
    # Count from the time.txt files (fix of 2026-11-18): runs with ERROR do not enter validation_summary,
    # so the old count (ok + error in the summary) never added up and the list of errors was not printed.
    missing, no_res = [], []
    for s in sorted(R["cenario"].unique()):
        for m in ALL_METHODS:
            d = _time_txt(os.path.join(where[s], m))
            if d is None:
                missing.append((s, m))
            elif d.get("status") != "ok":
                no_res.append((s, m, d.get("status", "?"), d.get("error", "")))
    txt.append(f"Runs with time.txt: {expected - len(missing)} of {expected}; with a result: {n_ok}; with status "
               f"other than ok: {len(no_res)}.")
    txt.append("")
    if n_scen < expected_scen or missing or n_ok + len(no_res) != expected:
        txt += ["**INCOMPLETE: the numbers below are not the R1-2c result.**", ""]
        if missing:
            txt += ["Without time.txt: " + ", ".join(f"{s}/{m}" for s, m in missing), ""]
    if no_res:
        txt += ["**Runs without a result (status in time.txt other than ok), left out of the method's mean:**", "",
                "| Scenario | Method | Status | Error (beginning) |", "|---|---|---|---|"]
        txt += [f"| {s} | {m} | {st} | {er[:140].replace('|', '/')} |" for s, m, st, er in no_res] + [""]
    T2 = R.pivot_table(index="familia", columns="model", values="f1", aggfunc="mean")
    T2.insert(0, "scenarios", R.groupby("familia")["cenario"].nunique())
    txt += ["## T2 — mean F1 per family", "", T2.round(3).to_markdown(), ""]
    no_result = R[~ok].groupby("model").size()
    if len(no_result):
        txt += ["Runs without a result (left out of the method's mean): " +
                ", ".join(f"{m} {n}" for m, n in no_result.items()), ""]
    for block_name, filt in (("(iii) all", None), ("(i) disk", "disco"), ("(ii) external", "externo")):
        scen = sorted(R["cenario"].unique()) if filt is None else sorted(R.loc[R.bloco == filt, "cenario"].unique())
        if not scen:
            continue
        Fb = F.loc[[x for x in scen if x in F.index]]
        txt += [f"## T3 — block {block_name}: {len(Fb)} scenarios", "", Fb.mean().round(3).to_frame("Mean F1").to_markdown(), ""]
        lin, _ = _test_table(Fb, c, SUBMITTED, BASELINES)
        txt += ["**Main test** (`patch_tau10` against each baseline; paired Wilcoxon, Holm over the 5):", ""] + lin + [""]
        for v2 in (FINAL, VARIANT_R1_3B):
            lin, _ = _test_table(Fb, c, v2, BASELINES)
            txt += [f"Secondary (added on 2026-11-14, before running): `{v2}` against each baseline (Holm over the 5):",
                    ""] + lin + [""]
        sec = []
        for v2 in (FINAL, VARIANT_R1_3B):
            if v2 in Fb and SUBMITTED in Fb:
                x = Fb[[SUBMITTED, v2]].dropna()
                sec.append(f"- Secondary: `{v2}` against `patch_tau10`: mean difference "
                           f"{x[v2].mean() - x[SUBMITTED].mean():+.3f}, better/equal/worse {c.wins(x[v2], x[SUBMITTED])}, "
                           f"Wilcoxon p {c.wilcoxon_p(x[v2], x[SUBMITTED]):.4f} ({len(x)} scenarios).")
        for opponent in (SUBMITTED, FINAL):
            if NO_S7 in Fb and opponent in Fb:
                x = Fb[[opponent, NO_S7]].dropna()
                sec.append(f"- Exploratory: `{NO_S7}` against `{opponent}`: mean difference "
                           f"{x[NO_S7].mean() - x[opponent].mean():+.3f}, better/equal/worse {c.wins(x[NO_S7], x[opponent])}, "
                           f"Wilcoxon p {c.wilcoxon_p(x[NO_S7], x[opponent]):.4f} ({len(x)} scenarios).")
        txt += sec + [""]
    floor = {}
    for s in R["cenario"].unique():
        gt = pd.read_csv(os.path.join(where[s], "ground_truth.csv"), skipinitialspace=True)
        pos = gt["causal_gt"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).sum()
        pr = pos / len(gt) if len(gt) else np.nan
        floor[s] = 2 * pr / (1 + pr) if pr == pr else np.nan
    floor = pd.Series(floor)
    floor.rename("piso_trivial").to_csv(os.path.join(base, "piso_por_cenario.csv"))
    below = {m: int((F[m].dropna() < floor[F[m].dropna().index] - 1e-9).sum()) for m in F.columns}
    main_methods = [m for m in MAIN_METHODS if m in F.columns]
    ranks = F[main_methods].rank(axis=1, ascending=False, method="average").mean()
    txt += ["## Trivial floor and mean rank (rank among the 8 main methods)", "",
            "| Method | Scenarios with a result | Below the floor | Mean rank |", "|---|---|---|---|"]
    for m in F.columns:
        txt.append(f"| {m} | {int(F[m].notna().sum())} | {below[m]} | "
                   f"{ranks[m]:.2f} |" if m in ranks else f"| {m} | {int(F[m].notna().sum())} | {below[m]} | — |")
    open(os.path.join(base, "relatorio.md"), "w", encoding="utf-8", newline="\r\n").write("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("carregar", "rodar", "exploratoria", "avaliar"):
        raise SystemExit(__doc__)
    from run_log import execution
    with execution(NAME, __file__):
        {"carregar": load, "rodar": run, "exploratoria": exploratory, "avaliar": evaluate}[mode]()

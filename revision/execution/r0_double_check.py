"""
Double check of round 2 -- stages D2, D2b, D4 and B.

Design (fixed before running)
  That of revision/PROTOCOL_double_check.md (section 3), frozen on
  2026-11-11 14:34. SHA-256 of the protocol (Portuguese original; the translation does not match it, see
  revision/README.md, Notes):
    file on disk, CRLF : 8336ca3b7da6b1203e12b6c8d3b6eb3080f96e1204553ba89430eda0a7999b09
    content, LF        : 2ad5837f665c2dece186e518fbf60e9ac7c15a1db308c4e8fe0ee9e18ce1d076
  If this file and the protocol differ, the protocol prevails.

  Nothing of the method changes. The specifications come from r1_2g_expanded_grid.py (SPECS),
  unchanged; the names of the output folders are the same as there, for the
  direct comparison with round 2 (another machine, the runner).

  D2   dynotears_lw0.05_la0.05_w0.3_a0.1_tau10 and cuts_plus_th0.5_h32_s{0,1,2}_tau10
       (the defaults of the paper) on the 90 scenarios, twice (rep1, rep2), with one
       thread per process, as in r1_2g (run_parallel).
  D2b  the same four, a third time, with the default number of threads of the
       libraries: sequential run (run_resuming), without fixing torch or
       BLAS to one thread, as in the runs of the paper.
  D4   the 39 reruns of PCMCI, PCMCI+ and VARLiNGAM of r1_2g on the 90 scenarios.
  B    the 267 reruns of DYNOTEARS and CUTS+ of r1_2g on 14 scenarios (one per
       family, rule of the protocol).

  Comparison (mode comparar), per run: status, error message, set
  of predicted pairs (results.csv) and TP/FP/FN recounted against
  ground_truth.csv, as in the tabular mode of r1_2g. Against:
    - round 2 (Results_Revisao/_outra_maquina/rodada2_grade_ampliada_2026-11-10);
    - between the repetitions here (rep1 x rep2; rep1 x D2b);
    - the run of the paper (only the defaults; CUTS+ at seed 0, that of the paper).
  The reading is the one fixed in the protocol; this script only counts.
  Its results are reported in Supplementary Material, Table S29.5 and Text ST16 (run-to-run variation).

  The order of the scenarios (heaviest first, by their duration in round 2) changes
  no result. The durations of this round are not a time measurement: they run
  beside other experiments (authors' lab notes, 2026-11-11).

Usage (from the repository root)
  python revision/execution/r0_double_check.py rodar d2  --paralelo 12 [--verificar]
  python revision/execution/r0_double_check.py rodar d2b [--verificar]
  python revision/execution/r0_double_check.py rodar d4  --paralelo 12 [--verificar]
  python revision/execution/r0_double_check.py rodar b   --paralelo 12 [--verificar]
  python revision/execution/r0_double_check.py comparar [--verificar]
  Folders: round 2 is looked for in _outra_maquina/rodada2_grade_ampliada_2026-11-10/ (main machine) and
  in r1_2g_grade_ampliada/ (other machine); the report says which one was used.
  --verificar: one scenario (benchmark_synthetic_var_1_seed11), output in
  r0_dupla_conferencia_verificacao/; it is a test of the script, not a result.
Output: Results_Revisao/r0_dupla_conferencia/ (file and column names kept in Portuguese; see revision/README.md, Notes)
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import r1_2g_expanded_grid as G  # noqa: E402

NAME = "r0_dupla_conferencia"
# Round 2: on the main machine, the copy in _outra_maquina/; on the other machine (the runner), the original folder.
ROUND2_CANDIDATES = [os.path.join("_outra_maquina", "rodada2_grade_ampliada_2026-11-10", "r1_2g_grade_ampliada", "run"),
                     os.path.join("r1_2g_grade_ampliada", "run")]


def round2() -> str:
    import common_exec as ce
    for r in ROUND2_CANDIDATES:
        p = os.path.join(ce.OUTPUT_ROOT, r)
        if os.path.isdir(p):
            return p
    raise SystemExit("round 2 (r1_2g_grade_ampliada/run) not found")

D2_SPECS = ["dynotears_lw0.05_la0.05_w0.3_a0.1_tau10", "cuts_plus_th0.5_h32_s0_tau10",
            "cuts_plus_th0.5_h32_s1_tau10", "cuts_plus_th0.5_h32_s2_tau10"]
# rerun -> model in the runs of the paper (the CUTS+ of the paper is seed 0)
IN_PAPER = {"dynotears_lw0.05_la0.05_w0.3_a0.1_tau10": "dynotears", "cuts_plus_th0.5_h32_s0_tau10": "cuts_plus"}
D2_HIGHLIGHTED = {
    "dynotears": ["benchmark_causaltime_pm25_sample0", "benchmark_causaltime_pm25_sample2",
                  "benchmark_netsim_sim3_subj1", "benchmark_rcaeval_RE1-OB_adservice_delay_5",
                  "real_pems_bay_trafego", "real_petshop_high_traffic"],
    "cuts_plus": ["benchmark_causaltime_traffic_sample1", "benchmark_rcaeval_RE1-OB_adservice_cpu_3",
                  "benchmark_rcaeval_RE1-OB_adservice_delay_5", "benchmark_rcaeval_RE1-OB_cartservice_delay_3",
                  "benchmark_rcaeval_RE1-OB_checkoutservice_disk_1", "benchmark_rcaeval_RE1-OB_checkoutservice_mem_5",
                  "benchmark_synthetic_var_5_seed55", "real_alarm_uti", "real_causal_chamber_lt_actuators_white",
                  "real_easyvista_web_2", "real_petshop_low_traffic", "real_petshop_temporal_traffic1"],
}
B_SCENARIOS = ["benchmark_synthetic_var_1_seed11", "benchmark_dream3_size10_Ecoli2",
               "benchmark_logistic_maps_convergence_seed3", "benchmark_tigramite_saturation_seed22",
               "real_easyvista_mom_1", "benchmark_lorenz96_K8_seed1", "benchmark_netsim_sim3_subj2",
               "benchmark_beeline_VSC_VSC-2000-10", "benchmark_causaltime_traffic_sample1", "real_sachs_proteins",
               "real_tigramite_climate_walker", "real_pems_bay_trafego",
               "real_causal_chamber_wt_actuators_random_walk_3",
               "benchmark_rcaeval_RE1-OB_productcatalogservice_cpu_2"]
SCEN_CHECK = "benchmark_synthetic_var_1_seed11"

# part -> (folder, selection of reruns, scenarios: None = the 90)
PARTS = {
    "d2_rep1": ("d2_rep1", lambda x: x["nome"] in D2_SPECS, None),
    "d2_rep2": ("d2_rep2", lambda x: x["nome"] in D2_SPECS, None),
    "d2b": ("d2b_threads_padrao", lambda x: x["nome"] in D2_SPECS, None),
    "d4": ("d4", lambda x: x["baseline"] in ("pcmci_tigramite", "pcmciplus", "varlingam"), None),
    "b": ("b", lambda x: x["baseline"] in ("dynotears", "cuts_plus"), B_SCENARIOS),
}
EXPECTED = {"d2_rep1": 4, "d2_rep2": 4, "d2b": 4, "d4": 39, "b": 267}


def _base(check: bool) -> str:
    import common_exec as ce
    return ce.output_dir(NAME + ("_verificacao" if check else ""))


def _grid(part: str) -> list:
    g = [x for x in G.GRID if PARTS[part][1](x)]
    assert len(g) == EXPECTED[part], (part, len(g))
    assert len({x["nome"] for x in g}) == len(g)
    return g


def _scenarios(part: str, check: bool) -> list:
    import common as c
    import common_exec as ce
    dfs = ce.paper_scenarios()
    assert len(dfs) == 90, len(dfs)
    if check:
        sel = [SCEN_CHECK]
    else:
        sel = PARTS[part][2]
    if sel is None:
        return dfs
    out = [d for d in dfs if d["name"] in sel or c.scenario_key(d["name"]) in sel]
    assert len(out) == len(sel), (part, len(out), len(sel))
    return out


def _round2_weight(dfs: list, grid: list) -> list:
    import common_exec as ce
    r2 = round2()
    weight = {}
    for d in dfs:
        ds = [G._duration(os.path.join(r2, d["name"], x["nome"])) for x in grid]
        weight[d["name"]] = sum(v for v in ds if v is not None)
    return sorted(dfs, key=lambda d: -weight[d["name"]])


def run():
    import common_exec as ce
    check = "--verificar" in sys.argv
    stage = sys.argv[2] if len(sys.argv) > 2 else ""
    parts = {"d2": ["d2_rep1", "d2_rep2"], "d2b": ["d2b"], "d4": ["d4"], "b": ["b"]}.get(stage)
    if parts is None:
        raise SystemExit(__doc__)
    if stage != "d2b" and "--paralelo" not in sys.argv:
        raise SystemExit("d2, d4 and b run with --paralelo N (one thread per process, as in r1_2g)")
    for part in parts:
        grid = _grid(part)
        specs = [(x["base"], x["nome"], x["override"]) for x in grid]
        dfs = _round2_weight(_scenarios(part, check), grid)
        folder = ce.output_dir(os.path.join(NAME + ("_verificacao" if check else ""), PARTS[part][0], "run"))
        print(f"[{NAME}] {part}: {len(dfs)} scenarios x {len(specs)} reruns -> {folder}", flush=True)
        if part == "d2b":
            ce.run_resuming(dfs, [ce.model(b, n, **o) for b, n, o in specs], folder)
        else:
            ce.run_parallel(dfs, specs, folder, int(sys.argv[sys.argv.index("--paralelo") + 1]))


# ---------------------------------------------------------------- comparar
def _run_record(ps: str, name: str, T: set) -> dict:
    """status, error, pairs and TP/FP/FN of one run (scenario folder ps, rerun name)."""
    pm = os.path.join(ps, name)
    t = G.read_time(pm)
    if t is None:
        return {"status": "sem_execucao", "erro": "", "pares": None, "TP": None, "FP": None, "FN": None}
    st = "error" if t.get("status", "").upper() == "ERROR" else t.get("status", "?")
    e = {"status": st, "erro": t.get("error", "") if st == "error" else "", "pares": None,
         "TP": None, "FP": None, "FN": None}
    if st == "ok":
        S = G.predicted_pairs(pm)
        e.update(pares=frozenset(S), TP=len(S & T), FP=len(S - T), FN=len(T - S))
    return e


def _equal(a: dict, b: dict):
    """True/False when both ran; the reason as text when they cannot be compared."""
    if a["status"] == "sem_execucao" or b["status"] == "sem_execucao":
        return "sem_execucao"
    if a["status"] != b["status"]:
        return False
    if a["status"] == "error":
        return a["erro"] == b["erro"]
    return a["pares"] == b["pares"]


def _part_table(part: str, check: bool) -> pd.DataFrame:
    import common as c
    import common_exec as ce
    base = _base(check)
    run_folder = os.path.join(base, PARTS[part][0], "run")
    r2 = round2()
    L = c.rows()
    rows = []
    if not os.path.isdir(run_folder):
        return pd.DataFrame()
    for scen in sorted(d for d in os.listdir(run_folder) if os.path.isdir(os.path.join(run_folder, d))):
        ps = os.path.join(run_folder, scen)
        if not os.path.isfile(os.path.join(ps, "ground_truth.csv")):
            continue
        T = G.folder_truth(ps)
        k = c.scenario_key(scen) or scen
        for x in _grid(part):
            here = _run_record(ps, x["nome"], T)
            there = _run_record(os.path.join(r2, scen), x["nome"], T) if os.path.isdir(os.path.join(r2, scen)) else \
                {"status": "sem_execucao", "erro": "", "pares": None, "TP": None, "FP": None, "FN": None}
            row = {"cenario": k, "baseline": x["baseline"], "reexecucao": x["nome"], "parte": part,
                   "status": here["status"], "TP": here["TP"], "FP": here["FP"], "FN": here["FN"],
                   "f1": G.f1c(here["TP"], here["FP"], here["FN"]) if here["status"] == "ok" else None,
                   "status_r2": there["status"], "TP_r2": there["TP"], "FP_r2": there["FP"], "FN_r2": there["FN"],
                   "f1_r2": G.f1c(there["TP"], there["FP"], there["FN"]) if there["status"] == "ok" else None,
                   "igual_r2": _equal(here, there), "erro": here["erro"]}
            m = IN_PAPER.get(x["nome"])
            if m is not None:
                art = L.get(k, {}).get(m) if k in L else None
                row.update(TP_artigo=art["tp"] if art else None, FP_artigo=art["fp"] if art else None,
                           FN_artigo=art["fn"] if art else None)
                row["igual_artigo"] = ("sem_artigo" if art is None else
                                       (here["status"] == "ok" and (art["tp"], art["fp"], art["fn"]) ==
                                        (here["TP"], here["FP"], here["FN"])))
            if x["nome"] in D2_SPECS:
                row["destacado"] = k in D2_HIGHLIGHTED[x["baseline"]]
            row["_pares"] = here["pares"]
            row["_erro_cmp"] = here["erro"]
            rows.append(row)
    return pd.DataFrame(rows)


def _count(s: pd.Series) -> str:
    v = s.dropna().map(lambda z: z if isinstance(z, str) else bool(z)).value_counts()
    return ", ".join(f"{k}: {n}" for k, n in v.items())


def compare():
    import common as c
    check = "--verificar" in sys.argv
    suf = "_verificacao" if check else ""
    rel = c.Report(NAME + suf, "Double check of round 2 — D2, D2b, D4 and B")
    import platform
    rel.p(f"Machine: {platform.node()}. Round 2 read from `{os.path.relpath(round2(), c.OUTPUT_ROOT)}`.")
    rel.p("Protocol: `revision/PROTOCOL_double_check.md` (frozen on 2026-11-11 14:34). "
          "Comparison by the set of predicted pairs (results.csv) and, for errors, by the message. "
          "TP/FP/FN recounted against ground_truth.csv, as in the `tabular` mode of r1_2g.")
    if check:
        rel.p("**Check of the script, one scenario. Not a result.**")
    T = {p: _part_table(p, check) for p in PARTS}
    for p, D in T.items():
        if len(D):
            rel.csv(f"{p}.csv", D.drop(columns=["_pares", "_erro_cmp"]))

    # ---- D2: rep1 x rep2, rep1 x D2b, against round 2 and against the paper
    rel.h("## D2 and D2b — defaults of DYNOTEARS and CUTS+")
    r1, r2_, tb = T["d2_rep1"], T["d2_rep2"], T["d2b"]
    if len(r1):
        key = ["cenario", "reexecucao"]
        M = r1[key + ["baseline", "status", "f1", "igual_r2", "igual_artigo", "destacado", "_pares", "_erro_cmp"]]
        for name, other in (("rep2", r2_), ("d2b", tb)):
            if len(other):
                O = other[key + ["status", "f1", "_pares", "_erro_cmp"]].rename(
                    columns={"status": f"status_{name}", "f1": f"f1_{name}", "_pares": f"_pares_{name}",
                             "_erro_cmp": f"_erro_{name}"})
                M = M.merge(O, on=key, how="left")
                M[f"rep1_igual_{name}"] = [
                    "sem_execucao" if pd.isna(s2) or s1 == "sem_execucao" or s2 == "sem_execucao" else
                    (s1 == s2 and (e1 == e2 if s1 == "error" else p1 == p2))
                    for s1, s2, p1, p2, e1, e2 in zip(M.status, M[f"status_{name}"], M._pares, M[f"_pares_{name}"],
                                                      M._erro_cmp, M[f"_erro_{name}"])]
        rel.csv("d2.csv", M.drop(columns=[k for k in M.columns if k.startswith("_")]))
        body = []
        for spec in D2_SPECS:
            G_ = M[M.reexecucao == spec]
            for group, H in (("all", G_), ("highlighted", G_[G_.destacado == True])):  # noqa: E712
                if not len(H):
                    continue
                body.append([spec, group, len(H),
                             _count(H["rep1_igual_rep2"]) if "rep1_igual_rep2" in H else "—",
                             _count(H["igual_r2"]),
                             _count(H["igual_artigo"]) if H["igual_artigo"].notna().any() else "—",
                             _count(H["rep1_igual_d2b"]) if "rep1_igual_d2b" in H else "—"])
        rel.table(["Rerun", "Scenarios", "n", "rep1 = rep2", "rep1 = round 2", "rep1 = paper",
                   "rep1 = D2b"], body)
        rel.p("\"highlighted\": the scenarios on which round 2 did not reproduce the paper (6 for DYNOTEARS, 12 for CUTS+; "
              "for CUTS+ the comparison with the paper holds only for seed 0).")
        dif = M[(M.igual_r2 != True) | (M.get("rep1_igual_rep2", True) != True)]  # noqa: E712
        if len(dif):
            rel.p("Runs with some difference (rep1 against round 2 or against rep2):")
            rel.table(["Scenario", "Rerun", "F1 rep1", "F1 round 2", "rep1=rep2", "F1 D2b"],
                      [[r.cenario, r.reexecucao, c.fmt(r.f1), c.fmt(T["d2_rep1"].set_index(key).loc[
                          (r.cenario, r.reexecucao), "f1_r2"]), str(getattr(r, "rep1_igual_rep2", "—")),
                        c.fmt(getattr(r, "f1_d2b", None))] for r in dif.itertuples()])
        rel.data["d2"] = {s: {k: (M[M.reexecucao == s][k].map(lambda z: z if isinstance(z, str) else bool(z))
                                  .value_counts().to_dict() if k in M else None)
                              for k in ("rep1_igual_rep2", "igual_r2", "igual_artigo", "rep1_igual_d2b")}
                          for s in D2_SPECS}

    # ---- D4: deterministic methods against round 2
    rel.h("## D4 — PCMCI, PCMCI+ and VARLiNGAM (39 reruns) against round 2")
    D = T["d4"]
    if len(D):
        body = [[b, len(H), _count(H.igual_r2), int((H.status == "error").sum()),
                 H.loc[H.status == "error", "cenario"].nunique()] for b, H in D.groupby("baseline")]
        rel.table(["Baseline", "Runs", "Equal to round 2", "Errors here", "Scenarios with an error"], body)
        err_here = set(D.loc[D.status == "error", "cenario"])
        err_there = set(D.loc[D.status_r2 == "error", "cenario"])
        rel.p(f"Scenarios with an error: here {len(err_here)}, in round 2 {len(err_there)}; "
              f"equal: {err_here == err_there}.")
        dif = D[D.igual_r2 != True]  # noqa: E712
        rel.data["d4"] = {"execucoes": int(len(D)), "iguais": int((D.igual_r2 == True).sum()),  # noqa: E712
                          "diferentes": int(len(dif)), "cenarios_erro_aqui": sorted(err_here),
                          "cenarios_erro_r2": sorted(err_there)}
        if len(dif):
            rel.table(["Scenario", "Rerun", "Status here/there", "F1 here", "F1 round 2", "Equal"],
                      [[r.cenario, r.reexecucao, f"{r.status}/{r.status_r2}", c.fmt(r.f1), c.fmt(r.f1_r2),
                        str(r.igual_r2)] for r in dif.itertuples()])

    # ---- B: full DYNOTEARS and CUTS+ grid on 14 scenarios (descriptive)
    rel.h("## B — grid of DYNOTEARS and CUTS+ on 14 scenarios (descriptive)")
    D = T["b"]
    if len(D):
        body = []
        for b, H in D.groupby("baseline"):
            ok = H[(H.status == "ok") & (H.status_r2 == "ok")]
            d = (ok.f1 - ok.f1_r2).abs()
            body.append([b, len(H), _count(H.igual_r2), c.fmt(d.max() if len(d) else None),
                         c.fmt(d[d > 0].median() if (d > 0).any() else 0.0)])
        rel.table(["Baseline", "Runs", "Equal to round 2", "Largest |ΔF1|", "Median |ΔF1| (≠ 0)"], body)
        per_scen = D.groupby(["baseline", "cenario"]).igual_r2.apply(lambda s: int((s == True).sum())).reset_index()  # noqa: E712
        rel.csv("b_por_cenario.csv", per_scen)
        rel.data["b"] = {b: {"execucoes": int(len(H)), "iguais": int((H.igual_r2 == True).sum())}  # noqa: E712
                         for b, H in D.groupby("baseline")}
    rel.save()


if __name__ == "__main__":
    from run_log import execution
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("rodar", "comparar"):
        raise SystemExit(__doc__)
    with execution(NAME, __file__):
        {"rodar": run, "comparar": compare}[mode]()

"""
R1-2c, block (iii) -- second benchmark after the freeze: coupled TimeGraph and CausalDynamics (SINGLE RUN).

Design: revision/PROTOCOL_third_party_benchmark_R1-2c.md (part A, frozen on 2026-11-22 before any download)
  and revision/ANNEX_C_R1-2c_third_party.md (structure of the files; hash in NOTAS.md). If this script and the protocol
  differ, the protocol prevails.

Scenarios (section 3; readers in r1_2c_third_party_readers.py; frozen list in revision/r1_2c_third_party_selection.json):
  timegraph       100 scenarios: 20 bases drawn per variant (B2, C2C, D2C, D3, D3C), _missing version when there is one.
  causaldynamics   90 scenarios: 30 coupled graphs drawn per N = 3, 5, 10; trajectory 0; node = entity.
Methods (section 4; tau_max = 10): patch_tau10 (default configuration: main comparison), patch_tau10_final (corrected
  variant: secondary) and the paper's 5 baselines. Limit of 60 min per run (+30 s), each run in its own
  process (the mechanism of r1_2c_yeast.py). PYTHONHASHSEED = 0 (the queue sets it; the script checks it).
Scoring (section 5): main.validate_all with the paper's parameters; in CausalDynamics, the pairs within the same node
  are removed from the ground truth and from the predictions (ignore_intra_entity=True), so each block has its own
  run_<block>/ folder.

Modes
  carregar           only the count allowed before the run (series, points, pairs and ground-truth edges per scenario)
                     and the comparison with the frozen list; no method runs.
  rodar [--blocos B ...]
                     the single run (orchestrator; each run in a child process). Order: timegraph ->
                     causaldynamics. In each block: per scenario, patch_tau10 and the 5 baselines; then, per scenario,
                     patch_tau10_final. Resumed via time.txt. Validation at the end of each block, outside the limit.
  um <block> <scenario> <model> <folder>
                     child process (internal use).
  fumaca             test of the mechanism (20 s limit) on one VALIDATION scenario of the first benchmark (wind
                     tunnel, walk 1), in Results_Revisao/r1_2c_benchmark_terceiros_verificacao/; never these data.
  avaliar            T1 to T3, tests and counts of section 5.

Usage (from the repository root, with PYTHONHASHSEED=0):  python revision/execution/r1_2c_third_party_benchmark.py <mode>
Output: Results_Revisao/r1_2c_benchmark_terceiros/
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
F2 = os.path.dirname(REV)
sys.path[:0] = [HERE, REV, F2]

SMOKE = "fumaca" in sys.argv[1:2]
NAME = "r1_2c_benchmark_terceiros" + ("_verificacao" if SMOKE else "")
FROZEN_HASH = "d0473069ebef1978"
LIMIT_S, SLACK_S = 3600, 30
BLOCKS = ("timegraph", "causaldynamics")
EXPECTED = {"timegraph": 100, "causaldynamics": 90}
INTRA_EXCLUDED = {"timegraph": False, "causaldynamics": True}   # section 3.2 of the protocol
SELECTION = os.path.join(REV, "r1_2c_third_party_selection.json")

SUBMITTED = "patch_tau10"
FINAL = "patch_tau10_final"
BASELINES = ["pcmci_tigramite_tau10", "pcmciplus_tau10", "dynotears_tau10", "cuts_plus_tau10", "varlingam_tau10"]
ALL_METHODS = [SUBMITTED, FINAL] + BASELINES


def arg(name, default=None):
    if name not in sys.argv:
        return default
    out = []
    for a in sys.argv[sys.argv.index(name) + 1:]:
        if a.startswith("--"):
            break
        out.append(a)
    return out


def _model(name):
    import common_exec as ce
    if name == FINAL:
        return ce.model("patch_tau10", FINAL, time_axis_rule="block_portmanteau", evidence_coherence_overlap="coherent")
    return ce.model(name)


FROZEN = os.path.join(REV, "FROZEN_third_party.json")
FROZEN_FILES = {   # what the congelar mode records and the rodar mode checks (paths relative to the repository root)
    "protocolo": "revision/PROTOCOL_third_party_benchmark_R1-2c.md",
    "anexo_c": "revision/ANNEX_C_R1-2c_third_party.md",
    "leitores": "revision/execution/r1_2c_third_party_readers.py",
    "script": "revision/execution/r1_2c_third_party_benchmark.py",
    "selecao": "revision/r1_2c_third_party_selection.json",
}


def _sha(rel_path):
    import hashlib
    with open(os.path.join(F2, rel_path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _check(require_frozen=True):
    import run_log
    h = run_log.method_code_hash()
    if h != FROZEN_HASH:
        print(f"WARNING: method_code_hash {h} != {FROZEN_HASH}, the hash of the frozen method code (the released files are English translations of the frozen ones; see revision/README.md)", file=sys.stderr)
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(f"PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED')!r}; this script requires 0 (the queue sets it).")
    if require_frozen:
        if not os.path.exists(FROZEN):
            raise SystemExit("revision/FROZEN_third_party.json missing: run the congelar mode (annex C ready) first.")
        rec = json.load(open(FROZEN, encoding="utf-8"))
        diff = [k for k, p in FROZEN_FILES.items() if rec.get(k) != _sha(p)]
        if diff:
            print(f"WARNING: SHA-256 different from revision/FROZEN_third_party.json for {diff} (the released files are English translations of the frozen ones; see revision/README.md)", file=sys.stderr)


def _valid(items):
    """The exclusion of section 3.2 (graph without an edge between distinct nodes) applies only to CausalDynamics."""
    return [x for x in items if not x.get("excluido_sem_aresta")]


def select():
    """Computes the selection of section 3 with the full data and writes the frozen list (refuses if it exists)."""
    import r1_2c_third_party_readers as L
    if os.path.exists(SELECTION):
        raise SystemExit(f"{SELECTION} already exists; the selection is computed only once. To check it, use carregar.")
    sel = {b: L.selection(b) for b in BLOCKS}
    for b in BLOCKS:
        if len(sel[b]) != EXPECTED[b]:
            raise SystemExit(f"{b}: {len(sel[b])} drawn, {EXPECTED[b]} expected (section 3); stop and note it.")
    # TimeGraph was drawn on the main machine, before the download, with only the commit listing: it must match
    exp = json.load(open(os.path.join(REV, "r1_2c_third_party_timegraph_expected.json"), encoding="utf-8"))["timegraph"]
    keys = ("name", "variante", "indice_na_variante", "base", "csv", "verdade")
    if [{k: x[k] for k in keys} for x in sel["timegraph"]] != [{k: x[k] for k in keys} for x in exp]:
        raise SystemExit("the TimeGraph selection differs from the one computed on the main machine before the download; "
                         "stop and note it.")
    sel["_meta"] = {"protocolo": FROZEN_FILES["protocolo"], "semente": L.SEED,
                    "gerado_em": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "excluidos_sem_aresta": [x["name"] for x in sel["causaldynamics"] if x.get("excluido_sem_aresta")]}
    with open(SELECTION, "w", encoding="utf-8", newline="\r\n") as f:
        json.dump(sel, f, ensure_ascii=False, indent=1)
    print(f"list written to {SELECTION}: " + ", ".join(f"{b} {len(sel[b])}" for b in BLOCKS) +
          f"; excluded without an edge: {sel['_meta']['excluidos_sem_aresta']}")


def freeze():
    """Records the SHA-256 of the protocol, annex C, the readers, the script and the list. Refuses if anything is open."""
    if os.path.exists(FROZEN):
        raise SystemExit("revision/FROZEN_third_party.json already exists; the freeze is done only once.")
    annex = os.path.join(F2, FROZEN_FILES["anexo_c"])
    if not os.path.exists(annex) or "A PREENCHER" in open(annex, encoding="utf-8").read():
        raise SystemExit("annex C missing or with 'A PREENCHER' fields.")
    if "NotImplementedError" in open(os.path.join(F2, FROZEN_FILES["leitores"]), encoding="utf-8").read():
        raise SystemExit("the readers still have unfilled functions (NotImplementedError).")
    if not os.path.exists(SELECTION):
        raise SystemExit("frozen list missing: run selecionar first.")
    rec = {k: _sha(p) for k, p in FROZEN_FILES.items()}
    rec["congelado_em"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(FROZEN, "w", encoding="utf-8", newline="\r\n") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))
    print("Note these hashes in NOTAS_outra_maquina.md and in the zip's RESUMO.md.")


def _folders():
    import common_exec as ce
    base = ce.output_dir(NAME)
    return base, {b: os.path.join(base, f"run_{b}") for b in BLOCKS}


def _frozen_selection():
    if not os.path.exists(SELECTION):
        raise SystemExit(f"frozen list missing: {SELECTION}")
    return json.load(open(SELECTION, encoding="utf-8"))


# ---------------------------------------------------------------------------------------------- carregar
def load():
    import r1_2c_third_party_readers as L
    import common_exec as ce
    rows, sel = [], {}
    for b in BLOCKS:
        items = L.selection(b)
        sel[b] = items
        for it in items:
            d = L.scenario(b, it["name"])
            df, gt = d["df"], d["gt"]
            props = [c for c in df.columns if c not in ("Date", "Name")]
            pos = gt["causal_gt"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
            inter = gt["source_entity"].astype(str) != gt["target_entity"].astype(str)
            rows.append({"bloco": b, "cenario": it["name"], "familia": L.family(it["name"]),
                         "entidades": int(df["Name"].nunique()), "propriedades": len(props),
                         "pontos": int(df["Date"].nunique()), "pares_verdade": int(len(gt)),
                         "arestas_verdade": int(pos.sum()), "arestas_entre_entidades": int((pos & inter).sum()),
                         "faltantes_interpolados": d["_params"].get("n_missing_interpolated"),
                         "colunas_constantes_retiradas": d["_params"].get("dropped_constant")})
    D = pd.DataFrame(rows)
    base = ce.output_dir(NAME)
    D.to_csv(os.path.join(base, "carregamento.csv"), index=False)
    with open(os.path.join(base, "selecao_recalculada.json"), "w", encoding="utf-8") as f:
        json.dump(sel, f, ensure_ascii=False, indent=1)
    print(D.to_string(index=False))
    for b in BLOCKS:
        n = int((D.bloco == b).sum())
        print(f"{b}: {n} scenarios ({EXPECTED[b]} expected)")
    if os.path.exists(SELECTION):
        frozen = _frozen_selection()
        equal = all(frozen[b] == json.loads(json.dumps(sel[b], ensure_ascii=False)) for b in BLOCKS)
        print("recomputed list == frozen list (names, files, indices and SHA-256):", equal)
        if not equal:
            raise SystemExit("the recomputed selection differs from the frozen one: stop and check the input data.")
    print("Count only; no model was run.")


# ---------------------------------------------------------------------------------------------- rodar
def _plan(blocks):
    frozen = _frozen_selection()
    steps = []
    for b in blocks:
        assert len(frozen[b]) == EXPECTED[b], (b, len(frozen[b]))
        names = [x["name"] for x in _valid(frozen[b])]
        steps += [(b, c, m) for c in names for m in [SUBMITTED] + BASELINES]
        steps += [(b, c, FINAL) for c in names]
    return steps


def one(block, scen, mod, folder):
    """Child process: runs ONE model on ONE scenario (without validate_all)."""
    import common_exec as ce
    if block == "fumaca":
        import r1_2c_new_benchmark as B
        d = [x for x in B.block_scenarios("fumaca") if x["name"] == scen][0]
    else:
        import r1_2c_third_party_readers as L
        d = L.scenario(block, scen)
    ce.main_mod.run_all(dataframes=[d], models=[_model(mod)], config=ce.cfg_exec.PIPELINE_CONFIG, output_dir=folder)


def _timeout_time_txt(model_folder, mod, start, end, limit_s):
    sec = (end - start).total_seconds()
    txt = (f"model      : {mod}\n"
           f"module     : desconhecido (interrompido pelo limite)\n"
           f"start      : {start:%Y-%m-%d %H:%M:%S}\n"
           f"end        : {end:%Y-%m-%d %H:%M:%S}\n"
           f"duration_s  : {sec:.4f}\n"
           f"status     : ERROR\n"
           f"error      : DEMORA EXCESSIVA (timeout): a execução passou do limite de {limit_s} s (protocolo do bloco "
           f"(iii), seção 4) e foi interrompida por revision/execution/r1_2c_third_party_benchmark.py; não é falha do método "
           f"nem dos dados\n")
    os.makedirs(model_folder, exist_ok=True)
    p = os.path.join(model_folder, "time.txt")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)


def _execute(steps, folders, prog, limit, slack):
    import common_exec as ce
    for b, scen, mod in steps:
        folder = folders[b]
        os.makedirs(folder, exist_ok=True)
        if ce.already_run(folder, scen, mod):
            continue
        start = datetime.datetime.now()
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "um", b, scen, mod, folder], cwd=F2)
        try:
            code, cut = p.wait(timeout=limit + slack), False
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
            p.wait()
            code, cut = None, True
        end = datetime.datetime.now()
        pm = os.path.join(folder, scen, mod)
        if cut:
            _timeout_time_txt(pm, mod, start, end, limit)
        st, tp = "?", os.path.join(pm, "time.txt")
        if os.path.exists(tp):
            st = next((ln.split(":", 1)[1].strip() for ln in open(tp, encoding="utf-8", errors="replace")
                       if ln.strip().startswith("status")), "?")
        rec = {"bloco": b, "cenario": scen, "modelo": mod, "inicio": f"{start:%Y-%m-%d %H:%M:%S}",
               "fim": f"{end:%Y-%m-%d %H:%M:%S}", "duracao_s": round((end - start).total_seconds(), 1),
               "cortado_limite": cut, "codigo": code, "status_time_txt": st,
               "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}
        with open(prog, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{NAME}] {b} / {scen} / {mod}: {st}{' (CUT at the limit)' if cut else ''} in {rec['duracao_s']} s",
              flush=True)


def _validate(folder, intra_excluded):
    import common_exec as ce
    ce.main_mod.validate_all(output_dir=folder, show_pairs=False, soft_threshold=ce.cfg_exec.SOFT_THRESHOLD,
                             ignore_intra_entity=True if intra_excluded else not ce.cfg_exec.DETECT_INTRA_ENTITY,
                             use_canonical_pair=ce.cfg_exec.USE_CANONICAL_PAIR)


def run():
    import common_exec as ce
    _check()
    blocks = arg("--blocos") or list(BLOCKS)
    base, folders = _folders()
    steps = _plan(blocks)
    print(f"[{NAME}] {len(steps)} runs; limit {LIMIT_S} s (+{SLACK_S} s); blocks {blocks}", flush=True)
    for b in blocks:
        _execute([p for p in steps if p[0] == b], folders, os.path.join(base, f"progresso_{b}.jsonl"),
                 LIMIT_S, SLACK_S)
        print(f"[{NAME}] validation of block {b} (outside the limit; pairs within the entity excluded: {INTRA_EXCLUDED[b]})",
              flush=True)
        _validate(folders[b], INTRA_EXCLUDED[b])
    missing = [(b, c, m) for b, c, m in steps if not ce.already_run(folders[b], c, m)]
    if missing:
        print(f"[{NAME}] WITHOUT time.txt: {missing[:20]} ... ({len(missing)})", flush=True)
        sys.exit(1)
    print(f"[{NAME}] done: {len(steps)} runs with time.txt", flush=True)


def smoke():
    import common_exec as ce
    import r1_2c_new_benchmark as B
    _check()
    base = ce.output_dir(NAME)
    folder = os.path.join(base, "run")
    scen = [d["name"] for d in B.block_scenarios("fumaca")][0]
    steps = [("fumaca", scen, "varlingam_tau10"), ("fumaca", scen, FINAL), ("fumaca", scen, SUBMITTED)]
    _execute(steps, {"fumaca": folder}, os.path.join(base, "progresso.jsonl"), 20, 0)
    _validate(folder, False)
    print(f"[{NAME}] smoke test done; check in {folder}: varlingam_tau10 with status ok; patch_tau10 (and, if it exceeds "
          f"20 s, patch_tau10_final) with time.txt 'DEMORA EXCESSIVA (timeout)', which shows the cutoff working.",
          flush=True)


# ---------------------------------------------------------------------------------------------- avaliar
def _time_txt(model_folder):
    f = os.path.join(model_folder, "time.txt")
    if not os.path.exists(f):
        return None
    d = {}
    for line in open(f, encoding="utf-8", errors="replace"):
        if ":" in line:
            k, v = line.split(":", 1)
            d.setdefault(k.strip(), v.strip())
    return d


def _tests(Fb, c, m, opponents):
    p = {}
    for b in opponents:
        if b in Fb and m in Fb:
            x = Fb[[m, b]].dropna()
            if len(x):
                p[b] = (c.wilcoxon_p(x[m], x[b]), len(x), x[m].mean() - x[b].mean(), c.wins(x[m], x[b]))
    ph = c.holm({k: v[0] for k, v in p.items()})
    lines = ["| Against | Paired scenarios | Mean F1 difference | Better/equal/worse | p | Holm p |",
             "|---|---|---|---|---|---|"]
    for b, (pv, n, dm, v) in p.items():
        lines.append(f"| {b} | {n} | {dm:+.3f} | {v[0]}/{v[1]}/{v[2]} | {pv:.4f} | {ph[b]:.4f} |")
    return lines


def evaluate():
    import common as c
    import common_exec as ce
    import r1_2c_third_party_readers as L
    base, folders = _folders()
    frozen = _frozen_selection()
    parts = []
    for b in BLOCKS:
        x = ce.summary(folders[b]) if os.path.isdir(folders[b]) else pd.DataFrame()
        if len(x):
            x.insert(0, "bloco", b)
            parts.append(x)
    R = pd.concat(parts, ignore_index=True)
    R["status"] = R["status"].astype(str)
    ok = R["status"] == "ok"
    R["f1"] = [ce.f1(a, b_, d) if s else np.nan for a, b_, d, s in zip(R.TP, R.FP, R.FN, ok)]
    R["precisao"] = np.where(ok & ((R.TP + R.FP) > 0), R.TP / (R.TP + R.FP).replace(0, np.nan), np.nan)
    R["revocacao"] = np.where(ok & ((R.TP + R.FN) > 0), R.TP / (R.TP + R.FN).replace(0, np.nan), np.nan)
    R["familia"] = R["cenario"].map(L.family)
    R.to_csv(os.path.join(base, "T1_por_cenario.csv"), index=False)
    F = R.pivot_table(index="cenario", columns="model", values="f1")
    expected_scen = sum(len(_valid(frozen[b])) for b in BLOCKS)
    expected = expected_scen * len(ALL_METHODS)
    excl = frozen.get("_meta", {}).get("excluidos_sem_aresta", [])
    missing, no_res = [], []
    for b in BLOCKS:
        for it in _valid(frozen[b]):
            for m in ALL_METHODS:
                d = _time_txt(os.path.join(folders[b], it["name"], m))
                if d is None:
                    missing.append((it["name"], m))
                elif d.get("status") != "ok":
                    no_res.append((it["name"], m, d.get("status", "?"), d.get("error", "")))
    txt = ["# R1-2c, block (iii) — coupled TimeGraph and CausalDynamics", "",
           "Protocol: `revision/PROTOCOL_third_party_benchmark_R1-2c.md` (part A) and `revision/ANNEX_C_R1-2c_third_party.md`.",
           "", f"CausalDynamics graphs excluded for having no edge between distinct nodes (section 3.2): {len(excl)} "
           f"{excl}.", "",
           f"Valid scenarios in the frozen list: {expected_scen}; with output: {R['cenario'].nunique()}. Expected runs: "
           f"{expected}; with time.txt: {expected - len(missing)}; with a result: {int(ok.sum())}; with status other "
           f"than ok: {len(no_res)}.", ""]
    if missing or int(ok.sum()) + len(no_res) != expected:
        txt += ["**INCOMPLETE: the numbers below are not the result of block (iii).**", ""]
        if missing:
            txt += ["Without time.txt: " + ", ".join(f"{s}/{m}" for s, m in missing[:50]), ""]
    if no_res:
        txt += ["**Runs without a result (left out of the method's mean):**", "",
                "| Scenario | Method | Status | Error (beginning) |", "|---|---|---|---|"]
        txt += [f"| {s} | {m} | {st} | {er[:140].replace('|', '/')} |" for s, m, st, er in no_res] + [""]
    T2 = R.pivot_table(index="familia", columns="model", values="f1", aggfunc="mean")
    T2.insert(0, "scenarios", R.groupby("familia")["cenario"].nunique())
    txt += ["## T2 — mean F1 per variant (TimeGraph) and per N stratum (CausalDynamics)", "", T2.round(3).to_markdown(), ""]
    for block_name, filt in (("(iii) both", None), ("(iii-a) TimeGraph", "timegraph"),
                             ("(iii-b) CausalDynamics", "causaldynamics")):
        scen = sorted(R["cenario"].unique()) if filt is None else sorted(R.loc[R.bloco == filt, "cenario"].unique())
        if not scen:
            continue
        Fb = F.loc[[x for x in scen if x in F.index]]
        txt += [f"## T3 — block {block_name}: {len(Fb)} scenarios", "",
                Fb.mean().round(3).to_frame("Mean F1").to_markdown(), "",
                "**Main test** (`patch_tau10` against each baseline; two-sided paired Wilcoxon, Holm over the 5):", ""]
        txt += _tests(Fb, c, SUBMITTED, BASELINES) + [""]
        txt += ["Secondary: `patch_tau10_final` against each baseline (Holm over the 5):", ""]
        txt += _tests(Fb, c, FINAL, BASELINES) + [""]
        if FINAL in Fb and SUBMITTED in Fb:
            x = Fb[[SUBMITTED, FINAL]].dropna()
            txt += [f"- Secondary: `{FINAL}` against `{SUBMITTED}`: mean difference {x[FINAL].mean() - x[SUBMITTED].mean():+.3f}, "
                    f"better/equal/worse {c.wins(x[FINAL], x[SUBMITTED])}, Wilcoxon p "
                    f"{c.wilcoxon_p(x[FINAL], x[SUBMITTED]):.4f} ({len(x)} scenarios).", ""]
    floor = {}
    for b in BLOCKS:
        for s in R.loc[R.bloco == b, "cenario"].unique():
            gt = pd.read_csv(os.path.join(folders[b], s, "ground_truth.csv"), skipinitialspace=True)
            if INTRA_EXCLUDED[b]:
                gt = gt[gt["source_entity"].astype(str).str.strip() != gt["target_entity"].astype(str).str.strip()]
            ents = sorted(set(gt["source_entity"].astype(str).str.strip()) | set(gt["target_entity"].astype(str).str.strip()))
            pos = {(a, z) for a, z, v in zip(gt["source_entity"].astype(str).str.strip(),
                                              gt["target_entity"].astype(str).str.strip(), gt["causal_gt"])
                   if str(v).strip().lower() in ("true", "1", "1.0")}
            # ordered pairs between distinct entities: in TimeGraph each entity has a single property (there is no pair
            # within the entity); in CausalDynamics those pairs are removed from scoring (section 3.2)
            n_pairs = len(ents) * (len(ents) - 1)
            pr = len(pos) / n_pairs if n_pairs else np.nan
            floor[s] = 2 * pr / (1 + pr) if pr == pr else np.nan
    floor = pd.Series(floor)
    floor.rename("piso_trivial").to_csv(os.path.join(base, "piso_por_cenario.csv"))
    below = {m: int((F[m].dropna() < floor[F[m].dropna().index] - 1e-9).sum()) for m in F.columns}
    main_methods = [m for m in ALL_METHODS if m in F.columns]
    ranks = F[main_methods].rank(axis=1, ascending=False, method="average").mean()
    txt += ["## Trivial floor and mean rank (among the 7 methods)", "",
            "| Method | Scenarios with a result | Below the floor | Mean rank |", "|---|---|---|---|"]
    for m in main_methods:
        txt.append(f"| {m} | {int(F[m].notna().sum())} | {below[m]} | {ranks[m]:.2f} |")
    open(os.path.join(base, "relatorio.md"), "w", encoding="utf-8", newline="\r\n").write("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "um":
        one(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif mode in ("selecionar", "carregar", "congelar", "rodar", "fumaca", "avaliar"):
        from run_log import execution
        with execution(NAME + ("_" + mode if mode in ("selecionar", "congelar") else ""), __file__):
            {"selecionar": select, "carregar": load, "congelar": freeze, "rodar": run, "fumaca": smoke,
             "avaliar": evaluate}[mode]()
    else:
        raise SystemExit(__doc__)

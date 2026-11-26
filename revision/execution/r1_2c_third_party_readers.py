"""
R1-2c, block (iii) -- readers for the coupled TimeGraph and CausalDynamics.

Design: revision/PROTOCOL_third_party_benchmark_R1-2c.md, section 3 (part A, frozen on 2026-11-22). If this file and
the protocol differ, the protocol prevails.

What has been written here since the main machine (2026-11-22) and does NOT change: the CSV -> ground-truth
correspondence, the version rule (_missing), the draw (seeds, order, quantities), the scenario names and the assembly
in the pipeline format. What depends on the internal format of the files, which was only read on the runner, is in the
functions marked "TO BE FILLED IN ON THE RUNNER (annex C)": they are written there AFTER the structure report
(revision/offline/r1_2c_third_party_structure.py estrutura) and BEFORE any method runs, and each decision taken in them
goes to annex C (revision/ANNEX_C_R1-2c_third_party.md), with the reason taken from the authors' documentation, code or
files -- never from results.

Interface used by r1_2c_third_party_benchmark.py:
  selection(block)        -> list of dicts (name, ... paths, sha256) in the order of the draw
  scenario(block, name)   -> dict in the Data.base.ScenarioEntry format (name, df, gt, _scenario, _params)
  family(name)            -> label of the variant (TimeGraph) or of the stratum (CausalDynamics)
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

SEED = 20261122
# ---- TimeGraph (section 3.1) ---------------------------------------------------------------------------------------
TG_DIR = Path("./external_data/timegraph/extraido")
VARIANTS = ("B2", "C2C", "D2C", "D3", "D3C")
EXPECTED_BASES = {"B2": 36, "C2C": 72, "D2C": 216, "D3": 432, "D3C": 432}
N_TG = 20
# ---- CausalDynamics (section 3.2) ----------------------------------------------------------------------------------
CD_DIR = Path("./external_data/causaldynamics/extraido_coupled")
N_STRATA = (3, 5, 10)
N_CD = 30


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _draw(n_total: int, k_draw: int, rng_seed: int) -> List[int]:
    """Indices drawn without replacement, in increasing order; all of them if there are at most k_draw."""
    if n_total <= k_draw:
        return list(range(n_total))
    rng = np.random.default_rng(rng_seed)
    return sorted(rng.choice(n_total, size=k_draw, replace=False).tolist())


# ================================================================================================= TimeGraph
@lru_cache(maxsize=1)
def _tg_root() -> Path:
    cands = list(TG_DIR.glob("TimeGraph-*/Datasets"))
    if len(cands) != 1:
        raise SystemExit(f"expected one TimeGraph-*/Datasets folder in {TG_DIR}; found {cands}")
    return cands[0].parent


@lru_cache(maxsize=1)
def _tg_bases() -> Dict[str, List[str]]:
    """Per variant, the bases (path relative to the root of the authors' repository, with '/', without suffix) that have
    a machine-readable ground truth in the same folder; in lexicographic order of the full path (section 3.1)."""
    root = _tg_root()
    all_files = sorted(str(p.relative_to(root)).replace("\\", "/") for p in (root / "Datasets").rglob("*") if p.is_file())
    S = set(all_files)
    bases = defaultdict(set)
    for p in all_files:
        if not p.endswith(".csv"):
            continue
        b = re.sub(r"_(complete|missing)$", "", p[:-4])
        if b + "_structure.txt" in S or b + "_causal_structure.txt" in S:
            bases[p.split("/")[1]].add(b)
    out = {v: sorted(bases[v]) for v in VARIANTS}
    count = {v: len(out[v]) for v in VARIANTS}
    if count != EXPECTED_BASES:
        raise SystemExit(f"TimeGraph bases {count} != {EXPECTED_BASES} (listing of the pinned commit); stop and note it")
    return out


def _tg_files(b: str) -> Tuple[str, str]:
    root = _tg_root()
    csv = b + "_missing.csv" if (root / (b + "_missing.csv")).exists() else b + ".csv"
    truth = b + "_structure.txt" if (root / (b + "_structure.txt")).exists() else b + "_causal_structure.txt"
    return csv, truth


@lru_cache(maxsize=1)
def _selection_tg() -> List[dict]:
    root = _tg_root()
    out = []
    for k, v in enumerate(VARIANTS):
        bs = _tg_bases()[v]
        for j, i in enumerate(_draw(len(bs), N_TG, SEED + k)):
            csv, truth = _tg_files(bs[i])
            out.append({"name": f"timegraph_{v}_{j:02d}", "variante": v, "indice_na_variante": i, "base": bs[i],
                        "csv": csv, "verdade": truth, "sha256_csv": _sha256(root / csv),
                        "sha256_verdade": _sha256(root / truth)})
    return out


# ---- TO BE FILLED IN ON THE RUNNER (annex C) -------------------------------------------------------------------------
def _tg_read_series(path: Path) -> Tuple[pd.DataFrame, dict]:
    """TO BE FILLED IN ON THE RUNNER (annex C).

    Reads a TimeGraph CSV and returns (X, info):
      X     DataFrame with only the OBSERVED variables, one column per variable, rows in file order (the time
            axis; section 3.1). A timestamp column, if any, does NOT enter X. Columns that the dataset
            designates as latent (the confounder of the variants with C) do NOT enter X; the designation comes from the
            authors' documentation, code or structure file (annex C).
      info  {"coluna_tempo": name or None, "latentes_retiradas": [...]}.
    Do not interpolate here (interpolation is done in scenario(), the same for all). Do not look at values beyond what is
    needed to identify columns (names, type).

    FILLED IN on the runner on 2026-11-22 (annex C, section 3). In the 1,188 bases the header is X1..Xn (n = 4, 6 or 8),
    then `U` only in the variants with C (C2C, D2C, D3C) and `time` last. `time` is the timestamp (irregular
    sampling) and does not enter. `U` is the confounder designated by the dataset (README: "Linear with confounder
    `U[t]`"; Codes/c2c.py, d2c.py and d3c.py: "df['U'] = U  # Add confounder column"; structure files: "U[t] = eU") and
    does not enter. Any header outside this pattern stops the run.
    """
    df = pd.read_csv(path)
    cols = [str(c) for c in df.columns]
    obs = [c for c in cols if re.fullmatch(r"X\d+", c)]
    variant = Path(path).as_posix().split("/Datasets/")[1].split("/")[0]
    expected = ["U", "time"] if variant.endswith("C") else ["time"]
    if not obs or obs != [f"X{i + 1}" for i in range(len(obs))] or cols != obs + expected:
        raise SystemExit(f"{path}: header {cols} outside the pattern X1..Xn{' + U' if 'U' in expected else ''} + time "
                         f"of variant {variant}; stop and note it (annex C)")
    return df[obs].copy(), {"coluna_tempo": "time", "latentes_retiradas": [c for c in expected if c == "U"]}


def _tg_read_truth(path: Path, observed: List[str]) -> Set[Tuple[str, str]]:
    """TO BE FILLED IN ON THE RUNNER (annex C).

    Reads the structure file and returns the set of directed pairs (source, target) between DISTINCT OBSERVED
    variables, with the same names as the columns of X, at any lag (section 3.1). Self-lags are left out. Edges
    that involve a latent are left out (the latent is not a variable of the scenario). If the file has a form that does
    not allow the direction to be read unambiguously, stop and note it in annex C (do not guess).

    FILLED IN on the runner on 2026-11-22 (annex C, section 3). The edge lines have three forms, all with an explicit
    direction (source first, target after):
      B2             ('X1', -2, 'X4') => 0.25                 ("Format: (source, lag, target) => weight")
      D2C            (X1, -2, X4) => 0.25 [base coefficient]  ("Format: (source, lag, target) => coefficient")
      C2C, D3, D3C   X1 --(cos, lag=2)--> X4: 0.25
    The lag is read and discarded (the ground truth is the pair, at any lag). Where the file has the "Equations"
    section (all variants except B2), the pairs of the edges must equal those of the equations (target = left-hand
    side; sources = the X# or U variables on the right-hand side); if they do not, it stops. It also stops if there is
    a line with "=>" or "-->" that is not read (other than the "Format:" line), if there is no edge at all, or if an edge
    mentions a variable that is neither an observed column nor U.
    """
    r_quoted_tuple = re.compile(r"^\(\s*'(\w+)'\s*,\s*(-?\d+)\s*,\s*'(\w+)'\s*\)\s*=>")
    r_tuple = re.compile(r"^\(\s*(\w+)\s*,\s*(-?\d+)\s*,\s*(\w+)\s*\)\s*=>")
    r_arrow = re.compile(r"^(\w+)\s*--\(\s*[^)]*?lag\s*=\s*(-?\d+)\s*\)-->\s*(\w+)\s*:")
    r_equation = re.compile(r"^(\w+)\[t\]\s*=\s*(.+)$")
    edges, eq_pairs, has_equations = set(), set(), False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = r_quoted_tuple.match(line) or r_tuple.match(line) or r_arrow.match(line)
        if m:
            edges.add((m.group(1), m.group(3)))
            continue
        me = r_equation.match(line)
        if me:
            has_equations = True
            eq_pairs |= {(source, me.group(1)) for source in re.findall(r"\b(X\d+|U)\[t(?:-\d+)?\]", me.group(2))}
            continue
        if ("=>" in line or "-->" in line) and not line.lower().startswith("format"):
            raise SystemExit(f"{path}: edge line not read: {line!r}; stop and note it (annex C)")
    known = set(observed) | {"U"}
    if not edges:
        raise SystemExit(f"{path}: no edge read; stop and note it (annex C)")
    if any(a not in known or b not in known for a, b in edges):
        raise SystemExit(f"{path}: edge with a variable outside {sorted(known)}: {sorted(edges)}; stop and note it")
    if has_equations and eq_pairs != edges:
        raise SystemExit(f"{path}: the edges {sorted(edges)} differ from the pairs of the equations {sorted(eq_pairs)}; "
                         f"ambiguous direction; stop and note it (annex C)")
    return {(a, b) for a, b in edges if a in observed and b in observed and a != b}


def _scenario_tg(it: dict) -> dict:
    from Data.base import ScenarioEntry, array_to_pipeline_format_single_property, drop_constant_columns
    root = _tg_root()
    X, info = _tg_read_series(root / it["csv"])
    n_nan = int(X.isna().sum().sum())
    X = X.apply(pd.to_numeric, errors="raise").interpolate(method="linear", limit_direction="both")
    if X.isna().any().any():
        raise SystemExit(f"{it['name']}: missing values remained after interpolation (entirely empty column?); note it")
    before = [str(c) for c in X.columns]
    truth = _tg_read_truth(root / it["verdade"], before)
    pos0 = {c: i for i, c in enumerate(before)}
    pairs0 = {(pos0[a], pos0[b]) for a, b in truth if a in pos0 and b in pos0 and a != b}
    # Data.base.drop_constant_columns(data, entity_names, causal_pair_idx) -> (data, names, pairs, dropped)
    arr, cols, pairs, dropped = drop_constant_columns(X.to_numpy(dtype=float), before, pairs0)
    names = [f"VAR_{c}" for c in cols]
    df, gt = array_to_pipeline_format_single_property(arr, names, pairs, start_date="2015-01-01", freq="D",
                                                      property_name="VAL")
    params = {"source": "timegraph", "commit": "4d93403b002664fa55607f308ad0e80623472ded", **it,
              "n_missing_interpolated": n_nan, "dropped_constant": dropped, "n_causal_links": len(pairs),
              "n_causal_links_lost_to_constant": len(pairs0) - len(pairs),
              **{f"leitor_{k}": v for k, v in info.items()}}
    return ScenarioEntry(name=it["name"], df=df, gt=gt, _scenario=f"timegraph_{it['variante']}",
                         _params=params).to_dict()


# ================================================================================================= CausalDynamics
# ---- TO BE FILLED IN ON THE RUNNER (annex C) -------------------------------------------------------------------------
def _cd_enumerate() -> List[dict]:
    """TO BE FILLED IN ON THE RUNNER (annex C).

    Returns ALL coupled graphs of the archive, in the CANONICAL ORDER of section 3.2: path of the .nc file relative to
    CD_DIR, in lexicographic order (with '/'); within the file, the index of the graph in the order in which it is
    stored. Each item: {"arquivo": relative path, "indice": int (0 if one graph per file), "N": number of observed
    nodes of the graph}. N comes from the file itself (shape of the adjacency matrix or attribute), not from the file
    name, unless annex C shows that they are equal in all. No series value is read here.

    FILLED IN on the runner on 2026-11-22 (annex C, sections 4 and 5). One graph per file (14,096 .nc files), index 0.
    DEVIATION from the sentence above, by the author's decision on 2026-11-22 (annex C, section 5, deviation 1): N is
    the GENERATOR's N, the `systems=` field of the folder (3, 5 or 10; the authors' Table 1, which is the stratum of
    section 3.2 of the protocol), and not the stored number of nodes. Reason: in the files with `confounder=True` the
    authors already removed a node before storing (scripts/generate_coupled.py -> utils.process_confounders), and the
    file has N - 1 nodes (2, 4 or 9) in 6,344 of the 7,058 graphs with a confounder; with the file's N those graphs would
    never enter the draw. The stored number of nodes is checked in _cd_read_graph (N without a confounder; N or N - 1
    with a confounder). No file is opened here.
    """
    out = []
    for rel in sorted(str(p.relative_to(CD_DIR)).replace("\\", "/") for p in CD_DIR.rglob("*.nc")):
        m = re.search(r"_systems=(\d+)_", rel)
        if not m or int(m.group(1)) not in N_STRATA:
            raise SystemExit(f"{rel}: no `systems=` in {N_STRATA} in the path; stop and note it (annex C)")
        out.append({"arquivo": rel, "indice": 0, "N": int(m.group(1))})
    if len(out) != 14096:
        raise SystemExit(f"{len(out)} .nc files in {CD_DIR}; 14096 expected (annex C); stop and note it")
    return out


def _cd_read_graph(item: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    """TO BE FILLED IN ON THE RUNNER (annex C).

    Returns (series, adj, info) of the graph `item`:
      series array (T, N, d): the FIRST stored trajectory (index 0), all steps, only the observed nodes;
      adj    binary array (N, N), adj[i, j] = 1 if node i causes node j (the orientation comes from the authors' code
             or documentation, annex C; if the stored matrix is the transpose, transpose here); diagonal as it is;
      info   {"configuracao": text of the generator configuration, if any; "confusor_retirado": indices of latent nodes
             removed (if the dataset designates any); "orientacao": "linha->coluna" or "transposta"}.

    FILLED IN on the runner on 2026-11-22 (annex C, sections 4 and 5).
      Ground truth: the field `adjacency_matrix_summary` (dimensions node_in, node_out; what the authors' eval.py uses),
        without transposing: A[i, j] = 1 means that node i causes node j (paper, section 2: "a_ki = 1 if x_i is causally
        impacted by x_k"; src/causaldynamics/scm.py: "A[i,j] indicates an edge from i to j"; checked on the structure:
        root_nodes matches the null columns in the 3,513 files without a confounder and without lag). The diagonal
        (self-edges of the roots) is left as it is; _scenario_cd is the one that discards it.
      Series: `time_series`, dimensions (time, system, node, dim) = (1000, 10, nodes, 3); `system` is the trajectory.
        DEVIATION from the sentence "the FIRST stored trajectory (index 0)", by the author's decision on 2026-11-22
        (annex C, section 5, deviation 2): returns the first ENTIRELY FINITE trajectory (the one with index 0 when it
        is finite). Reason: in 3,948 of the 14,096 graphs trajectory 0 has non-finite values from step 0 to the end (in
        3,912, a whole node). If none of the 10 trajectories is entirely finite, returns the one with index 0 as it is
        and sets info["sem_trajetoria_finita"] = True: selection() excludes the graph, without replacement.
        info["series_constantes"] counts the series (dimension of a node) without variation in the returned trajectory;
        selection() excludes the graph that has any (author's decision on 2026-11-22; annex C, section 5, deviation 4).
      Confounder: no node is removed here ("confusor_retirado" = []). In the files with `confounder=True` the authors
        already removed the node before storing; the file has N - 1 nodes (or N, when their post-processing found no
        node to remove). info["no_retirado_pelos_autores"] says which of the two cases.
    """
    import h5py
    rel = item["arquivo"]
    m = re.search(r"_systems=(\d+)_confounder=(True|False)_", rel)
    if item["indice"] != 0 or not m or int(m.group(1)) != item["N"]:
        raise SystemExit(f"{rel}: item {item} does not match the path (one graph per file; N = systems=); note it")
    n_generator, confounder = int(m.group(1)), m.group(2) == "True"
    with h5py.File(CD_DIR / rel, "r") as f:
        A = np.asarray(f["adjacency_matrix_summary"][()], dtype=float)
        ts = f["time_series"]
        dims = [ts.dims[i][0].name.rsplit("/", 1)[-1] for i in range(ts.ndim)]
        all_traj = np.asarray(ts[()], dtype=float)
    n = A.shape[0]
    if (dims != ["time", "system", "node", "dim"] or A.shape != (n, n) or all_traj.shape[2] != n
            or n not in ((n_generator - 1, n_generator) if confounder else (n_generator,))
            or np.isnan(A).any() or set(np.unique(A).tolist()) - {0.0, 1.0}):
        raise SystemExit(f"{rel}: unexpected structure (dimensions {dims}, series {all_traj.shape}, matrix {A.shape}, "
                         f"systems={n_generator}, confounder={confounder}); stop and note it (annex C)")
    finite = [bool(np.isfinite(all_traj[:, s]).all()) for s in range(all_traj.shape[1])]
    trajectory = finite.index(True) if any(finite) else 0
    series = all_traj[:, trajectory]
    # constant series (one dimension of a node without variation) in the returned trajectory: only the count, for the
    # exclusion decided by the author on 2026-11-22 (annex C, section 5, deviation 4), applied in selection()
    constants = int((series.max(axis=0) == series.min(axis=0)).sum()) if any(finite) else 0
    info = {"configuracao": rel.split("/")[1], "confusor_retirado": [], "orientacao": "linha->coluna",
            "trajetoria": trajectory, "trajetorias_inteiramente_finitas": int(sum(finite)),
            "sem_trajetoria_finita": not any(finite), "series_constantes": constants, "nos_no_arquivo": int(n),
            "no_retirado_pelos_autores": bool(confounder and n == n_generator - 1)}
    return series, (A != 0).astype(int), info


@lru_cache(maxsize=1)
def _selection_cd() -> List[dict]:
    all_graphs = _cd_enumerate()
    out = []
    for k, n in enumerate(N_STRATA):
        stratum = [g for g in all_graphs if g["N"] == n]
        for j, i in enumerate(_draw(len(stratum), N_CD, SEED + 10 + k)):
            g = stratum[i]
            out.append({"name": f"causaldynamics_N{n:02d}_{j:02d}", "N": n, "indice_no_estrato": i,
                        "arquivo": g["arquivo"], "indice": g["indice"], "sha256_arquivo": _sha256(CD_DIR / g["arquivo"])})
    return out


def _scenario_cd(it: dict) -> dict:
    import itertools
    from Data.base import ScenarioEntry
    series, adj, info = _cd_read_graph({"arquivo": it["arquivo"], "indice": it["indice"], "N": it["N"]})
    T, N, d = series.shape
    # Change outside the 4 functions, authorized by the author on 2026-11-22 (annex C, section 5, deviation 1): it["N"] is
    # the generator's N; the file has N nodes, or N - 1 when the authors removed the confounder node. Before: `N != it["N"]`.
    if N not in (it["N"], it["N"] - 1) or adj.shape != (N, N):
        raise SystemExit(f"{it['name']}: unexpected shape {series.shape} / {adj.shape}; note it")
    ents = [f"NODE_{i}" for i in range(N)]
    props = [f"F_{f}" for f in range(d)]
    dates = pd.date_range("2015-01-01", periods=T, freq="D")
    rows = []
    for t in range(T):
        for i in range(N):
            row = {"Date": dates[t], "Name": ents[i]}
            for f in range(d):
                row[props[f]] = float(series[t, i, f])
            rows.append(row)
    df = pd.DataFrame(rows)
    between = {(i, j) for i in range(N) for j in range(N) if i != j and adj[i, j] != 0}
    gt = []
    for (i, fi), (j, fj) in itertools.permutations([(i, f) for i in range(N) for f in range(d)], 2):
        gt.append({"source_entity": ents[i], "source_property": props[fi], "target_entity": ents[j],
                   "target_property": props[fj], "causal_gt": (i != j) and ((i, j) in between)})
    params = {"source": "causaldynamics_coupled", "revision": "e42a6eebb4736f982be5332dbba91670b21e861e", **it,
              "T": T, "d": d, "n_causal_links_between_nodes": len(between),
              "intra_entity_pairs": "fora da pontuação (seção 3.2; validate_all com ignore_intra_entity=True)",
              **{f"leitor_{k}": v for k, v in info.items()}}
    return ScenarioEntry(name=it["name"], df=df, gt=pd.DataFrame(gt), _scenario=f"causaldynamics_N{N:02d}",
                         _params=params).to_dict()


# ================================================================================================= interface
def selection(block: str) -> List[dict]:
    if block == "timegraph":
        # Change outside the 4 functions, by the author's decision of 2026-11-22 (annex C, section 5, deviation 3): a
        # TimeGraph scenario with a non-finite (infinite) value after interpolation is removed, without replacement. It
        # uses the same marker `excluido_sem_aresta` (the only one r1_2c_third_party_benchmark.py reads);
        # `motivo_exclusao` gives the reason. Before this change the function returned only
        # `[dict(x) for x in _selection_tg()]`.
        sel = [dict(x) for x in _selection_tg()]
        root = _tg_root()
        for it in sel:
            X, _ = _tg_read_series(root / it["csv"])
            X = X.apply(pd.to_numeric, errors="raise").interpolate(method="linear", limit_direction="both")
            non_finite = bool(not np.isfinite(X.to_numpy(dtype=float)).all())
            it["motivo_exclusao"] = "valor não finito depois da interpolação" if non_finite else None
            it["excluido_sem_aresta"] = non_finite
        return sel
    if block == "causaldynamics":
        sel = [dict(x) for x in _selection_cd()]
        excluded = []
        # Change outside the 4 functions, by the author's decisions of 2026-11-22 (annex C, section 5, deviations 2 and
        # 4): besides the exclusion of section 3.2, the graph without any entirely finite trajectory and the graph with
        # some constant series in the trajectory used are also removed, also without replacement. The three exclusions
        # use the marker `excluido_sem_aresta` (the only one r1_2c_third_party_benchmark.py reads); `motivo_exclusao`
        # says which.
        for it in sel:     # exclusion of section 3.2: graph without an edge between distinct observed nodes
            _, adj, info = _cd_read_graph({"arquivo": it["arquivo"], "indice": it["indice"], "N": it["N"]})
            reasons = []
            if not any(adj[i, j] for i in range(adj.shape[0]) for j in range(adj.shape[0]) if i != j):
                reasons.append("sem aresta entre nós distintos")
            if info["sem_trajetoria_finita"]:
                reasons.append("sem trajetória inteiramente finita")
            if info["series_constantes"]:
                reasons.append(f"{info['series_constantes']} série(s) constante(s) na trajetória usada")
            if reasons:
                excluded.append(it["name"])
            it["motivo_exclusao"] = "; ".join(reasons) or None
            it["trajetoria"] = None if info["sem_trajetoria_finita"] else info["trajetoria"]
            it["nos_no_arquivo"] = info["nos_no_arquivo"]
        for it in sel:
            it["excluido_sem_aresta"] = it["name"] in excluded
        return sel
    raise ValueError(block)


def scenario(block: str, name: str) -> dict:
    sel = {x["name"]: x for x in (_selection_tg() if block == "timegraph" else _selection_cd())}
    if name not in sel:
        raise SystemExit(f"{name} is not in the selection of block {block}")
    return _scenario_tg(sel[name]) if block == "timegraph" else _scenario_cd(sel[name])


def family(name: str) -> str:
    m = re.match(r"timegraph_([A-Z0-9]+)_\d+$", name)
    if m:
        return f"TimeGraph {m.group(1)}"
    m = re.match(r"causaldynamics_N(\d+)_\d+$", name)
    if m:
        return f"CausalDynamics N = {int(m.group(1))}"
    return "outro"


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    b = sys.argv[1] if len(sys.argv) > 1 else "timegraph"
    print(json.dumps(selection(b), indent=1, ensure_ascii=False))

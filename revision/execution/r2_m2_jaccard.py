"""
R2-m2 -- Stability of the graph under subsampling (Jaccard).

Request
  R2: "the sensitivity of the resulting edge set to noise perturbations or
       sub-sampling is not reported. Stability metrics, such as Jaccard
       similarity across bootstrap runs, would increase confidence".

Design (fixed before running)
  For each scenario of the paper, B = 20 subsamples of one contiguous block with 80%
  of the dates (drawn start, seed 7000 + CRC32 of the scenario name, stable across runs); the paper
  pipeline (patch_tau10) runs on each one. A contiguous block preserves the temporal
  dependence; resampling loose observations would destroy it.
  Graph: the set of reported directional pairs (the same one used for scoring).
  Reporting per scenario and per family: mean Jaccard between each subsample and the
  full-sample graph (paper run); mean Jaccard between pairs of
  subsamples; fraction of the full-sample edges that appear in at
  least 80% of the subsamples; and the mean F1 of the subsamples.
  Cost: about 18 h for the 90 scenarios; --filtro restricts the scope and the
  script resumes from where it stopped.
  Stages (modular strategy, NOTAS.md, 2026-11-10): rodar --B 5, then --B 10, then --B 20; the first
  subsamples of a larger B are the same as those of a smaller B. avaliar labels as PARTIAL any result with
  fewer than 20 subsamples in some scenario or fewer than 90 scenarios (addendum of 2026-11-12, before running; it
  only changes the report).

Usage
  python revision/execution/r2_m2_jaccard.py rodar [--B 20] [--filtro name ...]
  python revision/execution/r2_m2_jaccard.py avaliar
Output: Results_Revisao/r2_m2_jaccard/
"""
from __future__ import annotations

import itertools
import zlib
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

NAME = "r2_m2_jaccard"
FRACTION = 0.8


def arg(name, default):
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    vals = []
    for a in sys.argv[i + 1:]:
        if a.startswith("--"):
            break
        vals.append(a)
    return vals


def subsamples(d, B, seed):
    dates = sorted(d["df"]["Date"].unique())
    L = int(round(FRACTION * len(dates)))
    rng = np.random.default_rng(seed)
    out = []
    for b in range(B):
        i0 = int(rng.integers(0, len(dates) - L + 1))
        window = set(dates[i0:i0 + L])
        out.append({**d, "name": f"{d['name']}__sub{b:02d}",
                    "df": d["df"][d["df"]["Date"].isin(window)].reset_index(drop=True)})
    return out


def run():
    import common_exec as ce
    B = int(arg("--B", ["20"])[0])
    name_filter = arg("--filtro", None)
    folder = ce.output_dir(os.path.join(NAME, "run"))
    dfs = ce.paper_scenarios(name_filter)
    for d in dfs:
        seed = 7000 + zlib.crc32(d["name"].encode("utf-8"))
        ce.run_resuming(subsamples(d, B, seed), ce.models(["patch_tau10"]), folder)


def jacc(a, b):
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def graph(path):
    if not os.path.exists(path):
        return None
    r = pd.read_csv(path)
    return set(zip(r["source_entity"].astype(str).str.strip(), r["target_entity"].astype(str).str.strip())) \
        if len(r) else set()


def evaluate():
    import common as c
    rel = c.Report(NAME, "R2-m2 — Stability of the graph under subsampling")
    run = os.path.join(c.OUTPUT_ROOT, NAME, "run")
    if not os.path.isdir(run):
        rel.p("No run; use the rodar mode.")
        rel.save()
        return
    groups = {}
    for s in os.listdir(run):
        m = re.match(r"(.+)__sub\d+$", s)
        if m:
            groups.setdefault(m.group(1), []).append(s)
    rows = []
    for base, subs in sorted(groups.items()):
        k = c.scenario_key(base)
        if not k:
            continue
        T, _ = c.truth(k)
        full = graph(os.path.join(c.folder(k), c.PIPE_FOLDER, "results.csv"))
        G = [g for g in (graph(os.path.join(run, s, "patch_tau10", "results.csv")) for s in sorted(subs)) if g is not None]
        if not G or full is None:
            continue
        freq = {e: np.mean([e in g for g in G]) for e in full}
        rows.append({"cenario": k, "familia": c.family(k), "heldout": c.is_heldout(k), "B": len(G),
                     "jaccard_vs_completa": np.mean([jacc(g, full) for g in G]),
                     "jaccard_entre_subamostras": np.mean([jacc(a, b) for a, b in itertools.combinations(G, 2)])
                     if len(G) > 1 else np.nan,
                     "arestas_estaveis_frac": np.mean([v >= 0.8 for v in freq.values()]) if freq else np.nan,
                     "f1_completa": c.f1_set(full, T), "f1_subamostras": np.mean([c.f1_set(g, T) for g in G])})
    D = pd.DataFrame(rows)
    rel.csv("por_cenario.csv", D)
    if D.empty:
        rel.save()
        return
    # Rule 2 of the modular strategy (NOTAS.md, 2026-11-10): an intermediate stage is labelled as partial.
    rel.p(f"Subsamples per scenario: minimum {int(D['B'].min())}, maximum {int(D['B'].max())}; "
          f"scenarios with a result: {len(D)} of 90; runs included: {int(D['B'].sum())}.")
    if int(D["B"].min()) < 20 or len(D) < 90:
        rel.p("**PARTIAL:** intermediate stage (the registered design is B = 20 on the 90 scenarios). This is not the "
              "final result of R2-m2.")
    cols = ["jaccard_vs_completa", "jaccard_entre_subamostras", "arestas_estaveis_frac", "f1_completa", "f1_subamostras"]
    F = D.groupby("familia")[cols].mean()
    F = F.loc[[f for f in c.FAMILIES_DEV + c.FAMILIES_HO if f in F.index]]
    rel.table(["Family"] + cols, [[f] + [c.fmt(F.loc[f, k]) for k in cols] for f in F.index])
    rel.p("")
    for part, m in (("development", ~D["heldout"]), ("held-out", D["heldout"])):
        if m.any():
            rel.p(f"- {part}: mean Jaccard against the full sample {D.loc[m, 'jaccard_vs_completa'].mean():.3f}; "
                  f"between subsamples {D.loc[m, 'jaccard_entre_subamostras'].mean():.3f} ({int(m.sum())} scenarios).")
    rel.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        if len(sys.argv) > 1 and sys.argv[1] == "rodar":
            run()
        else:
            evaluate()

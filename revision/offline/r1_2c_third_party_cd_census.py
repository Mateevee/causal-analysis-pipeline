"""
R1-2c, block (iii) -- STRUCTURE census of the 14,096 files of the coupled CausalDynamics, for annex C.

Written on the runner on 2026-11-22, after the extraction and BEFORE any method ran. Complements
revision/offline/r1_2c_third_party_structure.py, whose `estrutura` mode only opens the first 3 files (all with
confounder=False). It changes no file frozen on the main machine.

It only reads structure and ground truth, never a series value:
  - from the path: the fields of the generator configuration (coupling, noise, systems, confounder, standardize, timelag,
    activation) and the file name;
  - from the file: the field names, the shape of adjacency_matrix, of adjacency_matrix_summary and of time_series, the
    count of nonzero entries of the summary matrix (diagonal and off it), whether it is lower triangular, and the sum of
    root_nodes;
  - from time_series: only the COUNT of non-finite values (NaN or infinity) in trajectory 0 (index 0 of the dimension
    `system`), which is the one the protocol uses (section 3.2). It is a count of missing values, like the one the
    `carregar` mode does on TimeGraph; no value is printed or written.

Usage (from the repository root):  python revision/offline/r1_2c_third_party_cd_census.py
Output: Results_Revisao/r1_2c_benchmark_terceiros/censo_causaldynamics.csv and censo_causaldynamics.md
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
CD_DIR = ROOT / "external_data" / "causaldynamics" / "extraido_coupled"
OUTPUT = ROOT / "Results_Revisao" / "r1_2c_benchmark_terceiros"


def main():
    ncs = sorted(str(p.relative_to(CD_DIR)).replace("\\", "/") for p in CD_DIR.rglob("*.nc"))
    rows = []
    for k, rel in enumerate(ncs):
        cfg = dict(re.findall(r"(coupling|noise|systems|confounder|standardize|timelag|activation)=([^_/]+)", rel))
        with h5py.File(CD_DIR / rel, "r") as f:
            fields = sorted(f.keys())
            A = f["adjacency_matrix_summary"][()] if "adjacency_matrix_summary" in f else None
            ts = f["time_series"]
            dims_ts = [d.label or "?" for d in ts.dims]
            if not any(x != "?" for x in dims_ts):
                dims_ts = []
                for i in range(ts.ndim):
                    try:
                        dims_ts.append(os.path.basename(ts.dims[i][0].name))
                    except Exception:
                        dims_ts.append("?")
            # only the mask of non-finite values (position and count); no value is kept
            bad = ~np.isfinite(ts[()])                       # (time, system, node, dim)
            bad0 = bad[:, 0]
            non_finite = int(bad0.sum())
            finite_traj = [bool(not bad[:, s].any()) for s in range(bad.shape[1])]
            bad_step = bad0.any(axis=(1, 2))               # per time step, in trajectory 0
            first = int(np.argmax(bad_step)) if bad_step.any() else -1
            extra = {"traj_inteiramente_finitas": int(sum(finite_traj)),
                     "primeira_traj_finita": finite_traj.index(True) if any(finite_traj) else -1,
                     "traj0_fracao_nao_finita": round(float(bad0.mean()), 6),
                     "traj0_primeiro_passo_nao_finito": first,
                     "traj0_passos_com_nao_finito": int(bad_step.sum()),
                     "traj0_nao_finito_ate_o_fim": bool(bad_step.any() and bad_step[first:].all()),
                     "traj0_nos_com_nao_finito": int(bad0.any(axis=(0, 2)).sum()),
                     "traj0_nos_inteiramente_nao_finitos": int(bad0.all(axis=(0, 2)).sum())}
            if "root_nodes" in f and "adjacency_matrix" in f:
                rn = np.asarray(f["root_nodes"][()]).astype(bool)
                A0 = np.nan_to_num(f["adjacency_matrix"][()]) != 0
                if A0.ndim == 2 and A0.shape[0] == A0.shape[1] == rn.shape[0]:
                    # root = node without an incoming edge. If A[i, j] = i -> j, root <=> entirely null column.
                    extra["raiz_igual_coluna_nula"] = bool((rn == ~A0.any(axis=0)).all())
                    extra["raiz_igual_linha_nula"] = bool((rn == ~A0.any(axis=1)).all())
                extra["forma_root_nodes"] = str(rn.shape)
            row = {"arquivo": rel, **{f"cfg_{a}": b for a, b in cfg.items()}, **extra,
                   "campos": " ".join(fields),
                   "forma_adjacency_matrix": str(f["adjacency_matrix"].shape) if "adjacency_matrix" in f else "",
                   "forma_summary": str(A.shape) if A is not None else "",
                   "forma_time_series": str(ts.shape), "dims_time_series": " ".join(dims_ts),
                   "nao_finitos_trajetoria0": non_finite,
                   "time_lag_gravado": int(f["time_lag"][()]) if "time_lag" in f else -1,
                   "soma_root_nodes": int(np.asarray(f["root_nodes"][()]).sum()) if "root_nodes" in f else -1}
            if A is not None and A.ndim == 2 and A.shape[0] == A.shape[1]:
                nz = np.nan_to_num(A) != 0
                diag = int(np.trace(nz))
                row.update({"n_nos_arquivo": int(A.shape[0]), "summary_diag": diag,
                            "summary_fora_diag": int(nz.sum() - diag),
                            "summary_acima_diag": int(np.triu(nz, 1).sum()),
                            "summary_abaixo_diag": int(np.tril(nz, -1).sum()),
                            "summary_nan": int(np.isnan(A).sum()),
                            "summary_valores": " ".join(str(v) for v in sorted(set(np.nan_to_num(A).ravel().tolist()))[:6])})
        rows.append(row)
        if (k + 1) % 2000 == 0:
            print(f"{k + 1} of {len(ncs)}", flush=True)
    D = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUTPUT / "censo_causaldynamics.csv", index=False)

    L = ["# Structure census of the coupled CausalDynamics — for annex C", "",
         "Generated by `revision/offline/r1_2c_third_party_cd_census.py`. Only structure, ground truth and count of non-finite "
         "values in trajectory 0; no series value; no method was run.", "",
         f"- .nc files: {len(D)} (one graph per file: `adjacency_matrix_summary` has two dimensions in "
         f"{int((D.forma_summary.str.count(',') == 1).sum())} of them).", ""]

    def table(title, series):
        out = [f"## {title}", "", "| Value | Files |", "|---|---|"]
        out += [f"| `{a}` | {b} |" for a, b in sorted(Counter(series.astype(str)).items())]
        return out + [""]

    L += table("Set of fields per file", D["campos"])
    L += table("Dimensions of time_series", D["dims_time_series"])
    L += table("Shape of time_series", D["forma_time_series"])
    L += table("Number of nodes in the file (shape of the summary matrix)", D["n_nos_arquivo"])
    L += table("`systems=` from the path", D["cfg_systems"])
    L += ["## `systems=` from the path × `confounder=` from the path × number of nodes in the file", "",
          pd.crosstab([D.cfg_systems.astype(int), D.cfg_confounder], D.n_nos_arquivo).to_markdown(), ""]
    L += ["## `timelag=` from the path × stored `time_lag`", "",
          pd.crosstab(D.cfg_timelag, D.time_lag_gravado).to_markdown(), ""]
    L += ["## Summary matrix", "",
          f"- distinct values: {dict(Counter(D.summary_valores))}", f"- files with NaN in the matrix: {int((D.summary_nan > 0).sum())}",
          f"- files with a nonzero entry ABOVE the diagonal: {int((D.summary_acima_diag > 0).sum())}",
          f"- files with a nonzero entry BELOW the diagonal: {int((D.summary_abaixo_diag > 0).sum())}",
          f"- files without any off-diagonal entry (graph without an edge between distinct nodes): "
          f"{int((D.summary_fora_diag == 0).sum())}",
          f"- files in which the nonzero diagonal differs from the sum of root_nodes: "
          f"{int((D.summary_diag != D.soma_root_nodes).sum())}", "",
          "Files without an off-diagonal edge, by `systems=` × `confounder=` × nodes in the file:", "",
          pd.crosstab([D.cfg_systems.astype(int), D.cfg_confounder],
                      [D.n_nos_arquivo, D.summary_fora_diag == 0]).to_markdown(), ""]
    L += ["## Non-finite values in trajectory 0 (count; no value read beyond that)", "",
          f"- files with some non-finite value in trajectory 0: {int((D.nao_finitos_trajetoria0 > 0).sum())} of {len(D)}", "",
          pd.crosstab([D.cfg_systems.astype(int), D.cfg_confounder], D.nao_finitos_trajetoria0 > 0).to_markdown(), ""]
    with_nf = D[D.nao_finitos_trajetoria0 > 0]
    L += ["### Pattern of the non-finite values in trajectory 0 (only in the files that have some)", "",
          f"- from the first step with a non-finite value on, ALL steps to the end have a non-finite value: "
          f"{int(with_nf.traj0_nao_finito_ate_o_fim.sum())} of {len(with_nf)} files",
          f"- first step with a non-finite value (of 1000): minimum {int(with_nf.traj0_primeiro_passo_nao_finito.min())}, median "
          f"{int(with_nf.traj0_primeiro_passo_nao_finito.median())}, maximum {int(with_nf.traj0_primeiro_passo_nao_finito.max())}",
          f"- fraction of trajectory 0 that is non-finite: minimum {with_nf.traj0_fracao_nao_finita.min():.4f}, median "
          f"{with_nf.traj0_fracao_nao_finita.median():.4f}, maximum {with_nf.traj0_fracao_nao_finita.max():.4f}",
          f"- files in which some node is entirely non-finite in trajectory 0: "
          f"{int((with_nf.traj0_nos_inteiramente_nao_finitos > 0).sum())}",
          f"- files in which ALL nodes have some non-finite value in trajectory 0: "
          f"{int((with_nf.traj0_nos_com_nao_finito == with_nf.n_nos_arquivo).sum())}", "",
          "Non-finite values in trajectory 0, by `noise=` and by `standardize=` from the path (files with some / total):", "",
          pd.crosstab([D.cfg_noise, D.cfg_standardize], D.nao_finitos_trajetoria0 > 0).to_markdown(), "",
          "By `coupling=` and by system prefix:", "",
          pd.crosstab([D.cfg_coupling, D["arquivo"].map(lambda x: x.rsplit('/', 1)[1].split('_')[0])],
                      D.nao_finitos_trajetoria0 > 0).to_markdown(), "",
          "### Entirely finite trajectories per file (out of 10)", "",
          pd.crosstab([D.cfg_systems.astype(int), D.cfg_confounder], D.traj_inteiramente_finitas).to_markdown(), "",
          f"- files without any entirely finite trajectory: {int((D.traj_inteiramente_finitas == 0).sum())}", "",
          "## Orientation: `root_nodes` against the null columns and rows of `adjacency_matrix`", "",
          "Only in the files in which `root_nodes` has the same length as the number of nodes (root = node without an "
          "incoming edge; if `A[i, j]` = edge from i to j, root <=> entirely null column):", "",
          f"- shapes of root_nodes: {dict(Counter(D.forma_root_nodes.astype(str)))}",
          f"- comparable files: {int(D.raiz_igual_coluna_nula.notna().sum())}",
          f"- root == null column in all nodes: {int((D.raiz_igual_coluna_nula == True).sum())}",   # noqa: E712
          f"- root == null row in all nodes: {int((D.raiz_igual_linha_nula == True).sum())}", ""]   # noqa: E712
    for c in ("cfg_coupling", "cfg_noise", "cfg_standardize", "cfg_activation"):
        L += table(f"`{c[4:]}=` from the path", D[c])
    L += table("Prefix of the file name (system)", D["arquivo"].map(lambda x: x.rsplit("/", 1)[1].split("_")[0]))
    (OUTPUT / "censo_causaldynamics.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\r\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

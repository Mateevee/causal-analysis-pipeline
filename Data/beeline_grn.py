"""
Data.beeline_grn
===============================================================================
SEMI-SYNTHETIC BENCHMARK — BEELINE (Pratapa, Jalihal, Law, Bharadwaj & Murali
2020, "Benchmarking algorithms for gene regulatory network inference from
single-cell transcriptomic data", Nature Methods 17:147-154). Standard
benchmark for gene regulatory network (GRN) inference from single-cell
RNA-seq-like data, with the CURATED networks 'mCAD' (5 genes), 'VSC'
(8 genes), 'HSC' (11 genes) and 'GSD' (19 genes) -- boolean models published
in the developmental-biology literature, simulated with BoolODE (the same
package/methodology used by the BEELINE authors).

WARNING — SAME REGIME AS DREAM4 (Data/dream4.py): the EXPRESSION here is
SIMULATED (BoolODE turns the published boolean rules into differential
equations and adds single-cell-like noise/dropout) -- it is not real
biological data, unlike Data/sachs.py. Only the NETWORK (regulatory
structure) is curated from the experimental literature.

DIFFERENCE from DREAM4: every simulated cell has a PSEUDOTIME
(PseudoTime.csv, computed by BoolODE itself from the simulated trajectory) --
this loader SORTS the cells by pseudotime before assembling the series, so the
row order reflects a genuine differentiation trajectory (not i.i.d. like
Sachs/ALARM), even though the expression is simulated.

DATA FORMAT (as officially distributed, see HOW TO GET THE DATA)
-------------------------------------------------------------------------------
For each (network, run) in <base_dir>/inputs/Curated/<network>/<network>-2000-<run>/:
    ExpressionData.csv : genes in the ROWS, cells in the COLUMNS (transposed
                          with respect to this pipeline's contract -- this
                          loader transposes it).
    PseudoTime.csv      : cells in the rows, 1+ 'PseudoTimeN' columns
                          (simulated branches/lineages; NaN if the cell does
                          not belong to that branch) -- the MEAN of the
                          non-null branches is used as the sorting pseudotime.
<base_dir>/inputs/Curated/<network>/GroundTruthNetwork.csv : columns
    Gene1,Gene2,Type (regulator,target,sign +/-) -- GT by construction
    (known boolean rules), not estimated.

HOW TO GET THE DATA
-------------------------------------------------------------------------------
Not auto-downloaded. Download BEELINE-data.zip (~260 MB) from Zenodo (DOI
10.5281/zenodo.3378975, which redirects to the latest record -- at the time
of writing https://zenodo.org/records/19009603):

    https://zenodo.org/records/19009603/files/BEELINE-data.zip?download=1
        (MD5 9ea93065004f2f12022adeeaf3728edd; the other file of the
         record, BEELINE-Networks.zip, is not needed)

The archive's top-level folder is 'BEELINE-data/', so extracting it inside
"./external_data/beeline_grn/" gives the default path directly:

    ./external_data/beeline_grn/BEELINE-data/inputs/Curated/<network>/
        GroundTruthNetwork.csv
        <network>-2000-<run>/ExpressionData.csv, PseudoTime.csv

(or point base_dir= / the BEELINE_DIR environment variable at any folder
holding the 'inputs/Curated/' subtree). See
https://github.com/Murali-group/Beeline for the full documentation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import ScenarioEntry, register, array_to_pipeline_format_single_property

_BASE_DEFAULT = Path(os.environ.get("BEELINE_DIR", "./external_data/beeline_grn/BEELINE-data"))
_DEFAULT_NETWORKS = ["mCAD", "VSC", "HSC", "GSD"]


def _load_network(network_dir: Path) -> pd.DataFrame:
    path = network_dir / "GroundTruthNetwork.csv"
    if not path.exists():
        raise FileNotFoundError(f"'{path}' not found.")
    return pd.read_csv(path)


def _load_run(run_dir: Path, max_cells: Optional[int]) -> tuple:
    """Reads ExpressionData.csv (genes x cells) + PseudoTime.csv of a run,
    returns (data (T, n_genes) sorted by pseudotime, gene_names)."""
    df_expr = pd.read_csv(run_dir / "ExpressionData.csv", index_col=0)
    genes = list(df_expr.index)

    pt_path = run_dir / "PseudoTime.csv"
    if pt_path.exists():
        df_pt = pd.read_csv(pt_path, index_col=0)
        pseudotime = df_pt.mean(axis=1, skipna=True)
        sorted_cells = pseudotime.reindex(df_expr.columns).sort_values().index
        sorted_cells = [c for c in sorted_cells if c in df_expr.columns]
    else:
        sorted_cells = list(df_expr.columns)

    if max_cells is not None and max_cells < len(sorted_cells):
        sorted_cells = sorted_cells[:max_cells]

    data = df_expr[sorted_cells].T.to_numpy(dtype=float)
    return data, genes


@register("beeline_grn")
def generate_beeline_grn_benchmark_scenarios(
    *,
    networks: Optional[List[str]] = None,
    max_runs_per_network: Optional[int] = 3,
    max_cells: Optional[int] = 500,
    base_dir: Optional[str] = None,
    name_prefix: str = "benchmark_beeline",
) -> List[dict]:
    """
    Generates scenarios from the CURATED BEELINE networks (Pratapa et al.
    2020) -- see the module docstring for the simulated-expression caveat
    (same regime as DREAM4) and the data format.

    Parameters
    ----------
    networks             : subset of ['mCAD','VSC','HSC','GSD'] to load
                           (default: all 4).
    max_runs_per_network : cap on the number of runs/scenarios per network
                           (each network has 10 independent runs in the
                           official dataset). None = all.
    max_cells            : cap on the number of cells (rows) per scenario,
                           KEEPING the pseudotime order (not a random
                           subsample) -- each run has ~2000 cells. None =
                           all.
    base_dir             : root holding 'inputs/Curated/', extracted from
                           BEELINE-data.zip (default: env BEELINE_DIR or
                           "./external_data/beeline_grn/BEELINE-data").
    name_prefix          : prefix of each generated scenario's name/folder.

    Returns a list of dicts (ScenarioEntry.to_dict()).
    """
    root = Path(base_dir) if base_dir else _BASE_DEFAULT
    curated_dir = root / "inputs" / "Curated"
    networks = networks if networks is not None else _DEFAULT_NETWORKS

    scenarios = []
    for network in networks:
        network_dir = curated_dir / network
        if not network_dir.exists():
            print(f"  [WARNING] network '{network}' not found in '{network_dir}' -- skipping.")
            continue

        df_network = _load_network(network_dir)
        col_reg, col_target = df_network.columns[0], df_network.columns[1]

        run_dirs = sorted(
            p for p in network_dir.iterdir()
            if p.is_dir() and p.name.startswith(f"{network}-")
        )
        if max_runs_per_network is not None:
            run_dirs = run_dirs[:max_runs_per_network]

        for run_dir in run_dirs:
            data, genes = _load_run(run_dir, max_cells)
            gene_idx = {g: i for i, g in enumerate(genes)}
            causal_pairs_idx = {
                (gene_idx[r], gene_idx[a])
                for r, a in zip(df_network[col_reg], df_network[col_target])
                if r in gene_idx and a in gene_idx and r != a
            }

            df_long, df_gt = array_to_pipeline_format_single_property(
                data, genes, causal_pairs_idx,
                start_date="2015-01-01", freq="min", property_name="EXPR",
            )

            scenario_name = f"{name_prefix}_{network}_{run_dir.name}"
            params = {
                "source": "beeline_grn",
                "reference": "Pratapa et al. 2020, Nature Methods 17:147-154",
                "network": network,
                "run": run_dir.name,
                "n_genes": len(genes),
                "n_cells_used": data.shape[0],
                "n_causal_links": len(causal_pairs_idx),
                "ground_truth_type": "boolean rules curated from the "
                                      "literature (by construction), not "
                                      "estimated",
                "caveat": "expression SIMULATED with BoolODE, not real data "
                          "-- same regime as DREAM4; cells SORTED by "
                          "simulated pseudotime (genuine trajectory, not "
                          "i.i.d.)",
            }

            scenarios.append(ScenarioEntry(
                name=scenario_name, df=df_long, gt=df_gt,
                _scenario=f"beeline_{network}", _params=params,
            ).to_dict())

        if run_dirs:
            print(f"  [semi-synthetic-benchmark] beeline_grn/{network} loaded — "
                  f"{len(run_dirs)} run(s), {len(genes)} genes | GT: "
                  f"{df_network.shape[0]} interactions (curated network, BoolODE)")

    return scenarios


if __name__ == "__main__":
    scenarios = generate_beeline_grn_benchmark_scenarios(max_runs_per_network=1, max_cells=200)
    for c in scenarios:
        n_edges = int(c["gt"]["causal_gt"].sum())
        print(f"  {c['name']}: df={c['df'].shape}, {n_edges}/{len(c['gt'])} real edges")

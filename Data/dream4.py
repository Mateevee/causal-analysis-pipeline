"""
Data.dream4
===============================================================================
SEMI-SYNTHETIC BENCHMARK — DREAM4 In Silico Network Challenge (Marbach,
Schaffter, Mattiussi & Floreano 2009, "Generating realistic in silico gene
networks for performance assessment of reverse engineering methods", J.
Comput. Biol. 16(2); Marbach et al. 2010, "Revealing strengths and
weaknesses of methods for gene network inference", PNAS 107(14)).

NOT real measurements: the series are SIMULATED by GeneNetWeaver, but on
network topologies extracted from REAL transcriptional regulatory networks
(E. coli and S. cerevisiae), with nonlinear kinetic models of transcription
and translation, stochastic dynamics (SDE) and measurement noise. It is the
classic gene-regulation benchmark of the causal discovery literature and
adds a biological, strongly nonlinear domain to the portfolio.

Five networks of 10 genes and five of 100 genes. GROUND TRUTH: the directed
regulatory network used to simulate each one (regulator -> target), known
BY CONSTRUCTION. Unlike most sources of this project it contains feedback
loops, so the graph is not acyclic.

TIME AXIS -- SHORT, INDEPENDENT SERIES
-------------------------------------------------------------------------------
Each network comes with several INDEPENDENT time-series experiments (5 for
size 10, 10 for size 100), each with only 21 time points (t = 0, 50, ...,
1000). In each experiment a perturbation is applied to about a third of the
genes at t = 0, held during the first half and removed at t = 500, after
which the network relaxes back; which genes were perturbed is not disclosed
in the time-series file, so the perturbation acts as an unobserved common
driver within each experiment.

21 points are far too few for this pipeline, so, by default
(mode="concatenated"), the experiments of a network are glued one after the
other into a single series (105 points for size 10, 210 for size 100) --
common practice with this benchmark, with a known cost: one transition in
every 21 (the junction between two experiments) is artificial, and the
series jumps there. mode="separate_experiments" returns each 21-point
experiment as its own scenario instead (same convention as
Data/causaltime.py), for methods that can handle series that short.

HOW TO GET THE DATA
-------------------------------------------------------------------------------
Not auto-downloaded. The data is on Synapse, project "DREAM4 - In Silico
Network Challenge" (syn3049712, https://www.synapse.org/Synapse:syn3049712);
downloading requires a free Synapse account. Under "sub challenge 1":

    training data/DREAM4_InSilico_Size10.zip            (time series, size 10)
    training data/DREAM4_InSilico_Size100.zip           (time series, size 100)
    gold standard/DREAM4_InSilicoNetworks_GoldStandard.zip  (true networks)

(the same files are also bundled, with the same names, in the archive
"DREAM4 in-silico challenge.zip" distributed with GeneNetWeaver,
https://gnw.sourceforge.net -- folders "DREAM4 training data" and "DREAM4
gold standards"). Some copies of the gold standards name the file of
network K "DREAM4_GoldStandard_InSilico_Size<S>_<K>.tsv" instead; the
content is the same -- rename it to the name below. The archives keep each
network in its own subfolder; from them, copy ONLY the
two files below per network, FLAT (no subfolders), into <base_dir>/
(default: env DREAM4_DIR or "./external_data/dream4/"):

    insilico_size<S>_<K>_timeseries.tsv    'Time' + one column per gene
                                           (G1..G<S>); experiments
                                           separated by a blank line
    insilico_size<S>_<K>_goldstandard.tsv  regulator, target, 0/1 (no
                                           header; one line per ordered pair)

with S in {10, 100} and K in 1..5 -- 20 files in total, e.g.

    ./external_data/dream4/insilico_size10_1_timeseries.tsv
    ./external_data/dream4/insilico_size10_1_goldstandard.tsv

The other files of the archives (knockouts, knockdowns, multifactorial,
wildtype, ...) are not used. Only size 10 is loaded by default, so the 10
size-10 files are the minimum setup.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .base import ScenarioEntry, register, array_to_pipeline_format_single_property

_BASE_DEFAULT = Path(os.environ.get("DREAM4_DIR", "./external_data/dream4"))

_POINTS_PER_EXPERIMENT = 21


def _load_experiments(path: Path) -> tuple:
    """Reads a DREAM4 time-series file and returns (gene names, list of
    (21, n_genes) arrays -- one per experiment). The experiments are told
    apart by the 'Time' column restarting, which does not depend on how the
    blank separator lines are parsed."""
    table = pd.read_csv(path, sep="\t", skip_blank_lines=True)
    genes = [c for c in table.columns if c != "Time"]
    time = table["Time"].to_numpy()
    starts = np.flatnonzero(np.r_[True, np.diff(time) < 0])
    experiments = np.split(table[genes].to_numpy(dtype=float), starts[1:])
    lengths = {len(e) for e in experiments}
    if lengths != {_POINTS_PER_EXPERIMENT}:
        raise ValueError(
            f"'{path}' has experiments with {sorted(lengths)} time points, "
            f"expected {_POINTS_PER_EXPERIMENT} each."
        )
    return genes, experiments


def _load_gold_standard(path: Path, genes: List[str]) -> set:
    table = pd.read_csv(path, sep="\t", header=None,
                        names=["regulator", "target", "edge"])
    unknown = (set(table["regulator"]) | set(table["target"])) - set(genes)
    if unknown:
        raise ValueError(
            f"'{path}' references genes absent from the time series: "
            f"{sorted(unknown)[:5]}..."
        )
    index = {gene: k for k, gene in enumerate(genes)}
    edges = table[table["edge"] == 1]
    return {(index[a], index[b])
            for a, b in zip(edges["regulator"], edges["target"]) if a != b}


@register("dream4")
def generate_dream4_benchmark_scenarios(
    *,
    sizes: Optional[Sequence[int]] = None,
    networks: Optional[Sequence[int]] = None,
    mode: str = "concatenated",
    base_dir: Optional[Union[str, Path]] = None,
    start_date: str = "2015-01-01",
    freq: str = "D",
    name_prefix: str = "benchmark_dream4",
) -> List[dict]:
    """
    Generates scenarios from the DREAM4 in silico gene networks, in the
    format consumed by run_all()/validate_dataset() (same signature
    described in Data.base). See the module docstring for the source, the
    ground truth and the caveat about the short, independent series.

    Parameters
    ----------
    sizes      : network sizes to load, 10 and/or 100 genes (default [10];
                 size 100 means 100 entities per scenario).
    networks   : which of the five networks of each size (default 1..5).
    mode       : "concatenated" (default, one scenario per network with all
                 its experiments glued in file order) or
                 "separate_experiments" (one 21-point scenario per
                 experiment, all sharing the network's ground truth).
    base_dir   : folder with the .tsv files (default: env DREAM4_DIR or
                 "./external_data/dream4").
    start_date, freq : synthetic dates for the 'Date' column (the simulated
                 time is in arbitrary units, 50 per sample).
    name_prefix : prefix for the name/folder of each generated scenario.

    Returns a list of dicts (ScenarioEntry.to_dict()).
    """
    if mode not in ("concatenated", "separate_experiments"):
        raise ValueError(
            f"mode='{mode}' is invalid. Use 'concatenated' or "
            f"'separate_experiments'."
        )

    root = Path(base_dir) if base_dir is not None else _BASE_DEFAULT
    sizes = list(sizes) if sizes is not None else [10]
    networks = list(networks) if networks is not None else [1, 2, 3, 4, 5]

    scenarios: List[dict] = []
    for size in sizes:
        for network in networks:
            stem = f"insilico_size{size}_{network}"
            ts_path = root / f"{stem}_timeseries.tsv"
            gt_path = root / f"{stem}_goldstandard.tsv"
            if not ts_path.exists() or not gt_path.exists():
                raise FileNotFoundError(
                    f"DREAM4 files not found in '{root}': '{ts_path.name}' / "
                    f"'{gt_path.name}'. Download the DREAM4 In Silico Network "
                    f"Challenge data (Synapse syn3049712) and copy those "
                    f"files, flat, into that folder (or point base_dir= / "
                    f"the DREAM4_DIR environment variable) -- see this "
                    f"module's docstring."
                )

            genes, experiments = _load_experiments(ts_path)
            causal_pairs_idx = _load_gold_standard(gt_path, genes)

            base_params = {
                "source": "dream4",
                "reference": "Marbach, Schaffter, Mattiussi & Floreano 2009, "
                             "J. Comput. Biol. 16(2); Marbach et al. 2010, "
                             "PNAS 107(14) (DREAM4 In Silico Network "
                             "Challenge)",
                "data_file": str(ts_path),
                "ground_truth_file": str(gt_path),
                "size": size,
                "network": network,
                "mode": mode,
                "n_experiments": len(experiments),
                "points_per_experiment": _POINTS_PER_EXPERIMENT,
                "n_entities": len(genes),
                "n_causal_links": len(causal_pairs_idx),
                "simulated": True,
                "ground_truth_type": "regulatory network used by the "
                                      "simulator (by construction); "
                                      "topology extracted from real "
                                      "E. coli / yeast networks, contains "
                                      "feedback loops",
            }

            if mode == "concatenated":
                blocks = [("", np.vstack(experiments), None)]
            else:
                blocks = [(f"_exp{k + 1}", data, k + 1)
                          for k, data in enumerate(experiments)]

            for suffix, data, experiment in blocks:
                name = f"{name_prefix}_size{size}_net{network}{suffix}"
                df_long, df_gt = array_to_pipeline_format_single_property(
                    data, genes, causal_pairs_idx,
                    start_date=start_date, freq=freq, property_name="EXPR",
                )
                scenarios.append(ScenarioEntry(
                    name=name, df=df_long, gt=df_gt,
                    _scenario=f"dream4_size{size}",
                    _params={**base_params, "experiment": experiment,
                             "n_points": data.shape[0]},
                ).to_dict())

            print(f"  [benchmark] {name_prefix}_size{size}_net{network} loaded "
                  f"— {len(genes)} genes, {len(experiments)} experiments x "
                  f"{_POINTS_PER_EXPERIMENT} points ({mode}) | "
                  f"{len(causal_pairs_idx)} causal links (ground truth by "
                  f"construction, DREAM4)")

    return scenarios


if __name__ == "__main__":
    # Quick manual test (needs the files in DREAM4_DIR / the default path).
    for c in generate_dream4_benchmark_scenarios(sizes=[10, 100]):
        n_edges = int(c["gt"]["causal_gt"].sum())
        print(f"  {c['name']}: df={c['df'].shape}, "
              f"{n_edges}/{len(c['gt'])} real edges")

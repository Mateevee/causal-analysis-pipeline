"""
Data.sachs
===============================================================================
REAL BENCHMARK — SACHS ET AL. 2005 ("Causal Protein-Signaling Networks
Derived from Multiparameter Single-Cell Data", Science 308(5721):523-529) —
the most cited real-biological-data benchmark for causal discovery (protein
signaling network via multiparameter flow cytometry).

GROUND TRUTH: an 11-node / 20-edge network established from biochemical
knowledge + pathway-specific intervention experiments (inhibitors/activators),
NOT by automated discovery. Reproduced here from the curated reconstruction
distributed by CMU's Causal Learning group (repository
`cmu-phil/example-causal-datasets`, used with the Tetrad software) --
`real/sachs/ground.truth/sachs.2005.ground.truth.graph.txt` -- the same
source the continuous data used here by default came from. It is a somewhat
denser reconstruction (20 edges) than the 17-edge diagram most commonly
reproduced from Fig. 3 of the original paper / the `sachs` dataset from the
R package `bnlearn` -- both versions appear in the causal-discovery
literature (e.g. NOTEARS, DAG-GNN, PC/GES/score-based benchmarks); this
module uses the 20-edge reconstruction because it ships alongside an already
verified, publicly accessible data file.

WARNING -- THIS IS NOT A TIME SERIES, UNLIKE THE OTHER PROVIDERS IN Data/.
Each row is one cell (i.i.d. sample), not a temporal sequence; row order is
treated as a fake "time" axis purely to fit the pipeline's common contract
(Date/Name/VAL). Use a low max_lag (or 0); this dataset is best suited for
the CONTEMPORANEOUS STAGE (DML/effect estimation) rather than lag-detection.

HOW TO OBTAIN THE DATA
-------------------------------------------------------------------------------
Recommended primary source (the same one used to generate this project's
local CSV): the `cmu-phil/example-causal-datasets` repository on GitHub
(maintained by CMU's Causal Learning group / Tetrad) --
    real/sachs/data/sachs.2005.continuous.txt        (7466 cells, tab-separated,
                                                        columns already lowercase:
                                                        raf mek plc pip2 pip3
                                                        erk akt pka pkc p38 jnk)
    real/sachs/ground.truth/sachs.2005.ground.truth.graph.txt  (20 edges)
Not auto-downloaded. Download the data .txt and convert tab->comma into the
CSV format this loader expects, saving it to <csv_path> (default in
main_heldout.py: SACHS_CSV_PATH = "./external_data/sachs/sachs.csv"). From
the repository root, this does the whole step (creates the folder, too):

    python -c "import os, pandas as pd; os.makedirs('external_data/sachs', exist_ok=True); pd.read_csv('https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/sachs/data/sachs.2005.continuous.txt', sep='\t').to_csv('external_data/sachs/sachs.csv', index=False)"

The resulting CSV has the header "raf,mek,plc,pip2,pip3,erk,akt,pka,pkc,
p38,jnk" and 7466 rows.

Equivalent alternatives (column names vary, but see
_RECOGNIZED_COLUMN_MAP below for the accepted mapping):
    - R package `bnlearn` (Scutari): install.packages("bnlearn"); data(sachs);
      write.csv(sachs, "sachs.csv", row.names=FALSE) -- produces columns Raf,
      Mek, Plcg, PIP2, PIP3, Erk, Akt, PKA, PKC, P38, Jnk.
    - Original paper's supplementary material (Sachs et al. 2005, Science,
      DOI 10.1126/science.1105809), with the original names: praf, pmek,
      plcg, PIP2, PIP3, p44/42, pakts473, PKA, PKC, P38, pjnk.

This loader accepts ANY of these column-naming conventions and normalizes
them to the canonical names used here: raf, mek, plc, pip2, pip3, erk, akt,
pka, pkc, p38, jnk (matching the cmu-phil file, the primary source).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import array_to_pipeline_format_single_property, register

# Canonical names (matching the cmu-phil file, the primary source -- see
# the module docstring) and the accepted variants when reading the CSV
# (key = canonical name, values = recognized variants, compared
# case-insensitively and ignoring spaces/underscores).
_RECOGNIZED_COLUMN_MAP = {
    "raf":  {"raf", "praf"},
    "mek":  {"mek", "pmek"},
    "plc":  {"plc", "plcg"},
    "pip2": {"pip2"},
    "pip3": {"pip3"},
    "erk":  {"erk", "p44/42", "p44.42", "p4442"},
    "akt":  {"akt", "pakts473"},
    "pka":  {"pka"},
    "pkc":  {"pkc"},
    "p38":  {"p38"},
    "jnk":  {"jnk", "pjnk"},
}

# Reference network -- 11 nodes, 20 edges -- reproduced from
# real/sachs/ground.truth/sachs.2005.ground.truth.graph.txt in the
# cmu-phil/example-causal-datasets repository (see the module docstring for
# the source and the note on the alternative 17-edge version used by other
# papers). Established by biochemical knowledge + specific interventions,
# NOT statistically estimated.
_SACHS_GT_EDGES = {
    ("erk", "akt"), ("mek", "erk"),
    ("pip2", "pkc"), ("pip3", "akt"), ("pip3", "pip2"), ("pip3", "plc"),
    ("pka", "akt"), ("pka", "erk"), ("pka", "jnk"), ("pka", "mek"),
    ("pka", "p38"), ("pka", "raf"),
    ("pkc", "jnk"), ("pkc", "mek"), ("pkc", "p38"), ("pkc", "pka"), ("pkc", "raf"),
    ("plc", "pip2"), ("plc", "pkc"),
    ("raf", "mek"),
}

_CANONICAL_NODES = list(_RECOGNIZED_COLUMN_MAP.keys())


def _normalize_column_name(col: str) -> str:
    return col.strip().lower().replace(" ", "").replace("_", "")


def _map_columns(csv_columns) -> dict:
    """Returns {canonical_name: original_name_in_csv} for the CSV columns
    that match some variant recognized in _RECOGNIZED_COLUMN_MAP."""
    normalized = {_normalize_column_name(c): c for c in csv_columns}
    mapping = {}
    for canonical, variants in _RECOGNIZED_COLUMN_MAP.items():
        for variant in variants:
            if variant in normalized:
                mapping[canonical] = normalized[variant]
                break
    return mapping


@register("sachs")
def load_sachs_benchmark(
    csv_path: str,
    name: str = "real_sachs_proteins",
    max_rows: Optional[int] = None,
    sampling_seed: int = 0,
) -> dict:
    """
    Loads the Sachs et al. 2005 benchmark (protein signaling network, flow
    cytometry) from a CSV with (at least) the 11 protein columns recognized
    in _RECOGNIZED_COLUMN_MAP (names from the cmu-phil file, from bnlearn, OR
    from the original paper -- see the module docstring).

    GROUND TRUTH: an 11-node / 20-edge network (see _SACHS_GT_EDGES),
    established from domain knowledge + interventions -- not estimated.

    IMPORTANT: this dataset has NO real time axis (each row is a cell,
    sampled essentially i.i.d. within each experimental condition) -- see the
    WARNING in the module docstring. The CSV row order is used as a
    pseudo-time solely to fit the pipeline's common contract; do not
    interpret lag findings as real dynamics.

    Parameters
    ----------
    csv_path : path to the CSV with the protein columns (see the module
               docstring for sources: cmu-phil/example-causal-datasets,
               bnlearn, or the original supplementary material).
    max_rows : if given, randomly subsamples down to this number of rows
               (the original dataset has ~5400-7466 cells -- running the
               full pipeline over all of them can be slow/unnecessary for a
               test run). None = use all rows.
    sampling_seed : RNG seed used for subsampling (if max_rows is given);
               ignored otherwise.

    Raises FileNotFoundError if the CSV does not exist, and ValueError if
    fewer than 11 recognized columns are found.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"File '{csv_path}' not found. Download "
            f"real/sachs/data/sachs.2005.continuous.txt from the "
            f"cmu-phil/example-causal-datasets repository (GitHub) and "
            f"convert tab->csv, or generate it via bnlearn (R: data(sachs); "
            f"write.csv(sachs, 'sachs.csv', row.names=FALSE)) -- see this "
            f"module's docstring."
        )

    df_raw = pd.read_csv(path)
    column_map = _map_columns(df_raw.columns)

    missing = [n for n in _CANONICAL_NODES if n not in column_map]
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' has no recognized columns for: "
            f"{missing}. Columns found: {list(df_raw.columns)}. "
            f"[FIX] add the missing name variants to "
            f"_RECOGNIZED_COLUMN_MAP (Data/sachs.py) if the CSV uses a "
            f"different naming convention."
        )

    df_proteins = df_raw[[column_map[n] for n in _CANONICAL_NODES]].copy()
    df_proteins.columns = _CANONICAL_NODES
    df_proteins = df_proteins.apply(pd.to_numeric, errors="coerce")

    n_original_rows = len(df_proteins)
    if max_rows is not None and max_rows < n_original_rows:
        rng = np.random.default_rng(sampling_seed)
        idx = np.sort(rng.choice(n_original_rows, size=max_rows, replace=False))
        df_proteins = df_proteins.iloc[idx].reset_index(drop=True)

    df_proteins = df_proteins.dropna()
    if df_proteins.empty:
        raise ValueError(
            f"CSV '{csv_path}' became empty after removing rows with "
            f"non-numeric/NaN values in the 11 protein columns."
        )

    data = df_proteins.to_numpy(dtype=float)
    causal_pairs_idx = {
        (_CANONICAL_NODES.index(o), _CANONICAL_NODES.index(d))
        for o, d in _SACHS_GT_EDGES
    }

    df_long, df_gt = array_to_pipeline_format_single_property(
        data, _CANONICAL_NODES, causal_pairs_idx,
        start_date="2015-01-01", freq="min", property_name="EXPR",
    )

    params = {
        "source": "sachs_2005",
        "reference": "Sachs et al. 2005, Science 308(5721):523-529 "
                     "(DOI 10.1126/science.1105809)",
        "data_file": str(path),
        "n_original_rows": n_original_rows,
        "n_rows_used": len(df_proteins),
        "n_entities": len(_CANONICAL_NODES),
        "n_causal_links": len(causal_pairs_idx),
        "ground_truth_type": "biochemical knowledge + specific "
                              "interventions; 20-edge graph via "
                              "cmu-phil/example-causal-datasets, not "
                              "estimated",
        "caveat": "per-cell i.i.d. data, with NO real time axis -- row "
                  "order used as pseudo-time only to fit the pipeline "
                  "format; there is no real lag dynamic to recover",
    }

    print(f"  [real-benchmark] {name} loaded — 11 proteins x "
          f"{len(df_proteins)} cells | GT: Sachs et al. 2005 consensus "
          f"network ({len(causal_pairs_idx)} edges) | WARNING: no real "
          f"time axis, see Data/sachs.py docstring")

    return {"name": name, "df": df_long, "gt": df_gt, "_scenario": None,
            "_params": params}

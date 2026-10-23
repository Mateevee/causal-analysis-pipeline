"""
Data.tuebingen_pairs
===============================================================================
REAL BENCHMARK — TUEBINGEN CAUSE-EFFECT PAIRS (Mooij, Peters, Janzing,
Zscheischler & Schoelkopf 2016, "Distinguishing Cause from Effect Using
Observational Data: Methods and Benchmarks", JMLR 17(32):1-102). THE
reference benchmark for bivariate causal DIRECTION inference — more than 100
pairs of real-world variables (climate, biology, economics, physics...), each
with the causal direction established by DOMAIN KNOWLEDGE (not estimated),
maintained by the Max Planck Institute for Biological Cybernetics.

GROUND TRUTH: for each pair, the direction (cause_variable ->
effect_variable) is fixed by the official metadata (pairmeta.txt: the column
range that is "cause" vs "effect" in each pairNNNN.txt file) — it does not
depend on the column order in the raw file nor on the variable names.

WARNING — SAME CAVEAT AS Data/sachs.py and Data/alarm.py: most pairs are
CROSS-SECTIONAL data (one observation per row — e.g. a weather station, an
individual, a car), NOT a time series. The row order is used as pseudo-time
only to fit the pipeline format; there is no real lag dynamic to recover.
This benchmark exercises the CONTEMPORANEOUS layer of the pipeline, not the
lag detection.

ONLY BIVARIATE PAIRS (1 cause column -> 1 effect column) are used here — of
the ~108 pairs of the dataset, a minority (6) is multivariate (e.g. pair0052
has 4 columns on each side); those are OUT of the scope of this loader — see
_load_bivariate_pairmeta.

HOW THE DATA IS OBTAINED
-------------------------------------------------------------------------------
Downloads automatically (with disk cache) the official MPI Tuebingen package:
    https://webdav.tuebingen.mpg.de/cause-effect/pairs.zip
    https://webdav.tuebingen.mpg.de/cause-effect/pairmeta.txt
Requires internet on the first run (uses cache_dir afterwards).
"""
from __future__ import annotations

import io
import os
import urllib.request as _urlreq_bench
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import ScenarioEntry, register, array_to_pipeline_format_single_property

_PAIRS_ZIP_URL = "https://webdav.tuebingen.mpg.de/cause-effect/pairs.zip"
_PAIRMETA_URL = "https://webdav.tuebingen.mpg.de/cause-effect/pairmeta.txt"
_BASE_DEFAULT = Path(os.environ.get("TUEBINGEN_PAIRS_DIR", "./external_data/tuebingen_pairs"))


def _download_and_extract(cache_dir: Path, timeout: int = 60) -> Path:
    """Downloads (with cache) pairs.zip + pairmeta.txt and extracts the zip
    into cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "pairmeta.txt"
    already_extracted = (cache_dir / "pair0001.txt").exists()

    if not already_extracted:
        try:
            with _urlreq_bench.urlopen(_PAIRS_ZIP_URL, timeout=timeout) as resp:
                zip_content = resp.read()
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {_PAIRS_ZIP_URL}. Check your internet "
                f"connection, or download/extract it manually into "
                f"'{cache_dir}'. Original error: {e}"
            ) from e
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            zf.extractall(cache_dir)

    if not meta_path.exists():
        try:
            with _urlreq_bench.urlopen(_PAIRMETA_URL, timeout=timeout) as resp:
                meta_path.write_bytes(resp.read())
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {_PAIRMETA_URL}. Check your internet "
                f"connection, or download it manually to '{meta_path}'. "
                f"Original error: {e}"
            ) from e

    return cache_dir


def _load_bivariate_pairmeta(meta_path: Path) -> pd.DataFrame:
    """Reads pairmeta.txt (columns: id, cause_start, cause_end, effect_start,
    effect_end, weight) and keeps only the BIVARIATE pairs (1 cause column,
    1 effect column)."""
    df = pd.read_csv(
        meta_path, sep=r"\s+", header=None,
        names=["id", "cause_start", "cause_end", "effect_start", "effect_end", "weight"],
    )
    bivariate = (df["cause_end"] == df["cause_start"]) & (df["effect_end"] == df["effect_start"])
    return df[bivariate].reset_index(drop=True)


@register("tuebingen_pairs")
def generate_tuebingen_pairs_benchmark_scenarios(
    *,
    max_pairs: Optional[int] = None,
    max_rows: Optional[int] = 500,
    sampling_seed: int = 0,
    cache_dir: Optional[str] = None,
    name_prefix: str = "benchmark_tuebingen",
) -> List[dict]:
    """
    Generates one ScenarioEntry per bivariate pair of the Tuebingen
    Cause-Effect Pairs benchmark (Mooij et al. 2016) — see the module
    docstring for the benchmark, the no-real-time-axis caveat and the
    automatic download (with cache).

    Parameters
    ----------
    max_pairs     : if given, uses only the first N bivariate pairs (by
                    dataset index); None = all (~102).
    max_rows      : if given, randomly subsamples up to this many rows per
                    pair (some pairs have >10000 observations — keeps the
                    runtime comparable to the other scenarios). None = all.
    sampling_seed : seed of the RNG used by the row subsampling.
    cache_dir     : download/cache folder (default: env TUEBINGEN_PAIRS_DIR
                    or "./external_data/tuebingen_pairs").
    name_prefix   : prefix of each generated scenario's name/folder.

    GROUND TRUTH: always causal_gt = {(0,1)} (the first entity, "CAUSE",
    causes the second, "EFFECT") — the real direction (which may correspond
    to either of the two original columns of the file) was already resolved
    by the official metadata (pairmeta.txt) before the entity was labelled.

    Returns a list of dicts (ScenarioEntry.to_dict()).
    """
    cache = Path(cache_dir) if cache_dir else _BASE_DEFAULT
    _download_and_extract(cache)
    bivariate_meta = _load_bivariate_pairmeta(cache / "pairmeta.txt")

    if max_pairs is not None:
        bivariate_meta = bivariate_meta.iloc[:max_pairs]

    rng = np.random.default_rng(sampling_seed)
    scenarios = []

    for _, row in bivariate_meta.iterrows():
        pair_id = int(row["id"])
        pair_path = cache / f"pair{pair_id:04d}.txt"
        if not pair_path.exists():
            continue

        raw = pd.read_csv(pair_path, sep=r"\s+", header=None)
        cause_col = int(row["cause_start"]) - 1
        effect_col = int(row["effect_start"]) - 1

        df_pair = raw[[cause_col, effect_col]].apply(pd.to_numeric, errors="coerce")
        df_pair.columns = ["CAUSE", "EFFECT"]
        df_pair = df_pair.dropna()
        if df_pair.empty:
            continue

        n_original_rows = len(df_pair)
        if max_rows is not None and max_rows < n_original_rows:
            idx = np.sort(rng.choice(n_original_rows, size=max_rows, replace=False))
            df_pair = df_pair.iloc[idx].reset_index(drop=True)

        data = df_pair.to_numpy(dtype=float)
        df_long, df_gt = array_to_pipeline_format_single_property(
            data, ["CAUSE", "EFFECT"], {(0, 1)},
            start_date="2015-01-01", freq="min", property_name="VAL",
        )

        params = {
            "source": "tuebingen_cause_effect_pairs",
            "reference": "Mooij et al. 2016, JMLR 17(32):1-102",
            "pair_id": pair_id,
            "dataset_weight": float(row["weight"]),
            "n_original_rows": n_original_rows,
            "n_rows_used": len(df_pair),
            "ground_truth_type": "causal direction established by domain "
                                  "knowledge (official metadata "
                                  "pairmeta.txt), not estimated",
            "caveat": "most pairs are cross-sectional (i.i.d.) data, with NO "
                      "real time axis -- row order used as pseudo-time only "
                      "to fit the pipeline format",
        }

        scenarios.append(ScenarioEntry(
            name=f"{name_prefix}_pair{pair_id:04d}",
            df=df_long, gt=df_gt, _scenario=f"tuebingen_pair{pair_id:04d}",
            _params=params,
        ).to_dict())

    print(f"  [real-benchmark] tuebingen_pairs loaded — {len(scenarios)} "
          f"bivariate pair(s) (of {len(bivariate_meta)} eligible) | GT: "
          f"cause->effect direction per pair (domain knowledge) | WARNING: "
          f"no real time axis, see Data/tuebingen_pairs.py docstring")

    return scenarios


if __name__ == "__main__":
    scenarios = generate_tuebingen_pairs_benchmark_scenarios(max_pairs=5)
    for c in scenarios:
        n_edges = int(c["gt"]["causal_gt"].sum())
        print(f"  {c['name']}: df={c['df'].shape}, {n_edges}/{len(c['gt'])} real edges")

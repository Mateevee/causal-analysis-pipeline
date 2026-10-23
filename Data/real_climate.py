"""
Data.real_climate
===============================================================================
REAL BENCHMARK — TIGRAMITE / CLIMATE (Walker Circulation, equatorial
Pacific), the same dataset used by Runge et al. in the official tigramite
tutorial (Nat. Rev. Earth Environ. 10, 2553, 2023). ERA5 reanalysis data
(winds/vertical motion) downloaded automatically from the tigramite GitHub
repository.

GROUND TRUTH: causal chain CPAC -> WCPAC -> WPAC, established by physical
domain knowledge (not by automatic discovery). IMPORTANT: in the original
study these relations are treated as mostly CONTEMPORANEOUS (same bimonthly
period), not necessarily lagged — unlike the X_(t-tau) -> Y_t regime this
pipeline was designed to detect. This is, in itself, a useful test: it
shows how the pipeline behaves when the real causality has a strong
contemporaneous component.

Requires internet access (raw.githubusercontent.com) on first run; uses an
on-disk cache if cache_dir is provided.
"""
from __future__ import annotations

import urllib.request as _urlreq_bench
from pathlib import Path
from typing import Optional

import numpy as np

from .base import array_to_pipeline_format_single_property, register

_CLIMATE_TIGRAMITE_BASE_URL = (
    "https://raw.githubusercontent.com/jakobrunge/tigramite/master/"
    "tutorials/case_studies/climate_data"
)
# Same file -> variable mapping as the tutorial (climate_case_study.ipynb:
# var_names = ['WPAC', 'WCPAC', 'CPAC'] for the three files below, in that
# order), and as the regions in each file's KNMI header: WPAC = ascending
# air (omega at 700 hPa) over the West Pacific, WCPAC = surface zonal wind
# in the West-Central Pacific, CPAC = descending air over the Central
# Pacific.
_CLIMATE_TIGRAMITE_FILES = {
    "CPAC":  "iera5_w700_-160--120E_5--5N_n.dat",
    "WCPAC":"iera5_u10m_160-180E_5--5N_n.dat",
    "WPAC":  "iera5_w700_130-150E_20-0N_n.dat",
}


def _download_text_file(url: str, cache_path: Optional[Path] = None,
                         timeout: int = 30) -> str:
    """Downloads a text file via HTTP, with optional on-disk cache."""
    if cache_path is not None and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    with _urlreq_bench.urlopen(url, timeout=timeout) as resp:
        content = resp.read().decode("utf-8")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding="utf-8")
    return content


def _seasonal_anomaly(data: np.ndarray, cycle_length: int = 12) -> np.ndarray:
    """
    Removes the seasonal mean and divides by the seasonal standard
    deviation, calendar month by calendar month -- the tutorial's
    `anomalize(..., divide_by_std=True)`. Rows must start at the first
    month of the cycle (January).
    """
    anomaly = np.array(data, dtype=float, copy=True)
    for m in range(cycle_length):
        block = anomaly[m::cycle_length]
        anomaly[m::cycle_length] = (block - block.mean(axis=0)) / block.std(axis=0)
    return anomaly


@register("real_climate")
def load_tigramite_climate_benchmark(
    n_years: int = 72,
    start_year: int = 1950,
    cache_dir: Optional[str] = None,
    name: str = "real_tigramite_climate_walker",
    seasonal_anomaly: bool = True,
) -> dict:
    """
    Loads the real Walker Circulation case study (equatorial Pacific), the
    same dataset used by Runge et al. in the official tigramite tutorial
    (Nat. Rev. Earth Environ. 10, 2553, 2023). ERA5 reanalysis data
    (winds/vertical motion) downloaded automatically from the tigramite
    GitHub repository.

    GROUND TRUTH: causal chain CPAC -> WCPAC -> WPAC, established by
    physical domain knowledge (not by automatic discovery) — see the
    module docstring for the citation.
    IMPORTANT: in the original study these relations are treated as mostly
    CONTEMPORANEOUS (same bimonthly period), not necessarily lagged —
    unlike the X_(t-tau) -> Y_t regime this pipeline was designed to
    detect. This is, in itself, a useful test: it shows how the pipeline
    behaves when the real causality has a strong contemporaneous
    component.

    PREPROCESSING (seasonal_anomaly=True, default): the raw monthly values
    carry a strong seasonal cycle, a deterministic common driver that the
    tutorial explicitly removes as a confounder before any causal analysis
    (it makes every pair Granger-significant on the raw data). Each series
    is therefore converted to anomalies: seasonal mean removed and divided
    by the seasonal standard deviation, month by month (see
    _seasonal_anomaly). The tutorial's other steps are NOT reproduced: the
    15-year Gaussian detrending, the bimonthly binning and the mask
    restricting the samples to Nov-Feb neutral/La Nina periods.
    seasonal_anomaly=False returns the raw values.

    Requires internet access (raw.githubusercontent.com) on first run; uses
    an on-disk cache if cache_dir is provided.
    """
    cache_base = Path(cache_dir) if cache_dir else None
    series = {}
    for var, file in _CLIMATE_TIGRAMITE_FILES.items():
        url = f"{_CLIMATE_TIGRAMITE_BASE_URL}/{file}"
        cache_path = (cache_base / file) if cache_base else None
        try:
            content = _download_text_file(url, cache_path=cache_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download real climate data '{file}' from {url}. "
                f"Check your internet connection or download it manually to "
                f"{cache_path}. Original error: {e}"
            ) from e
        lines = [l for l in content.splitlines() if l.strip() and not l.startswith("#")]
        values = []
        for line in lines:
            parts = line.split()
            values.extend(float(v) for v in parts[1:])
        series[var] = np.array(values)

    length = min(12 * n_years, *[len(v) for v in series.values()])
    data = np.vstack([series[v][:length] for v in ["CPAC", "WCPAC", "WPAC"]]).T

    if np.any(~np.isfinite(data)):
        raise ValueError(
            "Climate data contains NaN/Inf after loading — possible "
            "incorrect truncation or corrupted cache file."
        )

    if seasonal_anomaly:
        data = _seasonal_anomaly(data)

    entity_names = ["CPAC", "WCPAC", "WPAC"]
    causal_pair_idx = {(0, 1), (1, 2)}  # CPAC->WCPAC->WPAC (indices 0,1,2)

    df_long, df_gt = array_to_pipeline_format_single_property(
        data, entity_names, causal_pair_idx,
        start_date=f"{start_year}-01-01", freq="MS", property_name="VAL",
    )

    params = {
        "source": "tigramite_real_climate",
        "reference": "Runge et al., Nat Rev Earth Environ 10:2553 (2023)",
        "n_points": length,
        "n_entities": 3,
        "ground_truth_type": "domain (physical) knowledge, not synthetic",
        "caveat": "mostly contemporaneous relations, not lagged",
        "preprocessing": ("seasonal anomaly (monthly mean/std)" if seasonal_anomaly
                          else "raw values"),
    }

    print(f"  [real-benchmark] {name} loaded — 3 variables (CPAC, WCPAC, "
          f"WPAC) x {length} monthly points "
          f"({'seasonal anomalies' if seasonal_anomaly else 'raw'}) | "
          f"GT: chain CPAC->WCPAC->WPAC (domain knowledge)")

    return {"name": name, "df": df_long, "gt": df_gt, "_scenario": None,
            "_params": params}

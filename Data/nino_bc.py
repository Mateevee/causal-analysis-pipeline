"""
Data.nino_bc
===============================================================================
REAL BENCHMARK — ENSO -> BRITISH COLUMBIA TELECONNECTION (illustrative).
Qualitative example discussed by Runge et al. 2019 ("Inferring causation from
time series in Earth system sciences", Nat. Commun. 10:2553): the Nino3.4
index (sea surface temperature in the central equatorial Pacific, the
standard proxy of the ENSO phenomenon) as a possible driver of the air
temperature in British Columbia (Canada), with a lag of a few months --
treated here as ILLUSTRATIVE/APPROXIMATE, not as an exact replication of the
pre-processing, geographic box or precise lag used in the original paper
(which were not reconstructed here).

The qualitative direction/sign of this teleconnection (El Nino phase
associated with a warmer winter in western Canada) is also backed by a wider
literature on ENSO-North America teleconnections (e.g. studies in the style
of Shabbar & Khandekar on ENSO and the climate of Canada) -- even though the
ground truth encoded here is an illustrative reproduction of that domain
knowledge, not a precise replication of any specific paper.

DATA SOURCES
-------------------------------------------------------------------------------
- NINO3.4 (monthly SST anomaly, 1991-2020 climatological base, ERSSTv5):
      https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii
  (NOAA/NWS Climate Prediction Center). Format: one header line followed by
  lines "YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM" -- the last
  column is the anomaly of the Nino3.4 region, which is what this loader
  uses.
- AIR TEMPERATURE (British Columbia): monthly mean air temperature at
  Vancouver Intl A -- the long-term historical station of the region (Climate
  ID 1108447, Station ID 889, 49.20N -123.18W), via Environment and Climate
  Change Canada (ECCC), bulk download (CSV) from the public portal:
      https://climate.weather.gc.ca/climate_data/bulk_data_e.html
          ?format=csv&stationID=889&timeframe=3&Year=<year>&Month=1&Day=1
  Coverage: 1937-01 to 2013-06 (the historical station was replaced by an
  automatic one -- Station ID 51442, Climate ID 1108395 -- which has no
  MONTHLY (MLY) climatological product published at this endpoint). There is
  a single known monthly gap in the record (2010-09, no reading); the loader
  fills small internal gaps by simple linear interpolation
  (`_interpolate_gaps`) and fails explicitly if any NaN/Inf remains after
  that (see the guard check in `load_nino_bc_benchmark`).

GROUND TRUTH: NINO34 -> BC_TEMP (unidirectional, no reverse edge, no cycle),
by domain knowledge (ENSO teleconnection), NOT by automatic discovery. The
exact lag is treated as approximate/illustrative -- see `_params["caveat"]`
in the returned dict.

Requires internet access (cpc.ncep.noaa.gov and climate.weather.gc.ca) on the
first run; uses the disk cache if cache_dir is given.
"""
from __future__ import annotations

import csv
import urllib.request as _urlreq_bench
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import array_to_pipeline_format_single_property, register

_NINO34_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
_BC_TEMP_URL_TEMPLATE = (
    "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    "?format=csv&stationID=889&Year={year}&Month=1&Day=1&timeframe=3"
    "&submit=Download+Data"
)


def _download_text_file(url: str, cache_path: Optional[Path] = None,
                        timeout: int = 30) -> str:
    """Downloads a text file over HTTP, with an optional disk cache."""
    if cache_path is not None and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    with _urlreq_bench.urlopen(url, timeout=timeout) as resp:
        content = resp.read().decode("utf-8")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding="utf-8")
    return content


def _parse_nino34_cpc(text: str) -> dict:
    """Parses the CPC ascii file (ersst5.nino.mth.91-20.ascii) and returns
    {(year, month): nino34_anomaly}. Format: the 1st line is a header; the
    other lines have 10 tokens "YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM
    NINO3.4 ANOM" -- the last column is the Nino3.4 anomaly used here."""
    series = {}
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) != 10:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            anom = float(parts[-1])
        except ValueError:
            continue
        if not (1 <= month <= 12):
            continue
        if abs(anom) >= 90:
            continue
        series[(year, month)] = anom
    if not series:
        raise ValueError(
            "No valid Nino3.4 record found in the file downloaded from the "
            "CPC -- the file format may have changed.")
    return series


def _parse_bc_temp_eccc(text: str) -> dict:
    """Parses the ECCC monthly CSV (Vancouver Intl A, Climate ID 1108447) and
    returns {(year, month): mean_temperature_c}. Skips months with no reading
    (empty field -- e.g. the known 2010-09 gap)."""
    lines = text.lstrip("\ufeff").splitlines()
    reader = csv.DictReader(lines)
    temp_col = None
    for column_name in reader.fieldnames or []:
        if column_name.startswith("Mean Temp (") and "Flag" not in column_name:
            temp_col = column_name
            break
    if temp_col is None:
        raise ValueError(
            f"Mean temperature column not found in the ECCC CSV. Columns "
            f"present: {reader.fieldnames}")

    series = {}
    for row in reader:
        value_str = (row.get(temp_col) or "").strip()
        if not value_str:
            continue
        try:
            year = int(row["Year"])
            month = int(row["Month"])
            value = float(value_str)
        except (ValueError, KeyError, TypeError):
            continue
        series[(year, month)] = value

    if not series:
        raise ValueError(
            "No valid British Columbia temperature record found in the CSV "
            "downloaded from the ECCC -- the format may have changed.")
    return series


def _interpolate_gaps(values: np.ndarray) -> np.ndarray:
    """Fills internal gaps (NaN) by simple linear interpolation between the
    neighbouring points (e.g. the known 2010-09 gap in the Vancouver
    temperature series). Does NOT extrapolate at the edges -- a gap at the
    start/end of the requested window stays NaN on purpose, so that the
    caller's NaN/Inf guard check reports that the window exceeds the real
    coverage of the source, instead of masking the problem."""
    series = pd.Series(values)
    series = series.interpolate(method="linear", limit_direction=None)
    return series.to_numpy()


@register("nino_bc")
def load_nino_bc_benchmark(
    n_years: int = 60,
    start_year: int = 1950,
    cache_dir: Optional[str] = None,
    name: str = "real_nino34_bc",
) -> dict:
    """
    Loads the illustrative real benchmark of the ENSO -> British Columbia
    teleconnection (Runge et al. 2019, Nat. Commun. 10:2553): Nino3.4
    (CPC/NOAA) and monthly mean air temperature at Vancouver Intl A (ECCC),
    aligned on the common monthly period starting at `start_year`.

    GROUND TRUTH: NINO34 -> BC_TEMP (unidirectional), by domain knowledge
    (ENSO teleconnection), NOT by automatic discovery -- see the module
    docstring for the caveats on the lag being approximate/illustrative.

    Requires internet access (cpc.ncep.noaa.gov and climate.weather.gc.ca)
    on the first run; uses the disk cache if cache_dir is given.

    The default n_years=60 from start_year=1950 (window 1950-2009) stays
    within the solid coverage of both sources (Nino3.4 CPC: 1950 onwards;
    Vancouver Intl A/ECCC: 1937-01 to 2013-06) and avoids the single known
    monthly gap of the temperature series (2010-09). Windows going beyond
    2013-06 or before 1950 fail the NaN/Inf guard check below.
    """
    cache_base = Path(cache_dir) if cache_dir else None

    nino_cache = cache_base / "nino34_cpc.txt" if cache_base else None
    try:
        nino_text = _download_text_file(_NINO34_URL, cache_path=nino_cache)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download the Nino3.4 index from {_NINO34_URL}. Check "
            f"your internet connection or download it manually to "
            f"{nino_cache}. Original error: {e}"
        ) from e
    nino_series = _parse_nino34_cpc(nino_text)

    bc_url = _BC_TEMP_URL_TEMPLATE.format(year=start_year)
    bc_cache = cache_base / "vancouver_mly_889.csv" if cache_base else None
    try:
        bc_text = _download_text_file(bc_url, cache_path=bc_cache)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download the Vancouver temperature (ECCC) from "
            f"{bc_url}. Check your internet connection or download it "
            f"manually to {bc_cache}. Original error: {e}"
        ) from e
    bc_series = _parse_bc_temp_eccc(bc_text)

    months = [(start_year + i // 12, 1 + i % 12) for i in range(12 * n_years)]
    nino_vals = _interpolate_gaps(np.array([nino_series.get(m, np.nan) for m in months]))
    bc_vals = _interpolate_gaps(np.array([bc_series.get(m, np.nan) for m in months]))

    data = np.column_stack([nino_vals, bc_vals])

    if np.any(~np.isfinite(data)):
        raise ValueError(
            f"NINO34/BC_TEMP series hold NaN/Inf after interpolation -- the "
            f"requested window (start_year={start_year}, n_years={n_years}) "
            f"probably exceeds the real coverage of one of the sources "
            f"(Nino3.4 CPC: from 1950; Vancouver Intl A/ECCC: 1937-01 to "
            f"2013-06). Adjust start_year/n_years.")

    entity_names = ["NINO34", "BC_TEMP"]
    causal_pairs_idx = {(0, 1)}

    df_long, df_gt = array_to_pipeline_format_single_property(
        data, entity_names, causal_pairs_idx,
        start_date=f"{start_year}-01-01", freq="MS", property_name="VAL",
    )

    params = {
        "source": "nino34_cpc + vancouver_intl_a_eccc",
        "reference": "Runge et al., Nat Commun 10:2553 (2019) [qualitative "
                     "example ENSO->North American climate]; Nino3.4 index: "
                     "NOAA/CPC (ersst5.nino.mth.91-20.ascii); temperature: "
                     "Environment and Climate Change Canada, station "
                     "Vancouver Intl A (Climate ID 1108447)",
        "n_points": len(months),
        "n_entities": 2,
        "ground_truth_type": "domain knowledge (ENSO teleconnection), "
                              "illustrative/approximate lag",
        "caveat": "illustrative reproduction of the ENSO -> North American "
                  "temperature teleconnection described qualitatively by "
                  "Runge et al. 2019 -- NOT an exact replication of the "
                  "geographic box, pre-processing or lag used in the "
                  "original paper.",
    }

    print(f"  [real-benchmark] {name} loaded — 2 variables (NINO34, BC_TEMP) x "
          f"{len(months)} monthly points | GT: NINO34->BC_TEMP (ENSO "
          f"teleconnection, domain knowledge, illustrative)")

    return {"name": name, "df": df_long, "gt": df_gt, "_scenario": None,
            "_params": params}

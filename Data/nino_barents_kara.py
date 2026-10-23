"""
Data.nino_barents_kara
===============================================================================
REAL BENCHMARK — CONTESTED TELECONNECTION ENSO -> BARENTS-KARA SEA ICE
EXTENT. UNLIKE the other real benchmarks of this package (e.g.
Data/real_climate.py, chain CPAC->WCPAC->WPAC; Data/nino_bc.py, ENSO->BC),
the ground truth encoded here is NOT consolidated domain knowledge -- it is a
HYPOTHESIS FROM THE LITERATURE, STILL IN DISPUTE, included deliberately to
test the behaviour of the pipeline in front of a real but NON-CONSENSUAL
causal claim.

HISTORY OF THE DISPUTE (read before treating the GT below as "truth"):
- Kretschmer et al. 2016 (J. Climate, "Using Causal Effect Networks to
  Analyze Different Arctic Drivers of Midlatitude Winter Circulation")
  reported, via a causal network (PCMCI), the tropical Pacific sea surface
  temperature (ENSO proxy) as one of the possible drivers of an
  Arctic-midlatitude network that includes the Barents-Kara sea ice.
- Dorr et al. 2024 (Geophysical Research Letters) re-examined that causal
  chain with more recent causal methods and did NOT find a robust causal
  link between ENSO and the Barents-Kara ice -- i.e. the most recent
  literature DISPUTES the existence of this link.
Conclusion: this benchmark must NOT be read as "established ground truth" in
the same sense as Data/real_climate.py or Data/nino_bc.py. The causal pair
NINO34->BARENTS_KARA here is a HYPOTHESIS, not a consensus -- see
`_params["ground_truth_type"]` and `_params["caveat"]` in the returned dict.

DATA SOURCES
-------------------------------------------------------------------------------
- NINO3.4 (monthly SST anomaly, ERSSTv5): same source/parser as
  Data/nino_bc.py (`_parse_nino34_cpc`), reused from there:
      https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii
- Barents-Kara SEA ICE EXTENT: NSIDC Sea Ice Index v4.0, monthly regional
  index (Excel workbook, one sheet per region):
      https://noaadata.apps.nsidc.org/NOAA/G02135/seaice_analysis/
          N_Sea_Ice_Index_Regional_Monthly_Data_G02135_v4.0.xlsx
  Sheets used: "Barents-Extent-km^2" and "Kara-Extent-km^2". Each sheet has a
  sub-header row ("extent"/"rank" per month) and a blank row before the
  numeric data starts -- see `_parse_extent_sheet`. There is no combined
  "Barents-Kara" sheet in the original file; the BARENTS_KARA index used here
  is the plain SUM (in km^2) of the Barents extent and the Kara extent, month
  by month -- the most direct way of combining two adjacent regions into a
  single regional index, since both are already in the same unit (km^2).
  There is a known, well documented monthly gap in the satellite record
  (SMMR->SSM/I sensor change) in 1987-12/1988-01, present in both sheets --
  filled by simple linear interpolation, in the same spirit as
  Data/nino_bc.py (`_interpolate_gaps`, reused from there).

GROUND TRUTH: NINO34 -> BARENTS_KARA (unidirectional), labelled as
"hypothesis from the literature, not consensus" -- see Kretschmer et al. 2016
vs. Dorr et al. 2024 above. This benchmark serves to observe if/how the
pipeline reacts to a real but contested causal pair, not to validate
precision against a reliable GT.

Requires internet access (cpc.ncep.noaa.gov and noaadata.apps.nsidc.org) on
the first run; uses the disk cache if cache_dir is given (the xlsx is a few
MB, so it is worth caching).
"""
from __future__ import annotations

import io
import urllib.request as _urlreq_bench
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import array_to_pipeline_format_single_property, register
from .nino_bc import _download_text_file, _interpolate_gaps, _parse_nino34_cpc

_NINO34_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
_SEAICE_URL = (
    "https://noaadata.apps.nsidc.org/NOAA/G02135/seaice_analysis/"
    "N_Sea_Ice_Index_Regional_Monthly_Data_G02135_v4.0.xlsx"
)

_REGION_SHEETS = ["Barents-Extent-km^2", "Kara-Extent-km^2"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]


def _download_binary_file(url: str, cache_path: Optional[Path] = None,
                          timeout: int = 60) -> bytes:
    """Downloads a binary file over HTTP, with an optional disk cache (same
    spirit as `_download_text_file` in Data/nino_bc.py, but keeping the bytes
    instead of decoding text -- needed for the NSIDC .xlsx)."""
    if cache_path is not None and cache_path.exists():
        return cache_path.read_bytes()
    with _urlreq_bench.urlopen(url, timeout=timeout) as resp:
        content = resp.read()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)
    return content


def _parse_extent_sheet(xls: pd.ExcelFile, sheet: str) -> dict:
    """Parses a regional extent sheet of the NSIDC Sea Ice Index v4.0 and
    returns {(year, month): extent_km2}.

    Layout of the sheet: row 0 = sub-header "extent"/"rank" alternating per
    month; row 1 = blank row; numeric data starts at row 2, with the year in
    the "Unnamed: 0" column and the extent of each month in the column named
    after the month in English (the interleaved "Unnamed: N" columns are the
    rank of the month in that year among all years -- not used here)."""
    df = pd.read_excel(xls, sheet_name=sheet, header=0)
    df = df.iloc[2:]
    df = df.rename(columns={df.columns[0]: "Year"})

    series = {}
    for _, row in df.iterrows():
        year = row["Year"]
        if pd.isna(year):
            continue
        year = int(year)
        for i, month_name in enumerate(_MONTHS, start=1):
            value = row[month_name]
            if pd.isna(value):
                continue
            series[(year, i)] = float(value)

    if not series:
        raise ValueError(
            f"No valid sea ice extent record found in the NSIDC sheet "
            f"'{sheet}' -- the layout of the workbook may have changed.")
    return series


@register("nino_barents_kara")
def load_nino_barents_kara_benchmark(
    n_years: int = 45,
    start_year: int = 1979,
    cache_dir: Optional[str] = None,
    name: str = "real_nino34_barents_kara",
) -> dict:
    """
    Loads the real benchmark of the CONTESTED teleconnection ENSO ->
    Barents-Kara sea ice: Nino3.4 (CPC/NOAA) and Barents+Kara ice extent
    (NSIDC Sea Ice Index v4.0), aligned on the common monthly period starting
    at `start_year`.

    WARNING -- unlike the other real benchmarks of this package, the GT here
    is NOT consolidated domain knowledge: it is a hypothesis from the
    literature (Kretschmer et al. 2016) that a more recent causal re-analysis
    (Dorr et al. 2024) could NOT confirm as a robust causal link. See the
    module docstring for the full discussion.

    Requires internet access (cpc.ncep.noaa.gov and noaadata.apps.nsidc.org)
    on the first run; uses the disk cache if cache_dir is given.

    The default n_years=45 from start_year=1979 (window 1979-2023) stays
    within the solid coverage of both sources (Nino3.4 CPC: from 1950; NSIDC
    Sea Ice Index: November 1978 onwards, used here from 1979 to avoid the
    first partial year) and includes the known 1987-12/1988-01 gap (filled by
    interpolation -- see `_interpolate_gaps`).
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

    seaice_cache = cache_base / "nsidc_seaice_regional_monthly_v4.xlsx" if cache_base else None
    try:
        seaice_bytes = _download_binary_file(_SEAICE_URL, cache_path=seaice_cache)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download the NSIDC regional Sea Ice Index from "
            f"{_SEAICE_URL}. Check your internet connection or download it "
            f"manually to {seaice_cache}. Original error: {e}"
        ) from e

    xls = pd.ExcelFile(io.BytesIO(seaice_bytes))
    barents_series = _parse_extent_sheet(xls, _REGION_SHEETS[0])
    kara_series = _parse_extent_sheet(xls, _REGION_SHEETS[1])

    months = [(start_year + i // 12, 1 + i % 12) for i in range(12 * n_years)]
    nino_vals = _interpolate_gaps(np.array([nino_series.get(m, np.nan) for m in months]))
    barents_vals = np.array([barents_series.get(m, np.nan) for m in months])
    kara_vals = np.array([kara_series.get(m, np.nan) for m in months])
    ice_vals = _interpolate_gaps(barents_vals + kara_vals)

    data = np.column_stack([nino_vals, ice_vals])

    if np.any(~np.isfinite(data)):
        raise ValueError(
            f"NINO34/BARENTS_KARA series hold NaN/Inf after interpolation -- "
            f"the requested window (start_year={start_year}, "
            f"n_years={n_years}) probably exceeds the real coverage of one "
            f"of the sources (Nino3.4 CPC: from 1950; NSIDC regional Sea Ice "
            f"Index: from Nov/1978). Adjust start_year/n_years.")

    entity_names = ["NINO34", "BARENTS_KARA"]
    causal_pairs_idx = {(0, 1)}

    df_long, df_gt = array_to_pipeline_format_single_property(
        data, entity_names, causal_pairs_idx,
        start_date=f"{start_year}-01-01", freq="MS", property_name="VAL",
    )

    params = {
        "source": "nino34_cpc + nsidc_seaice_index_v4_barents_kara",
        "reference": "Kretschmer et al., J. Climate 29:4069-4081 (2016) "
                     "[original hypothesis: ENSO as a driver of an "
                     "Arctic-midlatitude network including Barents-Kara]; "
                     "Dorr et al., Geophys. Res. Lett. (2024) [causal "
                     "re-analysis that does NOT confirm a robust link -- see "
                     "caveat]; sea ice: NSIDC Sea Ice Index v4.0 (G02135), "
                     "regional monthly extent, Barents + Kara",
        "n_points": len(months),
        "n_entities": 2,
        "ground_truth_type": "hypothesis from the literature, not consensus",
        "caveat": "the causal link ENSO -> Barents-Kara ice is CONTESTED in "
                  "the literature: Kretschmer et al. 2016 reported this "
                  "driver in an Arctic-midlatitude causal network, but Dorr "
                  "et al. 2024 re-examined it with more recent causal "
                  "methods and did NOT find a robust causal link. Do not "
                  "read the score on this scenario as precision against a "
                  "reliable ground truth.",
    }

    print(f"  [real-benchmark] {name} loaded — 2 variables (NINO34, "
          f"BARENTS_KARA) x {len(months)} monthly points | GT: "
          f"NINO34->BARENTS_KARA (HYPOTHESIS FROM THE LITERATURE, not "
          f"consensus -- see Kretschmer 2016 vs. Dorr 2024)")

    return {"name": name, "df": df_long, "gt": df_gt, "_scenario": None,
            "_params": params}

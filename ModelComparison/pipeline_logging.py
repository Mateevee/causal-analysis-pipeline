"""
pipeline_logging.py — Error-diagnostics infrastructure ("logs/whocausederror")
=================================================================================
Shared by main.py and by any pipeline module that needs to log an
execution error without treating it as an unhandled crash.
"""

import datetime
import traceback
from pathlib import Path

import pandas as pd
from typing import Callable

LOGS_DIR = Path(__file__).resolve().parent / "logs" / "failures"

def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def _describe_df(df: pd.DataFrame, date_col: str = None, entity_col: str = None) -> str:
    lines = []
    lines.append(f"shape            : {df.shape}")
    lines.append(f"columns          : {list(df.columns)}")
    lines.append("")
    lines.append("dtypes:")
    for col, dt in df.dtypes.items():
        lines.append(f"  {col:<25}: {dt}")

    if date_col and date_col in df.columns:
        lines.append("")
        lines.append(f"Date column ('{date_col}'):")
        lines.append(f"  original dtype          : {df[date_col].dtype}")
        lines.append(f"  sample (first 5)        : {df[date_col].head(5).tolist()}")
        lines.append(f"  sample (last 5)         : {df[date_col].tail(5).tolist()}")
        lines.append(f"  n_nulls (before conv.)  : {int(df[date_col].isna().sum())}")

        try:
            already_datetime = pd.api.types.is_datetime64_any_dtype(df[date_col])
            lines.append(f"  already datetime64?     : {already_datetime}")

            conv_dayfirst = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
            n_nat_dayfirst = int(conv_dayfirst.isna().sum())
            lines.append(
                f"  to_datetime(dayfirst=True, errors='coerce') -> "
                f"NaT in {n_nat_dayfirst}/{len(df)} rows"
            )

            conv_default = pd.to_datetime(df[date_col], errors='coerce')
            n_nat_default = int(conv_default.isna().sum())
            lines.append(
                f"  to_datetime(errors='coerce')                -> "
                f"NaT in {n_nat_default}/{len(df)} rows"
            )

            if n_nat_dayfirst == len(df) and len(df) > 0:
                lines.append(
                    "  >>> SUSPECT: ALL dates became NaT with dayfirst=True."
                )
                lines.append(
                    "      This empties the dataframe after dropna(subset=[date_col]),"
                )
                lines.append(
                    "      which makes pivot_table() fail with "
                    "'need at least one array to concatenate'."
                )
        except Exception as e:
            lines.append(f"  [error testing date conversion] {e}")

    if entity_col and entity_col in df.columns:
        lines.append("")
        lines.append(f"Entity column ('{entity_col}'):")
        lines.append(f"  n_unique                : {df[entity_col].nunique()}")
        lines.append(f"  values                  : {sorted(df[entity_col].dropna().unique().tolist())[:20]}")
        lines.append(f"  n_nulls                 : {int(df[entity_col].isna().sum())}")

    lines.append("")
    lines.append("head(5):")
    lines.append(df.head(5).to_string())

    return "\n".join(lines)


def _log_pipeline_error(
    df_name: str,
    model_name: str,
    pipeline: Callable,
    df: pd.DataFrame,
    cfg: dict,
    exc: Exception,
):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name  = f"{_slug(df_name)}__{_slug(model_name)}__{timestamp}.log"
    log_path   = LOGS_DIR / file_name

    date_col = cfg.get('date_col', 'Data')
    entity_col = cfg.get('entity_col', 'Nome')

    lines = []
    lines.append("=" * 70)
    lines.append("ERROR LOG — PIPELINE")
    lines.append("=" * 70)
    lines.append(f"dataset    : {df_name}")
    lines.append(f"model      : {model_name}")
    lines.append(f"module     : {pipeline.__module__}")
    lines.append(f"timestamp  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"error      : {type(exc).__name__}: {exc}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("FULL TRACEBACK")
    lines.append("-" * 70)
    lines.append(traceback.format_exc())
    lines.append("")

    lines.append("-" * 70)
    lines.append("EFFECTIVE CONFIG (cfg) USED IN THIS CALL")
    lines.append("-" * 70)
    for k, v in cfg.items():
        lines.append(f"  {k:<25}: {v!r}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("INPUT DATAFRAME")
    lines.append("-" * 70)
    try:
        lines.append(_describe_df(df, date_col=date_col, entity_col=entity_col))
    except Exception as e:
        lines.append(f"[error describing dataframe] {e}")
    lines.append("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [LOG] Error diagnostics saved to: {log_path}")
    return log_path
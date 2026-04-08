from __future__ import annotations

from pathlib import Path
import pandas as pd

from .types import TimeSeriesData


def load_hourly_csv(path: str | Path) -> TimeSeriesData:
    """Load a CSV with required columns:
    timestamp, load_kw, solar_cf, grid_price_per_kwh

    timestamp can be any pandas-parsable datetime string.
    """
    df = pd.read_csv(path)
    required = {"timestamp", "load_kw", "solar_cf", "grid_price_per_kwh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return TimeSeriesData(
        timestamps=pd.DatetimeIndex(df["timestamp"]),
        load_kw=df["load_kw"].to_numpy(dtype=float),
        solar_cf=df["solar_cf"].to_numpy(dtype=float),
        grid_price_per_kwh=df["grid_price_per_kwh"].to_numpy(dtype=float),
    )

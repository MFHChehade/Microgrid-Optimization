from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DataConfig
from .types import TimeSeriesData


def _smooth_random(size: int, rng: np.random.Generator, scale: float = 0.08, window: int = 24) -> np.ndarray:
    raw = rng.normal(0.0, scale, size=size)
    kernel = np.ones(window) / window
    smooth = np.convolve(raw, kernel, mode="same")
    return smooth


def generate_synthetic_year(cfg: DataConfig) -> TimeSeriesData:
    rng = np.random.default_rng(cfg.seed)
    hours = cfg.year_hours
    timestamps = pd.date_range("2025-01-01", periods=hours, freq="h")

    hod = timestamps.hour.to_numpy()
    doy = timestamps.dayofyear.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()

    # Solar capacity factor: daylight sinusoid times seasonal modulation and smooth clouds.
    daylight = np.sin(np.pi * np.clip((hod - 6) / 12.0, 0.0, 1.0))
    daylight = np.where((hod >= 6) & (hod <= 18), daylight, 0.0)
    seasonal = 0.55 + 0.35 * np.sin(2.0 * np.pi * (doy - 80) / 365.0)
    cloud = 1.0 + _smooth_random(hours, rng, scale=0.15, window=12)
    solar_cf = np.clip(daylight * seasonal * cloud, 0.0, 1.0)

    # Load: base + morning shoulder + evening peak + seasonal HVAC + smooth noise.
    morning_peak = 120.0 * np.exp(-0.5 * ((hod - 8) / 2.0) ** 2)
    evening_peak = cfg.daily_peak_kw * np.exp(-0.5 * ((hod - 19) / 3.0) ** 2)
    seasonal_load = cfg.seasonal_peak_kw * (0.5 + 0.5 * np.sin(2.0 * np.pi * (doy - 172) / 365.0) ** 2)
    weekend_reduction = np.where(weekday >= 5, -60.0, 0.0)
    noise = _smooth_random(hours, rng, scale=40.0, window=6)
    load_kw = np.clip(cfg.base_load_kw + morning_peak + evening_peak + seasonal_load + weekend_reduction + noise, 300.0, None)

    # Time-of-use grid price with higher summer and weekday evening peaks.
    base_price = 0.10 * np.ones(hours)
    shoulder = np.where((hod >= 12) & (hod <= 16), 0.05, 0.0)
    peak = np.where((hod >= 17) & (hod <= 21), 0.16, 0.0)
    weekend_discount = np.where(weekday >= 5, -0.015, 0.0)
    summer = np.where((timestamps.month >= 6) & (timestamps.month <= 9), cfg.summer_price_adder, 0.0)
    price_noise = _smooth_random(hours, rng, scale=0.004, window=24)
    grid_price = np.clip(base_price + shoulder + peak + weekend_discount + summer + price_noise, 0.05, None)

    return TimeSeriesData(
        timestamps=timestamps,
        load_kw=load_kw.astype(float),
        solar_cf=solar_cf.astype(float),
        grid_price_per_kwh=grid_price.astype(float),
    )

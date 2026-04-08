from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Design:
    battery_kwh: float
    pv_kw: float

    @property
    def label(self) -> str:
        return f"B{self.battery_kwh:.0f}_PV{self.pv_kw:.0f}"


@dataclass
class TimeSeriesData:
    timestamps: pd.DatetimeIndex
    load_kw: np.ndarray
    solar_cf: np.ndarray
    grid_price_per_kwh: np.ndarray


@dataclass
class DispatchResult:
    success: bool
    status: int
    objective_value: float
    message: str
    final_soc_kwh: float
    schedule: pd.DataFrame | None = None

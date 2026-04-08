from __future__ import annotations

import pandas as pd

from .config import ExperimentConfig
from .dispatch import WeekSlice, solve_dispatch_week
from .types import Design, TimeSeriesData, DispatchResult


def iter_week_slices(data: TimeSeriesData, horizon_hours: int):
    total = len(data.load_kw)
    for start in range(0, total, horizon_hours):
        stop = min(start + horizon_hours, total)
        yield WeekSlice(
            timestamps=data.timestamps[start:stop],
            load_kw=data.load_kw[start:stop],
            solar_cf=data.solar_cf[start:stop],
            grid_price_per_kwh=data.grid_price_per_kwh[start:stop],
        )


def evaluate_design_year(
    data: TimeSeriesData,
    design: Design,
    cfg: ExperimentConfig,
    accurate: bool = False,
    keep_schedule: bool = False,
) -> tuple[float, float, DispatchResult | None]:
    soc = design.battery_kwh * cfg.operation.initial_soc_fraction
    total_cost = 0.0
    schedules = []

    for week in iter_week_slices(data, cfg.operation.horizon_hours):
        res = solve_dispatch_week(
            week=week,
            design=design,
            operation_cfg=cfg.operation,
            cost_cfg=cfg.costs,
            initial_soc_kwh=soc,
            accurate=accurate,
        )
        if not res.success:
            return float("inf"), float("nan"), res
        total_cost += res.objective_value
        soc = res.final_soc_kwh
        if keep_schedule and res.schedule is not None:
            schedules.append(res.schedule)

    dispatch_result = None
    if keep_schedule:
        dispatch_result = DispatchResult(
            success=True,
            status=0,
            objective_value=total_cost,
            message="aggregated",
            final_soc_kwh=soc,
            schedule=pd.concat(schedules, axis=0, ignore_index=True) if schedules else None,
        )
    return total_cost, soc, dispatch_result

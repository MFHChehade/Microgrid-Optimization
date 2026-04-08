from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy import sparse

from .config import OperationConfig, CostConfig
from .types import Design, DispatchResult


@dataclass(frozen=True)
class WeekSlice:
    timestamps: pd.DatetimeIndex
    load_kw: np.ndarray
    solar_cf: np.ndarray
    grid_price_per_kwh: np.ndarray


@dataclass(frozen=True)
class _IndexMap:
    pv: slice
    diesel: slice
    grid: slice
    ch: slice
    dis: slice
    soc: slice
    u: slice | None
    total_vars: int


def _make_index_map(T: int, accurate: bool) -> _IndexMap:
    start = 0
    pv = slice(start, start + T); start += T
    diesel = slice(start, start + T); start += T
    grid = slice(start, start + T); start += T
    ch = slice(start, start + T); start += T
    dis = slice(start, start + T); start += T
    soc = slice(start, start + (T + 1)); start += (T + 1)
    u = None
    if accurate:
        u = slice(start, start + T); start += T
    return _IndexMap(pv=pv, diesel=diesel, grid=grid, ch=ch, dis=dis, soc=soc, u=u, total_vars=start)


def solve_dispatch_week(
    week: WeekSlice,
    design: Design,
    operation_cfg: OperationConfig,
    cost_cfg: CostConfig,
    initial_soc_kwh: float | None = None,
    accurate: bool = False,
) -> DispatchResult:
    T = len(week.load_kw)
    idx = _make_index_map(T, accurate)
    n = idx.total_vars

    initial_soc_kwh = (
        design.battery_kwh * operation_cfg.initial_soc_fraction
        if initial_soc_kwh is None
        else float(initial_soc_kwh)
    )
    soc_min = operation_cfg.soc_min_fraction * design.battery_kwh
    soc_max = operation_cfg.soc_max_fraction * design.battery_kwh
    p_ch_max = operation_cfg.charge_power_fraction_per_hour * design.battery_kwh
    p_dis_max = operation_cfg.discharge_power_fraction_per_hour * design.battery_kwh

    c = np.zeros(n)
    c[idx.grid] = week.grid_price_per_kwh
    c[idx.diesel] = cost_cfg.diesel_cost_per_kwh

    lb = np.zeros(n)
    ub = np.full(n, np.inf)

    ub[idx.pv] = design.pv_kw * week.solar_cf
    ub[idx.diesel] = operation_cfg.diesel_max_kw
    ub[idx.grid] = operation_cfg.grid_max_kw
    ub[idx.ch] = p_ch_max
    ub[idx.dis] = p_dis_max
    lb[idx.soc] = soc_min
    ub[idx.soc] = soc_max
    lb[idx.soc.start] = initial_soc_kwh
    ub[idx.soc.start] = initial_soc_kwh

    if operation_cfg.terminal_soc_target_fraction is not None:
        terminal_soc = operation_cfg.terminal_soc_target_fraction * design.battery_kwh
        lb[idx.soc.stop - 1] = terminal_soc
        ub[idx.soc.stop - 1] = terminal_soc

    if accurate and idx.u is not None:
        lb[idx.u] = 0.0
        ub[idx.u] = 1.0

    Aeq = sparse.lil_matrix((2 * T, n), dtype=float)
    beq = np.zeros(2 * T)

    # Power balance
    for t in range(T):
        Aeq[t, idx.pv.start + t] = 1.0
        Aeq[t, idx.diesel.start + t] = 1.0
        Aeq[t, idx.grid.start + t] = 1.0
        Aeq[t, idx.dis.start + t] = 1.0
        Aeq[t, idx.ch.start + t] = -1.0
        beq[t] = week.load_kw[t]

    # SOC dynamics
    eta_c = operation_cfg.charge_efficiency
    eta_d = operation_cfg.discharge_efficiency
    for t in range(T):
        row = T + t
        Aeq[row, idx.soc.start + t + 1] = 1.0
        Aeq[row, idx.soc.start + t] = -1.0
        Aeq[row, idx.ch.start + t] = -eta_c
        Aeq[row, idx.dis.start + t] = 1.0 / eta_d
        beq[row] = 0.0

    if not accurate:
        res = linprog(
            c=c,
            A_eq=Aeq.tocsr(),
            b_eq=beq,
            bounds=list(zip(lb, ub, strict=False)),
            method="highs",
        )
    else:
        Aub = sparse.lil_matrix((2 * T, n), dtype=float)
        bub = np.zeros(2 * T)
        for t in range(T):
            # diesel_t - diesel_max * u_t <= 0
            Aub[t, idx.diesel.start + t] = 1.0
            Aub[t, idx.u.start + t] = -operation_cfg.diesel_max_kw
            bub[t] = 0.0
            # -diesel_t + diesel_min * u_t <= 0
            Aub[T + t, idx.diesel.start + t] = -1.0
            Aub[T + t, idx.u.start + t] = operation_cfg.diesel_min_kw
            bub[T + t] = 0.0

        constraints = [
            LinearConstraint(Aeq.tocsr(), beq, beq),
            LinearConstraint(Aub.tocsr(), -np.inf * np.ones_like(bub), bub),
        ]
        integrality = np.zeros(n, dtype=int)
        integrality[idx.u] = 1
        res = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=constraints,
            options={"disp": False},
        )

    if not res.success:
        return DispatchResult(
            success=False,
            status=int(getattr(res, "status", -1)),
            objective_value=float("inf"),
            message=str(getattr(res, "message", "solver failed")),
            final_soc_kwh=float("nan"),
            schedule=None,
        )

    x = np.asarray(res.x, dtype=float)
    schedule = pd.DataFrame(
        {
            "timestamp": week.timestamps,
            "load_kw": week.load_kw,
            "solar_cf": week.solar_cf,
            "grid_price_per_kwh": week.grid_price_per_kwh,
            "pv_kw": x[idx.pv],
            "diesel_kw": x[idx.diesel],
            "grid_kw": x[idx.grid],
            "charge_kw": x[idx.ch],
            "discharge_kw": x[idx.dis],
            "soc_kwh": x[idx.soc][1:],
        }
    )
    if accurate and idx.u is not None:
        schedule["diesel_on"] = np.rint(x[idx.u]).astype(int)

    return DispatchResult(
        success=True,
        status=int(getattr(res, "status", 0)),
        objective_value=float(res.fun),
        message=str(getattr(res, "message", "optimal")),
        final_soc_kwh=float(x[idx.soc.stop - 1]),
        schedule=schedule,
    )

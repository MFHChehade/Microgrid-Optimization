from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .dispatch import WeekSlice
from .types import Design, TimeSeriesData


def greedy_dispatch_year(data: TimeSeriesData, design: Design, cfg: ExperimentConfig) -> tuple[float, pd.DataFrame]:
    soc = design.battery_kwh * cfg.operation.initial_soc_fraction
    soc_min = cfg.operation.soc_min_fraction * design.battery_kwh
    soc_max = cfg.operation.soc_max_fraction * design.battery_kwh
    p_ch_max = cfg.operation.charge_power_fraction_per_hour * design.battery_kwh
    p_dis_max = cfg.operation.discharge_power_fraction_per_hour * design.battery_kwh
    eta_c = cfg.operation.charge_efficiency
    eta_d = cfg.operation.discharge_efficiency
    records = []
    total_cost = 0.0

    price_threshold = float(np.quantile(data.grid_price_per_kwh, 0.70))

    for t in range(len(data.load_kw)):
        load = float(data.load_kw[t])
        pv_avail = design.pv_kw * float(data.solar_cf[t])
        price = float(data.grid_price_per_kwh[t])

        pv_to_load = min(load, pv_avail)
        load_after_pv = load - pv_to_load
        pv_surplus = pv_avail - pv_to_load

        charge = 0.0
        if pv_surplus > 0:
            headroom = max(0.0, soc_max - soc)
            charge = min(p_ch_max, pv_surplus, headroom / eta_c)
            soc += eta_c * charge
            pv_surplus -= charge

        discharge = 0.0
        if load_after_pv > 0 and price >= price_threshold:
            available = max(0.0, soc - soc_min)
            discharge = min(p_dis_max, load_after_pv, available * eta_d)
            soc -= discharge / eta_d
            load_after_pv -= discharge

        # Choose the cheaper external source; avoid diesel min-output overgeneration for very small residuals.
        diesel = 0.0
        grid = 0.0
        if load_after_pv > 1e-9:
            if cfg.costs.diesel_cost_per_kwh < price and load_after_pv >= cfg.operation.diesel_min_kw:
                diesel = min(load_after_pv, cfg.operation.diesel_max_kw)
                load_after_pv -= diesel
            grid = min(load_after_pv, cfg.operation.grid_max_kw)

        cost = price * grid + cfg.costs.diesel_cost_per_kwh * diesel
        total_cost += cost
        records.append(
            {
                "timestamp": data.timestamps[t],
                "load_kw": load,
                "pv_kw": pv_to_load + charge,
                "diesel_kw": diesel,
                "grid_kw": grid,
                "charge_kw": charge,
                "discharge_kw": discharge,
                "soc_kwh": soc,
                "grid_price_per_kwh": price,
            }
        )
    return total_cost, pd.DataFrame(records)


def dp_dispatch_year(
    data: TimeSeriesData,
    design: Design,
    cfg: ExperimentConfig,
    num_soc_states: int = 31,
    num_actions: int = 9,
) -> tuple[float, pd.DataFrame]:
    # Approximate DP baseline:
    # state = SOC, action = net battery power, residual demand met by cheapest external source.
    soc_min = cfg.operation.soc_min_fraction * design.battery_kwh
    soc_max = cfg.operation.soc_max_fraction * design.battery_kwh
    eta_c = cfg.operation.charge_efficiency
    eta_d = cfg.operation.discharge_efficiency
    p_ch_max = cfg.operation.charge_power_fraction_per_hour * design.battery_kwh
    p_dis_max = cfg.operation.discharge_power_fraction_per_hour * design.battery_kwh

    soc_grid = np.linspace(soc_min, soc_max, num_soc_states)
    charge_levels = np.linspace(0.0, p_ch_max, max(2, num_actions // 2 + 1))
    discharge_levels = np.linspace(0.0, p_dis_max, max(2, num_actions // 2 + 1))
    actions = np.unique(np.concatenate([-discharge_levels[1:], [0.0], charge_levels[1:]]))

    V = np.zeros((len(data.load_kw) + 1, num_soc_states))
    policy_idx = np.zeros((len(data.load_kw), num_soc_states), dtype=int)

    def ext_cost(residual_kw: float, price: float) -> float:
        if residual_kw <= 0:
            return 0.0
        if cfg.costs.diesel_cost_per_kwh < price and residual_kw >= cfg.operation.diesel_min_kw:
            diesel = min(residual_kw, cfg.operation.diesel_max_kw)
            grid = max(0.0, residual_kw - diesel)
        else:
            diesel = 0.0
            grid = residual_kw
        return price * grid + cfg.costs.diesel_cost_per_kwh * diesel

    for t in range(len(data.load_kw) - 1, -1, -1):
        load = float(data.load_kw[t])
        pv_avail = design.pv_kw * float(data.solar_cf[t])
        price = float(data.grid_price_per_kwh[t])

        for s_idx, soc in enumerate(soc_grid):
            best_cost = float("inf")
            best_a_idx = 0
            for a_idx, action in enumerate(actions):
                # action > 0 means charging from surplus / external energy; action < 0 means discharge to load.
                if action >= 0:
                    next_soc = soc + eta_c * action
                else:
                    next_soc = soc + action / eta_d  # action negative
                if next_soc < soc_min - 1e-9 or next_soc > soc_max + 1e-9:
                    continue
                discharge = max(0.0, -action)
                charge = max(0.0, action)
                residual = load + charge - pv_avail - discharge
                stage_cost = ext_cost(residual_kw=max(0.0, residual), price=price)
                next_idx = int(np.argmin(np.abs(soc_grid - next_soc)))
                total = stage_cost + V[t + 1, next_idx]
                if total < best_cost:
                    best_cost = total
                    best_a_idx = a_idx
            V[t, s_idx] = best_cost
            policy_idx[t, s_idx] = best_a_idx

    # Forward simulation
    soc = design.battery_kwh * cfg.operation.initial_soc_fraction
    records = []
    total_cost = 0.0
    for t in range(len(data.load_kw)):
        s_idx = int(np.argmin(np.abs(soc_grid - soc)))
        action = actions[policy_idx[t, s_idx]]
        load = float(data.load_kw[t])
        pv_avail = design.pv_kw * float(data.solar_cf[t])
        price = float(data.grid_price_per_kwh[t])

        charge = max(0.0, action)
        discharge = max(0.0, -action)
        if charge > 0:
            soc = min(soc_max, soc + eta_c * charge)
        elif discharge > 0:
            soc = max(soc_min, soc - discharge / eta_d)

        residual = load + charge - pv_avail - discharge
        if residual <= 0:
            diesel = 0.0
            grid = 0.0
        elif cfg.costs.diesel_cost_per_kwh < price and residual >= cfg.operation.diesel_min_kw:
            diesel = min(residual, cfg.operation.diesel_max_kw)
            grid = max(0.0, residual - diesel)
        else:
            diesel = 0.0
            grid = residual
        cost = price * grid + cfg.costs.diesel_cost_per_kwh * diesel
        total_cost += cost
        records.append(
            {
                "timestamp": data.timestamps[t],
                "load_kw": load,
                "pv_kw": min(load, pv_avail) + charge,
                "diesel_kw": diesel,
                "grid_kw": grid,
                "charge_kw": charge,
                "discharge_kw": discharge,
                "soc_kwh": soc,
                "grid_price_per_kwh": price,
            }
        )
    return total_cost, pd.DataFrame(records)

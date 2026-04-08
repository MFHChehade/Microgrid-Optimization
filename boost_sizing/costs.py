from __future__ import annotations

from .config import CostConfig
from .types import Design


def capital_recovery_factor(rate: float, years: int) -> float:
    if years <= 0:
        raise ValueError("years must be positive")
    if abs(rate) < 1e-12:
        return 1.0 / years
    return rate * (1.0 + rate) ** years / ((1.0 + rate) ** years - 1.0)


def annualized_pv_cost(design: Design, cfg: CostConfig) -> float:
    crf = capital_recovery_factor(cfg.discount_rate, cfg.pv_lifetime_years)
    return design.pv_kw * (cfg.pv_capex_per_kw * crf + cfg.pv_fixed_om_per_kw_year)


def annualized_battery_cost(design: Design, cfg: CostConfig) -> float:
    crf = capital_recovery_factor(cfg.discount_rate, cfg.battery_lifetime_years)
    return design.battery_kwh * (
        cfg.battery_capex_per_kwh * crf + cfg.battery_fixed_om_per_kwh_year
    )


def total_annual_cost(operational_cost: float, design: Design, cfg: CostConfig) -> float:
    return operational_cost + annualized_pv_cost(design, cfg) + annualized_battery_cost(design, cfg)


def lcoe(total_cost: float, annual_load_kwh: float) -> float:
    if annual_load_kwh <= 0:
        raise ValueError("annual_load_kwh must be positive")
    return total_cost / annual_load_kwh

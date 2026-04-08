from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class CostConfig:
    # Editable defaults chosen for a runnable demo.
    pv_capex_per_kw: float = 1100.0
    battery_capex_per_kwh: float = 350.0
    pv_fixed_om_per_kw_year: float = 18.0
    battery_fixed_om_per_kwh_year: float = 8.0
    diesel_cost_per_kwh: float = 0.22
    discount_rate: float = 0.08
    pv_lifetime_years: int = 25
    battery_lifetime_years: int = 12


@dataclass
class OperationConfig:
    horizon_hours: int = 24 * 7
    initial_soc_fraction: float = 0.50
    soc_min_fraction: float = 0.10
    soc_max_fraction: float = 0.90
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95

    # User-preference-aligned default: max charge/discharge power is 10% of energy capacity per hour.
    charge_power_fraction_per_hour: float = 0.10
    discharge_power_fraction_per_hour: float = 0.10

    diesel_min_kw: float = 150.0
    diesel_max_kw: float = 1500.0
    grid_max_kw: float = 2500.0
    terminal_soc_target_fraction: float | None = None


@dataclass
class DataConfig:
    year_hours: int = 24 * 365
    seed: int = 7
    base_load_kw: float = 900.0
    daily_peak_kw: float = 350.0
    seasonal_peak_kw: float = 180.0
    summer_price_adder: float = 0.03


@dataclass
class DesignSpaceConfig:
    battery_sizes_kwh: List[float] = field(
        default_factory=lambda: np.linspace(500.0, 5000.0, 10).round(3).tolist()
    )
    pv_sizes_kw: List[float] = field(
        default_factory=lambda: np.linspace(500.0, 2500.0, 10).round(3).tolist()
    )
    sample_seed: int = 42


@dataclass
class OrdinalOptimizationConfig:
    probability_top_alpha_hit: float = 0.99
    alpha_fraction: float = 0.05
    good_design_count: int = 10
    overlap_k: int = 1
    alignment_target: float = 0.90


@dataclass
class ExperimentConfig:
    costs: CostConfig = field(default_factory=CostConfig)
    operation: OperationConfig = field(default_factory=OperationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    design_space: DesignSpaceConfig = field(default_factory=DesignSpaceConfig)
    oo: OrdinalOptimizationConfig = field(default_factory=OrdinalOptimizationConfig)

    # By default, full-year data are generated. For quick smoke tests set to e.g. 24*56.
    simulation_hours: int | None = None


def default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig()

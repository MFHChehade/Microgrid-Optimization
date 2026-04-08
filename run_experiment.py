from __future__ import annotations

import argparse
from pathlib import Path

from boost_sizing import default_experiment_config, generate_synthetic_year
from boost_sizing.boost import run_boost_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a clean BOOST-style microgrid sizing experiment.")
    parser.add_argument("--out_dir", type=str, default="results/run_full")
    parser.add_argument("--simulation_hours", type=int, default=None, help="Override the number of hours simulated.")
    parser.add_argument("--battery_grid", type=int, default=10, help="Number of battery capacity grid points.")
    parser.add_argument("--pv_grid", type=int, default=10, help="Number of PV size grid points.")
    parser.add_argument("--no_baselines", action="store_true")
    args = parser.parse_args()

    cfg = default_experiment_config()
    if args.simulation_hours is not None:
        cfg.simulation_hours = args.simulation_hours

    # Replace design grids if requested.
    import numpy as np
    cfg.design_space.battery_sizes_kwh = np.linspace(500.0, 5000.0, args.battery_grid).round(3).tolist()
    cfg.design_space.pv_sizes_kw = np.linspace(500.0, 2500.0, args.pv_grid).round(3).tolist()

    data = generate_synthetic_year(cfg.data)
    summary = run_boost_experiment(
        data=data,
        cfg=cfg,
        out_dir=Path(args.out_dir),
        evaluate_baselines=not args.no_baselines,
    )
    print("Finished.")
    print(summary)


if __name__ == "__main__":
    main()

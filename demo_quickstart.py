from __future__ import annotations

from pathlib import Path

from boost_sizing import default_experiment_config, generate_synthetic_year
from boost_sizing.boost import run_boost_experiment


def main() -> None:
    cfg = default_experiment_config()
    cfg.simulation_hours = 24 * 56  # 8-week smoke test
    cfg.design_space.battery_sizes_kwh = [500, 1000, 1500, 2000, 2500]
    cfg.design_space.pv_sizes_kw = [500, 1000, 1500, 2000, 2500]
    cfg.oo.good_design_count = 5

    data = generate_synthetic_year(cfg.data)
    summary = run_boost_experiment(
        data=data,
        cfg=cfg,
        out_dir=Path("results/quickstart"),
        evaluate_baselines=True,
    )
    print(summary)


if __name__ == "__main__":
    main()

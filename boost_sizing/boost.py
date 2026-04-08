from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import pandas as pd

from .baselines import dp_dispatch_year, greedy_dispatch_year
from .config import ExperimentConfig
from .costs import total_annual_cost, lcoe
from .design_space import enumerate_designs, sample_designs
from .oo import compute_n, choose_s, alignment_probability
from .types import Design, TimeSeriesData
from .yearly import evaluate_design_year


def _effective_data(data: TimeSeriesData, simulation_hours: int | None) -> TimeSeriesData:
    if simulation_hours is None or simulation_hours >= len(data.load_kw):
        return data
    return TimeSeriesData(
        timestamps=data.timestamps[:simulation_hours],
        load_kw=data.load_kw[:simulation_hours],
        solar_cf=data.solar_cf[:simulation_hours],
        grid_price_per_kwh=data.grid_price_per_kwh[:simulation_hours],
    )


def run_boost_experiment(
    data: TimeSeriesData,
    cfg: ExperimentConfig,
    out_dir: str | Path,
    evaluate_baselines: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(exist_ok=True)

    data = _effective_data(data, cfg.simulation_hours)
    total_load = float(data.load_kw.sum())

    all_designs = enumerate_designs(cfg.design_space)
    n_theory = compute_n(cfg.oo.probability_top_alpha_hit, cfg.oo.alpha_fraction)
    sampled_designs = sample_designs(all_designs, n=min(n_theory, len(all_designs)), seed=cfg.design_space.sample_seed)
    n_used = len(sampled_designs)

    g = min(cfg.oo.good_design_count, n_used)
    s = choose_s(cfg.oo.overlap_k, g, n_used, cfg.oo.alignment_target)
    ap = alignment_probability(cfg.oo.overlap_k, g, s, n_used)

    lp_rows = []
    for design in sampled_designs:
        op_cost, _, _ = evaluate_design_year(data=data, design=design, cfg=cfg, accurate=False, keep_schedule=False)
        total_cost = total_annual_cost(op_cost, design, cfg.costs)
        lp_rows.append(
            {
                "design": design.label,
                "battery_kwh": design.battery_kwh,
                "pv_kw": design.pv_kw,
                "lp_operational_cost": op_cost,
                "lp_total_cost": total_cost,
                "lp_lcoe": lcoe(total_cost, total_load),
            }
        )
    lp_df = pd.DataFrame(lp_rows).sort_values("lp_total_cost", ascending=True).reset_index(drop=True)
    lp_df["lp_rank"] = range(1, len(lp_df) + 1)
    lp_df.to_csv(out_dir / "lp_rankings.csv", index=False)

    top_s = lp_df.head(s).copy()
    accurate_rows = []
    best_schedule = None
    best_design = None
    for _, row in top_s.iterrows():
        design = Design(battery_kwh=float(row["battery_kwh"]), pv_kw=float(row["pv_kw"]))
        op_cost, _, disp = evaluate_design_year(data=data, design=design, cfg=cfg, accurate=True, keep_schedule=True)
        total_cost = total_annual_cost(op_cost, design, cfg.costs)
        accurate_rows.append(
            {
                "design": design.label,
                "battery_kwh": design.battery_kwh,
                "pv_kw": design.pv_kw,
                "accurate_operational_cost": op_cost,
                "accurate_total_cost": total_cost,
                "accurate_lcoe": lcoe(total_cost, total_load),
            }
        )
        if best_schedule is None or total_cost < min(r["accurate_total_cost"] for r in accurate_rows[:-1] or [{"accurate_total_cost": math.inf}]):
            best_schedule = disp.schedule if disp is not None else None
            best_design = design

    accurate_df = pd.DataFrame(accurate_rows).sort_values("accurate_total_cost", ascending=True).reset_index(drop=True)
    accurate_df["accurate_rank"] = range(1, len(accurate_df) + 1)
    merged = accurate_df.merge(lp_df[["design", "lp_rank", "lp_lcoe"]], on="design", how="left")
    merged["order_gain"] = merged["lp_rank"] - merged["accurate_rank"]
    merged.to_csv(out_dir / "accurate_top_s.csv", index=False)

    if best_schedule is not None:
        best_schedule.to_csv(out_dir / "best_schedule.csv", index=False)

    # Plots
    plt.figure(figsize=(6, 4))
    plt.scatter(merged["lp_rank"], merged["accurate_rank"])
    lim = max(int(merged["lp_rank"].max()), int(merged["accurate_rank"].max())) + 1
    plt.plot([1, lim], [1, lim], linestyle="--")
    plt.xlabel("LP rank")
    plt.ylabel("Accurate rank")
    plt.title("Ranking stability inside top-s set")
    plt.tight_layout()
    plt.savefig(fig_dir / "rank_stability.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plot_df = merged.nsmallest(min(5, len(merged)), "accurate_total_cost")
    labels = [f"B{int(b/1000)}MWh\nPV{int(p)}kW" for b, p in zip(plot_df["battery_kwh"], plot_df["pv_kw"])]
    plt.bar(labels, plot_df["accurate_lcoe"] * 100)
    plt.ylabel("LCOE (c/kWh)")
    plt.title("Top accurate designs")
    plt.tight_layout()
    plt.savefig(fig_dir / "top_designs_lcoe.png", dpi=180)
    plt.close()

    baseline_summary = {}
    if evaluate_baselines and best_design is not None:
        greedy_cost, greedy_sched = greedy_dispatch_year(data, best_design, cfg)
        greedy_total = total_annual_cost(greedy_cost, best_design, cfg.costs)
        greedy_sched.to_csv(out_dir / "greedy_schedule.csv", index=False)

        dp_cost, dp_sched = dp_dispatch_year(data, best_design, cfg)
        dp_total = total_annual_cost(dp_cost, best_design, cfg.costs)
        dp_sched.to_csv(out_dir / "dp_schedule.csv", index=False)

        boost_total = float(accurate_df.iloc[0]["accurate_total_cost"])
        baseline_summary = {
            "BOOST": {"total_cost": float(boost_total), "lcoe_c_per_kwh": float(100.0 * lcoe(boost_total, total_load))},
            "DP": {"total_cost": float(dp_total), "lcoe_c_per_kwh": float(100.0 * lcoe(dp_total, total_load))},
            "greedy": {"total_cost": float(greedy_total), "lcoe_c_per_kwh": float(100.0 * lcoe(greedy_total, total_load))},
        }
        pd.DataFrame(
            [{"method": k, **v} for k, v in baseline_summary.items()]
        ).to_csv(out_dir / "baseline_comparison.csv", index=False)

        plt.figure(figsize=(5, 4))
        methods = list(baseline_summary.keys())
        values = [baseline_summary[m]["lcoe_c_per_kwh"] for m in methods]
        plt.bar(methods, values)
        plt.ylabel("LCOE (c/kWh)")
        plt.title("Method comparison on best BOOST design")
        plt.tight_layout()
        plt.savefig(fig_dir / "baseline_comparison.png", dpi=180)
        plt.close()

    summary = {
        "n_theory": n_theory,
        "n_used": n_used,
        "g": g,
        "s": s,
        "alignment_probability": ap,
        "best_design": None if best_design is None else {"battery_kwh": best_design.battery_kwh, "pv_kw": best_design.pv_kw},
        "best_accurate_lcoe_c_per_kwh": None if accurate_df.empty else 100.0 * float(accurate_df.iloc[0]["accurate_lcoe"]),
        "baseline_summary": baseline_summary,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

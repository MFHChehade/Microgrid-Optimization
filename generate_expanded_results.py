from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from boost_sizing import default_experiment_config, generate_synthetic_year
from boost_sizing.boost import run_boost_experiment
from boost_sizing.costs import total_annual_cost, lcoe
from boost_sizing.design_space import enumerate_designs
from boost_sizing.oo import compute_n, choose_s
from boost_sizing.types import Design, TimeSeriesData
from boost_sizing.yearly import evaluate_design_year

OUT_DIR = Path('results/expanded_arxiv')
FIG_DIR = OUT_DIR / 'figs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'lines.linewidth': 2.0,
})


def evaluate_design_set(cfg, data, designs, accurate=False, keep_best_schedule=False):
    total_load = float(data.load_kw.sum())
    rows = []
    best_total = math.inf
    best_design = None
    best_schedule = None
    t0 = time.perf_counter()
    for i, d in enumerate(designs, 1):
        op_cost, _, disp = evaluate_design_year(data, d, cfg, accurate=accurate, keep_schedule=keep_best_schedule)
        total_cost = total_annual_cost(op_cost, d, cfg.costs)
        rows.append({
            'design': d.label,
            'battery_kwh': d.battery_kwh,
            'pv_kw': d.pv_kw,
            'operational_cost': op_cost,
            'total_cost': total_cost,
            'lcoe_c_per_kwh': 100.0 * lcoe(total_cost, total_load),
        })
        if total_cost < best_total:
            best_total = total_cost
            best_design = d
            if keep_best_schedule and disp is not None and disp.schedule is not None:
                best_schedule = disp.schedule.copy()
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows).sort_values('total_cost').reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)
    return df, best_design, best_schedule, elapsed


def scenario_data_and_cfg(name: str):
    cfg = default_experiment_config()
    data = generate_synthetic_year(cfg.data)
    if name == 'base':
        pass
    elif name == 'cheap_battery':
        cfg.costs.battery_capex_per_kwh = 180.0
        cfg.costs.battery_fixed_om_per_kwh_year = 5.0
    elif name == 'cheap_pv':
        cfg.costs.pv_capex_per_kw = 700.0
        cfg.costs.pv_fixed_om_per_kw_year = 12.0
    elif name == 'expensive_diesel':
        cfg.costs.diesel_cost_per_kwh = 0.36
    elif name == 'high_peak_tariff':
        hod = data.timestamps.hour.to_numpy()
        price = data.grid_price_per_kwh + np.where((hod >= 17) & (hod <= 21), 0.12, 0.0)
        data = TimeSeriesData(
            timestamps=data.timestamps,
            load_kw=data.load_kw,
            solar_cf=data.solar_cf,
            grid_price_per_kwh=price,
        )
    else:
        raise ValueError(name)
    return cfg, data


# 1) Base full-grid exhaustive evaluation for stronger screening analysis.
base_cfg, base_data = scenario_data_and_cfg('base')
all_designs = enumerate_designs(base_cfg.design_space)
print('Running exhaustive LP over full design grid...')
lp_df, _, _, lp_elapsed = evaluate_design_set(base_cfg, base_data, all_designs, accurate=False, keep_best_schedule=False)
lp_df = lp_df.rename(columns={
    'operational_cost': 'lp_operational_cost',
    'total_cost': 'lp_total_cost',
    'lcoe_c_per_kwh': 'lp_lcoe_c_per_kwh',
    'rank': 'lp_rank',
})
print('Running exhaustive accurate MILP over full design grid...')
acc_df, acc_best_design, acc_best_schedule, acc_elapsed = evaluate_design_set(base_cfg, base_data, all_designs, accurate=True, keep_best_schedule=True)
acc_df = acc_df.rename(columns={
    'operational_cost': 'accurate_operational_cost',
    'total_cost': 'accurate_total_cost',
    'lcoe_c_per_kwh': 'accurate_lcoe_c_per_kwh',
    'rank': 'accurate_rank',
})
merged = lp_df.merge(acc_df, on=['design', 'battery_kwh', 'pv_kw'])
merged['rank_gap'] = merged['lp_rank'] - merged['accurate_rank']
merged['cost_gap_pct'] = 100.0 * (merged['lp_total_cost'] - merged['accurate_total_cost']) / merged['accurate_total_cost']
merged.to_csv(OUT_DIR / 'full_grid_lp_vs_accurate.csv', index=False)
if acc_best_schedule is not None:
    acc_best_schedule.to_csv(OUT_DIR / 'best_accurate_schedule.csv', index=False)

rho = spearmanr(merged['lp_rank'], merged['accurate_rank']).statistic

# Recall curve: how many true top-g accurate designs are recovered in the LP top-s set?
g = min(base_cfg.oo.good_design_count, len(merged))
true_top_g = set(acc_df.head(g)['design'])
recall_rows = []
for s in range(1, len(merged) + 1):
    lp_top_s = set(lp_df.head(s)['design'])
    recall = len(true_top_g.intersection(lp_top_s)) / g
    contains_true_best = int(acc_df.iloc[0]['design'] in lp_top_s)
    recall_rows.append({'s': s, 'top_g_recall': recall, 'contains_true_best': contains_true_best})
recall_df = pd.DataFrame(recall_rows)
recall_df.to_csv(OUT_DIR / 'screening_recall_curve.csv', index=False)

# BOOST runtime and outcome on the sampled 90-design set used in the paper-style procedure.
print('Running actual BOOST pipeline for runtime comparison...')
boost_cfg = default_experiment_config()
boost_start = time.perf_counter()
boost_summary = run_boost_experiment(base_data, boost_cfg, OUT_DIR / 'boost_pipeline', evaluate_baselines=True)
boost_elapsed = time.perf_counter() - boost_start

boost_best_design = Design(
    battery_kwh=float(boost_summary['best_design']['battery_kwh']),
    pv_kw=float(boost_summary['best_design']['pv_kw']),
)
true_best_matches_boost = (
    abs(boost_best_design.battery_kwh - acc_best_design.battery_kwh) < 1e-9 and
    abs(boost_best_design.pv_kw - acc_best_design.pv_kw) < 1e-9
)

# 2) Scenario analysis using the BOOST pipeline.
scenario_names = ['base', 'cheap_battery', 'cheap_pv', 'expensive_diesel', 'high_peak_tariff']
scenario_rows = []
for name in scenario_names:
    print(f'Running scenario: {name}')
    cfg, data = scenario_data_and_cfg(name)
    out_dir = OUT_DIR / f'scenario_{name}'
    start = time.perf_counter()
    summary = run_boost_experiment(data, cfg, out_dir, evaluate_baselines=False)
    elapsed = time.perf_counter() - start
    scenario_rows.append({
        'scenario': name,
        'battery_kwh': summary['best_design']['battery_kwh'],
        'pv_kw': summary['best_design']['pv_kw'],
        'lcoe_c_per_kwh': summary['best_accurate_lcoe_c_per_kwh'],
        'n_used': summary['n_used'],
        's': summary['s'],
        'alignment_probability': summary['alignment_probability'],
        'runtime_sec': elapsed,
    })
scenario_df = pd.DataFrame(scenario_rows)
scenario_df.to_csv(OUT_DIR / 'scenario_summary.csv', index=False)

# 3) Publication-quality figures.
# Figure A: LP vs accurate landscape.
def pivot_grid(df, value_col):
    p = df.pivot(index='battery_kwh', columns='pv_kw', values=value_col)
    return p.sort_index().sort_index(axis=1)

lp_grid = pivot_grid(merged, 'lp_lcoe_c_per_kwh')
acc_grid = pivot_grid(merged, 'accurate_lcoe_c_per_kwh')
vmin = min(lp_grid.min().min(), acc_grid.min().min())
vmax = max(lp_grid.max().max(), acc_grid.max().max())
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
for ax, grid, title in zip(axes, [lp_grid, acc_grid], ['Simple LP screening landscape', 'Accurate MILP landscape']):
    im = ax.imshow(grid.values, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('PV size (kW)')
    ax.set_ylabel('Battery size (kWh)')
    ax.set_xticks(np.arange(len(grid.columns)))
    ax.set_xticklabels([f'{int(v)}' for v in grid.columns], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(grid.index)))
    ax.set_yticklabels([f'{int(v)}' for v in grid.index])
    # Mark the true accurate optimum.
    bx = list(grid.columns).index(acc_best_design.pv_kw)
    by = list(grid.index).index(acc_best_design.battery_kwh)
    ax.scatter([bx], [by], marker='*', s=220, edgecolor='black', linewidth=0.8)
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, label='LCOE (¢/kWh)')
fig.savefig(FIG_DIR / 'design_landscape_lp_vs_accurate.png', bbox_inches='tight')
plt.close(fig)

# Figure B: screening recall curve.
fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
ax.plot(recall_df['s'], 100.0 * recall_df['top_g_recall'], label='Recall of true top-10 accurate designs')
ax.plot(recall_df['s'], 100.0 * recall_df['contains_true_best'], label='True accurate optimum included', linestyle='--')
n_theory = compute_n(base_cfg.oo.probability_top_alpha_hit, base_cfg.oo.alpha_fraction)
s_theory = choose_s(base_cfg.oo.overlap_k, g, n_theory, base_cfg.oo.alignment_target)
ax.axvline(s_theory, linestyle=':', linewidth=1.7, label=f'Paper-style choice: s={s_theory}')
ax.set_xlabel('Number of designs re-evaluated with MILP (s)')
ax.set_ylabel('Recovery (%)')
ax.set_title(f'Screening quality of the LP ranking (Spearman ρ = {rho:.3f})')
ax.set_ylim(0, 105)
ax.legend(frameon=True)
fig.savefig(FIG_DIR / 'screening_recall_curve.png', bbox_inches='tight')
plt.close(fig)

# Figure C: runtime comparison.
runtime_df = pd.DataFrame([
    {'method': 'Exhaustive LP\n(100 designs)', 'seconds': lp_elapsed},
    {'method': 'Exhaustive accurate\n(100 designs)', 'seconds': acc_elapsed},
    {'method': 'BOOST pipeline\n(90 LP + 18 MILP)', 'seconds': boost_elapsed},
])
runtime_df.to_csv(OUT_DIR / 'runtime_summary.csv', index=False)
fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
bars = ax.bar(runtime_df['method'], runtime_df['seconds'])
ax.set_ylabel('Wall-clock time (s)')
ax.set_title('Runtime benefit of screening before accurate re-evaluation')
for b, val in zip(bars, runtime_df['seconds']):
    ax.text(b.get_x() + b.get_width()/2, val + 0.02*runtime_df['seconds'].max(), f'{val:.1f}', ha='center', va='bottom')
fig.savefig(FIG_DIR / 'runtime_comparison.png', bbox_inches='tight')
plt.close(fig)

# Figure D: one representative summer week dispatch.
if acc_best_schedule is not None:
    sched = acc_best_schedule.copy()
    sched['timestamp'] = pd.to_datetime(sched['timestamp'])
    week_id = sched['timestamp'].dt.isocalendar().week.astype(int)
    summer_mask = sched['timestamp'].dt.month.isin([6, 7, 8])
    week_scores = sched.loc[summer_mask].groupby(week_id)['load_kw'].mean().sort_values(ascending=False)
    target_week = int(week_scores.index[0]) if len(week_scores) else int(week_id.iloc[0])
    week_df = sched.loc[week_id == target_week].reset_index(drop=True)
    x = np.arange(len(week_df))
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 5.8), sharex=True, constrained_layout=True)
    axes[0].stackplot(
        x,
        week_df['pv_kw'].to_numpy(),
        week_df['diesel_kw'].to_numpy(),
        week_df['grid_kw'].to_numpy(),
        week_df['discharge_kw'].to_numpy(),
        labels=['PV', 'Diesel', 'Grid', 'Battery discharge'],
        alpha=0.88,
    )
    axes[0].plot(x, week_df['load_kw'].to_numpy(), label='Load', linewidth=1.7)
    axes[0].set_ylabel('Power (kW)')
    axes[0].set_title(f'Representative summer-week dispatch for best accurate design ({acc_best_design.label})')
    axes[0].legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.22), frameon=True)

    axes[1].plot(x, week_df['soc_kwh'].to_numpy(), label='Battery SOC (kWh)')
    axes[1].plot(x, 1000.0 * week_df['grid_price_per_kwh'].to_numpy(), label='Grid price (mills/kWh)', linestyle='--')
    axes[1].set_xlabel('Hour within week')
    axes[1].set_ylabel('SOC / price')
    axes[1].legend(frameon=True)
    fig.savefig(FIG_DIR / 'dispatch_representative_summer_week.png', bbox_inches='tight')
    plt.close(fig)

# Figure E: scenario sensitivity.
scenario_order = scenario_names
scenario_df['scenario'] = pd.Categorical(scenario_df['scenario'], categories=scenario_order, ordered=True)
scenario_df = scenario_df.sort_values('scenario').reset_index(drop=True)
fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), sharex=True, constrained_layout=True)
axes[0].plot(scenario_df['scenario'], scenario_df['battery_kwh'], marker='o')
axes[0].set_ylabel('Battery (kWh)')
axes[0].set_title('Best design selected by BOOST under cost/tariff scenarios')
axes[1].plot(scenario_df['scenario'], scenario_df['pv_kw'], marker='o')
axes[1].set_ylabel('PV (kW)')
axes[2].plot(scenario_df['scenario'], scenario_df['lcoe_c_per_kwh'], marker='o')
axes[2].set_ylabel('LCOE (¢/kWh)')
axes[2].set_xlabel('Scenario')
for ax in axes:
    ax.tick_params(axis='x', rotation=20)
fig.savefig(FIG_DIR / 'scenario_sensitivity.png', bbox_inches='tight')
plt.close(fig)

# Additional scatter figure for cost agreement.
fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
ax.scatter(merged['lp_total_cost'] / 1e6, merged['accurate_total_cost'] / 1e6, alpha=0.75)
mins = min(merged['lp_total_cost'].min(), merged['accurate_total_cost'].min()) / 1e6
maxs = max(merged['lp_total_cost'].max(), merged['accurate_total_cost'].max()) / 1e6
ax.plot([mins, maxs], [mins, maxs], linestyle='--')
ax.set_xlabel('LP-screened total cost (million $/year)')
ax.set_ylabel('Accurate total cost (million $/year)')
ax.set_title('Cost ordering remains close after adding diesel commitment logic')
fig.savefig(FIG_DIR / 'lp_vs_accurate_cost_scatter.png', bbox_inches='tight')
plt.close(fig)

# 4) Compact results summary for writing.
summary = {
    'full_grid_true_best_design': {'battery_kwh': acc_best_design.battery_kwh, 'pv_kw': acc_best_design.pv_kw},
    'full_grid_true_best_lcoe_c_per_kwh': float(acc_df.iloc[0]['accurate_lcoe_c_per_kwh']),
    'boost_best_design': {'battery_kwh': boost_best_design.battery_kwh, 'pv_kw': boost_best_design.pv_kw},
    'boost_best_matches_full_grid_true_best': bool(true_best_matches_boost),
    'spearman_rank_correlation_lp_vs_accurate': float(rho),
    'lp_elapsed_sec_full_grid': float(lp_elapsed),
    'accurate_elapsed_sec_full_grid': float(acc_elapsed),
    'boost_elapsed_sec': float(boost_elapsed),
    'runtime_reduction_vs_exhaustive_accurate_pct': float(100.0 * (1.0 - boost_elapsed / acc_elapsed)),
    'paper_style_n': int(n_theory),
    'paper_style_s': int(s_theory),
    'recall_at_paper_style_s': float(recall_df.loc[recall_df['s'] == s_theory, 'top_g_recall'].iloc[0]),
    'contains_true_best_at_paper_style_s': bool(recall_df.loc[recall_df['s'] == s_theory, 'contains_true_best'].iloc[0]),
    'scenario_results': scenario_df.to_dict(orient='records'),
    'baseline_summary_on_boost_best_design': boost_summary.get('baseline_summary', {}),
}
with open(OUT_DIR / 'expanded_results_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

# Writing-oriented markdown note.
md = []
md.append('# Expanded BOOST results note')
md.append('')
md.append('## Main takeaways')
md.append(f"- On the full 10x10 design grid, the true accurate optimum is **B={acc_best_design.battery_kwh:.0f} kWh, PV={acc_best_design.pv_kw:.0f} kW** with **LCOE = {acc_df.iloc[0]['accurate_lcoe_c_per_kwh']:.3f} ¢/kWh**.")
md.append(f"- The paper-style BOOST pipeline selected **B={boost_best_design.battery_kwh:.0f} kWh, PV={boost_best_design.pv_kw:.0f} kW**, and the sampled-set optimum {'matches' if true_best_matches_boost else 'does not match'} the full-grid accurate optimum.")
md.append(f"- Across the full grid, LP and accurate ranks have **Spearman ρ = {rho:.3f}**.")
md.append(f"- At the paper-style choice **s={s_theory}**, the LP top-s set recovers **{100.0 * recall_df.loc[recall_df['s'] == s_theory, 'top_g_recall'].iloc[0]:.1f}%** of the true top-{g} accurate designs and {'contains' if recall_df.loc[recall_df['s'] == s_theory, 'contains_true_best'].iloc[0] else 'does not contain'} the global accurate optimum.")
md.append(f"- Measured runtime: exhaustive accurate = **{acc_elapsed:.1f} s**, BOOST pipeline = **{boost_elapsed:.1f} s**, a reduction of **{100.0 * (1.0 - boost_elapsed / acc_elapsed):.1f}%** against evaluating every design accurately.")
md.append('')
md.append('## Scenario observations')
for row in scenario_df.to_dict(orient='records'):
    md.append(f"- **{row['scenario']}**: best design = ({row['battery_kwh']:.0f} kWh, {row['pv_kw']:.0f} kW), LCOE = {row['lcoe_c_per_kwh']:.3f} ¢/kWh.")
(Path(OUT_DIR / 'expanded_results_note.md')).write_text('\n'.join(md), encoding='utf-8')

print('Wrote expanded results to', OUT_DIR)
print(json.dumps(summary, indent=2))

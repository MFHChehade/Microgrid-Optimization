# BOOST-style microgrid sizing reimplementation

This is a **clean reimplementation** of the core idea in **BOOST: Microgrid Sizing using Ordinal Optimization**:

1. evaluate many candidate `(battery size, PV size)` designs with a **simple LP**;
2. rank them by **annual operating cost + annualized investment cost**;
3. re-evaluate only the top-`s` designs with a more accurate **MILP** that adds the diesel minimum-output logic.

## What is included

- `boost_sizing/dispatch.py`  
  Weekly dispatch solver:
  - LP = simple model
  - MILP = accurate model with diesel on/off binary

- `boost_sizing/yearly.py`  
  Rolls the weekly dispatch solver across the full year with **SOC carryover** from week to week.

- `boost_sizing/oo.py`  
  Implements:
  - the `N` formula from the paper
  - the alignment-probability formula
  - automatic choice of `s`

- `boost_sizing/boost.py`  
  End-to-end experiment runner:
  - sample `N` designs
  - solve LP for each
  - rank designs
  - solve MILP on top `s`
  - export CSVs, PNG plots, and JSON summary

- `boost_sizing/baselines.py`  
  Two comparison baselines:
  - greedy dispatch
  - approximate DP dispatch

- `boost_sizing/synthetic_data.py`  
  Creates an hourly synthetic year of:
  - load
  - solar capacity factor
  - time-of-use grid price

## Why this version is slightly more explicit than the paper

The paper is short, so a few engineering details have to be made explicit in code. This implementation therefore adds or fixes the following:

- **battery charge/discharge power limits**  
  The paper shows SOC dynamics and SOC bounds, but does not visibly spell out explicit battery power-rate constraints in the short formulation. Here they are added directly.

- **weekly SOC carryover**  
  The paper says a new optimization is solved every week. This code carries the final SOC from one week into the next.

- **annualized costs**  
  PV and battery investment costs are annualized using the standard capital recovery factor.

- **synthetic demo data by default**  
  This keeps the code runnable out of the box. You can swap in real hourly CSVs easily by replacing the `generate_synthetic_year(...)` call with your own loader.

## Default modeling choices

These are editable in `boost_sizing/config.py`.

- battery max charge power = `0.10 * battery_kwh` kW
- battery max discharge power = `0.10 * battery_kwh` kW
- SOC bounds = 10% to 90%
- charge / discharge efficiencies = 95%
- diesel minimum output = 150 kW
- diesel maximum output = 1500 kW
- grid maximum import = 2500 kW

The 10%-per-hour battery power limit matches a conservative engineering default and can be changed in one place.

## Quick start

```bash
python demo_quickstart.py
```

This runs an **8-week smoke test** and writes outputs under:

```bash
results/quickstart/
```

## Full(er) run

```bash
python run_experiment.py --out_dir results/run_full
```

For the default 10x10 design grid, the theoretical ordinal-optimization formula gives about 90 candidate designs for the paper's `P=99%`, `alpha=5%` setting.

## Main outputs

- `lp_rankings.csv`  
  LP-screened designs

- `accurate_top_s.csv`  
  top-`s` designs after accurate MILP re-evaluation

- `best_schedule.csv`  
  yearly schedule for the best accurate design

- `baseline_comparison.csv`  
  LCOE comparison of BOOST vs. DP vs. greedy on the selected best design

- `summary.json`  
  overall experiment summary

- `figs/*.png`  
  rank-stability and LCOE plots

## Notes on faithfulness

This is **method-faithful**, not a byte-for-byte clone of the original public codebase. In particular:

- it uses **SciPy / HiGHS** instead of the original repository's implementation stack;
- it uses synthetic demo data unless you provide your own real data;
- the DP baseline is intentionally approximate and documented as such.

That said, the core paper logic is preserved:
- simple LP screening,
- top-`s` refinement with a more accurate MILP,
- ordinal-optimization formulas for `N` and `s`,
- total-cost ranking,
- LCOE reporting.

## Suggested next upgrades

1. swap synthetic data for real hourly load / irradiance / tariff CSVs;
2. replace the approximate DP baseline with the exact DP from the author's earlier microgrid work;
3. add battery degradation and curtailment penalties;
4. add explicit grid outage/intermittency modeling if desired.

# BOOST Microgrid Sizing — Clean Reimplementation

This repository contains a **clean Python reimplementation** of the core BOOST workflow for residential microgrid sizing:

- **stage 1:** evaluate many candidate designs with a cheaper **LP** dispatch model
- **stage 2:** re-evaluate only the top-ranked designs with a more accurate **MILP**
- **output:** choose the best design based on annualized total cost / LCOE

The implementation is designed to be **readable, editable, and runnable out of the box**.

## What this code is

This project is a **method-faithful reimplementation** of the paper’s main idea:

- outer **ordinal-optimization (OO)** screening
- inner **weekly dispatch optimization**
- full-year cost aggregation
- comparison against simple baselines

It is **not** an exact reproduction of the authors’ original numbers from the paper. The current code uses:

- a **synthetic hourly benchmark**
- explicit, editable engineering assumptions
- a simplified but consistent economic model

That makes it useful for:
- understanding the method
- generating new experiments
- creating figures quickly
- extending the workflow to real CSV data later

## Important note about the zip files

The **full runnable project** is the one that contains:

- `boost_sizing/`
- `requirements.txt`
- `run_experiment.py`
- `demo_quickstart.py`
- `generate_expanded_results.py`

If you downloaded a zip that only contains the three top-level scripts and **does not** include `boost_sizing/`, that archive is incomplete for running the code. Use the full runnable bundle instead.

## Repository structure

```text
boost_reimplementation_full/
├── boost_sizing/
│   ├── __init__.py
│   ├── baselines.py
│   ├── boost.py
│   ├── config.py
│   ├── costs.py
│   ├── csv_data.py
│   ├── design_space.py
│   ├── dispatch.py
│   ├── oo.py
│   ├── synthetic_data.py
│   ├── types.py
│   └── yearly.py
├── demo_quickstart.py
├── run_experiment.py
├── generate_expanded_results.py
├── requirements.txt
└── results/
```

## Main capabilities

### 1. Synthetic benchmark generation
The code can generate a synthetic year of hourly data for:

- load
- solar availability
- grid price

This lets the full pipeline run immediately without any external dataset.

### 2. Fixed-design dispatch
For a given design `(battery_kwh, pv_kw)`, the code solves:

- a **simple LP** dispatch model
- a more accurate **MILP** dispatch model

The dispatch is evaluated in **weekly windows** and rolled across the year.

### 3. Ordinal optimization
The OO stage:

- computes the theoretical sample size `N`
- chooses the retained set size `s`
- screens candidate designs with the LP
- re-evaluates top designs with the MILP

### 4. Baselines
The code includes lightweight baselines for comparison:

- **greedy**
- **approximate DP**
- **BOOST**

### 5. Expanded plotting / paper support
The project includes scripts used to generate the expanded benchmark outputs and figures used in the paper workflow.

## Installation

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or on a simple existing Python environment:

```powershell
pip install -r requirements.txt
```

## Quick start

### Fast sanity check
Runs a small end-to-end experiment and writes outputs to `results/quickstart/`.

```powershell
python .\demo_quickstart.py
```

### Main experiment
Runs the standard experiment pipeline.

```powershell
python .\run_experiment.py --out_dir results\run_full
```

### Expanded results / figure generation
Runs the expanded benchmark workflow and writes outputs and figures.

```powershell
python .\generate_expanded_results.py
```

## Expected outputs

Typical output artifacts include:

- LP rankings
- accurate top-`s` rankings
- best design summary
- baseline comparison CSV
- schedule CSVs
- figures for rank stability / LCOE / runtime / sensitivity / dispatch

Examples already included in the bundle:

```text
results/quickstart/
├── accurate_top_s.csv
├── baseline_comparison.csv
├── best_schedule.csv
├── dp_schedule.csv
├── greedy_schedule.csv
├── lp_rankings.csv
├── summary.json
└── figs/
```

## Data model

### Current default
The default workflow uses **synthetic hourly data** so that the project is self-contained.

### Real CSV support
The package includes `csv_data.py` to make it easier to switch to real data later. The intended real-data inputs are hourly series for:

- load
- solar / PV availability
- grid price

The codebase is structured so you can replace the synthetic generator with CSV-backed inputs without rewriting the optimization logic.

## Method summary

For each candidate design:

1. solve weekly dispatch across the year
2. aggregate annual operating cost
3. add annualized battery and PV cost
4. compute total cost / LCOE

Then BOOST does:

1. evaluate many designs with the **LP**
2. rank them
3. keep the top `s`
4. re-evaluate only those with the **MILP**
5. select the best refined design

## Modeling notes

This code makes several explicit choices that are reasonable for a clean engineering implementation but may not exactly match the original paper implementation:

- battery charge/discharge limits are modeled explicitly
- annualized cost terms are written in a transparent way
- weekly optimization is chained across the year
- the synthetic benchmark is used for repeatability and rapid experimentation

So this repo should be read as a **clean reimplementation**, not a claim of exact archival reproduction.

## Recommended workflow for future figure changes

To avoid rerunning expensive experiments repeatedly:

1. run the experiments once
2. save outputs to **CSV**
3. regenerate only the plots
4. recompile the paper

That is the workflow used in the later paper iterations.

## If you want to upload this to GitHub

A clean GitHub workflow is:

1. clone your repository fresh
2. mirror this project into the repo folder
3. `git add -A`
4. commit
5. push

Example PowerShell skeleton:

```powershell
git clone https://github.com/MFHChehade/Microgrid-Optimization.git
# copy / mirror the new project into the cloned folder
git add -A
git commit -m "Replace repository contents with BOOST reimplementation"
git push origin HEAD
```

## Known limitations

- not an exact reproduction of the original paper’s published numerical results
- current default benchmark is synthetic
- figure / paper artifacts may evolve separately from the base code
- approximate DP baseline is lightweight and intended mainly for comparative illustration

## Recommended citation / description

A good one-line description for the repo is:

> Clean Python reimplementation of BOOST-style ordinal-optimization microgrid sizing with LP screening, MILP refinement, synthetic benchmarks, and paper-ready figure generation.


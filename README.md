# CERES-Maize-NSGA-III Optimization of Irrigation and Nitrogen Scheduling

This repository contains the code and research materials associated with the manuscript:

> **Pareto-Front-Based Optimization of Irrigation and Nitrogen Application Schedule in Sandy-Soil Maize: Trade-offs among Yield, Water Productivity, Nitrogen Physiological Efficiency, and Grain Nitrogen Concentration**

The workflow couples the CERES-Maize model in DSSAT v4.8.2 with the NSGA-III many-objective evolutionary algorithm to optimize irrigation and nitrogen application schedules for shallow-buried drip-irrigated maize grown on sandy soil. Four objectives are maximized simultaneously:

- grain yield;
- crop water productivity (WPc);
- nitrogen physiological efficiency (PEN); and
- grain nitrogen concentration (GNC).

The repository also contains scripts for selecting candidate schedules from the Pareto front, identifying a distance-to-utopia compromise solution, and reproducing the weather-matched partial-budget economic sensitivity analysis.

## Repository structure

```text
.
|-- requirements.txt
|-- optimization/
|   |-- NSGA_III32.py
|   |-- data/
|   |   |-- DSDS2301.WTH
|   |   |-- MZCER048.CUL
|   |   |-- ORDO2302.MZX
|   |   `-- SoilData.txt
|   `-- postprocessing/
|       |-- Utopia00.py
|       `-- Yieldlt12000_01.py
`-- economic_analysis/
    |-- weather_matched_economic_analysis.py
    `-- data/
        `-- economic_analysis_input.xlsx
```

### Main files

| File | Purpose |
|---|---|
| `optimization/NSGA_III32.py` | Runs the CERES-Maize-NSGA-III optimization and exports the final population, first Pareto front, hypervolume history, and run configuration. |
| `optimization/data/DSDS2301.WTH` | Mean daily weather series derived from the 2019-2023 weather records and used for optimization. |
| `optimization/data/MZCER048.CUL` | Calibrated maize cultivar coefficients used by CERES-Maize. |
| `optimization/data/ORDO2302.MZX` | DSSAT maize experiment template modified by the optimization script for each candidate schedule. |
| `optimization/data/SoilData.txt` | Note identifying the soil data used in the study. |
| `optimization/postprocessing/Utopia00.py` | Normalizes the four objectives and identifies the solution with the minimum Euclidean distance to the utopia point. |
| `optimization/postprocessing/Yieldlt12000_01.py` | Selects Pareto-optimal schedules with simulated yield greater than the historical farmer benchmark of 11,823 kg/ha and decodes their irrigation and nitrogen schedules. |
| `economic_analysis/data/economic_analysis_input.xlsx` | Candidate-management data, annual candidate and farmer simulations, and the low/medium/high economic parameters used in the partial-budget analysis. |
| `economic_analysis/weather_matched_economic_analysis.py` | Reproduces the weather-matched agronomic comparison, 243 price-cost scenarios, break-even analysis, output tables, and publication figures. |

## Terminology used in the code

Some output column names retain labels from earlier versions of the analysis. They correspond to the manuscript terminology as follows:

| Code/output label | Manuscript term |
|---|---|
| `Yield` | Grain yield |
| `WUE` | Crop water productivity (WPc) |
| `NUE` | Nitrogen physiological efficiency (PEN) |
| `GrainN` | Grain nitrogen concentration (GNC) |

## Software requirements

### DSSAT

- DSSAT v4.8.2 with the CERES-Maize model
- Windows, using the default installation directory `C:\DSSAT48`
- The DSSAT executable `C:\DSSAT48\DSCSM048.EXE`

DSSAT is not distributed in this repository and must be installed separately. The current optimization script uses Windows-specific DSSAT paths. If DSSAT is installed elsewhere, update the following paths in `optimization/NSGA_III32.py`:

```text
C:\DSSAT48\DSCSM048.EXE
C:\DSSAT48\Maize\DSSBatch.v48
C:\DSSAT48\Maize\ORDO2302.MZX
```

### Python

Python 3.10 or later is recommended. Install the required packages with:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The principal Python dependencies are DEAP, NumPy, pandas, pymoo, SciPy, Matplotlib, and openpyxl.

## DSSAT input setup

Before running the optimization, prepare the DSSAT installation as follows:

1. Copy `optimization/data/DSDS2301.WTH` to the DSSAT weather directory.
2. Add the cultivar entry in `optimization/data/MZCER048.CUL` to the DSSAT maize cultivar file, preserving a backup of the original DSSAT file.
3. Ensure that the soil profile referenced by the experiment file (`ORordos003`) is present in the DSSAT soil database.
4. Create `optimization/MZXFile/` and copy `optimization/data/ORDO2302.MZX` to `optimization/MZXFile/ORDO2302.MZX`.
5. Ensure that `C:\DSSAT48\Maize\DSSBatch.v48` is configured to run the `ORDO2302.MZX` maize experiment.
6. Create an empty `optimization/results/` directory if it does not already exist.

The optimization script modifies a working copy of `ORDO2302.MZX`, moves it to the DSSAT maize directory, executes CERES-Maize, and reads `Summary.OUT` and `PlantN.OUT` to obtain the four objective values.

## Reproducing the optimization

Run the optimization from the `optimization` directory so that all relative paths resolve correctly:

```bash
cd optimization
python NSGA_III32.py
```

The supplied script uses the following configuration:

| Setting | Value |
|---|---:|
| Objectives | 4, all maximized |
| Decision variables | 32 |
| Reference-direction divisions (`P`) | 28 |
| Population size | 4,500 |
| Maximum generations | 50 |
| Crossover | Simulated binary bounded crossover |
| Crossover probability | 1.0 |
| Crossover distribution index | 30 |
| Mutation | Polynomial bounded mutation |
| Mutation probability | 1.0 |
| Per-variable mutation probability | 1/32 |
| Mutation distribution index | 20 |
| Hypervolume improvement tolerance | 1e-8 |
| Early-stopping patience | 8 generations |

### Random seeds

Five independent optimization runs were conducted using the fixed seeds:

```text
202501, 202502, 202503, 202504, 202505
```

The distributed script sets `DEFAULT_RANDOM_SEED = 202501`, corresponding to one reproducible run. To repeat the other runs, change `DEFAULT_RANDOM_SEED` to the required value before execution and save each run in a separate results directory. The script writes the active seed and completed generation count to `results/run_configuration.csv`.

### Optimization outputs

The following files are generated in `optimization/results/`:

| Output | Description |
|---|---|
| `run_last.pkl` | Serialized final population, logbook, objective matrix, Pareto ranks, first-front indices, random seed, and population size. |
| `pop_fit.npy` | Objective values for the final population. |
| `first_front_objectives.npy` | Objective values for the first Pareto front. |
| `first_front_objectives.csv` | First-front objective values in a tabular format. |
| `progress_hv.csv` | Normalized hypervolume by generation. |
| `run_configuration.csv` | Population size, random seed, and number of completed generations. |

Running all five optimizations can be computationally expensive because every individual evaluation launches DSSAT. The wall-clock time depends on the computer, DSSAT configuration, and number of generations completed before early stopping.

## Pareto-front post-processing

Run both post-processing scripts from the `optimization` directory after `results/run_last.pkl` and `results/first_front_objectives.csv` have been generated.

### Distance-to-utopia compromise solution

```bash
python postprocessing/Utopia00.py
```

This command generates:

```text
results/first_front_with_utopia_distance.csv
```

The script min-max normalizes each maximization objective, calculates the Euclidean distance to the utopia point `(1, 1, 1, 1)`, and prints the solution with the minimum distance.

### High-yield candidate schedules

```bash
python postprocessing/Yieldlt12000_01.py
```

The selection rule is:

```text
Yield > 11,823 kg/ha
```

The script exports:

- `results/filtered_objectives.csv`;
- `results/filtered_management_summary.csv`;
- `results/filtered_irrigation_events.csv`; and
- `results/filtered_fertilization_events.csv`.

The irrigation totals produced by this post-processing script include only the regular-season irrigation events generated explicitly by NSGA-III. Establishment irrigation is accounted for separately in the weather-matched economic-analysis input workbook.

## Reproducing the weather-matched economic analysis

The economic-analysis module can be run independently because its workbook contains the selected candidate schedules, candidate simulations under the five individual weather years, corresponding farmer simulations, and economic assumptions.

From the repository root, run:

```bash
python economic_analysis/weather_matched_economic_analysis.py \
  --input economic_analysis/data/economic_analysis_input.xlsx \
  --output-dir economic_analysis/economic_sensitivity_results
```

On Windows PowerShell, the same command can be entered on one line:

```powershell
python economic_analysis/weather_matched_economic_analysis.py --input economic_analysis/data/economic_analysis_input.xlsx --output-dir economic_analysis/economic_sensitivity_results
```

### Economic-analysis workbook

`economic_analysis_input.xlsx` contains four worksheets:

| Worksheet | Contents |
|---|---|
| `Candidate_Management` | Mean-weather objective values and management characteristics of the compromise and high-yield candidate schedules. |
| `Candidate_Year_Results` | Simulated performance of each candidate schedule under the 2019-2023 individual weather years, including establishment-irrigation accounting. |
| `Farmer_Year_Results` | Simulated farmer-management outcomes and management inputs for 2019-2023. |
| `Economic_Parameters` | Low, medium, and high levels of maize price and four input-cost factors. |

The partial net return is calculated as:

```text
PNR = Py * Y - CN * N - CI * I - Cirr * Firr - Cfert * Ffert
```

where `Py` is maize price, `Y` is grain yield, `CN` is nitrogen cost, `N` is total applied nitrogen, `CI` is irrigation electricity cost per unit depth, `I` is irrigation depth, `Cirr` is the operation cost per irrigation event, `Firr` is irrigation frequency, `Cfert` is the operation cost per nitrogen application event, and `Ffert` is nitrogen application frequency.

No GNC-based grain-price premium is included. Candidate schedules are compared with the farmer-managed schedule under the same weather year.

### Expected economic-analysis checks

A successful run should report:

| Check | Expected value |
|---|---:|
| Candidate schedules | 17 |
| Weather years | 5 |
| Candidate-year rows | 85 |
| Price-cost scenarios | 243 |
| Weather-matched comparisons | 20,655 |
| Break-even cost scenarios | 81 |
| Break-even candidate-year results | 6,885 |

The output directory contains detailed CSV tables, JSON metadata, a Markdown summary, and publication figures in PDF, PNG, and TIFF formats. Principal outputs include:

- `Fig_10_establishment_adjusted_comparison.*`;
- `Fig_11_weather_year_agronomic_performance.*`;
- `Fig_12_weather_matched_economic_analysis.*`;
- `weather_matched_all_results.csv`;
- `economic_summary_by_schedule.csv`;
- `economic_factor_effect_summary.csv`;
- `break_even_summary_by_schedule_year.csv`;
- `analysis_summary.md`; and
- `run_metadata.json`.

## Reproducibility workflow

The complete analysis sequence is:

1. Install and configure DSSAT v4.8.2.
2. Place the supplied weather, cultivar, and experiment inputs in the required DSSAT locations, and install the exact `ORordos003` soil profile in the DSSAT soil database.
3. Run `optimization/NSGA_III32.py` using the recorded random seeds.
4. Retain the representative run and execute the two Pareto-front post-processing scripts.
5. Use the supplied annual candidate and farmer simulations in `economic_analysis_input.xlsx` to reproduce the weather-matched agronomic and economic analyses.

## Citation

If you use this repository, please cite the associated manuscript. The complete bibliographic citation will be added after publication.

## Contact

For questions about the repository or reproduction workflow, contact:

**Zhigang Wang**  
College of Agronomy, Inner Mongolia Agricultural University  
Email: zgwang@imau.edu.cn

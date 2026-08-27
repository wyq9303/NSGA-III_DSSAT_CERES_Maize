# -*- coding: utf-8 -*-
"""Weather-matched agronomic and partial-budget analysis.

The script reads the fixed input workbook ``economic_analysis_input.xlsx``
(or a path supplied with ``--input``), pairs every candidate schedule with the
Farmer schedule under the same weather year, and evaluates all 3^5 = 243
price-cost scenarios.

Partial net return (PNR, yuan/ha):

    PNR = Py*Y - CN*N - CI*I - Cirr*Firr - Cfert*Ffert

No GNC-based price premium is included. Candidate irrigation cost accounting
uses the year-specific establishment irrigation depth and one establishment
irrigation event already included in ``total_irrigation_accounted_mm`` and
``total_irrigation_frequency_accounted``.

In addition to the full CSV result set, this publication version exports Fig. 10,
Fig. 11, and Fig. 12 in editable PDF and in 600-dpi PNG/TIFF formats. TIFF output
uses LZW compression.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


REQUIRED_SHEETS = [
    "Candidate_Management",
    "Candidate_Year_Results",
    "Farmer_Year_Results",
    "Economic_Parameters",
]

AGRONOMIC_METRICS = {
    "yield_kg_ha": ("Yield", "kg/ha"),
    "wpc_kg_m3": ("WPc", "kg/m$^3$"),
    "pen_kg_kg": ("PEN", "kg/kg"),
    "gnc_pct": ("GNC", "%"),
}

LEVEL_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weather-matched partial-budget analysis for candidate INASs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("economic_analysis_input.xlsx"),
        help="Input Excel workbook (default: economic_analysis_input.xlsx).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("economic_sensitivity_results"),
        help="Output directory (default: economic_sensitivity_results).",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], sheet_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{sheet_name} is missing columns: {missing}")


def validate_input(
    management: pd.DataFrame,
    candidates: pd.DataFrame,
    farmers: pd.DataFrame,
    parameters: pd.DataFrame,
) -> dict:
    require_columns(
        management,
        [
            "schedule_id",
            "candidate_type",
            "yield_mean_weather_kg_ha",
            "wpc_mean_weather_kg_m3",
            "pen_mean_weather_kg_kg",
            "gnc_mean_weather_pct",
        ],
        "Candidate_Management",
    )
    require_columns(
        candidates,
        [
            "schedule_id",
            "candidate_type",
            "weather_year",
            "yield_kg_ha",
            "wpc_kg_m3",
            "pen_kg_kg",
            "gnc_pct",
            "total_irrigation_accounted_mm",
            "total_n_kg_ha",
            "total_irrigation_frequency_accounted",
            "nitrogen_application_frequency",
            "simulation_status",
        ],
        "Candidate_Year_Results",
    )
    require_columns(
        farmers,
        [
            "weather_year",
            "yield_kg_ha",
            "wpc_kg_m3",
            "pen_kg_kg",
            "gnc_pct",
            "total_irrigation_mm",
            "total_n_kg_ha",
            "irrigation_frequency",
            "nitrogen_application_frequency",
            "simulation_status",
        ],
        "Farmer_Year_Results",
    )
    require_columns(parameters, ["parameter", "level", "value", "unit"], "Economic_Parameters")

    candidate_ids = management["schedule_id"].astype(str).tolist()
    weather_years = sorted(farmers["weather_year"].astype(int).tolist())
    expected = {(schedule, year) for schedule in candidate_ids for year in weather_years}
    observed = set(
        zip(
            candidates["schedule_id"].astype(str),
            candidates["weather_year"].astype(int),
        )
    )

    if candidates.duplicated(["schedule_id", "weather_year"]).any():
        duplicates = candidates.loc[
            candidates.duplicated(["schedule_id", "weather_year"], keep=False),
            ["schedule_id", "weather_year"],
        ]
        raise ValueError(f"Duplicate candidate-year rows found:\n{duplicates}")
    if farmers.duplicated(["weather_year"]).any():
        raise ValueError("Duplicate Farmer weather-year rows found.")
    if expected != observed:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(f"Candidate-year grid mismatch. Missing={missing}; unexpected={unexpected}")

    candidate_status = candidates["simulation_status"].fillna("").astype(str).str.upper()
    farmer_status = farmers["simulation_status"].fillna("").astype(str).str.upper()
    if not candidate_status.eq("OK").all():
        bad = candidates.loc[~candidate_status.eq("OK"), ["schedule_id", "weather_year", "simulation_status"]]
        raise ValueError(f"Candidate simulations not marked OK:\n{bad}")
    if not farmer_status.eq("OK").all():
        bad = farmers.loc[~farmer_status.eq("OK"), ["weather_year", "simulation_status"]]
        raise ValueError(f"Farmer simulations not marked OK:\n{bad}")

    numeric_candidate_columns = list(AGRONOMIC_METRICS) + [
        "total_irrigation_accounted_mm",
        "total_n_kg_ha",
        "total_irrigation_frequency_accounted",
        "nitrogen_application_frequency",
    ]
    numeric_farmer_columns = list(AGRONOMIC_METRICS) + [
        "total_irrigation_mm",
        "total_n_kg_ha",
        "irrigation_frequency",
        "nitrogen_application_frequency",
    ]
    if candidates[numeric_candidate_columns].isna().any().any():
        raise ValueError("Candidate_Year_Results contains missing numeric analysis inputs.")
    if farmers[numeric_farmer_columns].isna().any().any():
        raise ValueError("Farmer_Year_Results contains missing numeric analysis inputs.")

    parameter_counts = parameters.groupby("parameter")["level"].nunique()
    if not (parameter_counts == 3).all():
        raise ValueError(f"Every economic parameter must contain three levels:\n{parameter_counts}")

    return {
        "n_candidates": len(candidate_ids),
        "n_years": len(weather_years),
        "n_candidate_year_rows": len(candidates),
        "candidate_ids": candidate_ids,
        "weather_years": weather_years,
    }


def load_inputs(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path.resolve()}")
    sheets = pd.read_excel(input_path, sheet_name=REQUIRED_SHEETS)
    management = sheets["Candidate_Management"].copy()
    candidates = sheets["Candidate_Year_Results"].copy()
    farmers = sheets["Farmer_Year_Results"].copy()
    parameters = sheets["Economic_Parameters"].copy()

    management["schedule_id"] = management["schedule_id"].astype(str)
    candidates["schedule_id"] = candidates["schedule_id"].astype(str)
    candidates["weather_year"] = candidates["weather_year"].astype(int)
    farmers["weather_year"] = farmers["weather_year"].astype(int)

    metadata = validate_input(management, candidates, farmers, parameters)
    return management, candidates, farmers, parameters, metadata


def generate_scenarios(parameters: pd.DataFrame) -> pd.DataFrame:
    parameter_order = [
        "maize_price",
        "N_cost",
        "irrigation_electricity_cost",
        "irrigation_operation_cost",
        "fertigation_operation_cost",
    ]
    available = set(parameters["parameter"].astype(str))
    if set(parameter_order) != available:
        raise ValueError(
            f"Economic parameters must be exactly {parameter_order}; found {sorted(available)}"
        )

    level_lookup = {}
    for parameter in parameter_order:
        subset = parameters.loc[parameters["parameter"] == parameter, ["level", "value"]].copy()
        subset["level_order"] = subset["level"].map(LEVEL_ORDER)
        if subset["level_order"].isna().any():
            raise ValueError(f"Unexpected level name for {parameter}; use Low, Medium, High.")
        subset = subset.sort_values("level_order")
        level_lookup[parameter] = list(zip(subset["level"], subset["value"].astype(float)))

    rows = []
    all_combinations = itertools.product(*(level_lookup[p] for p in parameter_order))
    for scenario_id, combination in enumerate(all_combinations, start=1):
        row = {"scenario_id": scenario_id}
        for parameter, (level, value) in zip(parameter_order, combination):
            row[parameter] = value
            row[f"{parameter}_level"] = level
        rows.append(row)
    scenarios = pd.DataFrame(rows)
    if len(scenarios) != 243:
        raise AssertionError(f"Expected 243 scenarios, generated {len(scenarios)}")
    return scenarios


def cross_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return left.merge(right, how="cross")


def calculate_pnr_components(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    output = df.copy()
    output[f"{prefix}_revenue_yuan_ha"] = output["maize_price"] * output[f"{prefix}_yield_kg_ha"]
    output[f"{prefix}_n_cost_yuan_ha"] = output["N_cost"] * output[f"{prefix}_total_n_kg_ha"]
    output[f"{prefix}_irrigation_electricity_cost_yuan_ha"] = (
        output["irrigation_electricity_cost"] * output[f"{prefix}_total_irrigation_mm"]
    )
    output[f"{prefix}_irrigation_operation_cost_yuan_ha"] = (
        output["irrigation_operation_cost"] * output[f"{prefix}_irrigation_frequency"]
    )
    output[f"{prefix}_fertigation_operation_cost_yuan_ha"] = (
        output["fertigation_operation_cost"] * output[f"{prefix}_nitrogen_application_frequency"]
    )
    output[f"{prefix}_variable_cost_yuan_ha"] = (
        output[f"{prefix}_n_cost_yuan_ha"]
        + output[f"{prefix}_irrigation_electricity_cost_yuan_ha"]
        + output[f"{prefix}_irrigation_operation_cost_yuan_ha"]
        + output[f"{prefix}_fertigation_operation_cost_yuan_ha"]
    )
    output[f"{prefix}_pnr_yuan_ha"] = (
        output[f"{prefix}_revenue_yuan_ha"] - output[f"{prefix}_variable_cost_yuan_ha"]
    )
    return output


def build_weather_matched_results(
    candidates: pd.DataFrame,
    farmers: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> pd.DataFrame:
    candidate_base = candidates.rename(
        columns={
            "yield_kg_ha": "candidate_yield_kg_ha",
            "wpc_kg_m3": "candidate_wpc_kg_m3",
            "pen_kg_kg": "candidate_pen_kg_kg",
            "gnc_pct": "candidate_gnc_pct",
            "total_irrigation_accounted_mm": "candidate_total_irrigation_mm",
            "total_n_kg_ha": "candidate_total_n_kg_ha",
            "total_irrigation_frequency_accounted": "candidate_irrigation_frequency",
            "nitrogen_application_frequency": "candidate_nitrogen_application_frequency",
        }
    )[
        [
            "schedule_id",
            "candidate_type",
            "weather_year",
            "candidate_yield_kg_ha",
            "candidate_wpc_kg_m3",
            "candidate_pen_kg_kg",
            "candidate_gnc_pct",
            "candidate_total_irrigation_mm",
            "candidate_total_n_kg_ha",
            "candidate_irrigation_frequency",
            "candidate_nitrogen_application_frequency",
        ]
    ]
    farmer_base = farmers.rename(
        columns={
            "yield_kg_ha": "farmer_yield_kg_ha",
            "wpc_kg_m3": "farmer_wpc_kg_m3",
            "pen_kg_kg": "farmer_pen_kg_kg",
            "gnc_pct": "farmer_gnc_pct",
            "total_irrigation_mm": "farmer_total_irrigation_mm",
            "total_n_kg_ha": "farmer_total_n_kg_ha",
            "irrigation_frequency": "farmer_irrigation_frequency",
            "nitrogen_application_frequency": "farmer_nitrogen_application_frequency",
        }
    )[
        [
            "weather_year",
            "farmer_yield_kg_ha",
            "farmer_wpc_kg_m3",
            "farmer_pen_kg_kg",
            "farmer_gnc_pct",
            "farmer_total_irrigation_mm",
            "farmer_total_n_kg_ha",
            "farmer_irrigation_frequency",
            "farmer_nitrogen_application_frequency",
        ]
    ]

    paired = candidate_base.merge(farmer_base, on="weather_year", how="left", validate="many_to_one")
    results = cross_join(paired, scenarios)
    results = calculate_pnr_components(results, "candidate")
    results = calculate_pnr_components(results, "farmer")

    results["delta_yield_kg_ha"] = results["candidate_yield_kg_ha"] - results["farmer_yield_kg_ha"]
    results["delta_wpc_kg_m3"] = results["candidate_wpc_kg_m3"] - results["farmer_wpc_kg_m3"]
    results["delta_pen_kg_kg"] = results["candidate_pen_kg_kg"] - results["farmer_pen_kg_kg"]
    results["delta_gnc_pct_point"] = results["candidate_gnc_pct"] - results["farmer_gnc_pct"]
    results["irrigation_saving_mm"] = (
        results["farmer_total_irrigation_mm"] - results["candidate_total_irrigation_mm"]
    )
    results["irrigation_saving_pct"] = (
        100 * results["irrigation_saving_mm"] / results["farmer_total_irrigation_mm"]
    )
    results["n_saving_kg_ha"] = results["farmer_total_n_kg_ha"] - results["candidate_total_n_kg_ha"]
    results["n_saving_pct"] = 100 * results["n_saving_kg_ha"] / results["farmer_total_n_kg_ha"]
    results["irrigation_event_saving"] = (
        results["farmer_irrigation_frequency"] - results["candidate_irrigation_frequency"]
    )
    results["n_application_event_saving"] = (
        results["farmer_nitrogen_application_frequency"]
        - results["candidate_nitrogen_application_frequency"]
    )

    results["delta_revenue_yuan_ha"] = (
        results["candidate_revenue_yuan_ha"] - results["farmer_revenue_yuan_ha"]
    )
    results["n_cost_saving_yuan_ha"] = (
        results["farmer_n_cost_yuan_ha"] - results["candidate_n_cost_yuan_ha"]
    )
    results["irrigation_electricity_cost_saving_yuan_ha"] = (
        results["farmer_irrigation_electricity_cost_yuan_ha"]
        - results["candidate_irrigation_electricity_cost_yuan_ha"]
    )
    results["irrigation_operation_cost_saving_yuan_ha"] = (
        results["farmer_irrigation_operation_cost_yuan_ha"]
        - results["candidate_irrigation_operation_cost_yuan_ha"]
    )
    results["fertigation_operation_cost_saving_yuan_ha"] = (
        results["farmer_fertigation_operation_cost_yuan_ha"]
        - results["candidate_fertigation_operation_cost_yuan_ha"]
    )
    results["variable_cost_saving_yuan_ha"] = (
        results["farmer_variable_cost_yuan_ha"] - results["candidate_variable_cost_yuan_ha"]
    )
    results["delta_pnr_yuan_ha"] = (
        results["candidate_pnr_yuan_ha"] - results["farmer_pnr_yuan_ha"]
    )
    results["delta_pnr_check_yuan_ha"] = (
        results["delta_revenue_yuan_ha"] + results["variable_cost_saving_yuan_ha"]
    )
    if not np.allclose(
        results["delta_pnr_yuan_ha"], results["delta_pnr_check_yuan_ha"], atol=1e-8
    ):
        raise AssertionError("Delta PNR component reconciliation failed.")

    expected_rows = len(candidates) * len(scenarios)
    if len(results) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} weather-matched rows; got {len(results)}")
    return results


def summarize_delta(group: pd.DataFrame) -> pd.Series:
    minimum_index = group["delta_pnr_yuan_ha"].idxmin()
    maximum_index = group["delta_pnr_yuan_ha"].idxmax()
    minimum_row = group.loc[minimum_index]
    maximum_row = group.loc[maximum_index]
    return pd.Series(
        {
            "n_weather_price_cost_combinations": len(group),
            "mean_delta_pnr_yuan_ha": group["delta_pnr_yuan_ha"].mean(),
            "median_delta_pnr_yuan_ha": group["delta_pnr_yuan_ha"].median(),
            "min_delta_pnr_yuan_ha": minimum_row["delta_pnr_yuan_ha"],
            "max_delta_pnr_yuan_ha": maximum_row["delta_pnr_yuan_ha"],
            "positive_ratio": (group["delta_pnr_yuan_ha"] > 0).mean(),
            "positive_ratio_pct": 100 * (group["delta_pnr_yuan_ha"] > 0).mean(),
            "all_combinations_positive": bool((group["delta_pnr_yuan_ha"] > 0).all()),
            "worst_weather_year": (
                int(minimum_row["weather_year"]) if "weather_year" in minimum_row.index else np.nan
            ),
            "worst_scenario_id": int(minimum_row["scenario_id"]),
            "best_weather_year": (
                int(maximum_row["weather_year"]) if "weather_year" in maximum_row.index else np.nan
            ),
            "best_scenario_id": int(maximum_row["scenario_id"]),
        }
    )


def build_economic_summaries(results: pd.DataFrame):
    schedule_summary = (
        results.groupby(["schedule_id", "candidate_type"], sort=False)
        .apply(summarize_delta, include_groups=False)
        .reset_index()
    )
    schedule_year_summary = (
        results.groupby(["schedule_id", "candidate_type", "weather_year"], sort=False)
        .apply(summarize_delta, include_groups=False)
        .reset_index()
    )

    group_summary_rows = []
    for label, subset in [
        ("All candidates", results),
        ("COMP", results.loc[results["schedule_id"] == "COMP"]),
        ("HY candidates", results.loc[results["candidate_type"] == "HY"]),
    ]:
        row = summarize_delta(subset).to_dict()
        row["group"] = label
        row["n_schedules"] = subset["schedule_id"].nunique()
        group_summary_rows.append(row)
    group_summary = pd.DataFrame(group_summary_rows)

    group_year_rows = []
    for label, subset in [
        ("COMP", results.loc[results["schedule_id"] == "COMP"]),
        ("HY01-HY16", results.loc[results["candidate_type"] == "HY"]),
    ]:
        for weather_year, year_subset in subset.groupby("weather_year", sort=True):
            row = summarize_delta(year_subset).to_dict()
            row["comparison_group"] = label
            row["weather_year"] = int(weather_year)
            row["n_schedules"] = year_subset["schedule_id"].nunique()
            group_year_rows.append(row)
    group_year_summary = pd.DataFrame(group_year_rows)

    medium_mask = (
        results["maize_price_level"].eq("Medium")
        & results["N_cost_level"].eq("Medium")
        & results["irrigation_electricity_cost_level"].eq("Medium")
        & results["irrigation_operation_cost_level"].eq("Medium")
        & results["fertigation_operation_cost_level"].eq("Medium")
    )
    baseline = results.loc[medium_mask].copy()
    if len(baseline) != 85:
        raise AssertionError(f"Expected 85 baseline rows; found {len(baseline)}")
    baseline_summary = (
        baseline.groupby(["schedule_id", "candidate_type"], sort=False)
        .agg(
            n_weather_years=("weather_year", "count"),
            mean_delta_pnr_yuan_ha=("delta_pnr_yuan_ha", "mean"),
            median_delta_pnr_yuan_ha=("delta_pnr_yuan_ha", "median"),
            min_delta_pnr_yuan_ha=("delta_pnr_yuan_ha", "min"),
            max_delta_pnr_yuan_ha=("delta_pnr_yuan_ha", "max"),
            positive_years=("delta_pnr_yuan_ha", lambda x: int((x > 0).sum())),
            positive_year_ratio_pct=("delta_pnr_yuan_ha", lambda x: 100 * (x > 0).mean()),
            mean_delta_yield_kg_ha=("delta_yield_kg_ha", "mean"),
            mean_irrigation_saving_mm=("irrigation_saving_mm", "mean"),
            mean_irrigation_saving_pct=("irrigation_saving_pct", "mean"),
            mean_n_saving_kg_ha=("n_saving_kg_ha", "mean"),
            mean_n_saving_pct=("n_saving_pct", "mean"),
            mean_variable_cost_saving_yuan_ha=("variable_cost_saving_yuan_ha", "mean"),
        )
        .reset_index()
    )
    return (
        schedule_summary,
        schedule_year_summary,
        group_summary,
        group_year_summary,
        baseline,
        baseline_summary,
    )


def build_agronomic_summary(
    management: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    mean_weather_map = {
        "yield_kg_ha": "yield_mean_weather_kg_ha",
        "wpc_kg_m3": "wpc_mean_weather_kg_m3",
        "pen_kg_kg": "pen_mean_weather_kg_kg",
        "gnc_pct": "gnc_mean_weather_pct",
    }
    rows = []
    management_lookup = management.set_index("schedule_id")
    for (schedule_id, group) in candidates.groupby("schedule_id", sort=False):
        row = {
            "schedule_id": schedule_id,
            "candidate_type": group["candidate_type"].iloc[0],
            "n_weather_years": len(group),
        }
        for metric in AGRONOMIC_METRICS:
            values = group[metric].astype(float)
            annual_mean = values.mean()
            annual_sd = values.std(ddof=1)
            mean_weather_value = float(management_lookup.loc[schedule_id, mean_weather_map[metric]])
            row[f"{metric}_annual_mean"] = annual_mean
            row[f"{metric}_annual_sd"] = annual_sd
            row[f"{metric}_cv_pct"] = 100 * annual_sd / annual_mean if annual_mean != 0 else np.nan
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
            row[f"{metric}_range"] = values.max() - values.min()
            row[f"{metric}_mean_weather"] = mean_weather_value
            row[f"{metric}_annual_mean_minus_mean_weather"] = annual_mean - mean_weather_value
        rows.append(row)
    return pd.DataFrame(rows)


def build_management_summary(results: pd.DataFrame) -> pd.DataFrame:
    base = results.drop_duplicates(["schedule_id", "weather_year"])
    return (
        base.groupby(["schedule_id", "candidate_type"], sort=False)
        .agg(
            mean_candidate_irrigation_mm=("candidate_total_irrigation_mm", "mean"),
            min_candidate_irrigation_mm=("candidate_total_irrigation_mm", "min"),
            max_candidate_irrigation_mm=("candidate_total_irrigation_mm", "max"),
            mean_irrigation_saving_mm=("irrigation_saving_mm", "mean"),
            mean_irrigation_saving_pct=("irrigation_saving_pct", "mean"),
            min_irrigation_saving_pct=("irrigation_saving_pct", "min"),
            max_irrigation_saving_pct=("irrigation_saving_pct", "max"),
            mean_n_saving_kg_ha=("n_saving_kg_ha", "mean"),
            mean_n_saving_pct=("n_saving_pct", "mean"),
            mean_irrigation_event_saving=("irrigation_event_saving", "mean"),
            mean_n_application_event_saving=("n_application_event_saving", "mean"),
        )
        .reset_index()
    )


def build_factor_summaries(results: pd.DataFrame):
    factors = [
        "maize_price",
        "N_cost",
        "irrigation_electricity_cost",
        "irrigation_operation_cost",
        "fertigation_operation_cost",
    ]
    level_rows = []
    for schedule_id, schedule_group in results.groupby("schedule_id", sort=False):
        candidate_type = schedule_group["candidate_type"].iloc[0]
        for factor in factors:
            for level, level_group in schedule_group.groupby(f"{factor}_level", sort=False):
                level_rows.append(
                    {
                        "schedule_id": schedule_id,
                        "candidate_type": candidate_type,
                        "factor": factor,
                        "level": level,
                        "mean_delta_pnr_yuan_ha": level_group["delta_pnr_yuan_ha"].mean(),
                        "min_delta_pnr_yuan_ha": level_group["delta_pnr_yuan_ha"].min(),
                        "max_delta_pnr_yuan_ha": level_group["delta_pnr_yuan_ha"].max(),
                        "positive_ratio_pct": 100 * (level_group["delta_pnr_yuan_ha"] > 0).mean(),
                    }
                )
    factor_level = pd.DataFrame(level_rows)

    pivot = factor_level.pivot_table(
        index=["schedule_id", "candidate_type", "factor"],
        columns="level",
        values="mean_delta_pnr_yuan_ha",
    ).reset_index()
    for level in ["Low", "Medium", "High"]:
        if level not in pivot.columns:
            pivot[level] = np.nan
    pivot["high_minus_low_yuan_ha"] = pivot["High"] - pivot["Low"]
    pivot["absolute_level_span_yuan_ha"] = pivot[["Low", "Medium", "High"]].max(axis=1) - pivot[
        ["Low", "Medium", "High"]
    ].min(axis=1)
    factor_effect = pivot.sort_values(
        ["schedule_id", "absolute_level_span_yuan_ha"], ascending=[True, False]
    )
    return factor_level, factor_effect


def build_break_even(
    candidates: pd.DataFrame,
    farmers: pd.DataFrame,
    parameters: pd.DataFrame,
):
    cost_parameters = [
        "N_cost",
        "irrigation_electricity_cost",
        "irrigation_operation_cost",
        "fertigation_operation_cost",
    ]
    level_lookup = {}
    for parameter in cost_parameters:
        subset = parameters.loc[parameters["parameter"] == parameter, ["level", "value"]].copy()
        subset["level_order"] = subset["level"].map(LEVEL_ORDER)
        subset = subset.sort_values("level_order")
        level_lookup[parameter] = list(zip(subset["level"], subset["value"].astype(float)))

    cost_rows = []
    combinations = itertools.product(*(level_lookup[p] for p in cost_parameters))
    for cost_scenario_id, combination in enumerate(combinations, start=1):
        row = {"cost_scenario_id": cost_scenario_id}
        for parameter, (level, value) in zip(cost_parameters, combination):
            row[parameter] = value
            row[f"{parameter}_level"] = level
        cost_rows.append(row)
    cost_scenarios = pd.DataFrame(cost_rows)
    if len(cost_scenarios) != 81:
        raise AssertionError(f"Expected 81 cost scenarios; generated {len(cost_scenarios)}")

    candidate_base = candidates[
        [
            "schedule_id",
            "candidate_type",
            "weather_year",
            "yield_kg_ha",
            "total_irrigation_accounted_mm",
            "total_n_kg_ha",
            "total_irrigation_frequency_accounted",
            "nitrogen_application_frequency",
        ]
    ].rename(
        columns={
            "yield_kg_ha": "candidate_yield_kg_ha",
            "total_irrigation_accounted_mm": "candidate_total_irrigation_mm",
            "total_n_kg_ha": "candidate_total_n_kg_ha",
            "total_irrigation_frequency_accounted": "candidate_irrigation_frequency",
            "nitrogen_application_frequency": "candidate_nitrogen_application_frequency",
        }
    )
    farmer_base = farmers[
        [
            "weather_year",
            "yield_kg_ha",
            "total_irrigation_mm",
            "total_n_kg_ha",
            "irrigation_frequency",
            "nitrogen_application_frequency",
        ]
    ].rename(
        columns={
            "yield_kg_ha": "farmer_yield_kg_ha",
            "total_irrigation_mm": "farmer_total_irrigation_mm",
            "total_n_kg_ha": "farmer_total_n_kg_ha",
            "irrigation_frequency": "farmer_irrigation_frequency",
            "nitrogen_application_frequency": "farmer_nitrogen_application_frequency",
        }
    )
    base = candidate_base.merge(farmer_base, on="weather_year", how="left", validate="many_to_one")
    break_even = cross_join(base, cost_scenarios)
    break_even["delta_yield_kg_ha"] = (
        break_even["candidate_yield_kg_ha"] - break_even["farmer_yield_kg_ha"]
    )
    break_even["delta_variable_cost_yuan_ha"] = (
        break_even["N_cost"]
        * (break_even["candidate_total_n_kg_ha"] - break_even["farmer_total_n_kg_ha"])
        + break_even["irrigation_electricity_cost"]
        * (break_even["candidate_total_irrigation_mm"] - break_even["farmer_total_irrigation_mm"])
        + break_even["irrigation_operation_cost"]
        * (break_even["candidate_irrigation_frequency"] - break_even["farmer_irrigation_frequency"])
        + break_even["fertigation_operation_cost"]
        * (
            break_even["candidate_nitrogen_application_frequency"]
            - break_even["farmer_nitrogen_application_frequency"]
        )
    )
    break_even["break_even_maize_price_yuan_kg"] = np.where(
        break_even["delta_yield_kg_ha"].abs() > 1e-12,
        break_even["delta_variable_cost_yuan_ha"] / break_even["delta_yield_kg_ha"],
        np.nan,
    )

    conditions = [
        (break_even["delta_yield_kg_ha"] > 0) & (break_even["delta_variable_cost_yuan_ha"] <= 0),
        (break_even["delta_yield_kg_ha"] < 0) & (break_even["delta_variable_cost_yuan_ha"] >= 0),
        (break_even["delta_yield_kg_ha"] > 0) & (break_even["delta_variable_cost_yuan_ha"] > 0),
        (break_even["delta_yield_kg_ha"] < 0) & (break_even["delta_variable_cost_yuan_ha"] < 0),
    ]
    choices = [
        "Higher yield and no higher cost: favorable at any positive maize price",
        "Lower yield and no lower cost: not favorable at any positive maize price",
        "Higher yield and higher cost: favorable when maize price exceeds break-even",
        "Lower yield and lower cost: favorable when maize price is below break-even",
    ]
    break_even["interpretation"] = np.select(conditions, choices, default="No yield difference")

    medium_mask = np.logical_and.reduce(
        [break_even[f"{parameter}_level"].eq("Medium") for parameter in cost_parameters]
    )
    baseline = break_even.loc[medium_mask].copy()
    baseline = baseline.rename(
        columns={
            "break_even_maize_price_yuan_kg": "baseline_break_even_maize_price_yuan_kg",
            "delta_variable_cost_yuan_ha": "baseline_delta_variable_cost_yuan_ha",
            "interpretation": "baseline_interpretation",
        }
    )[
        [
            "schedule_id",
            "candidate_type",
            "weather_year",
            "delta_yield_kg_ha",
            "baseline_delta_variable_cost_yuan_ha",
            "baseline_break_even_maize_price_yuan_kg",
            "baseline_interpretation",
        ]
    ]
    ranges = (
        break_even.groupby(["schedule_id", "candidate_type", "weather_year"], sort=False)
        .agg(
            n_cost_scenarios=("cost_scenario_id", "count"),
            min_break_even_maize_price_yuan_kg=("break_even_maize_price_yuan_kg", "min"),
            mean_break_even_maize_price_yuan_kg=("break_even_maize_price_yuan_kg", "mean"),
            max_break_even_maize_price_yuan_kg=("break_even_maize_price_yuan_kg", "max"),
        )
        .reset_index()
    )
    summary = baseline.merge(
        ranges, on=["schedule_id", "candidate_type", "weather_year"], how="left", validate="one_to_one"
    )
    return cost_scenarios, break_even, summary


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def dataframe_payload(df: pd.DataFrame) -> dict:
    clean = df.astype(object).where(pd.notna(df), None)
    return {"headers": list(clean.columns), "rows": clean.values.tolist()}


def set_plot_style() -> None:
    """Use a compact Times-compatible journal style for all publication figures."""
    preferred_fonts = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    font_name = next((name for name in preferred_fonts if name in installed), "DejaVu Serif")
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "figure.dpi": 200,
            "savefig.dpi": 600,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.025,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        clip_on=False,
    )


def save_publication_figure(fig: plt.Figure, stem: Path) -> None:
    """Write editable PDF plus 600-dpi PNG and LZW-compressed TIFF."""
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    # Write the vector file last so later raster backends cannot leave it partially open.
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def plot_descriptive_figure10(
    management: pd.DataFrame,
    farmers: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Rebuild Fig. 10 using establishment-adjusted irrigation accounting."""
    set_plot_style()
    hy = management.loc[management["candidate_type"].eq("HY")].copy()
    mean_establishment = float(farmers["establishment_irrigation_mm"].mean())

    hy["accounted_irrigation_mm"] = (
        hy["optimized_irrigation_excl_establishment_mm"] + mean_establishment
    )
    hy["accounted_irrigation_frequency"] = (
        hy["optimized_irrigation_frequency_excl_establishment"] + 1
    )

    panels = [
        ("yield_mean_weather_kg_ha", "yield_kg_ha", "Yield (kg/ha)", "#5B8E7D"),
        ("wpc_mean_weather_kg_m3", "wpc_kg_m3", "WPc (kg/m$^3$)", "#4F81A8"),
        ("pen_mean_weather_kg_kg", "pen_kg_kg", "PEN (kg/kg)", "#D09A43"),
        ("gnc_mean_weather_pct", "gnc_pct", "GNC (%)", "#B85C5C"),
        ("accounted_irrigation_mm", "total_irrigation_mm", "Accounted irrigation (mm)", "#738B9A"),
        ("total_n_kg_ha", "total_n_kg_ha", "Total nitrogen (kg N/ha)", "#8DA85C"),
        (
            "accounted_irrigation_frequency",
            "irrigation_frequency",
            "Accounted irrigation frequency",
            "#A77DB3",
        ),
        (
            "nitrogen_application_frequency",
            "nitrogen_application_frequency",
            "Nitrogen application frequency",
            "#C69C6D",
        ),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(7.1, 4.55))
    for index, (candidate_col, farmer_col, ylabel, color) in enumerate(panels):
        ax = axes.ravel()[index]
        candidate_values = hy[candidate_col].astype(float).to_numpy()
        farmer_values = farmers[farmer_col].astype(float).to_numpy()
        box = ax.boxplot(
            candidate_values,
            positions=[1.0],
            widths=0.50,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#555555", "linewidth": 0.8},
            boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.8, "alpha": 0.82},
            whiskerprops={"color": color, "linewidth": 0.8},
            capprops={"color": color, "linewidth": 0.8},
            flierprops={
                "marker": "o",
                "markerfacecolor": color,
                "markeredgecolor": color,
                "markersize": 2.2,
                "alpha": 0.50,
            },
        )
        _ = box
        offsets = np.linspace(-0.13, 0.13, len(farmer_values))
        ax.scatter(
            1.0 + offsets,
            farmer_values,
            marker="^",
            s=18,
            facecolors="none",
            edgecolors="black",
            linewidths=0.7,
            zorder=4,
            label="Farmer-managed",
        )
        ax.set_xlim(0.55, 1.45)
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", color="#D8D8D8", linestyle=(0, (2, 2)), linewidth=0.45)
        panel_label(ax, f"({chr(97 + index)})")
        if index in (0, 4):
            ax.legend(frameon=False, loc="lower left", handletextpad=0.3, borderaxespad=0.2)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.08, top=0.96, wspace=0.62, hspace=0.34)
    save_publication_figure(fig, output_dir / "Fig_10_establishment_adjusted_comparison")
    plt.close(fig)


def plot_agronomic_heatmaps(candidates: pd.DataFrame, schedule_order: list[str], output_dir: Path) -> None:
    set_plot_style()
    years = sorted(candidates["weather_year"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 7.7))
    cmaps = ["YlGn", "YlGnBu", "PuBuGn", "YlOrBr"]
    for panel_index, ((metric, (label, unit)), ax, cmap) in enumerate(
        zip(AGRONOMIC_METRICS.items(), axes.ravel(), cmaps)
    ):
        matrix = (
            candidates.pivot(index="schedule_id", columns="weather_year", values=metric)
            .reindex(index=schedule_order, columns=years)
        )
        image = ax.imshow(matrix.values, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xticks(range(len(years)), [str(year) for year in years])
        ax.set_yticks(range(len(schedule_order)), schedule_order)
        ax.set_xlabel("Weather year")
        ax.set_ylabel("Candidate schedule")
        panel_label(ax, f"({chr(97 + panel_index)}) {label} ({unit})")
        ax.axhline(0.5, color="#333333", linewidth=0.8)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.037, pad=0.025)
        colorbar.ax.tick_params(labelsize=6.5, width=0.6, length=2)
        colorbar.outline.set_linewidth(0.6)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.07, top=0.97, wspace=0.44, hspace=0.24)
    save_publication_figure(fig, output_dir / "Fig_11_weather_year_agronomic_performance")
    plt.close(fig)


def plot_economic_results(
    results: pd.DataFrame,
    schedule_summary: pd.DataFrame,
    schedule_year_summary: pd.DataFrame,
    break_even_summary: pd.DataFrame,
    parameters: pd.DataFrame,
    schedule_order: list[str],
    output_dir: Path,
) -> None:
    """Rebuild the complete four-panel main-text Fig. 12."""
    set_plot_style()
    color_comp = "#D9903D"
    color_comp_dark = "#9C5B17"
    color_hy = "#5E9F69"
    color_hy_dark = "#2E7D3E"
    color_gray = "#666666"

    comp_values = results.loc[results["schedule_id"].eq("COMP"), "delta_pnr_yuan_ha"].to_numpy()
    hy_values = results.loc[results["candidate_type"].eq("HY"), "delta_pnr_yuan_ha"].to_numpy()
    all_schedule_summary = (
        schedule_summary
        .sort_values("mean_delta_pnr_yuan_ha", ascending=True)
        .reset_index(drop=True)
    )
    comp_be = (
        break_even_summary.loc[break_even_summary["schedule_id"].eq("COMP")]
        .sort_values("weather_year", ascending=False)
        .reset_index(drop=True)
    )
    hy10_be = (
        break_even_summary.loc[break_even_summary["schedule_id"].eq("HY10")]
        .sort_values("weather_year", ascending=False)
        .reset_index(drop=True)
    )

    group_year_rows = []
    for label, subset in [
        ("COMP", results.loc[results["schedule_id"].eq("COMP")]),
        ("HY01-HY16", results.loc[results["candidate_type"].eq("HY")]),
    ]:
        for weather_year, year_subset in subset.groupby("weather_year", sort=True):
            group_year_rows.append(
                {
                    "comparison_group": label,
                    "weather_year": int(weather_year),
                    "mean": year_subset["delta_pnr_yuan_ha"].mean(),
                    "minimum": year_subset["delta_pnr_yuan_ha"].min(),
                    "maximum": year_subset["delta_pnr_yuan_ha"].max(),
                }
            )
    group_year = pd.DataFrame(group_year_rows)

    fig = plt.figure(figsize=(7.1, 5.55))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.03, 1.15], height_ratios=[1.08, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # (a) Overall distributions across all weather-price-cost combinations.
    box = ax_a.boxplot(
        [comp_values, hy_values],
        positions=[1, 2],
        widths=0.56,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        boxprops={"edgecolor": "black", "linewidth": 0.8},
    )
    for patch, face in zip(box["boxes"], [color_comp, color_hy]):
        patch.set_facecolor(face)
        patch.set_alpha(0.78)
    ax_a.scatter([1, 2], [comp_values.mean(), hy_values.mean()], marker="D", s=18,
                 facecolor="white", edgecolor="black", linewidth=0.7, zorder=4)
    ax_a.axhline(0, color=color_gray, linestyle=(0, (4, 3)), linewidth=0.7)
    ax_a.axhspan(0, max(hy_values.max(), 1), color="#EEF7EF", alpha=0.6, zorder=0)
    ax_a.axhspan(min(comp_values.min(), -1), 0, color="#FAEEEE", alpha=0.6, zorder=0)
    ax_a.set_xticks([1, 2], ["COMP", "HY01-HY16"])
    ax_a.set_ylabel("$\\Delta$PNR (yuan/ha)")
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax_a.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_a.grid(axis="y", color="#D5D5D5", linestyle=(0, (2, 2)), linewidth=0.45)
    panel_label(ax_a, "(a)")

    # (b) Schedule-specific means and full ranges for all 17 candidates.
    y = np.arange(len(all_schedule_summary))
    for candidate_type, color, error_color in [
        ("HY", color_hy_dark, color_hy),
        ("Compromise", color_comp_dark, color_comp),
    ]:
        subset = all_schedule_summary.loc[all_schedule_summary["candidate_type"].eq(candidate_type)]
        positions = subset.index.to_numpy()
        centers = subset["mean_delta_pnr_yuan_ha"].to_numpy()
        lower = centers - subset["min_delta_pnr_yuan_ha"].to_numpy()
        upper = subset["max_delta_pnr_yuan_ha"].to_numpy() - centers
        ax_b.errorbar(
            centers,
            positions,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color=color,
            ecolor=error_color,
            capsize=2.0,
            markersize=3.0,
            linewidth=0.8,
        )
    ax_b.axvline(0, color=color_gray, linestyle=(0, (4, 3)), linewidth=0.7)
    ax_b.axvspan(0, 1800, color="#EEF7EF", alpha=0.65, zorder=0)
    ax_b.axvspan(-1200, 0, color="#FAEEEE", alpha=0.65, zorder=0)
    ax_b.set_yticks(y, all_schedule_summary["schedule_id"])
    ax_b.set_xlim(-1150, 1800)
    ax_b.set_xlabel("$\\Delta$PNR (yuan/ha)")
    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_b.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax_b.grid(axis="x", color="#D5D5D5", linestyle=(0, (2, 2)), linewidth=0.45)
    panel_label(ax_b, "(b)")

    # (c) Weather-year performance of both comparison groups.
    years = np.array(sorted(group_year["weather_year"].unique()))
    for label, offset, color, error_color, marker in [
        ("COMP", -0.08, color_comp_dark, color_comp, "o"),
        ("HY01-HY16", 0.08, color_hy_dark, color_hy, "s"),
    ]:
        subset = group_year.loc[group_year["comparison_group"].eq(label)].sort_values("weather_year")
        centers = subset["mean"].to_numpy()
        lower = centers - subset["minimum"].to_numpy()
        upper = subset["maximum"].to_numpy() - centers
        ax_c.errorbar(
            years + offset,
            centers,
            yerr=np.vstack([lower, upper]),
            fmt=marker,
            color=color,
            ecolor=error_color,
            capsize=2.5,
            markersize=3.5,
            linewidth=0.85,
            label=label,
        )
    ax_c.axhline(0, color=color_gray, linestyle=(0, (4, 3)), linewidth=0.7)
    ax_c.axhspan(0, 600, color="#EEF7EF", alpha=0.6, zorder=0)
    ax_c.axhspan(-1200, 0, color="#FAEEEE", alpha=0.6, zorder=0)
    ax_c.set_xticks(years)
    ax_c.set_ylim(-1150, 1800)
    ax_c.set_xlabel("Weather year")
    ax_c.set_ylabel("$\\Delta$PNR (yuan/ha)")
    ax_c.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax_c.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_c.grid(axis="y", color="#D5D5D5", linestyle=(0, (2, 2)), linewidth=0.45)
    ax_c.legend(frameon=False, loc="upper right", handletextpad=0.4)
    panel_label(ax_c, "(c)")

    # (d) Positive break-even thresholds. Only COMP and HY10 have meaningful
    # positive thresholds; the other 15 HY schedules dominate Farmer at any
    # positive maize price because they have higher yield and no higher cost.
    y = np.arange(len(comp_be))
    for label, subset, offset, color, error_color, marker in [
        ("COMP", comp_be, -0.10, color_comp_dark, color_comp, "o"),
        ("HY10", hy10_be, 0.10, color_hy_dark, color_hy, "s"),
    ]:
        centers = subset["baseline_break_even_maize_price_yuan_kg"].to_numpy()
        lower = centers - subset["min_break_even_maize_price_yuan_kg"].to_numpy()
        upper = subset["max_break_even_maize_price_yuan_kg"].to_numpy() - centers
        ax_d.errorbar(
            centers,
            y + offset,
            xerr=np.vstack([lower, upper]),
            fmt=marker,
            color=color,
            ecolor=error_color,
            capsize=2.2,
            markersize=3.4,
            linewidth=0.85,
            label=label,
        )
    medium_price = float(
        parameters.loc[
            parameters["parameter"].eq("maize_price") & parameters["level"].eq("Medium"), "value"
        ].iloc[0]
    )
    ax_d.axvline(medium_price, color=color_gray, linestyle=(0, (4, 3)), linewidth=0.7)
    ax_d.text(medium_price + 0.08, len(y) - 0.45, "Baseline price", ha="left", va="center",
              fontsize=6.4, color=color_gray)
    ax_d.set_yticks(y, comp_be["weather_year"].astype(int))
    ax_d.set_xlim(0, 7.0)
    ax_d.set_xlabel("Break-even maize price (yuan/kg)")
    ax_d.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_d.grid(axis="x", color="#D5D5D5", linestyle=(0, (2, 2)), linewidth=0.45)
    ax_d.legend(frameon=False, loc="upper right", handletextpad=0.4)
    panel_label(ax_d, "(d)")

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.09, top=0.97, wspace=0.43, hspace=0.36)
    save_publication_figure(fig, output_dir / "Fig_12_weather_matched_economic_analysis")
    plt.close(fig)


def plot_compromise_break_even(
    break_even_summary: pd.DataFrame,
    parameters: pd.DataFrame,
    output_dir: Path,
) -> None:
    comp = break_even_summary.loc[break_even_summary["schedule_id"] == "COMP"].sort_values("weather_year")
    prices = (
        parameters.loc[parameters["parameter"] == "maize_price", ["level", "value"]]
        .assign(level_order=lambda x: x["level"].map(LEVEL_ORDER))
        .sort_values("level_order")
    )
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    x = np.arange(len(comp))
    center = comp["baseline_break_even_maize_price_yuan_kg"].to_numpy()
    lower = center - comp["min_break_even_maize_price_yuan_kg"].to_numpy()
    upper = comp["max_break_even_maize_price_yuan_kg"].to_numpy() - center
    ax.errorbar(
        x,
        center,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        color="#264653",
        ecolor="#7A7A7A",
        capsize=4,
        linewidth=1.2,
        markersize=5,
        label="Break-even price (medium costs; range across 81 cost scenarios)",
    )
    price_colors = {"Low": "#457B9D", "Medium": "#E76F51", "High": "#9B2226"}
    for _, row in prices.iterrows():
        ax.axhline(
            row["value"],
            linestyle="--",
            linewidth=1.0,
            color=price_colors[row["level"]],
            label=f"{row['level']} maize price ({row['value']:.2f} yuan/kg)",
        )
    ax.set_xticks(x, comp["weather_year"].astype(str))
    ax.set_xlabel("Weather year")
    ax.set_ylabel("Maize price (yuan/kg)")
    ax.set_title("Break-even maize price of the compromise schedule", fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    fig.savefig(output_dir / "Fig_break_even_compromise.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "Fig_break_even_compromise.png",
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)


def write_summary_markdown(
    output_path: Path,
    metadata: dict,
    schedule_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    management_summary: pd.DataFrame,
    break_even_summary: pd.DataFrame,
) -> None:
    comp_schedule = schedule_summary.loc[schedule_summary["schedule_id"] == "COMP"].iloc[0]
    hy_schedule = schedule_summary.loc[schedule_summary["candidate_type"] == "HY"].copy()
    comp_baseline = baseline_summary.loc[baseline_summary["schedule_id"] == "COMP"].iloc[0]
    hy_management = management_summary.loc[management_summary["candidate_type"] == "HY"].copy()
    comp_be = break_even_summary.loc[break_even_summary["schedule_id"] == "COMP"].copy()

    lines = [
        "# Weather-matched economic analysis summary",
        "",
        "## Analysis scope",
        "",
        f"- {metadata['n_candidates']} candidate schedules × {metadata['n_years']} weather years × 243 price-cost scenarios.",
        "- Every candidate is paired with Farmer management under the same weather year.",
        "- Candidate irrigation totals include the year-specific establishment irrigation depth and one establishment event for cost accounting.",
        "- No GNC-based price premium is included.",
        "",
        "## Key results",
        "",
        f"- COMP: mean ΔPNR = {comp_schedule['mean_delta_pnr_yuan_ha']:.1f} yuan/ha; positive ratio = {comp_schedule['positive_ratio_pct']:.1f}% across 1,215 matched combinations.",
        f"- COMP under the all-medium scenario: mean ΔPNR = {comp_baseline['mean_delta_pnr_yuan_ha']:.1f} yuan/ha; positive in {int(comp_baseline['positive_years'])}/5 weather years.",
        f"- HY schedules positive in all 1,215 combinations: {int(hy_schedule['all_combinations_positive'].sum())}/{len(hy_schedule)}.",
        f"- HY positive-ratio range: {hy_schedule['positive_ratio_pct'].min():.1f}%–{hy_schedule['positive_ratio_pct'].max():.1f}%.",
        f"- HY mean irrigation saving range: {hy_management['mean_irrigation_saving_pct'].min():.1f}%–{hy_management['mean_irrigation_saving_pct'].max():.1f}% (negative means greater irrigation than Farmer).",
        "",
        "## COMP break-even price",
        "",
    ]
    for _, row in comp_be.sort_values("weather_year").iterrows():
        lines.append(
            f"- {int(row['weather_year'])}: {row['baseline_break_even_maize_price_yuan_kg']:.2f} yuan/kg under medium costs "
            f"(range {row['min_break_even_maize_price_yuan_kg']:.2f}–{row['max_break_even_maize_price_yuan_kg']:.2f} across 81 cost scenarios)."
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    management, candidates, farmers, parameters, metadata = load_inputs(args.input)
    scenarios = generate_scenarios(parameters)
    results = build_weather_matched_results(candidates, farmers, scenarios)
    (
        schedule_summary,
        schedule_year_summary,
        group_summary,
        group_year_summary,
        baseline,
        baseline_summary,
    ) = build_economic_summaries(results)
    agronomic_summary = build_agronomic_summary(management, candidates)
    management_summary = build_management_summary(results)
    factor_level, factor_effect = build_factor_summaries(results)
    cost_scenarios, break_even, break_even_summary = build_break_even(candidates, farmers, parameters)

    output_tables = {
        "economic_scenarios_243.csv": scenarios,
        "weather_matched_all_results.csv": results,
        "economic_summary_by_schedule.csv": schedule_summary,
        "economic_summary_by_schedule_year.csv": schedule_year_summary,
        "economic_summary_by_group.csv": group_summary,
        "economic_summary_by_group_year.csv": group_year_summary,
        "baseline_medium_weather_matched.csv": baseline,
        "baseline_medium_summary_by_schedule.csv": baseline_summary,
        "agronomic_interannual_summary.csv": agronomic_summary,
        "management_weather_matched_summary.csv": management_summary,
        "economic_factor_level_summary.csv": factor_level,
        "economic_factor_effect_summary.csv": factor_effect,
        "break_even_cost_scenarios_81.csv": cost_scenarios,
        "break_even_all_candidates.csv": break_even,
        "break_even_summary_by_schedule_year.csv": break_even_summary,
        "economic_parameters_used.csv": parameters,
    }
    for filename, dataframe in output_tables.items():
        save_csv(dataframe, output_dir / filename)

    workbook_tables = {
        "schedule_summary": dataframe_payload(schedule_summary),
        "schedule_year_summary": dataframe_payload(schedule_year_summary),
        "group_summary": dataframe_payload(group_summary),
        "group_year_summary": dataframe_payload(group_year_summary),
        "baseline": dataframe_payload(baseline),
        "baseline_summary": dataframe_payload(baseline_summary),
        "agronomic_summary": dataframe_payload(agronomic_summary),
        "management_summary": dataframe_payload(management_summary),
        "factor_effect": dataframe_payload(factor_effect),
        "break_even_summary": dataframe_payload(break_even_summary),
        "parameters": dataframe_payload(parameters),
    }
    (output_dir / "workbook_tables.json").write_text(
        json.dumps(workbook_tables, ensure_ascii=False), encoding="utf-8"
    )

    schedule_order = management["schedule_id"].tolist()
    plot_descriptive_figure10(management, farmers, output_dir)
    plot_agronomic_heatmaps(candidates, schedule_order, output_dir)
    plot_economic_results(
        results,
        schedule_summary,
        schedule_year_summary,
        break_even_summary,
        parameters,
        schedule_order,
        output_dir,
    )
    write_summary_markdown(
        output_dir / "analysis_summary.md",
        metadata,
        schedule_summary,
        baseline_summary,
        management_summary,
        break_even_summary,
    )

    run_metadata = {
        **metadata,
        "input_file": str(args.input.resolve()),
        "output_directory": str(output_dir.resolve()),
        "n_economic_scenarios": len(scenarios),
        "n_weather_matched_results": len(results),
        "n_break_even_cost_scenarios": len(cost_scenarios),
        "n_break_even_results": len(break_even),
        "delta_pnr_reconciliation_max_abs_error": float(
            (results["delta_pnr_yuan_ha"] - results["delta_pnr_check_yuan_ha"]).abs().max()
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Weather-matched economic analysis completed.")
    print(json.dumps(run_metadata, ensure_ascii=False, indent=2))
    print("\nSchedule-level summary:")
    print(
        schedule_summary[
            [
                "schedule_id",
                "mean_delta_pnr_yuan_ha",
                "min_delta_pnr_yuan_ha",
                "max_delta_pnr_yuan_ha",
                "positive_ratio_pct",
                "all_combinations_positive",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

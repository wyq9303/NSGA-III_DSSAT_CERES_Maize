# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:29:13 2026

@author: 92593
"""

# -*- coding: utf-8 -*-
"""
Distance-to-utopia analysis for the first Pareto front

Input:
    first_front_objectives.csv

Output:
    1. first_front_with_utopia_distance.csv
    2. Printed compromise solution information
"""

from pathlib import Path

import numpy as np
import pandas as pd


def min_max_normalize(series: pd.Series) -> pd.Series:
    """
    Min-max normalization for a maximization objective.
    Maps values to [0, 1].
    """
    s_min = series.min()
    s_max = series.max()

    # Avoid division by zero if all values are identical
    if np.isclose(s_max, s_min):
        return pd.Series(np.ones(len(series)), index=series.index)

    return (series - s_min) / (s_max - s_min)


def main():
    # ===== 1. File path (relative to optimization/) =====
    folder_path = Path(__file__).resolve().parent.parent / "results"
    input_file = folder_path / "first_front_objectives.csv"
    output_file = folder_path / "first_front_with_utopia_distance.csv"

    # ===== 2. Read data =====
    df = pd.read_csv(input_file)

    # Expected objective columns
    obj_cols = ["Yield", "WUE", "NUE", "GrainN"]

    # Check columns
    missing_cols = [col for col in obj_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    # ===== 3. Normalize each objective =====
    for col in obj_cols:
        df[f"{col}_norm"] = min_max_normalize(df[col])

    # ===== 4. Compute Euclidean distance to utopia point (1,1,1,1) =====
    norm_cols = [f"{col}_norm" for col in obj_cols]

    df["distance_to_utopia"] = np.sqrt(
        ((df[norm_cols] - 1.0) ** 2).sum(axis=1)
    )

    # ===== 5. Identify compromise solution =====
    best_idx = df["distance_to_utopia"].idxmin()
    best_row = df.loc[best_idx]

    # ===== 6. Save results =====
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    # ===== 7. Print results =====
    print("=" * 60)
    print("Distance-to-utopia analysis completed.")
    print(f"Input file : {input_file}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    print(f"Number of Pareto-optimal solutions: {len(df)}")
    print(f"Compromise solution row index   : {best_idx}")
    print(f"Minimum distance to utopia      : {best_row['distance_to_utopia']:.6f}")
    print("-" * 60)
    print("Compromise solution objective values:")
    print(f"Yield  = {best_row['Yield']}")
    print(f"WUE    = {best_row['WUE']}")
    print(f"NUE    = {best_row['NUE']}")
    print(f"GrainN = {best_row['GrainN']}")
    print("-" * 60)
    print("Normalized objective values:")
    print(f"Yield_norm  = {best_row['Yield_norm']:.6f}")
    print(f"WUE_norm    = {best_row['WUE_norm']:.6f}")
    print(f"NUE_norm    = {best_row['NUE_norm']:.6f}")
    print(f"GrainN_norm = {best_row['GrainN_norm']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
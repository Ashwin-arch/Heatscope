import sys
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS, FIGURES


INPUT_GRID = RESULTS / "heatscope_spatial_blocks.gpkg"
INPUT_PRED = RESULTS / "reduced_spatial_ml_predictions.csv"

OUTPUT_GPKG = RESULTS / "heatscope_spatial_errors.gpkg"
OUTPUT_CSV = RESULTS / "spatial_error_summary.csv"

MODEL = "HistGradientBoosting"


print("=" * 70)
print("HeatScope Spatial Error Analysis")
print("=" * 70)


# =========================================================
# LOAD GRID
# =========================================================

gdf = gpd.read_file(
    INPUT_GRID,
    layer="spatial_blocks",
)

pred = pd.read_csv(
    INPUT_PRED
)


pred = pred[
    pred["model"] == MODEL
].copy()


print(
    "\nPrediction records:",
    f"{len(pred):,}"
)

print(
    "Grid cells:",
    f"{len(gdf):,}"
)


# =========================================================
# MERGE
# =========================================================

keep = [
    "cell_id",
    "actual_lst_p95",
    "predicted_lst_p95",
    "fold",
]

pred = pred[keep].copy()

pred["residual"] = (
    pred["actual_lst_p95"]
    - pred["predicted_lst_p95"]
)

pred["absolute_error"] = (
    np.abs(pred["residual"])
)

pred["squared_error"] = (
    pred["residual"] ** 2
)


gdf = gdf.merge(
    pred,
    on="cell_id",
    how="left",
)


# =========================================================
# OVERALL METRICS
# =========================================================

valid = gdf[
    "actual_lst_p95"
].notna()

actual = gdf.loc[
    valid,
    "actual_lst_p95"
]

predicted = gdf.loc[
    valid,
    "predicted_lst_p95"
]


mae = mean_absolute_error(
    actual,
    predicted,
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted,
    )
)

r2 = r2_score(
    actual,
    predicted,
)


print(
    "\nOverall spatial CV performance:"
)

print(
    f"MAE  : {mae:.4f} °C"
)

print(
    f"RMSE : {rmse:.4f} °C"
)

print(
    f"R²   : {r2:.4f}"
)


# =========================================================
# FOLD METRICS
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "FOLD ERROR ANALYSIS"
)

print(
    "=" * 70
)


fold_rows = []


for fold in sorted(
    gdf.loc[
        valid,
        "fold"
    ].dropna().unique()
):

    d = gdf[
        valid
        & (gdf["fold"] == fold)
    ]

    y_true = d[
        "actual_lst_p95"
    ]

    y_pred = d[
        "predicted_lst_p95"
    ]

    fold_mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    fold_rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    fold_r2 = r2_score(
        y_true,
        y_pred,
    )

    fold_rows.append(
        {
            "fold": int(fold),
            "cells": len(d),
            "mae": fold_mae,
            "rmse": fold_rmse,
            "r2": fold_r2,
            "mean_residual":
                d["residual"].mean(),
            "mean_absolute_error":
                d["absolute_error"].mean(),
            "p95_absolute_error":
                d["absolute_error"].quantile(
                    0.95
                ),
        }
    )

    print(
        f"Fold {int(fold)} | "
        f"cells={len(d):,} | "
        f"MAE={fold_mae:.4f} | "
        f"RMSE={fold_rmse:.4f} | "
        f"R²={fold_r2:.4f} | "
        f"bias={d['residual'].mean():+.4f}"
    )


fold_df = pd.DataFrame(
    fold_rows
)


# =========================================================
# SPATIAL ERROR QUANTILES
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "ERROR DISTRIBUTION"
)

print(
    "=" * 70
)


for q in [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
]:

    print(
        f"Absolute error P{int(q*100):02d}: "
        f"{gdf.loc[valid, 'absolute_error'].quantile(q):.4f} °C"
    )


# =========================================================
# WORST CELLS
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP 20 WORST-PREDICTED CELLS"
)

print(
    "=" * 70
)


worst = gdf[
    valid
].sort_values(
    "absolute_error",
    ascending=False,
).head(20)


print(
    worst[
        [
            "cell_id",
            "fold",
            "longitude",
            "latitude",
            "actual_lst_p95",
            "predicted_lst_p95",
            "residual",
            "absolute_error",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# =========================================================
# SYSTEMATIC BIAS
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "RESIDUAL BIAS"
)

print(
    "=" * 70
)


positive = (
    gdf.loc[
        valid,
        "residual"
    ] > 0
).mean() * 100


negative = (
    gdf.loc[
        valid,
        "residual"
    ] < 0
).mean() * 100


print(
    f"Under-predicted cells : "
    f"{positive:.2f}%"
)

print(
    f"Over-predicted cells  : "
    f"{negative:.2f}%"
)

print(
    f"Mean residual         : "
    f"{gdf.loc[valid, 'residual'].mean():+.4f} °C"
)


# =========================================================
# ERROR BY TARGET QUANTILE
# =========================================================

gdf["target_quantile"] = pd.qcut(
    gdf["actual_lst_p95"],
    q=5,
    labels=[
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
    ],
    duplicates="drop",
)


print(
    "\n" + "=" * 70
)

print(
    "ERROR BY OBSERVED LST-P95 QUINTILE"
)

print(
    "=" * 70
)


quantile_summary = (
    gdf.loc[valid]
    .groupby(
        "target_quantile",
        observed=True,
    )
    .agg(
        cells=(
            "cell_id",
            "count",
        ),
        mean_lst=(
            "actual_lst_p95",
            "mean",
        ),
        mean_prediction=(
            "predicted_lst_p95",
            "mean",
        ),
        mae=(
            "absolute_error",
            "mean",
        ),
        rmse=(
            "squared_error",
            lambda x:
            np.sqrt(np.mean(x)),
        ),
        mean_residual=(
            "residual",
            "mean",
        ),
    )
    .reset_index()
)


print(
    quantile_summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# =========================================================
# SAVE
# =========================================================

gdf.to_file(
    OUTPUT_GPKG,
    layer="spatial_errors",
    driver="GPKG",
)

fold_df.to_csv(
    OUTPUT_CSV,
    index=False,
)


print(
    "\nSaved:"
)

print(
    OUTPUT_GPKG
)

print(
    OUTPUT_CSV
)


print(
    "\n" + "=" * 70
)

print(
    "SPATIAL ERROR ANALYSIS COMPLETE"
)

print(
    "=" * 70
)

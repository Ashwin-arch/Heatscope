import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


PREDICTIONS_FILE = (
    RESULTS / "context_spatial_ml_predictions.csv"
)

GRID_FILE = (
    RESULTS / "heatscope_spatial_blocks.gpkg"
)

OUTPUT_CSV = (
    RESULTS / "spatial_uncertainty_results.csv"
)

OUTPUT_GPKG = (
    RESULTS / "heatscope_spatial_uncertainty.gpkg"
)

MODEL = "HistGradientBoosting"


print("=" * 70)
print("HeatScope Spatial Uncertainty Analysis")
print("=" * 70)


# =========================================================
# LOAD OUT-OF-FOLD PREDICTIONS
# =========================================================

pred = pd.read_csv(
    PREDICTIONS_FILE
)

pred = pred[
    pred["feature_set"].eq("Context_9")
].copy()


if pred.empty:
    raise RuntimeError(
        "No Context_9 predictions found."
    )


required = [
    "cell_id",
    "fold",
    "actual_lst_p95",
    "predicted_lst_p95",
    "residual",
    "absolute_error",
]

missing = [
    c for c in required
    if c not in pred.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )


print(
    "\nPrediction cells:",
    f"{len(pred):,}"
)

print(
    "Folds:",
    sorted(pred["fold"].unique())
)


# =========================================================
# BUILD FOLD-WISE SPATIALLY CROSS-FITTED INTERVALS
# =========================================================
#
# For each test fold:
#
#   calibration = residuals from the other folds
#   test        = current fold
#
# This prevents the test fold's own errors from being used
# to construct its interval.
#
# Because this is spatial data, we describe the result as
# a spatial cross-validated prediction interval rather than
# claiming formal distribution-free conformal coverage.
# =========================================================

fold_results = []
interval_frames = []


for test_fold in sorted(
    pred["fold"].unique()
):

    test = pred[
        pred["fold"] == test_fold
    ].copy()

    calibration = pred[
        pred["fold"] != test_fold
    ].copy()


    if calibration.empty:
        raise RuntimeError(
            f"No calibration data for fold {test_fold}."
        )


    print(
        "\n" + "-" * 70
    )

    print(
        f"TEST FOLD: {int(test_fold)}"
    )

    print(
        f"Test cells       : {len(test):,}"
    )

    print(
        f"Calibration cells: {len(calibration):,}"
    )


    # -----------------------------------------------------
    # Absolute residual quantiles
    # -----------------------------------------------------

    q90 = np.quantile(
        np.abs(
            calibration["residual"]
        ),
        0.90,
    )

    q95 = np.quantile(
        np.abs(
            calibration["residual"]
        ),
        0.95,
    )


    # -----------------------------------------------------
    # Symmetric intervals
    # -----------------------------------------------------

    test[
        "lower_90"
    ] = (
        test["predicted_lst_p95"]
        - q90
    )

    test[
        "upper_90"
    ] = (
        test["predicted_lst_p95"]
        + q90
    )


    test[
        "lower_95"
    ] = (
        test["predicted_lst_p95"]
        - q95
    )

    test[
        "upper_95"
    ] = (
        test["predicted_lst_p95"]
        + q95
    )


    # -----------------------------------------------------
    # Coverage
    # -----------------------------------------------------

    actual = test[
        "actual_lst_p95"
    ]


    covered_90 = (
        (actual >= test["lower_90"])
        &
        (actual <= test["upper_90"])
    )

    covered_95 = (
        (actual >= test["lower_95"])
        &
        (actual <= test["upper_95"])
    )


    coverage_90 = (
        covered_90.mean()
    )

    coverage_95 = (
        covered_95.mean()
    )


    width_90 = (
        test["upper_90"]
        - test["lower_90"]
    )

    width_95 = (
        test["upper_95"]
        - test["lower_95"]
    )


    # -----------------------------------------------------
    # Hot quintile coverage
    # -----------------------------------------------------

    test["target_quantile"] = pd.qcut(
        test["actual_lst_p95"],
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


    q5 = test[
        test["target_quantile"] == "Q5"
    ]


    if len(q5) > 0:

        q5_coverage_95 = (
            (
                (q5["actual_lst_p95"] >= q5["lower_95"])
                &
                (q5["actual_lst_p95"] <= q5["upper_95"])
            ).mean()
        )

    else:

        q5_coverage_95 = np.nan


    fold_results.append(
        {
            "fold": int(test_fold),
            "test_cells": len(test),
            "calibration_cells": len(calibration),
            "q90_abs_error": q90,
            "q95_abs_error": q95,
            "coverage_90": coverage_90,
            "coverage_95": coverage_95,
            "mean_width_90": width_90.mean(),
            "mean_width_95": width_95.mean(),
            "median_width_95": np.median(width_95),
            "q5_coverage_95": q5_coverage_95,
        }
    )


    interval_frames.append(
        test
    )


    print(
        f"Calibration 90% error: {q90:.4f} °C"
    )

    print(
        f"Calibration 95% error: {q95:.4f} °C"
    )

    print(
        f"90% interval coverage: {coverage_90 * 100:.2f}%"
    )

    print(
        f"95% interval coverage: {coverage_95 * 100:.2f}%"
    )

    print(
        f"Mean 95% interval width: {width_95.mean():.4f} °C"
    )

    print(
        f"Q5 95% coverage: {q5_coverage_95 * 100:.2f}%"
    )


# =========================================================
# COMBINE
# =========================================================

intervals = pd.concat(
    interval_frames,
    ignore_index=True,
)

fold_summary = pd.DataFrame(
    fold_results
)


# =========================================================
# OVERALL COVERAGE
# =========================================================

actual = intervals[
    "actual_lst_p95"
]


coverage_90 = (
    (
        (actual >= intervals["lower_90"])
        &
        (actual <= intervals["upper_90"])
    ).mean()
)

coverage_95 = (
    (
        (actual >= intervals["lower_95"])
        &
        (actual <= intervals["upper_95"])
    ).mean()
)


width_90 = (
    intervals["upper_90"]
    - intervals["lower_90"]
)

width_95 = (
    intervals["upper_95"]
    - intervals["lower_95"]
)


print(
    "\n" + "=" * 70
)

print(
    "OVERALL UNCERTAINTY RESULTS"
)

print(
    "=" * 70
)

print(
    f"90% interval coverage : "
    f"{coverage_90 * 100:.2f}%"
)

print(
    f"95% interval coverage : "
    f"{coverage_95 * 100:.2f}%"
)

print(
    f"Mean 90% width        : "
    f"{width_90.mean():.4f} °C"
)

print(
    f"Mean 95% width        : "
    f"{width_95.mean():.4f} °C"
)

print(
    f"Median 95% width      : "
    f"{np.median(width_95):.4f} °C"
)

print(
    f"P95 95% width         : "
    f"{np.quantile(width_95, 0.95):.4f} °C"
)


# =========================================================
# ERROR BY TARGET QUINTILE
# =========================================================

intervals[
    "target_quantile"
] = pd.qcut(
    intervals["actual_lst_p95"],
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


quantile_rows = []


for q in [
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q5",
]:

    d = intervals[
        intervals["target_quantile"] == q
    ]


    if len(d) == 0:
        continue


    cov = (
        (
            (d["actual_lst_p95"] >= d["lower_95"])
            &
            (d["actual_lst_p95"] <= d["upper_95"])
        ).mean()
    )


    quantile_rows.append(
        {
            "target_quantile": q,
            "cells": len(d),
            "coverage_95": cov,
            "mean_interval_width":
                (
                    d["upper_95"]
                    - d["lower_95"]
                ).mean(),
            "mean_actual":
                d["actual_lst_p95"].mean(),
            "mean_prediction":
                d["predicted_lst_p95"].mean(),
        }
    )


quantile_summary = pd.DataFrame(
    quantile_rows
)


print(
    "\n" + "=" * 70
)

print(
    "95% COVERAGE BY OBSERVED TEMPERATURE QUINTILE"
)

print(
    "=" * 70
)

print(
    quantile_summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# =========================================================
# SAVE TABULAR RESULTS
# =========================================================

fold_summary.to_csv(
    RESULTS / "spatial_uncertainty_by_fold.csv",
    index=False,
)


intervals.to_csv(
    OUTPUT_CSV,
    index=False,
)


# =========================================================
# LOAD GEOMETRY AND SAVE GPKG
# =========================================================

grid = gpd.read_file(
    GRID_FILE,
    layer="spatial_blocks",
)


spatial = grid.merge(
    intervals[
        [
            "cell_id",
            "fold",
            "actual_lst_p95",
            "predicted_lst_p95",
            "residual",
            "absolute_error",
            "lower_90",
            "upper_90",
            "lower_95",
            "upper_95",
            "target_quantile",
        ]
    ],
    on="cell_id",
    how="left",
    validate="one_to_one",
)


spatial[
    "interval_width_90"
] = (
    spatial["upper_90"]
    - spatial["lower_90"]
)


spatial[
    "interval_width_95"
] = (
    spatial["upper_95"]
    - spatial["lower_95"]
)


spatial[
    "covered_90"
] = (
    (
        spatial["actual_lst_p95"]
        >= spatial["lower_90"]
    )
    &
    (
        spatial["actual_lst_p95"]
        <= spatial["upper_90"]
    )
).astype("Int8")


spatial[
    "covered_95"
] = (
    (
        spatial["actual_lst_p95"]
        >= spatial["lower_95"]
    )
    &
    (
        spatial["actual_lst_p95"]
        <= spatial["upper_95"]
    )
).astype("Int8")


spatial.to_file(
    OUTPUT_GPKG,
    layer="spatial_uncertainty",
    driver="GPKG",
)


# =========================================================
# FINAL
# =========================================================

print(
    "\nSaved:"
)

print(
    OUTPUT_CSV
)

print(
    RESULTS / "spatial_uncertainty_by_fold.csv"
)

print(
    OUTPUT_GPKG
)


print(
    "\n" + "=" * 70
)

print(
    "SPATIAL UNCERTAINTY ANALYSIS COMPLETE"
)

print(
    "=" * 70
)

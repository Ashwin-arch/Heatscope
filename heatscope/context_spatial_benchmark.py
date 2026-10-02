import sys
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


MODEL_FILE = RESULTS / "heatscope_spatial_blocks.gpkg"
CONTEXT_FILE = RESULTS / "heatscope_spatial_context.gpkg"

OUTPUT_RESULTS = RESULTS / "context_spatial_ml_results.csv"
OUTPUT_PREDICTIONS = RESULTS / "context_spatial_ml_predictions.csv"


TARGET = "lst_celsius_p95"

BASE_FEATURES = [
    "ndvi_mean",
    "ndmi_mean",
    "population_density",
    "building_coverage_pct",
    "building_count",
]

CONTEXT_FEATURES = [
    "ndvi_mean_nbr_mean",
    "ndmi_mean_nbr_mean",
    "population_density_nbr_mean",
    "building_coverage_pct_nbr_mean",
]


print("=" * 70)
print("HeatScope Spatial Context Benchmark")
print("=" * 70)


# =========================================================
# LOAD PRIMARY MODEL DATA
# =========================================================

data = gpd.read_file(
    MODEL_FILE,
    layer="spatial_blocks",
)

print(
    "\nModel dataset:",
    f"{len(data):,} cells"
)

print(
    "Model columns:",
    len(data.columns)
)


# =========================================================
# LOAD ONLY CONTEXT ATTRIBUTES
# =========================================================

context = gpd.read_file(
    CONTEXT_FILE,
    layer="spatial_context",
)

context_cols = [
    "cell_id"
] + CONTEXT_FEATURES

context = context[
    context_cols
].copy()


print(
    "Context dataset:",
    f"{len(context):,} cells"
)


# =========================================================
# CHECK CELL-ID UNIQUENESS
# =========================================================

if data["cell_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate cell_id values in model dataset."
    )

if context["cell_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate cell_id values in context dataset."
    )


# =========================================================
# MERGE
# =========================================================

data = data.merge(
    context,
    on="cell_id",
    how="left",
    validate="one_to_one",
)


if TARGET not in data.columns:

    raise RuntimeError(
        f"Target '{TARGET}' missing after merge."
    )


# =========================================================
# VALID DATA
# =========================================================

valid = (
    data[TARGET].notna()
    & data["spatial_block"].notna()
)


data = data.loc[
    valid
].copy()


print(
    "\nValid cells:",
    f"{len(data):,}"
)

print(
    "Spatial blocks:",
    data["spatial_block"].nunique()
)


# =========================================================
# CONTEXT COMPLETENESS
# =========================================================

print(
    "\nContext missing values:"
)

for feature in CONTEXT_FEATURES:

    missing = data[
        feature
    ].isna().sum()

    print(
        f"{feature:35s}: {missing}"
    )


# =========================================================
# TARGET QUINTILES
# =========================================================

data["_target_q"] = pd.qcut(
    data[TARGET],
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


# =========================================================
# FEATURE SETS
# =========================================================

feature_sets = {

    "Base_5": BASE_FEATURES,

    "Context_9": (
        BASE_FEATURES
        + CONTEXT_FEATURES
    ),
}


# =========================================================
# MODEL
# =========================================================

def make_model():

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


# =========================================================
# IDENTICAL SPATIAL CV
# =========================================================

groups = data[
    "spatial_block"
].astype(str)

cv = GroupKFold(
    n_splits=5
)


all_results = []
all_predictions = []


# =========================================================
# RUN BENCHMARK
# =========================================================

for set_name, features in feature_sets.items():

    print(
        "\n" + "-" * 70
    )

    print(
        "FEATURE SET:",
        set_name
    )

    print(
        "Predictors:",
        len(features)
    )


    X = data[
        features
    ].copy()

    y = data[
        TARGET
    ].astype(float)


    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        cv.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):

        train_blocks = set(
            groups.iloc[
                train_idx
            ]
        )

        test_blocks = set(
            groups.iloc[
                test_idx
            ]
        )


        if train_blocks & test_blocks:

            raise RuntimeError(
                "Spatial leakage detected."
            )


        model = make_model()


        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]


        model.fit(
            X_train,
            y_train,
        )


        pred = model.predict(
            X_test
        )


        # -------------------------------------------------
        # OVERALL METRICS
        # -------------------------------------------------

        fold_mae = mean_absolute_error(
            y_test,
            pred,
        )

        fold_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred,
            )
        )

        fold_r2 = r2_score(
            y_test,
            pred,
        )


        # -------------------------------------------------
        # HOTTEST QUINTILE
        # -------------------------------------------------

        test_rows = data.iloc[
            test_idx
        ]

        q5_mask = (
            test_rows["_target_q"]
            .values == "Q5"
        )


        if q5_mask.sum() > 0:

            y_q5 = y_test.values[
                q5_mask
            ]

            p_q5 = pred[
                q5_mask
            ]


            q5_mae = mean_absolute_error(
                y_q5,
                p_q5,
            )

            q5_rmse = np.sqrt(
                mean_squared_error(
                    y_q5,
                    p_q5,
                )
            )

            q5_bias = np.mean(
                y_q5 - p_q5
            )

        else:

            q5_mae = np.nan
            q5_rmse = np.nan
            q5_bias = np.nan


        all_results.append(
            {
                "feature_set": set_name,
                "fold": fold,
                "cells": len(test_idx),
                "mae": fold_mae,
                "rmse": fold_rmse,
                "r2": fold_r2,
                "q5_mae": q5_mae,
                "q5_rmse": q5_rmse,
                "q5_bias": q5_bias,
            }
        )


        all_predictions.append(
            pd.DataFrame(
                {
                    "feature_set": set_name,
                    "fold": fold,
                    "cell_id":
                        test_rows[
                            "cell_id"
                        ].values,
                    "actual_lst_p95":
                        y_test.values,
                    "predicted_lst_p95":
                        pred,
                    "residual":
                        y_test.values - pred,
                    "absolute_error":
                        np.abs(
                            y_test.values - pred
                        ),
                }
            )
        )


        print(
            f"Fold {fold}: "
            f"MAE={fold_mae:.4f} | "
            f"RMSE={fold_rmse:.4f} | "
            f"R2={fold_r2:.4f} | "
            f"Q5-MAE={q5_mae:.4f} | "
            f"Q5-bias={q5_bias:+.4f}"
        )


# =========================================================
# SUMMARY
# =========================================================

results = pd.DataFrame(
    all_results
)

predictions = pd.concat(
    all_predictions,
    ignore_index=True,
)


summary = (
    results
    .groupby("feature_set")
    .agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        q5_mae_mean=("q5_mae", "mean"),
        q5_rmse_mean=("q5_rmse", "mean"),
        q5_bias_mean=("q5_bias", "mean"),
    )
    .sort_values(
        "rmse_mean"
    )
)


print(
    "\n" + "=" * 70
)

print(
    "CONTEXT BENCHMARK SUMMARY"
)

print(
    "=" * 70
)

print(
    summary.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# =========================================================
# DIRECT COMPARISON
# =========================================================

base = summary.loc[
    "Base_5"
]

context_model = summary.loc[
    "Context_9"
]


print(
    "\n" + "=" * 70
)

print(
    "DIRECT COMPARISON"
)

print(
    "=" * 70
)


print(
    f"Overall RMSE change : "
    f"{context_model['rmse_mean'] - base['rmse_mean']:+.4f}"
)

print(
    f"Overall R2 change   : "
    f"{context_model['r2_mean'] - base['r2_mean']:+.4f}"
)

print(
    f"Q5 MAE change       : "
    f"{context_model['q5_mae_mean'] - base['q5_mae_mean']:+.4f}"
)

print(
    f"Q5 bias change      : "
    f"{context_model['q5_bias_mean'] - base['q5_bias_mean']:+.4f}"
)


# =========================================================
# SAVE
# =========================================================

results.to_csv(
    OUTPUT_RESULTS,
    index=False,
)

predictions.to_csv(
    OUTPUT_PREDICTIONS,
    index=False,
)


print(
    "\nSaved:"
)

print(
    OUTPUT_RESULTS
)

print(
    OUTPUT_PREDICTIONS
)


print(
    "\n" + "=" * 70
)

print(
    "CONTEXT BENCHMARK COMPLETE"
)

print(
    "=" * 70
)

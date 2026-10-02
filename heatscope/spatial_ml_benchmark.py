import sys
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


INPUT_FILE = RESULTS / "heatscope_spatial_blocks.gpkg"

RESULTS_FILE = RESULTS / "spatial_ml_results.csv"


TARGET = "lst_celsius_p95"

FEATURES = [
    "ndvi_mean",
    "ndvi_median",
    "ndbi_mean",
    "ndbi_median",
    "ndmi_mean",
    "ndmi_median",
    "population",
    "population_density",
    "building_coverage_pct",
    "building_area_m2",
    "building_count",
]


print("=" * 70)
print("HeatScope Spatial ML Benchmark")
print("=" * 70)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading dataset...")

gdf = gpd.read_file(
    INPUT_FILE,
    layer="spatial_blocks",
)

print(
    "Total cells:",
    f"{len(gdf):,}"
)


# =========================================================
# VALID TARGET
# =========================================================

valid = (
    gdf[TARGET].notna()
    & gdf["spatial_block"].notna()
)


data = gdf.loc[
    valid
].copy()


print(
    "Valid cells:",
    f"{len(data):,}"
)


# =========================================================
# FEATURE CHECK
# =========================================================

missing_features = [
    f for f in FEATURES
    if f not in data.columns
]

if missing_features:

    raise RuntimeError(
        "Missing features: "
        + ", ".join(missing_features)
    )


X = data[FEATURES].copy()

y = data[TARGET].astype(float)

groups = data["spatial_block"].astype(str)


print(
    "Features:",
    len(FEATURES)
)

print(
    "Spatial blocks:",
    groups.nunique()
)


# =========================================================
# MODELS
# =========================================================

models = {}


models["LinearRegression"] = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LinearRegression(),
        ),
    ]
)


models["RandomForest"] = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


models["ExtraTrees"] = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "model",
            ExtraTreesRegressor(
                n_estimators=300,
                max_features=1.0,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


models["HistGradientBoosting"] = Pipeline(
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
# OPTIONAL XGBOOST
# =========================================================

try:

    from xgboost import XGBRegressor

    models["XGBoost"] = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nXGBoost detected.")

except ImportError:

    print(
        "\nXGBoost not installed. "
        "Continuing without it."
    )


# =========================================================
# SPATIAL CROSS VALIDATION
# =========================================================

N_SPLITS = 5

cv = GroupKFold(
    n_splits=N_SPLITS
)


print("\n" + "=" * 70)
print("Spatial Cross-Validation")
print("=" * 70)

print(
    f"\nUsing {N_SPLITS}-fold GroupKFold"
)

print(
    "Grouping variable: spatial_block"
)


# =========================================================
# BENCHMARK
# =========================================================

all_results = []

predictions = []


for model_name, model in models.items():

    print("\n" + "-" * 70)

    print(
        "MODEL:",
        model_name
    )

    fold_mae = []
    fold_rmse = []
    fold_r2 = []

    fold_rows = []


    for fold, (train_idx, test_idx) in enumerate(
        cv.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]


        train_blocks = set(
            groups.iloc[train_idx]
        )

        test_blocks = set(
            groups.iloc[test_idx]
        )


        overlap = (
            train_blocks
            & test_blocks
        )

        if overlap:

            raise RuntimeError(
                "Spatial leakage detected!"
            )


        model.fit(
            X_train,
            y_train,
        )


        pred = model.predict(
            X_test
        )


        mae = mean_absolute_error(
            y_test,
            pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred,
            )
        )

        r2 = r2_score(
            y_test,
            pred,
        )


        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)


        fold_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "train_cells": len(train_idx),
                "test_cells": len(test_idx),
                "train_blocks": len(
                    train_blocks
                ),
                "test_blocks": len(
                    test_blocks
                ),
            }
        )


        predictions.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "fold": fold,
                    "cell_id": data.iloc[
                        test_idx
                    ]["cell_id"].values,
                    "actual_lst_p95":
                        y_test.values,
                    "predicted_lst_p95":
                        pred,
                }
            )
        )


        print(
            f"Fold {fold}: "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"R2={r2:.4f}"
        )


    # -----------------------------------------------------
    # Aggregate
    # -----------------------------------------------------

    all_results.extend(
        fold_rows
    )


    print(
        "\nMean:"
        f" MAE={np.mean(fold_mae):.4f}"
        f" | RMSE={np.mean(fold_rmse):.4f}"
        f" | R2={np.mean(fold_r2):.4f}"
    )


# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.concat(
    predictions,
    ignore_index=True,
)


results_df.to_csv(
    RESULTS_FILE,
    index=False,
)


predictions_df.to_csv(
    RESULTS / "spatial_ml_predictions.csv",
    index=False,
)


# =========================================================
# SUMMARY
# =========================================================

summary = (
    results_df
    .groupby("model")
    .agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
    )
    .sort_values(
        "rmse_mean"
    )
)


print("\n" + "=" * 70)
print("SPATIAL ML RESULTS")
print("=" * 70)

print(
    summary.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# =========================================================
# ACCEPTANCE GATE
# =========================================================

print("\n" + "=" * 70)
print("INITIAL ACCEPTANCE GATE")
print("=" * 70)


best_model = summary.index[0]

best_r2 = summary.loc[
    best_model,
    "r2_mean"
]

best_rmse = summary.loc[
    best_model,
    "rmse_mean"
]


print(
    f"Best model: {best_model}"
)

print(
    f"Spatial CV RMSE: {best_rmse:.4f} °C"
)

print(
    f"Spatial CV R²: {best_r2:.4f}"
)


if best_r2 > 0:

    print(
        "\nPASS: "
        "Model explains positive spatially held-out variance."
    )

else:

    print(
        "\nWARNING: "
        "No model achieved positive mean spatial R²."
    )

print(
    "\nSaved:",
    RESULTS_FILE
)

print(
    "Saved:",
    RESULTS / "spatial_ml_predictions.csv"
)

print("\n" + "=" * 70)
print("SPATIAL ML BENCHMARK COMPLETE")
print("=" * 70)

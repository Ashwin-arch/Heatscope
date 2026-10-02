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

RESULTS_FILE = (
    RESULTS
    / "reduced_spatial_ml_results.csv"
)

PREDICTIONS_FILE = (
    RESULTS
    / "reduced_spatial_ml_predictions.csv"
)


TARGET = "lst_celsius_p95"


# ---------------------------------------------------------
# REDUCED SCIENTIFIC FEATURE SET
# ---------------------------------------------------------

FEATURES = [
    "ndvi_mean",
    "ndmi_mean",
    "population_density",
    "building_coverage_pct",
    "building_count",
]


print("=" * 70)
print("HeatScope Reduced-Feature Spatial ML Benchmark")
print("=" * 70)


# =========================================================
# LOAD
# =========================================================

gdf = gpd.read_file(
    INPUT_FILE,
    layer="spatial_blocks",
)


valid = (
    gdf[TARGET].notna()
    & gdf["spatial_block"].notna()
)


data = gdf.loc[
    valid
].copy()


X = data[FEATURES].copy()

y = data[TARGET].astype(float)

groups = data[
    "spatial_block"
].astype(str)


print(
    "\nCells:",
    f"{len(data):,}"
)

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

models = {

    "LinearRegression": Pipeline(
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
    ),

    "RandomForest": Pipeline(
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
    ),

    "ExtraTrees": Pipeline(
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
    ),

    "HistGradientBoosting": Pipeline(
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
    ),
}


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
        "\nXGBoost unavailable."
    )


# =========================================================
# SPATIAL CV
# =========================================================

cv = GroupKFold(
    n_splits=5
)


all_results = []

all_predictions = []


# =========================================================
# RUN
# =========================================================

for model_name, model in models.items():

    print("\n" + "-" * 70)

    print(
        "MODEL:",
        model_name
    )

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


        all_results.append(
            {
                "model": model_name,
                "fold": fold,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )


        all_predictions.append(
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


# =========================================================
# RESULTS
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
print("REDUCED FEATURE RESULTS")
print("=" * 70)

print(
    summary.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# =========================================================
# SAVE
# =========================================================

results.to_csv(
    RESULTS_FILE,
    index=False,
)

predictions.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


print("\nSaved:")
print(RESULTS_FILE)
print(PREDICTIONS_FILE)


# =========================================================
# COMPARISON WITH ORIGINAL
# =========================================================

ORIGINAL_RMSE = 1.5909
ORIGINAL_R2 = 0.6356

best_model = summary.index[0]

new_rmse = summary.loc[
    best_model,
    "rmse_mean"
]

new_r2 = summary.loc[
    best_model,
    "r2_mean"
]


rmse_change = (
    new_rmse
    - ORIGINAL_RMSE
)

r2_change = (
    new_r2
    - ORIGINAL_R2
)


print("\n" + "=" * 70)
print("REDUCTION CHECK")
print("=" * 70)

print(
    f"Previous best RMSE : "
    f"{ORIGINAL_RMSE:.4f}"
)

print(
    f"Reduced best RMSE  : "
    f"{new_rmse:.4f}"
)

print(
    f"RMSE change        : "
    f"{rmse_change:+.4f}"
)

print()

print(
    f"Previous best R2   : "
    f"{ORIGINAL_R2:.4f}"
)

print(
    f"Reduced best R2    : "
    f"{new_r2:.4f}"
)

print(
    f"R2 change          : "
    f"{r2_change:+.4f}"
)


if abs(r2_change) <= 0.03:

    print(
        "\nPASS: "
        "Reduced feature set maintains "
        "approximately comparable predictive performance."
    )

elif r2_change > 0:

    print(
        "\nPASS: "
        "Reduced feature set improves predictive performance."
    )

else:

    print(
        "\nREVIEW: "
        "Feature reduction caused a meaningful performance decrease."
    )


print("\n" + "=" * 70)
print("REDUCED FEATURE BENCHMARK COMPLETE")
print("=" * 70)

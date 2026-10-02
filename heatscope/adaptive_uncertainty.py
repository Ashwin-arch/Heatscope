import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


MODEL_DATA = RESULTS / "heatscope_spatial_blocks.gpkg"
PREDICTIONS = RESULTS / "context_spatial_ml_predictions.csv"

OUTPUT_FILE = (
    RESULTS / "adaptive_uncertainty_results.csv"
)


TARGET = "lst_celsius_p95"

BASE_FEATURES = [
    "ndvi_mean",
    "ndmi_mean",
    "population_density",
    "building_coverage_pct",
    "building_count",
    "ndvi_mean_nbr_mean",
    "ndmi_mean_nbr_mean",
    "population_density_nbr_mean",
    "building_coverage_pct_nbr_mean",
]


print("=" * 70)
print("HeatScope Adaptive Uncertainty Model")
print("=" * 70)


# =========================================================
# LOAD MODEL DATA
# =========================================================

import geopandas as gpd

data = gpd.read_file(
    MODEL_DATA,
    layer="spatial_blocks",
)

context = gpd.read_file(
    RESULTS / "heatscope_spatial_context.gpkg",
    layer="spatial_context",
)

context = context[
    [
        "cell_id",
        "ndvi_mean_nbr_mean",
        "ndmi_mean_nbr_mean",
        "population_density_nbr_mean",
        "building_coverage_pct_nbr_mean",
    ]
]

data = data.merge(
    context,
    on="cell_id",
    how="left",
    validate="one_to_one",
)

data = data[
    data[TARGET].notna()
].copy()


# =========================================================
# LOAD OUT-OF-FOLD PRIMARY PREDICTIONS
# =========================================================

pred = pd.read_csv(
    PREDICTIONS
)

pred = pred[
    pred["feature_set"].eq("Context_9")
].copy()

pred = pred[
    [
        "cell_id",
        "fold",
        "actual_lst_p95",
        "predicted_lst_p95",
        "residual",
        "absolute_error",
    ]
].copy()


data = data.merge(
    pred,
    on="cell_id",
    how="inner",
    validate="one_to_one",
)


print(
    "\nCells:",
    f"{len(data):,}"
)


# =========================================================
# CREATE UNCERTAINTY TARGET
# =========================================================
#
# We model absolute prediction error.
#
# IMPORTANT:
# These are out-of-fold primary-model errors.
# =========================================================

data["error_target"] = np.abs(
    data["residual"]
)


# =========================================================
# UNCERTAINTY FEATURES
# =========================================================
#
# We do NOT use actual temperature.
#
# The uncertainty model receives only information available
# before observing the target.
# =========================================================

UNCERTAINTY_FEATURES = (
    BASE_FEATURES
    + [
        "predicted_lst_p95",
    ]
)


X = data[
    UNCERTAINTY_FEATURES
].copy()

y = data[
    "error_target"
].astype(float)


groups = data[
    "spatial_block"
].astype(str)


# =========================================================
# CROSS-FITTED UNCERTAINTY MODEL
# =========================================================

cv = GroupKFold(
    n_splits=5
)


all_pred = []


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


    uncertainty_model = Pipeline(
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
                    max_iter=250,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    loss="absolute_error",
                    random_state=42,
                ),
            ),
        ]
    )


    uncertainty_model.fit(
        X_train,
        y_train,
    )


    predicted_error = (
        uncertainty_model.predict(
            X_test
        )
    )


    predicted_error = np.maximum(
        predicted_error,
        0.01,
    )


    test = data.iloc[
        test_idx
    ].copy()


    test[
        "predicted_abs_error"
    ] = predicted_error


    test[
        "error_calibration_ratio"
    ] = (
        test["error_target"]
        /
        test["predicted_abs_error"]
    )


    test[
        "uncertainty_fold"
    ] = fold


    all_pred.append(
        test[
            [
                "cell_id",
                "uncertainty_fold",
                "predicted_abs_error",
                "error_calibration_ratio",
            ]
        ]
    )


    print(
        f"Fold {fold}: "
        f"mean predicted error="
        f"{predicted_error.mean():.4f} °C | "
        f"actual MAE="
        f"{y_test.mean():.4f} °C"
    )


adaptive = pd.concat(
    all_pred,
    ignore_index=True,
)


data = data.merge(
    adaptive,
    on="cell_id",
    how="left",
    validate="one_to_one",
)


# =========================================================
# CALIBRATION
# =========================================================
#
# We need a single multiplicative calibration factor.
#
# It is estimated from out-of-fold predictions.
# =========================================================

calibration_factor = np.quantile(
    data["error_calibration_ratio"],
    0.95,
)


print(
    "\n" + "=" * 70
)

print(
    "ADAPTIVE ERROR CALIBRATION"
)

print(
    "=" * 70
)

print(
    "95th-percentile calibration factor:",
    f"{calibration_factor:.4f}"
)


# =========================================================
# BUILD INTERVAL
# =========================================================

data[
    "adaptive_half_width"
] = (
    data["predicted_abs_error"]
    * calibration_factor
)


data[
    "adaptive_lower_95"
] = (
    data["predicted_lst_p95"]
    - data["adaptive_half_width"]
)


data[
    "adaptive_upper_95"
] = (
    data["predicted_lst_p95"]
    + data["adaptive_half_width"]
)


# =========================================================
# COVERAGE
# =========================================================

actual = data[
    TARGET
]


covered = (
    (actual >= data["adaptive_lower_95"])
    &
    (actual <= data["adaptive_upper_95"])
)


overall_coverage = (
    covered.mean()
)


mean_width = (
    data["adaptive_upper_95"]
    -
    data["adaptive_lower_95"]
).mean()


# =========================================================
# QUINTILES
# =========================================================

data[
    "target_quantile"
] = pd.qcut(
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


print(
    "\nOverall coverage:",
    f"{overall_coverage * 100:.2f}%"
)

print(
    "Mean interval width:",
    f"{mean_width:.4f} °C"
)


print(
    "\n" + "=" * 70
)

print(
    "COVERAGE BY TEMPERATURE QUINTILE"
)

print(
    "=" * 70
)


for q in [
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q5",
]:

    d = data[
        data["target_quantile"] == q
    ]

    cov = (
        (
            (d[TARGET] >= d["adaptive_lower_95"])
            &
            (d[TARGET] <= d["adaptive_upper_95"])
        ).mean()
    )

    width = (
        d["adaptive_upper_95"]
        - d["adaptive_lower_95"]
    ).mean()

    print(
        f"{q}: "
        f"coverage={cov * 100:.2f}% | "
        f"width={width:.4f} °C"
    )


# =========================================================
# UNCERTAINTY RANGE
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "PREDICTED UNCERTAINTY DISTRIBUTION"
)

print(
    "=" * 70
)


half_width = data[
    "adaptive_half_width"
]


for q in [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
]:

    print(
        f"P{int(q * 100):02d} half-width: "
        f"{np.quantile(half_width, q):.4f} °C"
    )


# =========================================================
# SAVE
# =========================================================

save_cols = [
    "cell_id",
    "fold",
    TARGET,
    "predicted_lst_p95",
    "residual",
    "absolute_error",
    "predicted_abs_error",
    "error_calibration_ratio",
    "adaptive_half_width",
    "adaptive_lower_95",
    "adaptive_upper_95",
    "target_quantile",
]


data[
    save_cols
].to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print(
    "\n" + "=" * 70
)

print(
    "ADAPTIVE UNCERTAINTY COMPLETE"
)

print(
    "=" * 70
)

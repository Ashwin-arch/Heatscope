import sys
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shap

from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


INPUT_FILE = RESULTS / "heatscope_spatial_blocks.gpkg"

OUTPUT_IMPORTANCE = (
    RESULTS / "reduced_shap_feature_importance.csv"
)

OUTPUT_VALUES = (
    RESULTS / "reduced_shap_values.csv"
)

OUTPUT_GPKG = (
    RESULTS / "heatscope_reduced_shap.gpkg"
)


TARGET = "lst_celsius_p95"

FEATURES = [
    "ndvi_mean",
    "ndmi_mean",
    "population_density",
    "building_coverage_pct",
    "building_count",
]


print("=" * 70)
print("HeatScope Reduced-Feature SHAP Analysis")
print("=" * 70)


# =========================================================
# LOAD DATA
# =========================================================

gdf = gpd.read_file(
    INPUT_FILE,
    layer="spatial_blocks",
)

valid = gdf[TARGET].notna()

data = gdf.loc[
    valid
].copy()

X = data[FEATURES].copy()

y = data[TARGET].astype(float)


print(
    "\nCells:",
    f"{len(data):,}"
)

print(
    "Features:",
    len(FEATURES)
)


# =========================================================
# IMPUTATION
# =========================================================

imputer = SimpleImputer(
    strategy="median"
)

X_clean = pd.DataFrame(
    imputer.fit_transform(X),
    columns=FEATURES,
    index=X.index,
)


# =========================================================
# FINAL MODEL
# =========================================================

model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42,
)

print(
    "\nTraining final HistGradientBoosting model..."
)

model.fit(
    X_clean,
    y,
)


# =========================================================
# SHAP
# =========================================================

print(
    "\nCalculating SHAP values..."
)

explainer = shap.Explainer(
    model,
    X_clean,
)

shap_values = explainer(
    X_clean
)

values = np.asarray(
    shap_values.values
)


print(
    "SHAP matrix:",
    values.shape
)


# =========================================================
# GLOBAL IMPORTANCE
# =========================================================

importance = pd.DataFrame(
    {
        "feature": FEATURES,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }
)

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False,
).reset_index(drop=True)

importance[
    "rank"
] = np.arange(
    1,
    len(importance) + 1
)

importance[
    "relative_importance_pct"
] = (
    importance["mean_abs_shap"]
    / importance["mean_abs_shap"].sum()
    * 100
)


print(
    "\n" + "=" * 70
)

print(
    "GLOBAL SHAP FEATURE IMPORTANCE"
)

print(
    "=" * 70
)

print(
    importance[
        [
            "rank",
            "feature",
            "mean_abs_shap",
            "relative_importance_pct",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}"
    )
)


# =========================================================
# SAVE IMPORTANCE
# =========================================================

importance.to_csv(
    OUTPUT_IMPORTANCE,
    index=False,
)


# =========================================================
# SAVE SHAP VALUES
# =========================================================

shap_df = pd.DataFrame(
    values,
    columns=[
        f"shap_{f}"
        for f in FEATURES
    ],
)

shap_df.insert(
    0,
    "cell_id",
    data["cell_id"].values,
)

shap_df[
    "lst_celsius_p95"
] = y.values


shap_df.to_csv(
    OUTPUT_VALUES,
    index=False,
)


# =========================================================
# SPATIAL SHAP DATA
# =========================================================

spatial = data[
    [
        "cell_id",
        "geometry",
    ]
].copy()

for feature in FEATURES:

    spatial[
        f"shap_{feature}"
    ] = values[
        :,
        FEATURES.index(feature)
    ]


spatial[
    "shap_abs_total"
] = np.abs(values).sum(
    axis=1
)


spatial.to_file(
    OUTPUT_GPKG,
    layer="shap",
    driver="GPKG",
)


# =========================================================
# DIRECTIONAL SUMMARY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "SHAP DIRECTIONAL SUMMARY"
)

print(
    "=" * 70
)

for feature in FEATURES:

    s = values[
        :,
        FEATURES.index(feature)
    ]

    positive_pct = (
        np.mean(s > 0)
        * 100
    )

    negative_pct = (
        np.mean(s < 0)
        * 100
    )

    print(
        f"{feature:28s} "
        f"mean={s.mean():+.4f} | "
        f"positive={positive_pct:6.2f}% | "
        f"negative={negative_pct:6.2f}%"
    )


# =========================================================
# SUMMARY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "SAVED:"
)

print(
    OUTPUT_IMPORTANCE
)

print(
    OUTPUT_VALUES
)

print(
    OUTPUT_GPKG
)

print(
    "\n" + "=" * 70
)

print(
    "REDUCED SHAP ANALYSIS COMPLETE"
)

print(
    "=" * 70
)

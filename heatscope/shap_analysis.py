import sys
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


INPUT_FILE = RESULTS / "heatscope_spatial_blocks.gpkg"

IMPORTANCE_FILE = RESULTS / "shap_feature_importance.csv"
VALUES_FILE = RESULTS / "shap_values.csv"


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
print("HeatScope SHAP Explainability Analysis")
print("=" * 70)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading dataset...")

gdf = gpd.read_file(
    INPUT_FILE,
    layer="spatial_blocks",
)

valid = (
    gdf[TARGET].notna()
)

data = gdf.loc[
    valid
].copy()

X = data[FEATURES].copy()

y = data[TARGET].astype(float)


print(
    "Cells:",
    f"{len(data):,}"
)


# =========================================================
# IMPUTE
# =========================================================

print("\nPreparing features...")

imputer = SimpleImputer(
    strategy="median"
)

X_array = imputer.fit_transform(
    X
)

X_clean = pd.DataFrame(
    X_array,
    columns=FEATURES,
    index=X.index,
)


# =========================================================
# TRAIN FINAL MODEL
# =========================================================

print("\nTraining HistGradientBoosting...")

model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42,
)

model.fit(
    X_clean,
    y,
)


print("Model trained.")


# =========================================================
# SHAP
# =========================================================

print("\nCalculating SHAP values...")

try:

    import shap

except ImportError:

    raise RuntimeError(
        "SHAP is not installed. "
        "Run: pip install shap"
    )


# SHAP's generic explainer automatically selects
# an appropriate explanation method.

explainer = shap.Explainer(
    model,
    X_clean,
)

shap_result = explainer(
    X_clean
)


shap_values = np.asarray(
    shap_result.values
)


print(
    "SHAP matrix:",
    shap_values.shape
)


# =========================================================
# GLOBAL IMPORTANCE
# =========================================================

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0,
)


importance = pd.DataFrame(
    {
        "feature": FEATURES,
        "mean_abs_shap": mean_abs_shap,
    }
).sort_values(
    "mean_abs_shap",
    ascending=False,
).reset_index(
    drop=True
)


importance["rank"] = (
    np.arange(
        1,
        len(importance) + 1,
    )
)


# =========================================================
# NORMALIZED IMPORTANCE
# =========================================================

total = importance[
    "mean_abs_shap"
].sum()

importance[
    "relative_importance_pct"
] = (
    100
    * importance["mean_abs_shap"]
    / total
)


# =========================================================
# PRINT
# =========================================================

print("\n" + "=" * 70)
print("GLOBAL SHAP IMPORTANCE")
print("=" * 70)

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
    IMPORTANCE_FILE,
    index=False,
)


# =========================================================
# SAVE CELL-LEVEL SHAP VALUES
# =========================================================

shap_df = pd.DataFrame(
    shap_values,
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

shap_df["actual_lst_p95"] = (
    y.values
)

shap_df["predicted_lst_p95"] = (
    model.predict(
        X_clean
    )
)


shap_df.to_csv(
    VALUES_FILE,
    index=False,
)


# =========================================================
# ADD TOP SHAP FEATURE TO GEOSPATIAL DATA
# =========================================================

top_feature = importance.iloc[
    0
]["feature"]


top_index = FEATURES.index(
    top_feature
)


gdf.loc[
    valid,
    "shap_top_feature"
] = top_feature


gdf.loc[
    valid,
    "shap_top_value"
] = shap_values[
    :,
    top_index
]


# =========================================================
# SAVE SPATIAL SHAP DATA
# =========================================================

SHAP_GPKG = (
    RESULTS
    / "heatscope_shap.gpkg"
)

gdf.to_file(
    SHAP_GPKG,
    layer="shap",
    driver="GPKG",
)


print(
    "\nTop explanatory feature:",
    top_feature
)

print(
    "\nSaved:",
    IMPORTANCE_FILE
)

print(
    "Saved:",
    VALUES_FILE
)

print(
    "Saved:",
    SHAP_GPKG
)

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)

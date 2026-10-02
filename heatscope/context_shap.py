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


MODEL_FILE = RESULTS / "heatscope_spatial_blocks.gpkg"
CONTEXT_FILE = RESULTS / "heatscope_spatial_context.gpkg"

OUTPUT_IMPORTANCE = (
    RESULTS / "context_shap_feature_importance.csv"
)

OUTPUT_VALUES = (
    RESULTS / "context_shap_values.csv"
)

OUTPUT_GPKG = (
    RESULTS / "heatscope_context_shap.gpkg"
)

OUTPUT_CHECK = (
    RESULTS / "context_shap_additivity_check.csv"
)


TARGET = "lst_celsius_p95"


FEATURES = [
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


SHAP_SAMPLE_SIZE = 5000


print("=" * 70)
print("HeatScope Context-Enhanced SHAP Analysis")
print("=" * 70)


# =========================================================
# LOAD MODEL DATA
# =========================================================

data = gpd.read_file(
    MODEL_FILE,
    layer="spatial_blocks",
)


context = gpd.read_file(
    CONTEXT_FILE,
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
].copy()


data = data.merge(
    context,
    on="cell_id",
    how="left",
    validate="one_to_one",
)


valid = data[TARGET].notna()

data = data.loc[
    valid
].copy()


X = data[
    FEATURES
].copy()

y = data[
    TARGET
].astype(float)


print(
    "\nTotal valid cells:",
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
    "\nTraining final Context-9 "
    "HistGradientBoosting model..."
)

model.fit(
    X_clean,
    y,
)


# =========================================================
# STRATIFIED SHAP SAMPLE
# =========================================================
#
# Keep representation from the whole target range.
# This is for interpretation only; model training still
# uses all valid cells.
# =========================================================

rng = np.random.default_rng(42)

if len(X_clean) <= SHAP_SAMPLE_SIZE:

    sample_idx = np.arange(
        len(X_clean)
    )

else:

    # Five target strata.
    strata = pd.qcut(
        y,
        q=5,
        labels=False,
        duplicates="drop",
    )

    selected = []

    per_stratum = SHAP_SAMPLE_SIZE // 5

    for s in sorted(
        pd.Series(strata).dropna().unique()
    ):

        candidates = np.where(
            strata.to_numpy() == s
        )[0]

        n_select = min(
            per_stratum,
            len(candidates)
        )

        selected.extend(
            rng.choice(
                candidates,
                size=n_select,
                replace=False,
            )
        )

    # Fill any remaining places.
    selected = np.array(
        selected,
        dtype=int,
    )

    remaining = np.setdiff1d(
        np.arange(len(X_clean)),
        selected,
    )

    n_remaining = (
        SHAP_SAMPLE_SIZE
        - len(selected)
    )

    if n_remaining > 0:

        selected = np.concatenate(
            [
                selected,
                rng.choice(
                    remaining,
                    size=n_remaining,
                    replace=False,
                ),
            ]
        )

    sample_idx = selected


X_shap = X_clean.iloc[
    sample_idx
].copy()

y_shap = y.iloc[
    sample_idx
].copy()


print(
    "\nSHAP sample:",
    f"{len(X_shap):,}"
)


# =========================================================
# SHAP EXPLAINER
# =========================================================

print(
    "\nCreating SHAP explainer..."
)

explainer = shap.Explainer(
    model,
    X_clean.sample(
        n=min(
            100,
            len(X_clean)
        ),
        random_state=42,
    ),
)


print(
    "Calculating SHAP values..."
)

shap_result = explainer(
    X_shap,
    check_additivity=False,
)

values = np.asarray(
    shap_result.values
)


print(
    "SHAP matrix:",
    values.shape
)


# =========================================================
# INDEPENDENT ADDITIVITY CHECK
# =========================================================
#
# SHAP should approximately satisfy:
#
# expected_value + sum(phi) ≈ model_prediction
#
# We measure the actual discrepancy ourselves.
# =========================================================

model_predictions = model.predict(
    X_shap
)


base_value = np.asarray(
    shap_result.base_values,
    dtype=float,
)

if base_value.ndim == 0:

    reconstructed = (
        float(base_value)
        + values.sum(axis=1)
    )

else:

    reconstructed = (
        base_value
        + values.sum(axis=1)
    )


additivity_error = (
    reconstructed
    - model_predictions
)


check = pd.DataFrame(
    {
        "cell_id": data.iloc[
            sample_idx
        ]["cell_id"].values,
        "model_prediction":
            model_predictions,
        "shap_reconstruction":
            reconstructed,
        "additivity_error":
            additivity_error,
        "absolute_additivity_error":
            np.abs(additivity_error),
    }
)


check.to_csv(
    OUTPUT_CHECK,
    index=False,
)


print(
    "\n" + "=" * 70
)

print(
    "INDEPENDENT SHAP ADDITIVITY CHECK"
)

print(
    "=" * 70
)

print(
    "Mean absolute error:",
    f"{np.abs(additivity_error).mean():.6f} °C"
)

print(
    "Median absolute error:",
    f"{np.median(np.abs(additivity_error)):.6f} °C"
)

print(
    "P95 absolute error:",
    f"{np.quantile(np.abs(additivity_error), .95):.6f} °C"
)

print(
    "Maximum absolute error:",
    f"{np.max(np.abs(additivity_error)):.6f} °C"
)


# =========================================================
# GLOBAL SHAP IMPORTANCE
# =========================================================

importance = pd.DataFrame(
    {
        "feature": FEATURES,
        "mean_abs_shap":
            np.abs(values).mean(axis=0),
    }
)


importance = (
    importance
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    .reset_index(drop=True)
)


importance["rank"] = np.arange(
    1,
    len(importance) + 1
)


importance[
    "relative_importance_pct"
] = (
    importance["mean_abs_shap"]
    /
    importance["mean_abs_shap"].sum()
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


for i, feature in enumerate(
    FEATURES
):

    s = values[:, i]

    print(
        f"{feature:35s} "
        f"mean={s.mean():+.4f} | "
        f"positive="
        f"{np.mean(s > 0) * 100:6.2f}% | "
        f"negative="
        f"{np.mean(s < 0) * 100:6.2f}%"
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
        f"shap_{feature}"
        for feature in FEATURES
    ],
)


shap_df.insert(
    0,
    "cell_id",
    data.iloc[
        sample_idx
    ]["cell_id"].values,
)


shap_df[
    TARGET
] = y_shap.values


shap_df.to_csv(
    OUTPUT_VALUES,
    index=False,
)


# =========================================================
# GEOSPATIAL OUTPUT
# =========================================================

spatial = data.iloc[
    sample_idx
][
    [
        "cell_id",
        "geometry",
    ]
].copy()


for i, feature in enumerate(
    FEATURES
):

    spatial[
        f"shap_{feature}"
    ] = values[:, i]


spatial[
    "shap_abs_total"
] = np.abs(
    values
).sum(axis=1)


spatial.to_file(
    OUTPUT_GPKG,
    layer="shap",
    driver="GPKG",
)


# =========================================================
# SAVE SUMMARY
# =========================================================

print(
    "\nSaved:"
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
    OUTPUT_CHECK
)


print(
    "\n" + "=" * 70
)

print(
    "CONTEXT SHAP ANALYSIS COMPLETE"
)

print(
    "=" * 70
)

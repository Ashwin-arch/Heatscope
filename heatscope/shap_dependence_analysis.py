import sys
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS, FIGURES


MODEL_FILE = RESULTS / "heatscope_spatial_blocks.gpkg"
CONTEXT_FILE = RESULTS / "heatscope_spatial_context.gpkg"

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


print("=" * 70)
print("HeatScope SHAP Dependence Analysis")
print("=" * 70)


# =========================================================
# LOAD
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


X = data[FEATURES].copy()
y = data[TARGET].astype(float)


print(
    "\nCells:",
    f"{len(data):,}"
)


# =========================================================
# IMPUTE
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
# MODEL
# =========================================================

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


# =========================================================
# SHAP SAMPLE
# =========================================================

rng = np.random.default_rng(42)

sample_size = min(
    5000,
    len(X_clean)
)

sample_idx = rng.choice(
    len(X_clean),
    size=sample_size,
    replace=False,
)

X_sample = X_clean.iloc[
    sample_idx
].copy()


# =========================================================
# SHAP
# =========================================================

print(
    "\nCalculating SHAP..."
)

explainer = shap.Explainer(
    model,
    X_clean.sample(
        min(100, len(X_clean)),
        random_state=42,
    ),
)

shap_result = explainer(
    X_sample,
    check_additivity=False,
)

values = np.asarray(
    shap_result.values
)


# =========================================================
# NUMERIC DEPENDENCE SUMMARY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "SHAP DEPENDENCE SUMMARY"
)

print(
    "=" * 70
)


rows = []


for i, feature in enumerate(
    FEATURES
):

    feature_values = X_sample[
        feature
    ].to_numpy()

    shap_values = values[
        :, i
    ]

    q = np.quantile(
        feature_values,
        [
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ],
    )

    bins = [
        -np.inf,
        q[0],
        q[1],
        q[2],
        q[3],
        q[4],
        np.inf,
    ]

    labels = [
        "Q<5",
        "Q5-25",
        "Q25-50",
        "Q50-75",
        "Q75-95",
        "Q>95",
    ]

    bin_id = pd.cut(
        feature_values,
        bins=bins,
        labels=labels,
        duplicates="drop",
    )

    temp = pd.DataFrame(
        {
            "feature": feature,
            "bin": bin_id,
            "feature_value": feature_values,
            "shap_value": shap_values,
        }
    )

    grouped = (
        temp
        .groupby(
            ["feature", "bin"],
            observed=True,
        )
        .agg(
            n=("shap_value", "size"),
            feature_median=(
                "feature_value",
                "median",
            ),
            mean_shap=(
                "shap_value",
                "mean",
            ),
            median_shap=(
                "shap_value",
                "median",
            ),
            positive_pct=(
                "shap_value",
                lambda x:
                np.mean(x > 0) * 100,
            ),
        )
        .reset_index()
    )

    rows.append(
        grouped
    )


dependence = pd.concat(
    rows,
    ignore_index=True,
)


output_csv = (
    RESULTS
    / "shap_dependence_summary.csv"
)

dependence.to_csv(
    output_csv,
    index=False,
)


print(
    dependence.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}"
    )
)


# =========================================================
# PLOTS
# =========================================================

print(
    "\nGenerating dependence plots..."
)


for i, feature in enumerate(
    FEATURES
):

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        X_sample[feature],
        values[:, i],
        s=8,
        alpha=0.35,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        feature
    )

    plt.ylabel(
        "SHAP value (°C)"
    )

    plt.title(
        f"SHAP Dependence: {feature}"
    )

    plt.tight_layout()

    filename = (
        feature
        .replace("/", "_")
        + "_shap_dependence.png"
    )

    plt.savefig(
        FIGURES / filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


print(
    "\nSaved plots to:"
)

print(
    FIGURES
)

print(
    "Saved summary:"
)

print(
    output_csv
)


print(
    "\n" + "=" * 70
)

print(
    "SHAP DEPENDENCE ANALYSIS COMPLETE"
)

print(
    "=" * 70
)

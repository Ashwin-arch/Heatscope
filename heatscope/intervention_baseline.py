import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


FILE = RESULTS / "heatscope_hvi.gpkg"


FEATURES = [
    "hvi",
    "hvi_percentile",
    "exposure_score",
    "sensitivity_score",
    "capacity_score",
    "population",
    "population_density",
    "building_coverage_pct",
    "ndvi_mean",
    "ndmi_mean",
    "lst_celsius_p95",
]


print("=" * 70)
print("HeatScope Intervention Baseline")
print("=" * 70)


gdf = gpd.read_file(
    FILE,
    layer="hvi",
)


print(
    "\nCells:",
    f"{len(gdf):,}"
)


valid = gdf["hvi"].notna()

data = gdf.loc[
    valid
].copy()


print(
    "Valid HVI cells:",
    f"{len(data):,}"
)


print(
    "\n" + "=" * 70
)

print(
    "FEATURE DISTRIBUTIONS"
)

print(
    "=" * 70
)


rows = []


for feature in FEATURES:

    if feature not in data.columns:
        print(
            f"\nWARNING: {feature} missing"
        )
        continue


    x = pd.to_numeric(
        data[feature],
        errors="coerce",
    ).dropna()


    rows.append(
        {
            "feature": feature,
            "min": x.min(),
            "p01": x.quantile(0.01),
            "median": x.median(),
            "mean": x.mean(),
            "p99": x.quantile(0.99),
            "max": x.max(),
        }
    )


summary = pd.DataFrame(
    rows
)


print(
    summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# =========================================================
# HVI CATEGORIES
# =========================================================

if "hvi_category" in data.columns:

    print(
        "\n" + "=" * 70
    )

    print(
        "HVI CATEGORY COUNTS"
    )

    print(
        "=" * 70
    )

    print(
        data[
            "hvi_category"
        ].value_counts()
    )


# =========================================================
# HIGH-VULNERABILITY POPULATION
# =========================================================

if "hvi_category" in data.columns:

    high = data[
        data["hvi_category"].isin(
            [
                "High",
                "Very High",
            ]
        )
    ]

    print(
        "\nPopulation in High + Very High HVI:"
    )

    print(
        f"{high['population'].sum():,.0f}"
    )

    print(
        "Cells:"
        f" {len(high):,}"
    )


# =========================================================
# CORRELATION
# =========================================================

numeric = [
    f for f in FEATURES
    if f in data.columns
]


print(
    "\n" + "=" * 70
)

print(
    "HVI RELATIONSHIPS"
)

print(
    "=" * 70
)


corr = data[
    numeric
].corr(
    method="spearman"
)


for feature in [
    "lst_celsius_p95",
    "population_density",
    "ndvi_mean",
    "ndmi_mean",
    "building_coverage_pct",
]:

    if feature not in corr.columns:
        continue

    print(
        f"\nCorrelation with {feature}:"
    )

    print(
        corr[
            feature
        ].sort_values(
            ascending=False
        ).to_string(
            float_format=lambda x:
            f"{x:.4f}"
        )
    )


# =========================================================
# SAVE
# =========================================================

output = (
    RESULTS
    / "intervention_baseline_summary.csv"
)

summary.to_csv(
    output,
    index=False,
)


print(
    "\nSaved:"
)

print(
    output
)

print(
    "\n" + "=" * 70
)

print(
    "INTERVENTION BASELINE COMPLETE"
)

print(
    "=" * 70
)

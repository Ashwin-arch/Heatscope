import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


INPUT_FILE = RESULTS / "heatscope_spatial_blocks.gpkg"

OUTPUT_CORR = RESULTS / "feature_correlation_matrix.csv"
OUTPUT_MI = RESULTS / "feature_mutual_information.csv"


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
print("HeatScope Feature Redundancy Audit")
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
)


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
# SPEARMAN CORRELATION
# =========================================================

print("\n" + "=" * 70)
print("SPEARMAN FEATURE CORRELATION")
print("=" * 70)


corr = X.corr(
    method="spearman"
)


corr.to_csv(
    OUTPUT_CORR
)


print(
    corr.to_string(
        float_format=lambda x:
        f"{x:7.3f}"
    )
)


# =========================================================
# HIGH CORRELATION PAIRS
# =========================================================

print("\n" + "=" * 70)
print("HIGH-CORRELATION FEATURE PAIRS")
print("=" * 70)


pairs = []


for i in range(len(FEATURES)):

    for j in range(i + 1, len(FEATURES)):

        a = FEATURES[i]
        b = FEATURES[j]

        r = corr.loc[a, b]

        if abs(r) >= 0.80:

            pairs.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "spearman_r": r,
                    "abs_r": abs(r),
                }
            )


if pairs:

    pairs_df = pd.DataFrame(
        pairs
    ).sort_values(
        "abs_r",
        ascending=False,
    )

    print(
        pairs_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

else:

    print(
        "No feature pair has |rho| >= 0.80."
    )


# =========================================================
# MUTUAL INFORMATION
# =========================================================

print("\n" + "=" * 70)
print("MUTUAL INFORMATION WITH LST-P95")
print("=" * 70)


# Replace missing values defensively.

X_mi = X.copy()

X_mi = X_mi.fillna(
    X_mi.median()
)


mi = mutual_info_regression(
    X_mi,
    y,
    random_state=42,
)


mi_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "mutual_information": mi,
    }
).sort_values(
    "mutual_information",
    ascending=False,
)


mi_df.to_csv(
    OUTPUT_MI,
    index=False,
)


print(
    mi_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}"
    )
)


# =========================================================
# TARGET CORRELATION
# =========================================================

print("\n" + "=" * 70)
print("FEATURE ↔ TARGET CORRELATION")
print("=" * 70)


target_corr = (
    data[
        FEATURES + [TARGET]
    ]
    .corr(
        method="spearman"
    )[TARGET]
    .drop(TARGET)
    .sort_values(
        ascending=False
    )
)


print(
    target_corr.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# =========================================================
# SAVE SUMMARY
# =========================================================

summary = pd.DataFrame(
    {
        "feature": FEATURES,
        "spearman_target_r": [
            target_corr[f]
            for f in FEATURES
        ],
        "mutual_information": [
            mi_df.set_index(
                "feature"
            ).loc[
                f,
                "mutual_information"
            ]
            for f in FEATURES
        ],
    }
).sort_values(
    "mutual_information",
    ascending=False,
)


summary.to_csv(
    RESULTS / "feature_audit_summary.csv",
    index=False,
)


print("\nSaved:")
print(OUTPUT_CORR)
print(OUTPUT_MI)
print(
    RESULTS
    / "feature_audit_summary.csv"
)


print("\n" + "=" * 70)
print("FEATURE AUDIT COMPLETE")
print("=" * 70)

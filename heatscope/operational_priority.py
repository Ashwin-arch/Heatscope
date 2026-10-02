import sys
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


MODEL_FILE = (
    RESULTS / "heatscope_spatial_blocks.gpkg"
)

CONTEXT_FILE = (
    RESULTS / "heatscope_spatial_context.gpkg"
)

PREDICTION_FILE = (
    RESULTS / "context_spatial_ml_predictions.csv"
)

UNCERTAINTY_FILE = (
    RESULTS / "adaptive_uncertainty_results.csv"
)

OUTPUT_GPKG = (
    RESULTS / "heatscope_operational_priority.gpkg"
)

OUTPUT_CSV = (
    RESULTS / "heatscope_operational_priority.csv"
)


print("=" * 70)
print("HeatScope Operational Priority Layer")
print("=" * 70)


# =========================================================
# LOAD PRIMARY DATA
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


# =========================================================
# LOAD OUT-OF-FOLD PREDICTIONS
# =========================================================

pred = pd.read_csv(
    PREDICTION_FILE
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
    ]
]


# =========================================================
# LOAD ADAPTIVE UNCERTAINTY
# =========================================================

unc = pd.read_csv(
    UNCERTAINTY_FILE
)


unc = unc[
    [
        "cell_id",
        "predicted_abs_error",
        "adaptive_half_width",
        "adaptive_lower_95",
        "adaptive_upper_95",
    ]
]


# =========================================================
# MERGE
# =========================================================

data = data.merge(
    pred,
    on="cell_id",
    how="inner",
    validate="one_to_one",
)


data = data.merge(
    unc,
    on="cell_id",
    how="inner",
    validate="one_to_one",
)


print(
    "\nOperational cells:",
    f"{len(data):,}"
)


# =========================================================
# NORMALIZATION
# =========================================================

def robust_norm(series):

    x = pd.to_numeric(
        series,
        errors="coerce"
    ).astype(float)

    lo = x.quantile(0.01)
    hi = x.quantile(0.99)

    if hi <= lo:

        return pd.Series(
            0.5,
            index=series.index
        )

    x = x.clip(
        lo,
        hi
    )

    return (
        (x - lo)
        /
        (hi - lo)
    )


# =========================================================
# STRUCTURAL VULNERABILITY
# =========================================================
#
# Crucially, this does NOT use observed LST-P95.
#
# Sensitivity comes from population density.
# Capacity comes from environmental/built variables.
#
# Therefore this can be calculated before observing the
# actual thermal target.
# =========================================================

sensitivity = robust_norm(
    np.log1p(
        data[
            "population_density"
        ]
    )
)


capacity = pd.to_numeric(
    data[
        "capacity_score"
    ],
    errors="coerce"
).clip(
    0,
    1
)


structural_vulnerability = (
    sensitivity
    *
    (1 - capacity)
)


data[
    "structural_vulnerability"
] = robust_norm(
    structural_vulnerability
)


data[
    "structural_vulnerability_percentile"
] = (
    data[
        "structural_vulnerability"
    ].rank(
        pct=True,
        method="average"
    )
)


# Top 30% structural vulnerability is our operational
# equity-priority group.
data[
    "equity_priority"
] = (
    data[
        "structural_vulnerability_percentile"
    ] >= 0.70
)


# =========================================================
# PREDICTED EXPOSURE
# =========================================================

data[
    "predicted_exposure"
] = data[
    "predicted_lst_p95"
]


data[
    "uncertainty_half_width"
] = data[
    "adaptive_half_width"
]


# =========================================================
# RISK-AWARE EXPOSURE
# =========================================================
#
# lambda = 0.0  -> prediction only
# lambda = 0.5  -> moderate uncertainty awareness
# lambda = 1.0  -> conservative upper-risk estimate
# =========================================================

data[
    "risk_exposure_lambda0"
] = (
    data["predicted_lst_p95"]
)

data[
    "risk_exposure_lambda05"
] = (
    data["predicted_lst_p95"]
    +
    0.5
    *
    data["adaptive_half_width"]
)

data[
    "risk_exposure_lambda1"
] = (
    data["predicted_lst_p95"]
    +
    data["adaptive_half_width"]
)


# =========================================================
# NORMALIZED RISK
# =========================================================

for col in [
    "risk_exposure_lambda0",
    "risk_exposure_lambda05",
    "risk_exposure_lambda1",
]:

    data[
        f"{col}_norm"
    ] = robust_norm(
        data[col]
    )


# =========================================================
# OPERATIONAL PRIORITY
# =========================================================
#
# Exposure + structural vulnerability.
#
# These scores deliberately avoid using observed LST-P95.
# =========================================================

for label in [
    "lambda0",
    "lambda05",
    "lambda1",
]:

    exposure_col = (
        f"risk_exposure_{label}_norm"
    )


    priority = (
        0.60
        *
        data[
            exposure_col
        ]
        +
        0.40
        *
        data[
            "structural_vulnerability"
        ]
    )


    data[
        f"operational_priority_{label}"
    ] = priority.clip(
        0,
        1
    )


    data[
        f"priority_percentile_{label}"
    ] = (
        data[
            f"operational_priority_{label}"
        ].rank(
            pct=True,
            method="average"
        )
    )


# =========================================================
# TOP PRIORITY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP OPERATIONAL PRIORITY CELLS"
)

print(
    "=" * 70
)


for label in [
    "lambda0",
    "lambda05",
    "lambda1",
]:

    col = (
        f"operational_priority_{label}"
    )

    top = data.nlargest(
        10,
        col
    )


    print(
        f"\n{label}:"
    )


    print(
        top[
            [
                "cell_id",
                "longitude",
                "latitude",
                "predicted_lst_p95",
                "uncertainty_half_width",
                "structural_vulnerability",
                col,
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )


# =========================================================
# PRIORITY OVERLAP
# =========================================================

sets = {}

for label in [
    "lambda0",
    "lambda05",
    "lambda1",
]:

    col = (
        f"operational_priority_{label}"
    )

    top = data.nlargest(
        100,
        col
    )

    sets[label] = set(
        top["cell_id"]
    )


print(
    "\n" + "=" * 70
)

print(
    "TOP-100 PRIORITY OVERLAP"
)

print(
    "=" * 70
)


print(
    "lambda0 vs lambda05:",
    f"{len(sets['lambda0'] & sets['lambda05'])}%"
)

print(
    "lambda05 vs lambda1:",
    f"{len(sets['lambda05'] & sets['lambda1'])}%"
)

print(
    "lambda0 vs lambda1:",
    f"{len(sets['lambda0'] & sets['lambda1'])}%"
)


# =========================================================
# EQUITY GROUP
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "OPERATIONAL EQUITY GROUP"
)

print(
    "=" * 70
)

print(
    "Equity-priority cells:",
    f"{data['equity_priority'].sum():,}"
)

print(
    "Share:",
    f"{data['equity_priority'].mean() * 100:.2f}%"
)

print(
    "Population covered:",
    f"{data.loc[data['equity_priority'], 'population'].sum():,.0f}"
)


# =========================================================
# CORRELATION WITH ACTUAL TARGET
# =========================================================
#
# This is ONLY for retrospective validation.
#
# It is not used to build the operational score.
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "RETROSPECTIVE VALIDATION"
)

print(
    "=" * 70
)


for label in [
    "lambda0",
    "lambda05",
    "lambda1",
]:

    col = (
        f"operational_priority_{label}"
    )


    rho = data[
        [
            col,
            "actual_lst_p95",
        ]
    ].corr(
        method="spearman"
    ).iloc[
        0,
        1
    ]


    print(
        f"{label}: "
        f"priority vs observed LST-P95 "
        f"Spearman rho = {rho:.4f}"
    )


# =========================================================
# SAVE
# =========================================================

csv_data = data.drop(
    columns=[
        "geometry"
    ],
    errors="ignore"
)


csv_data.to_csv(
    OUTPUT_CSV,
    index=False
)


data.to_file(
    OUTPUT_GPKG,
    layer="operational_priority",
    driver="GPKG",
)


print(
    "\nSaved:"
)

print(
    OUTPUT_CSV
)

print(
    OUTPUT_GPKG
)


print(
    "\n" + "=" * 70
)

print(
    "OPERATIONAL PRIORITY COMPLETE"
)

print(
    "=" * 70
)

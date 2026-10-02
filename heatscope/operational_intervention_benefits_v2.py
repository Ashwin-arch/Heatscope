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


INPUT_FILE = (
    RESULTS / "heatscope_operational_priority.gpkg"
)

OUTPUT_GPKG = (
    RESULTS / "heatscope_operational_interventions_v2.gpkg"
)

OUTPUT_CSV = (
    RESULTS / "heatscope_operational_interventions_v2.csv"
)


INTERVENTIONS = [
    "tree",
    "shade",
    "cool_roof",
    "cooling",
]


# These remain explicit scenario assumptions.
# They are NOT measured temperature reductions.
EFFECT = {
    "tree": 1.00,
    "shade": 1.20,
    "cool_roof": 1.10,
    "cooling": 1.30,
}


print("=" * 70)
print("HeatScope Operational Intervention Model V2")
print("=" * 70)


# =========================================================
# LOAD
# =========================================================

gdf = gpd.read_file(
    INPUT_FILE,
    layer="operational_priority",
)

gdf = gdf[
    gdf["predicted_lst_p95"].notna()
].copy()

gdf = gdf.reset_index(
    drop=True
)


print(
    "\nCells:",
    f"{len(gdf):,}"
)


# =========================================================
# ROBUST NORMALIZATION
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
        / (hi - lo)
    )


# =========================================================
# COMMON NEED
# =========================================================
#
# This is calculated ONCE.
#
# We deliberately do not put population or vulnerability
# into every intervention's suitability score and then
# multiply them again later.
# =========================================================

thermal_risk = robust_norm(
    gdf[
        "risk_exposure_lambda05"
    ]
)

structural_vulnerability = (
    gdf[
        "structural_vulnerability"
    ]
    .clip(0, 1)
)


need = (
    0.60 * thermal_risk
    +
    0.40 * structural_vulnerability
).clip(
    0,
    1
)


# Population factor is kept separate and compressed.
# This provides additional benefit weighting without
# allowing population count to dominate the entire model.
population_factor = robust_norm(
    np.log1p(
        gdf["population"]
    )
)


# =========================================================
# ENVIRONMENTAL VARIABLES
# =========================================================

ndvi = robust_norm(
    gdf["ndvi_mean"]
)

ndmi = robust_norm(
    gdf["ndmi_mean"]
)

building = robust_norm(
    gdf["building_coverage_pct"]
)

population_density = robust_norm(
    np.log1p(
        gdf["population_density"]
    )
)

neighbor_ndvi = robust_norm(
    gdf["ndvi_mean_nbr_mean"]
)

neighbor_ndmi = robust_norm(
    gdf["ndmi_mean_nbr_mean"]
)

neighbor_population = robust_norm(
    np.log1p(
        gdf[
            "population_density_nbr_mean"
        ]
    )
)

neighbor_building = robust_norm(
    gdf[
        "building_coverage_pct_nbr_mean"
    ]
)


# =========================================================
# INTERVENTION OPPORTUNITIES
# =========================================================
#
# NOTE:
# These contain only intervention-specific physical
# opportunity factors.
#
# They do NOT contain the common need variables.
# =========================================================


# ---------------------------------------------------------
# TREE
# ---------------------------------------------------------
#
# Opportunity increases when vegetation/moisture are low.
# ---------------------------------------------------------

tree_opportunity = (
    0.40 * (1 - ndvi)
    +
    0.25 * (1 - neighbor_ndvi)
    +
    0.20 * (1 - ndmi)
    +
    0.15 * (1 - neighbor_ndmi)
)


# ---------------------------------------------------------
# SHADE
# ---------------------------------------------------------
#
# Opportunity increases with people and limited vegetation.
# ---------------------------------------------------------

shade_opportunity = (
    0.35 * population_density
    +
    0.20 * neighbor_population
    +
    0.25 * (1 - ndvi)
    +
    0.20 * (1 - neighbor_ndvi)
)


# ---------------------------------------------------------
# COOL ROOF
# ---------------------------------------------------------
#
# This intervention specifically responds to built intensity.
# ---------------------------------------------------------

cool_roof_opportunity = (
    0.55 * building
    +
    0.45 * neighbor_building
)


# ---------------------------------------------------------
# COOLING / WATER
# ---------------------------------------------------------

cooling_opportunity = (
    0.40 * (1 - ndmi)
    +
    0.25 * (1 - neighbor_ndmi)
    +
    0.20 * population_density
    +
    0.15 * neighbor_population
)


opportunity = {
    "tree":
        tree_opportunity.clip(0, 1),

    "shade":
        shade_opportunity.clip(0, 1),

    "cool_roof":
        cool_roof_opportunity.clip(0, 1),

    "cooling":
        cooling_opportunity.clip(0, 1),
}


# =========================================================
# BENEFIT
# =========================================================
#
# Need × intervention opportunity × effect × population
# weighting.
#
# Each component has a distinct role.
# =========================================================

for intervention in INTERVENTIONS:

    gdf[
        f"{intervention}_opportunity"
    ] = opportunity[
        intervention
    ]

    gdf[
        f"{intervention}_operational_benefit"
    ] = (
        need
        *
        opportunity[
            intervention
        ]
        *
        EFFECT[
            intervention
        ]
        *
        population_factor
    )


# =========================================================
# BEST INTERVENTION
# =========================================================

benefit_columns = [
    f"{i}_operational_benefit"
    for i in INTERVENTIONS
]


benefit_matrix = gdf[
    benefit_columns
]


gdf[
    "best_operational_intervention_v2"
] = (
    benefit_matrix
    .idxmax(axis=1)
    .str.replace(
        "_operational_benefit",
        "",
        regex=False,
    )
)


gdf[
    "best_operational_benefit_v2"
] = (
    benefit_matrix.max(
        axis=1
    )
)


# =========================================================
# WINNING MARGIN
# =========================================================
#
# Difference between the best and second-best intervention.
#
# A large margin means the recommendation is clearer.
# =========================================================

sorted_benefits = np.sort(
    benefit_matrix.to_numpy(),
    axis=1
)

gdf[
    "intervention_margin_v2"
] = (
    sorted_benefits[:, -1]
    -
    sorted_benefits[:, -2]
)


# =========================================================
# WINNING RATIO
# =========================================================

gdf[
    "intervention_confidence_v2"
] = (
    gdf[
        "intervention_margin_v2"
    ]
    /
    (
        gdf[
            "best_operational_benefit_v2"
        ]
        + 1e-9
    )
)


# =========================================================
# PRIORITY
# =========================================================

gdf[
    "operational_benefit_percentile_v2"
] = (
    gdf[
        "best_operational_benefit_v2"
    ].rank(
        pct=True,
        method="average",
    )
)


# =========================================================
# SUMMARY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "COMMON NEED"
)

print(
    "=" * 70
)

print(
    f"Median: "
    f"{need.median():.4f}"
)

print(
    f"P90: "
    f"{need.quantile(.90):.4f}"
)

print(
    f"Maximum: "
    f"{need.max():.4f}"
)


print(
    "\n" + "=" * 70
)

print(
    "INTERVENTION OPPORTUNITY"
)

print(
    "=" * 70
)


for intervention in INTERVENTIONS:

    x = opportunity[
        intervention
    ]

    print(
        f"{intervention:12s} "
        f"median={x.median():.4f} "
        f"P90={x.quantile(.90):.4f} "
        f"max={x.max():.4f}"
    )


# =========================================================
# BEST INTERVENTION COUNTS
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "BEST OPERATIONAL INTERVENTION V2"
)

print(
    "=" * 70
)

print(
    gdf[
        "best_operational_intervention_v2"
    ].value_counts()
)


# =========================================================
# TOP 10%
# =========================================================

top10 = gdf[
    gdf[
        "operational_benefit_percentile_v2"
    ] >= 0.90
]


print(
    "\nTop 10% cells:",
    f"{len(top10):,}"
)

print(
    "Population:",
    f"{top10['population'].sum():,.0f}"
)


print(
    "\nTop-10% intervention mix:"
)

print(
    top10[
        "best_operational_intervention_v2"
    ].value_counts()
)


# =========================================================
# INTERVENTION CONFIDENCE
# =========================================================

confidence = gdf[
    "intervention_confidence_v2"
]


print(
    "\n" + "=" * 70
)

print(
    "INTERVENTION CONFIDENCE"
)

print(
    "=" * 70
)

print(
    f"Median: "
    f"{confidence.median():.4f}"
)

print(
    f"P25: "
    f"{confidence.quantile(.25):.4f}"
)

print(
    f"P75: "
    f"{confidence.quantile(.75):.4f}"
)

print(
    f"P90: "
    f"{confidence.quantile(.90):.4f}"
)


# =========================================================
# TOP 10 CELLS
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP 10 OPERATIONAL INTERVENTION CELLS V2"
)

print(
    "=" * 70
)


top = gdf.nlargest(
    10,
    "best_operational_benefit_v2"
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
            "best_operational_intervention_v2",
            "best_operational_benefit_v2",
            "intervention_confidence_v2",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}",
    )
)


# =========================================================
# SAVE
# =========================================================

csv = gdf.drop(
    columns=["geometry"],
    errors="ignore",
)


csv.to_csv(
    OUTPUT_CSV,
    index=False,
)


gdf.to_file(
    OUTPUT_GPKG,
    layer="operational_interventions_v2",
    driver="GPKG",
)


print(
    "\nSaved:"
)

print(
    OUTPUT_GPKG
)

print(
    OUTPUT_CSV
)


print(
    "\n" + "=" * 70
)

print(
    "OPERATIONAL INTERVENTION MODEL V2 COMPLETE"
)

print(
    "=" * 70
)

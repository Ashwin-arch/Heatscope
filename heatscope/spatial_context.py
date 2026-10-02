import sys
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


INPUT_FILE = RESULTS / "heatscope_spatial_blocks.gpkg"

OUTPUT_GPKG = RESULTS / "heatscope_spatial_context.gpkg"
OUTPUT_CSV = RESULTS / "heatscope_spatial_context.csv"

# Independent predictors whose neighborhood context will be calculated.
CONTEXT_FEATURES = [
    "ndvi_mean",
    "ndmi_mean",
    "population_density",
    "building_coverage_pct",
]


print("=" * 70)
print("HeatScope Polygon-Adjacency Spatial Context")
print("=" * 70)


# =========================================================
# LOAD
# =========================================================

gdf = gpd.read_file(
    INPUT_FILE,
    layer="spatial_blocks",
)

print("\nCells:", f"{len(gdf):,}")
print("CRS:", gdf.crs)


# =========================================================
# PREPARE GEOMETRIES
# =========================================================

gdf = gdf.reset_index(drop=True)

geometries = list(gdf.geometry)

tree = STRtree(
    geometries
)


# =========================================================
# BUILD QUEEN ADJACENCY
# =========================================================
#
# STRtree query returns candidate geometries whose
# bounding boxes intersect. We then test actual geometric
# intersection after a tiny tolerance buffer.
#
# A cell is considered a neighbor when it touches another
# polygon or overlaps because of boundary geometry.
# =========================================================

print("\nBuilding polygon adjacency...")

neighbors = [
    set()
    for _ in range(len(gdf))
]


for i, geom in enumerate(geometries):

    if geom is None or geom.is_empty:
        continue

    # Candidate neighbors from spatial index.
    candidates = tree.query(
        geom
    )

    for j in candidates:

        j = int(j)

        if j == i:
            continue

        other = geometries[j]

        if other is None or other.is_empty:
            continue

        # Actual topological relationship.
        if geom.touches(other) or geom.intersects(other):

            neighbors[i].add(j)


neighbor_counts = np.array(
    [len(x) for x in neighbors]
)


print(
    "Neighbor count:"
)

print(
    f"min    : {neighbor_counts.min()}"
)

print(
    f"median : {np.median(neighbor_counts):.0f}"
)

print(
    f"mean   : {neighbor_counts.mean():.2f}"
)

print(
    f"max    : {neighbor_counts.max()}"
)

print(
    "Cells with zero neighbors:",
    int((neighbor_counts == 0).sum())
)


# =========================================================
# CALCULATE CONTEXT FEATURES
# =========================================================

for feature in CONTEXT_FEATURES:

    print(
        f"\nProcessing:",
        feature
    )

    values = pd.to_numeric(
        gdf[feature],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    neighbor_mean = np.full(
        len(gdf),
        np.nan,
        dtype=float
    )

    neighbor_std = np.full(
        len(gdf),
        np.nan,
        dtype=float
    )

    for i, idxs in enumerate(neighbors):

        if not idxs:
            continue

        idxs = np.fromiter(
            idxs,
            dtype=np.int64
        )

        vals = values[idxs]

        vals = vals[
            np.isfinite(vals)
        ]

        if len(vals) == 0:
            continue

        neighbor_mean[i] = np.mean(
            vals
        )

        if len(vals) > 1:
            neighbor_std[i] = np.std(
                vals,
                ddof=0
            )

    gdf[
        f"{feature}_nbr_mean"
    ] = neighbor_mean

    gdf[
        f"{feature}_local_contrast"
    ] = (
        values
        - neighbor_mean
    )

    gdf[
        f"{feature}_nbr_std"
    ] = neighbor_std


# =========================================================
# SAVE
# =========================================================

attribute_columns = [
    "cell_id"
]

for feature in CONTEXT_FEATURES:

    attribute_columns.extend(
        [
            f"{feature}_nbr_mean",
            f"{feature}_local_contrast",
            f"{feature}_nbr_std",
        ]
    )


out_csv = gdf[
    attribute_columns
].copy()


out_csv.to_csv(
    OUTPUT_CSV,
    index=False
)


gdf.to_file(
    OUTPUT_GPKG,
    layer="spatial_context",
    driver="GPKG",
)


# =========================================================
# SUMMARY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "SPATIAL CONTEXT SUMMARY"
)

print(
    "=" * 70
)

for feature in CONTEXT_FEATURES:

    col = (
        f"{feature}_local_contrast"
    )

    values = gdf[col].dropna()

    print(
        f"\n{feature}"
    )

    print(
        "  contrast median:",
        f"{values.median():.6f}"
    )

    print(
        "  contrast P05:",
        f"{values.quantile(.05):.6f}"
    )

    print(
        "  contrast P95:",
        f"{values.quantile(.95):.6f}"
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
    "SPATIAL CONTEXT COMPLETE"
)

print(
    "=" * 70
)

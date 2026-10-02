import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


FILE = RESULTS / "heatscope_operational_interventions_v2.gpkg"


INTERVENTIONS = [
    "tree",
    "shade",
    "cool_roof",
    "cooling",
]


print("=" * 70)
print("HeatScope Intervention Distribution Audit")
print("=" * 70)


gdf = gpd.read_file(
    FILE,
    layer="operational_interventions_v2",
)


print(
    "\nCells:",
    f"{len(gdf):,}"
)


for intervention in INTERVENTIONS:

    col = (
        f"{intervention}_opportunity"
    )

    x = pd.to_numeric(
        gdf[col],
        errors="coerce"
    ).dropna()


    print(
        "\n" + "-" * 70
    )

    print(
        intervention.upper()
    )

    print(
        f"min    = {x.min():.4f}"
    )

    print(
        f"P05    = {x.quantile(.05):.4f}"
    )

    print(
        f"P25    = {x.quantile(.25):.4f}"
    )

    print(
        f"median = {x.median():.4f}"
    )

    print(
        f"P75    = {x.quantile(.75):.4f}"
    )

    print(
        f"P90    = {x.quantile(.90):.4f}"
    )

    print(
        f"P95    = {x.quantile(.95):.4f}"
    )

    print(
        f"max    = {x.max():.4f}"
    )


print(
    "\n" + "=" * 70
)

print(
    "CROSS-INTERVENTION CORRELATION"
)

print(
    "=" * 70
)


opportunity_cols = [
    f"{i}_opportunity"
    for i in INTERVENTIONS
]


corr = gdf[
    opportunity_cols
].corr(
    method="spearman"
)


print(
    corr.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)


print(
    "\n" + "=" * 70
)

print(
    "BEST OPPORTUNITY VALUE DISTRIBUTION"
)

print(
    "=" * 70
)


opportunities = gdf[
    opportunity_cols
].to_numpy()


best = opportunities.argmax(
    axis=1
)


for j, intervention in enumerate(
    INTERVENTIONS
):

    share = (
        np.mean(best == j)
        * 100
    )

    print(
        f"{intervention:12s}: "
        f"{share:.2f}%"
    )


print(
    "\n" + "=" * 70
)

print(
    "DISTRIBUTION AUDIT COMPLETE"
)

print(
    "=" * 70
)

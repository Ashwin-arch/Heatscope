import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask

from config import DATA_RAW, RESULTS


WORLDPOP = DATA_RAW / "worldpop_india_2025.tif"
GRID_FILE = RESULTS / "heatscope_grid.gpkg"


print("=" * 70)
print("HeatScope — WorldPop Population Aggregation")
print("=" * 70)


# =========================================================
# Load HeatScope grid
# =========================================================

print("\nLoading HeatScope grid...")

grid = gpd.read_file(
    GRID_FILE,
    layer="grid"
)

print("Grid cells:", len(grid))
print("Grid CRS:", grid.crs)


# =========================================================
# Open WorldPop
# =========================================================

print("\nOpening WorldPop...")

with rasterio.open(WORLDPOP) as src:

    print("WorldPop CRS:", src.crs)
    print("Resolution:", src.res)
    print("NoData:", src.nodata)

    # Transform grid to WorldPop CRS
    grid_wgs84 = grid.to_crs(src.crs)

    # -----------------------------------------------------
    # Process each grid cell
    # -----------------------------------------------------

    population = np.zeros(
        len(grid_wgs84),
        dtype=np.float64
    )

    covered_area = np.zeros(
        len(grid_wgs84),
        dtype=np.float64
    )

    print("\nAggregating population...")

    total = len(grid_wgs84)

    for idx, geometry in enumerate(
        grid_wgs84.geometry
    ):

        try:

            data, transform = mask(
                src,
                [geometry],
                crop=True,
                filled=True,
                nodata=src.nodata,
                all_touched=False,
            )

            values = data[0]

            valid = (
                np.isfinite(values)
                & (values != src.nodata)
                & (values >= 0)
            )

            if valid.any():

                population[idx] = (
                    values[valid].sum()
                )

            else:
                population[idx] = 0.0

        except Exception:
            population[idx] = 0.0

        if (
            (idx + 1) % 500 == 0
            or idx == total - 1
        ):

            print(
                f"\rProcessed "
                f"{idx + 1:,}/{total:,}",
                end=""
            )


print("\n\nAggregation complete.")


# =========================================================
# Add population variables
# =========================================================

grid["population"] = population

grid["population_density_km2"] = (
    grid["population"]
    / (grid["area_m2"] / 1_000_000)
)


# =========================================================
# Population statistics
# =========================================================

print("\n" + "-" * 70)
print("POPULATION QUALITY CHECK")
print("-" * 70)

print(
    "Total population:",
    f"{grid.population.sum():,.0f}"
)

print(
    "Mean population/cell:",
    f"{grid.population.mean():,.2f}"
)

print(
    "Median population/cell:",
    f"{grid.population.median():,.2f}"
)

print(
    "Maximum population/cell:",
    f"{grid.population.max():,.2f}"
)

print(
    "Population density mean:",
    f"{grid.population_density_km2.mean():,.2f}",
    "people/km²"
)

print(
    "Population density max:",
    f"{grid.population_density_km2.max():,.2f}",
    "people/km²"
)


# =========================================================
# Save updated grid
# =========================================================

output = RESULTS / "heatscope_population.gpkg"

if output.exists():
    output.unlink()

grid.to_file(
    output,
    layer="population",
    driver="GPKG",
)


csv_output = RESULTS / "heatscope_population.csv"

grid.drop(
    columns="geometry"
).to_csv(
    csv_output,
    index=False
)


print("\nSaved:")
print(" ", output)
print(" ", csv_output)

print("\n" + "=" * 70)
print("POPULATION AGGREGATION COMPLETE")
print("=" * 70)

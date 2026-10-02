import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS, DATA_PROCESSED


GRID_FILE = RESULTS / "heatscope_grid.gpkg"

OUTPUT_FILE = RESULTS / "heatscope_landsat_features.gpkg"
CSV_FILE = RESULTS / "heatscope_landsat_features.csv"


RASTERS = {
    "lst_celsius": DATA_PROCESSED / "lst_celsius.tif",
    "ndvi": DATA_PROCESSED / "ndvi.tif",
    "ndbi": DATA_PROCESSED / "ndbi.tif",
    "ndmi": DATA_PROCESSED / "ndmi.tif",
}


print("=" * 70)
print("HeatScope Landsat -> 250 m Feature Aggregation")
print("=" * 70)


# =========================================================
# LOAD GRID
# =========================================================

print("\nLoading HeatScope grid...")

grid = gpd.read_file(
    GRID_FILE,
    layer="grid",
)

print("Grid cells:", f"{len(grid):,}")
print("Grid CRS:", grid.crs)


# =========================================================
# OPEN RASTERS
# =========================================================

print("\nOpening rasters...")

sources = {}

for name, path in RASTERS.items():

    src = rasterio.open(path)
    sources[name] = src

    print(
        f"{name:15s} "
        f"{src.width} x {src.height} | "
        f"{src.res} | "
        f"{src.crs}"
    )


# =========================================================
# VERIFY ALIGNMENT
# =========================================================

reference = sources["lst_celsius"]

reference_crs = reference.crs
reference_transform = reference.transform
reference_shape = (
    reference.height,
    reference.width,
)
reference_bounds = reference.bounds


for name, src in sources.items():

    if src.crs != reference_crs:
        raise RuntimeError(
            f"{name}: CRS mismatch"
        )

    if (
        src.height,
        src.width,
    ) != reference_shape:

        raise RuntimeError(
            f"{name}: raster shape mismatch"
        )

    if src.transform != reference_transform:
        raise RuntimeError(
            f"{name}: transform mismatch"
        )


print("\nAll rasters aligned.")


# =========================================================
# REPROJECT GRID IF NECESSARY
# =========================================================

if grid.crs != reference_crs:

    print(
        "\nGrid CRS differs from raster CRS."
        " Reprojecting..."
    )

    grid = grid.to_crs(reference_crs)


# =========================================================
# RESULT ARRAYS
# =========================================================

n_cells = len(grid)

metrics = {}

for variable in RASTERS:

    metrics[f"{variable}_mean"] = np.full(
        n_cells,
        np.nan,
        dtype=np.float32,
    )

    metrics[f"{variable}_median"] = np.full(
        n_cells,
        np.nan,
        dtype=np.float32,
    )


# LST upper-tail metrics
metrics["lst_celsius_p90"] = np.full(
    n_cells,
    np.nan,
    dtype=np.float32,
)

metrics["lst_celsius_p95"] = np.full(
    n_cells,
    np.nan,
    dtype=np.float32,
)


# =========================================================
# SAFE WINDOW FUNCTION
# =========================================================

def safe_window(geom, src):
    """
    Convert geometry bounds to a valid raster window.

    Uses floor/ceil explicitly so that partially covered
    edge cells do not become zero-sized windows.
    """

    raw = from_bounds(
        geom.bounds[0],
        geom.bounds[1],
        geom.bounds[2],
        geom.bounds[3],
        transform=src.transform,
    )

    col_start = int(np.floor(raw.col_off))
    row_start = int(np.floor(raw.row_off))

    col_stop = int(
        np.ceil(
            raw.col_off + raw.width
        )
    )

    row_stop = int(
        np.ceil(
            raw.row_off + raw.height
        )
    )

    # Clip against raster extent
    col_start = max(
        0,
        min(src.width, col_start),
    )

    row_start = max(
        0,
        min(src.height, row_start),
    )

    col_stop = max(
        0,
        min(src.width, col_stop),
    )

    row_stop = max(
        0,
        min(src.height, row_stop),
    )

    width = col_stop - col_start
    height = row_stop - row_start

    if width <= 0 or height <= 0:
        return None

    return Window(
        col_start,
        row_start,
        width,
        height,
    )


# =========================================================
# ZONAL STATISTICS
# =========================================================

print("\nCalculating zonal statistics...")
print()


empty_cells = 0
valid_cells = 0


for idx, geom in enumerate(grid.geometry):

    if geom is None or geom.is_empty:
        empty_cells += 1
        continue


    # -----------------------------------------------------
    # Get raster window
    # -----------------------------------------------------

    window = safe_window(
        geom,
        reference,
    )

    if window is None:

        empty_cells += 1
        continue


    # -----------------------------------------------------
    # Window transform
    # -----------------------------------------------------

    transform = reference.window_transform(
        window
    )


    # -----------------------------------------------------
    # Exact cell geometry mask
    # -----------------------------------------------------

    mask = geometry_mask(
        [geom],
        out_shape=(
            int(window.height),
            int(window.width),
        ),
        transform=transform,
        invert=True,
    )


    cell_had_data = False


    # -----------------------------------------------------
    # Read each raster
    # -----------------------------------------------------

    for variable, src in sources.items():

        data = src.read(
            1,
            window=window,
            masked=False,
        ).astype(
            np.float32,
            copy=False,
        )


        valid = (
            mask
            & np.isfinite(data)
        )


        values = data[valid]


        if values.size == 0:
            continue


        cell_had_data = True


        # -------------------------------------------------
        # Mean
        # -------------------------------------------------

        metrics[
            f"{variable}_mean"
        ][idx] = np.mean(values)


        # -------------------------------------------------
        # Median
        # -------------------------------------------------

        metrics[
            f"{variable}_median"
        ][idx] = np.median(values)


        # -------------------------------------------------
        # LST tail statistics
        # -------------------------------------------------

        if variable == "lst_celsius":

            metrics[
                "lst_celsius_p90"
            ][idx] = np.percentile(
                values,
                90,
            )

            metrics[
                "lst_celsius_p95"
            ][idx] = np.percentile(
                values,
                95,
            )


    if cell_had_data:
        valid_cells += 1


    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if (idx + 1) % 500 == 0:

        print(
            f"\rProcessed: "
            f"{idx + 1:,} / {n_cells:,}",
            end="",
            flush=True,
        )


print()


# =========================================================
# CLOSE RASTERS
# =========================================================

for src in sources.values():
    src.close()


# =========================================================
# ADD FEATURES
# =========================================================

print("\nAdding features to grid...")

for name, values in metrics.items():

    grid[name] = values


# =========================================================
# SANITY CHECKS
# =========================================================

print("\n" + "=" * 70)
print("Feature Statistics")
print("=" * 70)

for name in metrics:

    values = grid[name].dropna()

    if len(values) == 0:

        print(
            f"{name:25s}: NO VALID DATA"
        )

        continue


    print(
        f"{name:25s} "
        f"min={values.min():8.3f} "
        f"median={values.median():8.3f} "
        f"mean={values.mean():8.3f} "
        f"max={values.max():8.3f}"
    )


# =========================================================
# MISSING VALUES
# =========================================================

print("\n" + "=" * 70)
print("Missing Values")
print("=" * 70)

for name in metrics:

    print(
        f"{name:25s}: "
        f"{grid[name].isna().sum():,}"
    )


print("\nValid cells:", f"{valid_cells:,}")
print("Empty/no-data cells:", f"{empty_cells:,}")


# =========================================================
# SAVE
# =========================================================

print("\nSaving GeoPackage...")

grid.to_file(
    OUTPUT_FILE,
    layer="landsat_features",
    driver="GPKG",
)

print("Saved:", OUTPUT_FILE)


print("\nSaving CSV...")

grid.drop(
    columns="geometry"
).to_csv(
    CSV_FILE,
    index=False,
)

print("Saved:", CSV_FILE)


print("\nDONE.")

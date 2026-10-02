from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Bengaluru study area
BBOX = [
    77.45,
    12.82,
    77.78,
    13.16,
]

CITY_LAT = 12.9716
CITY_LON = 77.5946

DAYS_BACK = 180
MAX_CLOUD = 25

# Analysis grid
GRID_SIZE_METERS = 250

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for p in [
    DATA_RAW,
    DATA_PROCESSED,
    MODELS,
    RESULTS,
    FIGURES,
]:
    p.mkdir(parents=True, exist_ok=True)

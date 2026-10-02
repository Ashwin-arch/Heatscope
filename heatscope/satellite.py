import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timedelta, timezone
import csv
import requests

import planetary_computer
from pystac_client import Client

from config import BBOX, DATA_RAW, DAYS_BACK, MAX_CLOUD


CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def download_asset(asset, output_path):
    signed = planetary_computer.sign(asset)

    response = requests.get(
        signed.href,
        stream=True,
        timeout=300,
    )
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)


print("=" * 65)
print("HeatScope — Real Landsat Data Acquisition")
print("=" * 65)

end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=DAYS_BACK)

print("\nAOI:", BBOX)
print("Date range:", start_date.date(), "to", end_date.date())
print("Maximum cloud:", MAX_CLOUD, "%")

catalog = Client.open(
    CATALOG_URL,
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=BBOX,
    datetime=(
        start_date.isoformat(),
        end_date.isoformat(),
    ),
    query={
        "eo:cloud_cover": {
            "lte": MAX_CLOUD
        }
    },
)

items = list(search.items())

print("\nScenes found:", len(items))

if not items:
    raise RuntimeError("No suitable Landsat scenes found.")

items.sort(
    key=lambda x: x.properties.get(
        "eo:cloud_cover",
        100
    )
)

item = items[0]

cloud = item.properties.get(
    "eo:cloud_cover",
    None
)

print("\nSelected scene:")
print("ID:", item.id)
print("Date:", item.datetime)
print("Cloud:", cloud, "%")

required = {
    "red": "red",
    "nir": "nir08",
    "swir": "swir16",
    "lst_raw": "lwir11",
    "qa": "qa_pixel",
}

print("\nDownloading:")

metadata = []

for output_name, asset_name in required.items():

    if asset_name not in item.assets:
        raise RuntimeError(
            f"Missing required asset: {asset_name}"
        )

    output = DATA_RAW / f"{output_name}.tif"

    print(f"  {asset_name:10s} -> {output.name}")

    download_asset(
        item.assets[asset_name],
        output,
    )

    metadata.append({
        "scene_id": item.id,
        "scene_datetime": item.datetime,
        "cloud_cover": cloud,
        "asset": asset_name,
        "output": output.name,
    })

metadata_file = DATA_RAW / "scene_metadata.csv"

with open(
    metadata_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=metadata[0].keys()
    )

    writer.writeheader()
    writer.writerows(metadata)

print("\nSaved metadata:", metadata_file)

print("\nRaw files:")

for p in sorted(DATA_RAW.glob("*")):
    if p.is_file():
        print(
            f"  {p.name:20s}"
            f"{p.stat().st_size / 1024 / 1024:8.2f} MB"
        )

print("\n" + "=" * 65)
print("SATELLITE ACQUISITION COMPLETE")
print("=" * 65)

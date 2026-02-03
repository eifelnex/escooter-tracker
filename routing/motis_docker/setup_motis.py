"""
Setup script for local MOTIS instance.

Downloads:
1. GTFS data for Germany (DELFI) and Switzerland
2. OSM data for Baden-Württemberg region

Then creates MOTIS config and imports data.
"""

import os
import subprocess
import urllib.request
from pathlib import Path
import shutil

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# GTFS Sources
GTFS_SOURCES = {
    # Germany - DELFI (all German public transport)
    "germany": "https://download.gtfs.de/germany/free/latest.zip",

    # Switzerland - official feed
    "switzerland": "https://opentransportdata.swiss/en/dataset/timetable-2025-gtfs2020/permalink",
}

# OSM - Baden-Württemberg + surrounding areas
OSM_URL = "https://download.geofabrik.de/europe/germany/baden-wuerttemberg-latest.osm.pbf"


def download_file(url: str, dest: Path, desc: str):
    """Download file with progress."""
    if dest.exists():
        print(f"  {desc}: Already exists, skipping")
        return

    print(f"  {desc}: Downloading from {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  {desc}: Done ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"  {desc}: Failed - {e}")


def create_config():
    """Create MOTIS config.yml"""
    config = """# MOTIS Configuration
server:
  host: "0.0.0.0"
  port: 8080
  web_folder: "ui"

import:
  paths:
    - germany.gtfs.zip
    - switzerland.gtfs.zip
  osm: baden-wuerttemberg.osm.pbf
"""
    config_path = DATA_DIR / "config.yml"
    config_path.write_text(config)
    print(f"Created config at {config_path}")


def main():
    print("=" * 60)
    print("MOTIS Local Setup")
    print("=" * 60)

    # Download GTFS
    print("\n1. Downloading GTFS data...")
    download_file(
        GTFS_SOURCES["germany"],
        DATA_DIR / "germany.gtfs.zip",
        "Germany GTFS"
    )

    # Switzerland GTFS
    print("\n   Note: For Switzerland, download manually from:")
    print("   https://opentransportdata.swiss/en/dataset/timetable-2025-gtfs2020")
    print("   Save as: data/switzerland.gtfs.zip")

    # Download OSM
    print("\n2. Downloading OSM data...")
    download_file(
        OSM_URL,
        DATA_DIR / "baden-wuerttemberg.osm.pbf",
        "Baden-Württemberg OSM"
    )

    # Create config
    print("\n3. Creating MOTIS config...")
    create_config()

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("""
Next steps:

1. If you need Switzerland data, download manually:
   https://opentransportdata.swiss/en/dataset/timetable-2025-gtfs2020
   Save to: routing/motis_docker/data/switzerland.gtfs.zip

2. Import data (run once, takes ~10-30 min):
   cd routing/motis_docker
   docker compose run motis motis import

3. Start the server:
   docker compose up -d

4. Test: http://localhost:8080

5. Update the Python client to use localhost:8080
""")


if __name__ == "__main__":
    main()

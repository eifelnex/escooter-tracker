# Routing Module

This module provides routing capabilities for comparing e-scooter trips with public transport alternatives.

## Components

- **motis.py** - Python client for querying MOTIS public transport routes
- **valhalla.py** - Valhalla routing client for road/path routing
- **motis_docker/** - Local MOTIS server setup

## MOTIS Setup

MOTIS is a multi-modal routing engine for public transport. Follow these steps to set it up on a new machine.

### Prerequisites

- Docker installed and running
- Python 3.x

### Setup Steps

1. **Download GTFS and OSM data**
   ```bash
   cd routing/motis_docker
   python setup_motis.py
   ```

2. **Manually download Switzerland GTFS** (optional but recommended):
   - Go to: https://opentransportdata.swiss/en/dataset/timetable-2025-gtfs2020
   - Save as: `routing/motis_docker/data/switzerland.gtfs.zip`

3. **Download DACH OSM data** (if using full config):
   - Download from: https://download.geofabrik.de/europe/dach-latest.osm.pbf
   - Save as: `routing/motis_docker/data/dach.osm.pbf`

4. **Import the data** (takes 10-30+ minutes):
   ```bash
   cd routing/motis_docker
   docker compose run motis motis import
   ```

5. **Start the server**:
   ```bash
   docker compose up -d
   ```

6. **Verify** at http://localhost:8080

### Docker Commands

```bash
# Start server
docker compose up -d

# Stop server
docker compose down

# View logs
docker compose logs -f

# Re-import after data update
docker compose run motis motis import
```

### Configuration

The MOTIS configuration is in `motis_docker/data/config.yml`:

```yaml
osm: dach.osm.pbf
timetable:
  first_day: TODAY
  num_days: 30
  datasets:
    germany:
      path: germany.gtfs.zip
    switzerland:
      path: switzerland.gtfs.zip
osr_footpath: true
street_routing: true
```

### Data Sources

| Data | Source | Update Frequency |
|------|--------|------------------|
| Germany GTFS | https://download.gtfs.de/germany/free/latest.zip | Weekly |
| Switzerland GTFS | https://opentransportdata.swiss | Yearly |
| DACH OSM | https://download.geofabrik.de/europe/dach-latest.osm.pbf | Daily |

### Usage in Python

```python
from routing.motis import get_pt_route, batch_pt_routes

# Single route query
result = get_pt_route(
    start_lat=48.7758,
    start_lon=9.1829,
    end_lat=48.7845,
    end_lon=9.1872,
    timestamp="2024-01-15 08:30:00"
)

# Batch processing with checkpointing
results = batch_pt_routes(trips_df, checkpoint_path="checkpoints/pt_routes.parquet")
```

### Troubleshooting

- **Port 8080 in use**: Change the port mapping in `docker-compose.yml`
- **Import fails**: Ensure you have enough disk space (~10GB) and RAM (~8GB)
- **No routes found**: Check that your query date falls within the timetable window (30 days from import date)
- **GTFS expired**: Re-download GTFS data and re-run import

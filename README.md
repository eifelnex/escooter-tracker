# E-Scooter GBFS Tracker

A comprehensive data analysis and research project for tracking e-scooter usage patterns in the Baden-Württemberg region of Germany, Switzerland, and surrounding areas. The project combines real-time GBFS (General Bikeshare Feed Specification) data collection with spatial analysis, public transport comparison, and business case modeling.

## Project Overview

- **Geographic Coverage**: Baden-Württemberg (Germany), Switzerland (Zurich, Basel, Bern, St. Gallen), and other German cities
- **Data Providers**: Dott, VOI, Bolt, Zeus, Lime, Hopp, Yoio
- **Collection Period**: October 2025 - ongoing

---

## Main Dataset: `vehicle_events_export.parquet`

The primary dataset containing all recorded e-scooter events.

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total Events | 16,476,198 |
| Unique Vehicles | 2,730,733 |
| Date Range | 2025-10-25 to 2026-01-18 |
| File Size | ~1.1 GB |

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | int64 | Unique event identifier |
| `vehicle_id` | string | Vehicle identifier (rotates after rides for privacy) |
| `timestamp` | datetime | Event timestamp |
| `provider` | string | Service provider (e.g., dott_stuttgart, voi_de, bolt_zurich) |
| `lat` | double | Latitude coordinate |
| `lon` | double | Longitude coordinate |
| `is_reserved` | int64 | Whether vehicle is currently reserved |
| `is_disabled` | int64 | Whether vehicle is disabled |
| `vehicle_type_id` | string | Type of vehicle |
| `current_range_meters` | double | Remaining range in meters |
| `current_fuel_percent` | double | Battery percentage (0-1) |
| `last_reported` | double | Unix timestamp of last report |
| `pricing_plan_id` | string | Pricing plan identifier |
| `change_types` | string | JSON array of change types detected |
| `distance_moved_meters` | double | Distance moved since last event |
| `disappeared` | bool | Vehicle disappeared (ride or maintenance) |
| `first_seen` | bool | First observation of this vehicle, amybe ride end |
| `is_maintenance` | bool | Flagged as maintenance operation due to recharge |
| `is_id_reset` | bool | Vehicle ID was reset but still same vehicle |
| `is_temporary_disappearance` | bool | Vehicle reappeared after disappearing |
| `closest_public_transport_distance_m` | double | Distance to nearest PT stop |
| `closest_public_transport_mode` | string | Type of nearest PT (bus, tram, etc.) |
| `closest_public_transport_name` | string | Name of nearest PT stop |
| `recalibrated_current_range_meters` | double | Adjusted range estimate |
| `clustered_provider` | string | Clustering for VOI Provider |
| `city` | string | City/service area name |

### Providers Covered (45 service areas)

**Germany**: Dott (Stuttgart, Heidelberg, Karlsruhe, Mannheim, Ulm, etc.), Bolt (Stuttgart, Karlsruhe), VOI, Zeus (Freiburg, Konstanz, etc.), Lime, Yoio

**Switzerland**: Dott (Zurich, Basel, Winterthur), Bolt (Zurich, Basel), VOI, Lime

**Austria**: Dott (Bregenz)

---

## Project Structure

### `collection/`
**Purpose**: Automated GBFS API data collection

- `collector.py` - Main collection script polling provider APIs every minute
- `escooter-tracker.service/.timer` - Systemd service configuration
- `EVENT_TYPES.md` - Documentation of all event types
- `DATA_QUIRKS.md` - Known data anomalies and provider-specific behaviors
- `query_examples.sql` - SQL query templates for analysis

**Key Features**:
- Detects location changes (>100m), battery changes, status changes
- Records "disappeared" events indicating rides or maintenance pickups
- Handles provider-specific data quirks (94% of vehicle IDs rotate after rides)

---

### `Business Case/`
**Purpose**: Economic feasibility analysis of e-scooter operations

- `business_case_analysis.ipynb` - Financial analysis notebook
- `business_case_summary.ipynb` - Comprehensive summary with visualizations
- `energy_consumption_analysis.ipynb` - Battery usage modeling
- `labor_cost_analysis.ipynb` - Workforce cost analysis
- `provider_specs.py` - Provider specifications and parameters
- `active_drain.parquet` / `idle_phases.parquet` - Battery drain data

---

### `city analysis/`
**Purpose**: City-level usage pattern analysis

- `city_matching_analysis.ipynb` - Provider coverage by city
- `Finalusagepattern.ipynb` - Comprehensive usage pattern analysis
- `temporal_and_intention_combined.pdf` - Temporal and user intention visualizations

---

### `matching/`
**Purpose**: Trip-to-route matching using statistical models

- `matcher.py` - Core matching algorithm with speed/distance priors
- `matching_pipeline.py` - End-to-end matching workflow
- `matching_candidates_scored.parquet` - Scored route candidates (~2.7 GB)
- `matching_output/` - Cluster visualizations and summary statistics

**Technical Details**:
- Speed priors: Truncated normal distribution, mean 14 km/h
- Distance priors: Log-normal distribution
- Uses BallTree for spatial queries

---

### `noise_detection/`
**Purpose**: Filter maintenance operations from ride data

- `non_ride_detection.py` - Flags non-ride events (rebalancing, maintenance)
- `maintenance_noise_detection.ipynb` - Detailed analysis
- `range_recalibration.py` - Range estimation adjustments

**Distinguishes**:
- Actual rides vs. maintenance pickups
- Temporary disappearances vs. final rides
- Battery recalibrations vs. actual charging

---

### `public_transport_analysis/`
**Purpose**: E-scooter vs. public transport comparison

- `pt_competition_analysis.ipynb` - Competition analysis
- `pt_route_generation.ipynb` - Route generation pipeline
- `scooter_availability_analysis.ipynb` - Fleet availability patterns
- `pt_comparison_all_trips.parquet` - Full trip comparison (~152 MB)
- `demand_maps/` - Interactive HTML maps for 26+ cities

**Visualizations**:
- `availability_vs_usage.pdf` - Fleet availability correlation
- `hourly_duration_comparison.pdf` - Time-based patterns
- `pt_mode_distribution.png` - Transport mode breakdown

---

### `routing/`
**Purpose**: Multi-modal routing engines

- `motis.py` - MOTIS public transit routing client
- `valhalla.py` - Valhalla street/path routing client
- `motis_docker/` - Docker setup for MOTIS server

**Data Sources**:
- Germany GTFS (weekly updates)
- Switzerland GTFS (yearly)
- DACH OSM data (daily updates)

---

### `spatial/`
**Purpose**: Geographic analysis and POI classification

- `poi_classifier.py` - Point-of-interest classification
- `pois.parquet` - Complete POI database (~2.3 GB)
- `flow_direction_analysis.ipynb` - Trip flow patterns
- `drive_intention_analysis.ipynb` - Inferred user intentions
- `closest_transit.parquet` - Nearest public transit stations
- `osm_db.yml` - PostGIS Docker configuration

**POI Categories**: Restaurants, shops, transit stops, residential areas, workplaces

---

## Root-Level Files

| File | Description |
|------|-------------|
| `city_info.py` | City information, population data, name normalization |
| `geo_utils.py` | Geographic utilities, DBSCAN clustering, BallTree queries |
| `load_escooter_data.ipynb` | Data loading and exploration |
| `Scooter Weather Final.ipynb` | Weather correlation analysis |
| `events_with_flags.parquet` | Events with detection flags (~977 MB) |

---

## Quick Start

### Loading the Main Dataset

```python
import pandas as pd

# Load vehicle events
df = pd.read_parquet('vehicle_events_export.parquet')

# Filter to actual rides (not maintenance)
rides = df[df['disappeared'] & ~df['is_maintenance'] & ~df['is_temporary_disappearance']]

# Filter by city
stuttgart_rides = rides[rides['city'] == 'Stuttgart']

# Filter by provider
dott_rides = rides[rides['provider'].str.startswith('dott_')]
```

### Useful Queries

```python
# Daily ride counts
daily_rides = rides.groupby(rides['timestamp'].dt.date).size()

# Average battery at ride start
avg_battery = rides.groupby('provider')['current_fuel_percent'].mean()

# Distance to nearest public transport
pt_proximity = rides['closest_public_transport_distance_m'].describe()
```

---

## Data Quality Notes

1. **Vehicle ID Rotation**: ~94% of vehicle IDs change after rides (GBFS v2.0 privacy)
2. **Battery Reporting**: Dott reports BMS recalibrations; Bolt reports actual battery swaps
3. **"Disappeared" Events**: Can indicate either rides or maintenance pickups
4. **Provider Differences**: Each provider has unique data reporting patterns

See `collection/DATA_QUIRKS.md` for detailed provider-specific documentation.

---

## Requirements

- Python 3.9+
- pandas, pyarrow, geopandas
- scikit-learn (for clustering)
- folium (for map visualizations)
- Docker (for routing engines)

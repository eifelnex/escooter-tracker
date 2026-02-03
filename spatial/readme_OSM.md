# OSM PostGIS Setup Guide

This guide sets up a PostGIS database with OpenStreetMap data for spatial analysis.

## Prerequisites

- Docker Desktop installed
- OSM data files in `C:\valhalla_tiles\` (e.g., `germany-latest.osm.pbf`)

## 1. Start the PostGIS Container

```powershell
cd escooter-gbfs-tracker\spatial
docker-compose -f osm_db.yml up -d
```

## 2. Enable hstore Extension

```powershell
docker exec -it osm-postgis psql -U postgres -d osm -c "CREATE EXTENSION IF NOT EXISTS hstore;"
```

## 3. Filter OSM Data (Optional - saves ~50% space)

The PostGIS container doesn't include osmium. Use one of these methods:

**Option A: Install osmium locally (conda)**
```powershell
conda install -c conda-forge osmium-tool
osmium tags-filter --invert-match C:\valhalla_tiles\germany-latest.osm.pbf w/highway -o C:\valhalla_tiles\germany-no-roads.osm.pbf
```

**Option B: Use osmium Docker image**
```powershell
docker run --rm -v C:\valhalla_tiles:/data stefda/osmium-tool osmium tags-filter --invert-match /data/germany-latest.osm.pbf w/highway -o /data/germany-no-roads.osm.pbf
```

## 4. Import into PostGIS

The PostGIS container doesn't include osm2pgsql. Use a separate container:

```powershell
docker run --rm --network=spatial_default -v C:\valhalla_tiles:/data openfirmware/osm2pgsql osm2pgsql -H osm-postgis -d osm -U postgres --create --slim -G --hstore /data/germany-no-roads.osm.pbf
```

Or import the full file (without road filtering):
```powershell
docker run --rm --network=spatial_default -v C:\valhalla_tiles:/data openfirmware/osm2pgsql osm2pgsql -H osm-postgis -d osm -U postgres --create --slim -G --hstore /data/germany-latest.osm.pbf
```

**Note:** Import takes ~8-9 hours for Germany. Requires ~30GB disk space (filtered) or ~60GB (full).

## 5. Add Switzerland (Optional)

Filter:
```powershell
docker run --rm -v C:\valhalla_tiles:/data stefda/osmium-tool osmium tags-filter --invert-match /data/switzerland-latest.osm.pbf w/highway -o /data/switzerland-no-roads.osm.pbf
```

Append to existing database:
```powershell
docker run --rm --network=spatial_default -v C:\valhalla_tiles:/data openfirmware/osm2pgsql osm2pgsql -H osm-postgis -d osm -U postgres --append --slim -G --hstore /data/switzerland-no-roads.osm.pbf
```

## 6. Test Connection

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="osm",
    user="postgres",
    password="postgres"
)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM planet_osm_point;")
print(cur.fetchone())  # Should show millions of POIs
```

## Database Tables

| Table | Contents |
|-------|----------|
| `planet_osm_point` | POIs (restaurants, bus stops, shops, etc.) |
| `planet_osm_polygon` | Buildings, areas, land use |
| `planet_osm_line` | Linear features (rivers, railways - not roads if filtered) |

## Container Management

```powershell
# Stop container
docker-compose -f osm_db.yml down

# Start container
docker-compose -f osm_db.yml up -d

# View logs
docker logs osm-postgis

# Remove container and data (WARNING: deletes all imported data)
docker-compose -f osm_db.yml down -v
```

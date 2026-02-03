# Valhalla Setup Guide (Windows + Docker)

## 1. Install Docker Desktop

- Download from https://www.docker.com/products/docker-desktop/
- Install and restart PC
- Launch Docker Desktop and wait for it to start

## 2. Create folder and download map data

```powershell
mkdir C:\valhalla_tiles
cd C:\valhalla_tiles

curl -L -o germany-latest.osm.pbf https://download.geofabrik.de/europe/germany-latest.osm.pbf
curl -L -o switzerland-latest.osm.pbf https://download.geofabrik.de/europe/switzerland-latest.osm.pbf
```

Or download manually in browser — faster for large files.

## 3. Build tiles (first time only)

```powershell
docker run -dt --name valhalla `
  -p 8002:8002 `
  -v C:\valhalla_tiles:/custom_files `
  -e build_elevation=True `
  -e build_admins=True `
  -e build_time_zones=True `
  --memory=12g `
  ghcr.io/gis-ops/docker-valhalla/valhalla:latest
```

Takes 2-4 hours. Monitor with `docker logs -f valhalla`.

## 4. Subsequent runs (fast startup)

```powershell
docker run -dt --name valhalla `
  -p 8002:8002 `
  -v C:\valhalla_tiles:/custom_files `
  -e use_tiles_ignore_pbf=True `
  --memory=12g `
  ghcr.io/gis-ops/docker-valhalla/valhalla:latest
```

## 5. Test

```powershell
curl "http://localhost:8002/route?json={\"locations\":[{\"lat\":48.52,\"lon\":9.05},{\"lat\":48.51,\"lon\":9.06}],\"costing\":\"bicycle\"}"
```

## Common commands

| Command | Description |
|---------|-------------|
| `docker logs -f valhalla` | Watch build progress |
| `docker stop valhalla` | Stop container |
| `docker rm -f valhalla` | Remove container (required before re-running) |
| `docker ps` | List running containers |

## Notes

- Don't use OneDrive folders (sync issues)
- Don't stop container during initial build
- Tiles are saved in `C:\valhalla_tiles` — reusable across rebuilds

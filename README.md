# E-Scooter Data Tracker for Raspberry Pi

Automated data collection system for tracking e-scooter positions and battery levels in the BW region over time.

## Features
## Data Collected

For each vehicle change event:
- `vehicle_id` - Unique identifier
- `timestamp` - ISO 8601 timestamp
- `provider` - E-scooter provider (dott, voi, bolt, zeus)
- `lat`, `lon` - GPS coordinates
- `is_reserved`, `is_disabled` - Status flags
- `vehicle_type_id` - Type of vehicle
- `current_range_meters` - Remaining range
- `current_fuel_percent` - Battery percentage
- `change_types` - What changed (location, battery, range, status)
- `distance_moved_meters` - Distance moved (for location changes)


## Installation

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Install system dependencies (Python 3, SQLite)
2. Create a Python virtual environment
3. Install required Python packages
4. Initialize the SQLite database
5. Set up systemd timer to run every minute



## Usage

### Querying the Database

```bash
# Open SQLite shell
sqlite3 escooter_data.db

# Run example queries
sqlite3 escooter_data.db < query_examples.sql
```

### Common Queries

**Get total events:**
```sql
SELECT COUNT(*) FROM vehicle_events;
```

**Get most active vehicles:**
```sql
SELECT
    vehicle_id,
    provider,
    COUNT(*) as movements,
    SUM(distance_moved_meters)/1000.0 as total_km
FROM vehicle_events
WHERE distance_moved_meters IS NOT NULL
GROUP BY vehicle_id
ORDER BY movements DESC
LIMIT 10;
```

**Get current positions:**
```sql
SELECT vehicle_id, provider, lat, lon, current_fuel_percent
FROM vehicle_events
WHERE (vehicle_id, timestamp) IN (
    SELECT vehicle_id, MAX(timestamp)
    FROM vehicle_events
    GROUP BY vehicle_id
);
```

### Exporting Data

**Export to CSV:**
```bash
sqlite3 -header -csv escooter_data.db "SELECT * FROM vehicle_events;" > data_export.csv
```

**Export latest positions:**
```bash
sqlite3 -header -csv escooter_data.db "
SELECT vehicle_id, provider, lat, lon, current_fuel_percent, timestamp
FROM vehicle_events
WHERE (vehicle_id, timestamp) IN (
    SELECT vehicle_id, MAX(timestamp) FROM vehicle_events GROUP BY vehicle_id
);" > latest_positions.csv
```

### Monitoring

**View live logs:**
```bash
tail -f collector.log
```

**Check systemd timer status:**
```bash
systemctl --user status escooter-tracker.timer
systemctl --user status escooter-tracker.service
```

**View collection statistics:**
```bash
sqlite3 escooter_data.db "
SELECT
    DATE(timestamp) as date,
    COUNT(*) as collections,
    AVG(vehicles_collected) as avg_vehicles,
    AVG(storage_rate) as avg_storage_rate
FROM collection_metadata
GROUP BY DATE(timestamp)
ORDER BY date DESC;
"
```


## Configuration

Edit `collector.py` to adjust settings:

```python
# Region bounding box
REGION_BBOX = {
    'min_lat': 48.40,
    'max_lat': 48.60,
    'min_lon': 8.95,
    'max_lon': 9.35
}

# Change detection thresholds
LOCATION_THRESHOLD_METERS = 5   # Minimum movement to record
BATTERY_THRESHOLD_PERCENT = 1   # Minimum battery change to record
RANGE_THRESHOLD_METERS = 100    # Minimum range change to record
```

## Maintenance


### Clean Old Data

```sql
-- Delete events older than 6 months
DELETE FROM vehicle_events
WHERE datetime(timestamp) < datetime('now', '-6 months');

-- Vacuum to reclaim space
VACUUM;
```

## Troubleshooting

### Systemd timer not running

```bash
# Check timer status
systemctl --user status escooter-tracker.timer

# Check service status
systemctl --user status escooter-tracker.service

# View recent logs
journalctl --user -u escooter-tracker.service -n 50

# Manually test the script
cd ~/raspberry-pi-escooter-tracker
source venv/bin/activate
python3 collector.py
```

### Stop/Start/Restart collector

```bash
# Stop collection
systemctl --user stop escooter-tracker.timer

# Start collection
systemctl --user start escooter-tracker.timer

# Restart service
systemctl --user restart escooter-tracker.service
```



## Data Analysis

See [query_examples.sql](query_examples.sql) for 12 pre-made SQL queries including:
- Vehicle activity patterns
- Battery drain analysis
- Hourly usage patterns
- Movement statistics
- Provider comparisons


## Files

- `collector.py` - Main data collection script
- `setup.sh` - Automated setup script (includes systemd configuration)
- `escooter-tracker.service` - Systemd service definition
- `escooter-tracker.timer` - Systemd timer (runs every minute)
- `requirements.txt` - Python dependencies
- `query_examples.sql` - Example SQL queries
- `DATA_QUIRKS.md` - Documentation of dataset quirks and analysis findings
- `escooter_data.db` - SQLite database (created after first run)
- `collector.log` - Application logs
- `systemd.log` - Systemd service logs


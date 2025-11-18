# E-Scooter Event Types

This document describes all the event types tracked by the collector, based on actual data analysis.

## Core Event Types

### Vehicle Lifecycle Events

- **`disappeared`** (stored as column, not in change_types)
  - Vehicle becomes unavailable in the GBFS feed
  - **Interpretation**: Vehicle was rented OR picked up for maintenance
  - **Cannot distinguish**: GBFS API doesn't differentiate between rides and maintenance pickups
  - **Use case**: Primary metric for counting usage events (rides + maintenance)
  - **ID rotation**: 94% of vehicles get new IDs after disappearing (GBFS v2.0 privacy requirement)

- **`first_seen`**
  - Vehicle appears in the feed (new ID or after disappearance)
  - **Interpretation**: Ride completed OR vehicle returned after maintenance
  - **Battery levels**: Provide clues about whether it was a ride (lower battery) or battery swap (>90%)
  - **Use case**: Track ride completions, identify battery swap events
  - **Timing**: Often appears 15-20 minutes after corresponding disappearance

### Change Detection Events

- **`location`** - Vehicle moved more than **100 meters**
- **`battery`** - Battery level changed (only for providers reporting battery %)
- **`range`** - Range changed (for providers reporting range in meters)
- **`disabled_status`** - Availability status changed
- **`reservation_status`** - Reservation state changed

### Smart Activity Classification

- **`ride`**
  - Location changed + Battery/range significantly decreased
  - **Interpretation**: Really rare

- **`transport`**
  - Location changed + Battery/range unchanged or minimal drop
  - **Interpretation**: Operator rebalancing (scooter on truck)

- **`recharged`** - **PROVIDER-SPECIFIC MEANING**
  - **Bolt**: Actual battery swap
    - Average 79% battery jump
    - From low battery (1-10%) to full (94-100%)
    - Takes 2-5 hours between pickup and return
  - **Dott**: BMS recalibration (NOT actual charging!)
    - Small 3-10% battery adjustment (avg 5.4%)
    - Occurs ~15 minutes after first_seen
    - Voltage recovery after ride, not real charging
  - **Cannot use "recharged" tag across providers for analysis!**

- **`battery_drain`**
  - Battery decreased while stationary
  - **Interpretation**: Idle battery drain

## Configuration Thresholds

Current detection thresholds in `collector.py`:

```python
LOCATION_THRESHOLD_METERS = 100  # Minimum movement to record (filters GPS drift)
BATTERY_THRESHOLD_PERCENT = 1    # Minimum battery % change (for Dott, Bolt)
RANGE_THRESHOLD_METERS = 100     # Minimum range change
```

## Provider-Specific Behaviors

### Battery Data Reporting

**Dott**
- Reports `current_fuel_percent` (0.0-1.0 decimal, where 1.0 = 100%)
- Maximum reported: 97% (never reaches 100%)
- "recharged" events: 3-10% BMS recalibrations only
- Real battery swaps: Identified by `first_seen` with >90% battery (4.9% of events)

**Bolt**
- Reports `current_fuel_percent` (0.0-1.0 decimal)
- Maximum reported: 100%
- "recharged" events: Real battery swaps (92.5% are >90% final battery)
- Average battery swap: 79% increase over 4.9 hours

**VOI & Others**
- Some providers report `current_range_meters` instead
- Range-based analysis uses different thresholds

### Vehicle ID Rotation

- **94% of vehicles** get new IDs after rides
- **GBFS v2.0 requirement**: Privacy protection
- **Analysis impact**: Cannot track individual vehicles long-term
- **Exceptions**: Some maintenance events reuse same ID

## Identifying Real Battery Swaps

Since "recharged" tag is unreliable across providers:

### Bolt Battery Swaps
```sql
-- Bolt explicitly tags battery swaps
SELECT * FROM vehicle_events
WHERE provider LIKE 'bolt%'
AND change_types LIKE '%recharged%';
```

### Dott Battery Swaps
```sql
-- Dott: Look for high-battery first_seen events
SELECT * FROM vehicle_events
WHERE provider LIKE 'dott%'
AND change_types = '["first_seen"]'
AND current_fuel_percent > 0.9;
```

## Counting Rides

### Method 1: Disappearances (includes some maintenance)
```sql
SELECT COUNT(*) FROM vehicle_events
WHERE disappeared = 1;
```
- **Accuracy**: ~85% are real rides
- **Contamination**: Includes maintenance pickups and bulk operations

### Method 2: Battery Analysis (more accurate)
```sql
-- Rides likely have battery drop
SELECT COUNT(*) FROM vehicle_events e1
WHERE e1.disappeared = 1
AND EXISTS (
  SELECT 1 FROM vehicle_events e2
  WHERE e2.change_types LIKE '%first_seen%'
  AND e2.timestamp > e1.timestamp
  AND e2.timestamp < datetime(e1.timestamp, '+2 hours')
  AND e2.current_fuel_percent < e1.current_fuel_percent - 0.02
);
```

### Method 3: Filter Bulk Operations
```sql
-- Remove known bulk maintenance patterns
SELECT COUNT(*) FROM vehicle_events
WHERE disappeared = 1
AND NOT (
  -- Exclude Dott Saarbrücken 01:34 AM bulk operations
  provider = 'dott_saarbruecken'
  AND strftime('%H:%M', timestamp) = '01:34'
)
AND NOT (
  -- Exclude suspicious ID resets (same location, same battery)
  -- See DATA_QUIRKS.md for detection logic
);
```

## Known Data Quirks

**See [DATA_QUIRKS.md](DATA_QUIRKS.md) for comprehensive analysis including:**
- Suspicious ID resets without rides (1,222 cases, 90% Dott)
- Nightly bulk operations (Dott Saarbrücken 01:34 AM pattern)
- Battery analysis proving 85% are real rides
- Provider-specific "recharged" tag meanings
- Temporal patterns and anomalies

## Example Queries

### Real Rides (Conservative Estimate)
```sql
-- Tübingen rides excluding bulk operations
SELECT COUNT(*) as estimated_rides
FROM vehicle_events
WHERE disappeared = 1
AND lat BETWEEN 48.47 AND 48.57
AND lon BETWEEN 8.99 AND 9.11
AND NOT (
  -- Exclude bulk patterns
  strftime('%H:%M', timestamp) BETWEEN '01:00' AND '02:00'
  AND provider LIKE 'dott%'
);
```

### Battery Swap Locations
```sql
-- Bolt battery swap locations
SELECT
  ROUND(lat, 3) as lat_approx,
  ROUND(lon, 3) as lon_approx,
  COUNT(*) as swap_count
FROM vehicle_events
WHERE provider LIKE 'bolt%'
AND change_types LIKE '%recharged%'
GROUP BY lat_approx, lon_approx
HAVING swap_count > 5
ORDER BY swap_count DESC;
```

### Hourly Usage Pattern
```sql
SELECT
  strftime('%H', timestamp) as hour,
  COUNT(*) as rides,
  AVG(current_fuel_percent * 100) as avg_battery_at_start
FROM vehicle_events
WHERE disappeared = 1
AND date(timestamp) >= date('now', '-7 days')
GROUP BY hour
ORDER BY hour;
```

## Database Schema Notes

- `current_fuel_percent`: Stored as decimal (1.0 = 100%, 0.5 = 50%)
  - **Always multiply by 100**: `current_fuel_percent * 100 as battery_pct`
- `disappeared`: INTEGER column (0 or 1), not in change_types JSON
- `change_types`: JSON array as string, e.g., `'["first_seen"]'` or `'["location", "battery", "ride"]'`

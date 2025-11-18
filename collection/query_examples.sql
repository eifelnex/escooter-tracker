-- Example SQL queries for analyzing e-scooter data

-- 1. Get total number of events
SELECT COUNT(*) as total_events FROM vehicle_events;

-- 2. Get events per vehicle
SELECT
    vehicle_id,
    provider,
    COUNT(*) as event_count,
    MIN(timestamp) as first_seen,
    MAX(timestamp) as last_seen
FROM vehicle_events
GROUP BY vehicle_id
ORDER BY event_count DESC
LIMIT 20;

-- 3. Get all location changes for a specific vehicle
SELECT
    timestamp,
    lat,
    lon,
    distance_moved_meters,
    change_types
FROM vehicle_events
WHERE vehicle_id = 'YOUR_VEHICLE_ID_HERE'
  AND change_types LIKE '%location%'
ORDER BY timestamp;

-- 4. Get most active vehicles (most movements)
SELECT
    vehicle_id,
    provider,
    COUNT(*) as movement_count,
    SUM(distance_moved_meters) as total_distance_meters,
    ROUND(SUM(distance_moved_meters) / 1000.0, 2) as total_distance_km
FROM vehicle_events
WHERE distance_moved_meters IS NOT NULL
GROUP BY vehicle_id
ORDER BY movement_count DESC
LIMIT 20;

-- 5. Get collection statistics
SELECT
    DATE(timestamp) as date,
    COUNT(*) as collections,
    AVG(vehicles_collected) as avg_vehicles,
    AVG(events_written) as avg_events_written,
    AVG(storage_rate) as avg_storage_rate
FROM collection_metadata
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- 6. Get battery drain events (battery decreasing)
SELECT
    e1.vehicle_id,
    e1.timestamp as time_start,
    e2.timestamp as time_end,
    e1.current_fuel_percent as battery_start,
    e2.current_fuel_percent as battery_end,
    (e1.current_fuel_percent - e2.current_fuel_percent) as battery_drain
FROM vehicle_events e1
JOIN vehicle_events e2 ON e1.vehicle_id = e2.vehicle_id
WHERE e2.timestamp > e1.timestamp
  AND e1.current_fuel_percent > e2.current_fuel_percent
  AND (e1.current_fuel_percent - e2.current_fuel_percent) >= 5
ORDER BY battery_drain DESC
LIMIT 50;

-- 7. Get vehicles by provider and status
SELECT
    provider,
    COUNT(DISTINCT vehicle_id) as unique_vehicles,
    SUM(CASE WHEN is_reserved = 1 THEN 1 ELSE 0 END) as currently_reserved,
    SUM(CASE WHEN is_disabled = 1 THEN 1 ELSE 0 END) as currently_disabled
FROM (
    SELECT vehicle_id, provider, is_reserved, is_disabled
    FROM vehicle_events
    WHERE (vehicle_id, timestamp) IN (
        SELECT vehicle_id, MAX(timestamp)
        FROM vehicle_events
        GROUP BY vehicle_id
    )
) latest
GROUP BY provider;

-- 8. Get hourly activity pattern
SELECT
    strftime('%H', timestamp) as hour,
    COUNT(*) as event_count,
    AVG(distance_moved_meters) as avg_distance_per_movement
FROM vehicle_events
WHERE distance_moved_meters IS NOT NULL
GROUP BY hour
ORDER BY hour;

-- 9. Get vehicles that moved the furthest in a single trip
SELECT
    vehicle_id,
    provider,
    timestamp,
    lat,
    lon,
    distance_moved_meters,
    ROUND(distance_moved_meters / 1000.0, 2) as distance_km
FROM vehicle_events
WHERE distance_moved_meters IS NOT NULL
ORDER BY distance_moved_meters DESC
LIMIT 20;

-- 10. Get recent activity (last 24 hours)
SELECT
    vehicle_id,
    provider,
    timestamp,
    lat,
    lon,
    change_types,
    distance_moved_meters
FROM vehicle_events
WHERE datetime(timestamp) >= datetime('now', '-1 day')
ORDER BY timestamp DESC
LIMIT 100;

-- 11. Export latest position of all vehicles
SELECT
    vehicle_id,
    provider,
    lat,
    lon,
    current_fuel_percent,
    current_range_meters,
    is_reserved,
    is_disabled,
    timestamp as last_updated
FROM vehicle_events
WHERE (vehicle_id, timestamp) IN (
    SELECT vehicle_id, MAX(timestamp)
    FROM vehicle_events
    GROUP BY vehicle_id
)
ORDER BY provider, vehicle_id;

-- 12. Calculate database size and statistics
SELECT
    (SELECT COUNT(*) FROM vehicle_events) as total_events,
    (SELECT COUNT(DISTINCT vehicle_id) FROM vehicle_events) as unique_vehicles,
    (SELECT COUNT(DISTINCT provider) FROM vehicle_events) as providers,
    (SELECT MIN(timestamp) FROM vehicle_events) as data_start,
    (SELECT MAX(timestamp) FROM vehicle_events) as data_end;

-- ============================================================================
-- PROVIDER/REGION CHANGE QUERIES
-- ============================================================================

-- 13. Find all provider/region changes
SELECT
    vehicle_id,
    provider,
    timestamp,
    lat,
    lon,
    ROUND(distance_moved_meters / 1000.0, 2) as distance_km,
    change_types
FROM vehicle_events
WHERE change_types LIKE '%provider_changed%'
ORDER BY timestamp DESC
LIMIT 50;

-- 14. Most cross-region vehicles
SELECT
    vehicle_id,
    COUNT(*) as region_changes,
    COUNT(DISTINCT provider) as unique_providers,
    ROUND(SUM(distance_moved_meters) / 1000.0, 2) as total_km_when_crossing
FROM vehicle_events
WHERE change_types LIKE '%provider_changed%'
GROUP BY vehicle_id
ORDER BY region_changes DESC
LIMIT 20;

-- 15. Provider change matrix (from -> to)
-- Note: This requires LAG window function for SQLite 3.25+
WITH provider_transitions AS (
    SELECT
        vehicle_id,
        timestamp,
        provider,
        LAG(provider) OVER (PARTITION BY vehicle_id ORDER BY timestamp) as prev_provider
    FROM vehicle_events
    WHERE change_types LIKE '%provider_changed%'
)
SELECT
    prev_provider as from_provider,
    provider as to_provider,
    COUNT(*) as transition_count
FROM provider_transitions
WHERE prev_provider IS NOT NULL
GROUP BY prev_provider, provider
ORDER BY transition_count DESC;

-- 16. Track provider history for a specific vehicle
SELECT
    timestamp,
    provider,
    lat,
    lon,
    change_types,
    ROUND(distance_moved_meters / 1000.0, 2) as distance_km
FROM vehicle_events
WHERE vehicle_id = 'YOUR_VEHICLE_ID_HERE'
  AND (change_types LIKE '%first_seen%'
       OR change_types LIKE '%provider_changed%')
ORDER BY timestamp;

-- 17. Provider distribution (current snapshot from latest events)
WITH latest_events AS (
    SELECT
        vehicle_id,
        provider,
        timestamp,
        ROW_NUMBER() OVER (PARTITION BY vehicle_id ORDER BY timestamp DESC) as rn
    FROM vehicle_events
)
SELECT
    provider,
    COUNT(*) as vehicle_count
FROM latest_events
WHERE rn = 1
GROUP BY provider
ORDER BY vehicle_count DESC;

-- 18. Vehicles in Tübingen-Reutlingen region (from latest events)
WITH latest_events AS (
    SELECT
        vehicle_id,
        provider,
        lat,
        lon,
        timestamp,
        ROW_NUMBER() OVER (PARTITION BY vehicle_id ORDER BY timestamp DESC) as rn
    FROM vehicle_events
)
SELECT
    provider,
    COUNT(*) as vehicle_count
FROM latest_events
WHERE rn = 1
  AND lat BETWEEN 48.40 AND 48.60
  AND lon BETWEEN 8.95 AND 9.35
GROUP BY provider
ORDER BY vehicle_count DESC;

-- 19. Daily provider change statistics
SELECT
    DATE(timestamp) as date,
    COUNT(*) as provider_changes,
    COUNT(DISTINCT vehicle_id) as unique_vehicles,
    AVG(distance_moved_meters/1000.0) as avg_distance_km
FROM vehicle_events
WHERE change_types LIKE '%provider_changed%'
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- 20. Find vehicles that frequently change providers
SELECT
    vehicle_id,
    COUNT(*) as provider_changes,
    MIN(timestamp) as first_change,
    MAX(timestamp) as last_change,
    COUNT(DISTINCT provider) as unique_providers
FROM vehicle_events
WHERE change_types LIKE '%provider_changed%'
GROUP BY vehicle_id
HAVING COUNT(*) > 2
ORDER BY provider_changes DESC
LIMIT 20;

-- 21. Provider market share by region
SELECT
    r.provider,
    r.current_region,
    COUNT(*) as vehicles,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY r.current_region), 2) as market_share_pct
FROM vehicle_region_registry r
GROUP BY r.provider, r.current_region
ORDER BY r.current_region, vehicles DESC;

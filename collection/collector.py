#!/usr/bin/env python3


import sqlite3
import json
import requests
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import logging
import sys
from pathlib import Path

# Change detection thresholds
LOCATION_THRESHOLD_METERS = 100  # 100 meters (filters GPS drift/noise)
BATTERY_THRESHOLD_PERCENT = 1    # 1%

# Database path
DB_PATH = Path(__file__).parent / 'escooter_data.db'

# Previous fetch cache path
CACHE_PATH = Path(__file__).parent / 'last_fetch_cache.json'

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'collector.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r


def init_database():
    """Initialize SQLite database with schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create vehicle_events table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            provider TEXT,
            lat REAL,
            lon REAL,
            is_reserved INTEGER,
            is_disabled INTEGER,
            vehicle_type_id TEXT,
            current_range_meters REAL,
            current_fuel_percent REAL,
            last_reported INTEGER,
            pricing_plan_id TEXT,
            change_types TEXT,
            distance_moved_meters REAL,
            disappeared INTEGER DEFAULT 0,
            UNIQUE(vehicle_id, timestamp)
        )
    ''')

    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_vehicle_timestamp
        ON vehicle_events(vehicle_id, timestamp DESC)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp
        ON vehicle_events(timestamp DESC)
    ''')

    # Create metadata table for tracking collection stats
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS collection_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            vehicles_collected INTEGER,
            events_written INTEGER,
            no_change_count INTEGER,
            storage_rate REAL,
            stations_collected INTEGER,
            station_events_written INTEGER,
            errors TEXT
        )
    ''')

    # Create station_events table for virtual stations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS station_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            provider TEXT,
            name TEXT,
            lat REAL,
            lon REAL,
            capacity INTEGER,
            num_bikes_available INTEGER,
            num_docks_available INTEGER,
            is_installed INTEGER,
            is_renting INTEGER,
            is_returning INTEGER,
            change_types TEXT,
            bikes_change INTEGER,
            UNIQUE(station_id, timestamp)
        )
    ''')

    # Create index for faster station queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_station_timestamp
        ON station_events(station_id, timestamp DESC)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_station_provider
        ON station_events(provider, timestamp DESC)
    ''')

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


def load_last_fetch_cache():
    """Load the previous fetch's vehicle IDs and data"""
    if not CACHE_PATH.exists():
        return {}

    try:
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load cache: {e}")
        return {}


def save_fetch_cache(vehicles):
    """Save current fetch's vehicle data to cache"""
    cache = {}
    for vehicle in vehicles:
        vehicle_id = vehicle.get('bike_id')
        if vehicle_id:
            cache[vehicle_id] = {
                'provider': vehicle.get('provider'),
                'lat': vehicle.get('lat'),
                'lon': vehicle.get('lon'),
                'current_range_meters': vehicle.get('current_range_meters'),
                'current_fuel_percent': vehicle.get('current_fuel_percent'),
                'is_reserved': vehicle.get('is_reserved', False),
                'is_disabled': vehicle.get('is_disabled', False),
            }

    try:
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        logger.warning(f"Could not save cache: {e}")


def get_last_event(conn, vehicle_id):
    """Get the most recent event for a vehicle"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM vehicle_events
        WHERE vehicle_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (vehicle_id,))

    row = cursor.fetchone()
    if not row:
        return None

    columns = [desc[0] for desc in cursor.description]
    return dict(zip(columns, row))


def get_last_station_event(conn, station_id):
    """Get the most recent event for a station"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM station_events
        WHERE station_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (station_id,))

    row = cursor.fetchone()
    if not row:
        return None

    columns = [desc[0] for desc in cursor.description]
    return dict(zip(columns, row))


def detect_changes(vehicle_id, current_data, last_event):
    """
    Detect if there are significant changes requiring a new row
    Returns: (should_store, change_types, distance_moved)
    """
    if not last_event:
        return (True, ['first_seen'], None)

    change_types = []
    distance_moved = None

    # Location change
    if current_data.get('lat') and last_event.get('lat'):
        distance = haversine_distance(
            last_event['lat'],
            last_event['lon'],
            current_data['lat'],
            current_data['lon']
        )

        if distance > LOCATION_THRESHOLD_METERS:
            change_types.append('location')
            distance_moved = round(distance, 2)

    # Battery change (for Dott, Bolt)
    old_battery = last_event.get('current_fuel_percent')
    new_battery = current_data.get('current_fuel_percent')

    if old_battery is not None and new_battery is not None:
        battery_diff = abs(new_battery - old_battery)
        if battery_diff >= BATTERY_THRESHOLD_PERCENT:
            change_types.append('battery')

    # Range change (for VOI, Zeus) - 300m threshold
    old_range = last_event.get('current_range_meters')
    new_range = current_data.get('current_range_meters')

    if old_range is not None and new_range is not None:
        range_diff = abs(new_range - old_range)
        # 300 meters threshold - detects rides/charging while filtering small variations
        if range_diff >= 300:
            change_types.append('range')

    # Status changes
    if current_data.get('is_reserved') != last_event.get('is_reserved'):
        change_types.append('reservation_status')

    if current_data.get('is_disabled') != last_event.get('is_disabled'):
        change_types.append('disabled_status')

    # Provider/region change (e.g., dott_tubingen -> dott_reutlingen)
    old_provider = last_event.get('provider')
    new_provider = current_data.get('provider')

    if old_provider and new_provider and old_provider != new_provider:
        change_types.append('provider_changed')

    # Smart event classification based on location + battery/range changes
    location_changed = 'location' in change_types
    energy_changed = 'battery' in change_types or 'range' in change_types

    if location_changed and energy_changed:
        # Location changed AND battery/range changed
        # Calculate energy drop/increase
        battery_drop = 0
        range_drop = 0
        battery_increase = 0
        range_increase = 0

        if new_battery is not None and old_battery is not None:
            battery_drop = old_battery - new_battery
            battery_increase = new_battery - old_battery
        if new_range is not None and old_range is not None:
            range_drop = old_range - new_range
            range_increase = new_range - old_range

        energy_increased = battery_drop < 0 or range_drop < 0

        if energy_increased:
            # Energy went up while moving
            # Check if it's significant recharge or just a small adjustment
            is_significant_recharge = (battery_increase > 0.03) or (range_increase > 3000)

            if is_significant_recharge:
                # Significant charge increase (>3% or >3km) = operator relocated & charged
                change_types.append('relocated_charged')
            else:
                # Small adjustment (≤3% battery or ≤3km range) = data correction, treat as transport
                change_types.append('transport')
        else:
            # Energy went down while moving - distinguish ride from transport
            # Transport: minimal energy drop (≤1% battery or <800m range)
            # Ride: significant energy drop (>1% battery or ≥800m range)
            # Note: 800m matches VOI range quantum size for better ride detection
            is_minimal_drain = (battery_drop <= 0.01) or (range_drop < 800)

            if is_minimal_drain:
                # Moved but barely used energy = truck transport or idle drain
                change_types.append('transport')
            else:
                # Moved with significant energy use = actual ride
                change_types.append('ride')

    elif location_changed and not energy_changed:
        # Location changed but battery/range stayed same = truck transport
        change_types.append('transport')

    elif not location_changed and energy_changed:
        # Battery/range changed but didn't move
        battery_increase = 0
        range_increase = 0

        if new_battery is not None and old_battery is not None:
            battery_increase = new_battery - old_battery
        if new_range is not None and old_range is not None:
            range_increase = new_range - old_range

        energy_increased = battery_increase > 0 or range_increase > 0

        if energy_increased:
            # Energy went up
            # Check if it's significant recharge or just a small adjustment
            is_significant_recharge = (battery_increase > 0.03) or (range_increase > 3000)

            if is_significant_recharge:
                # Significant charge increase (>3% or >3km) = actual recharging
                change_types.append('recharged')
            else:
                # Small adjustment (≤3% battery or ≤3km range) = data correction/adjustment
                change_types.append('range_adjustment')
        else:
            # Energy went down without moving = battery drain while parked
            change_types.append('battery_drain')

    should_store = len(change_types) > 0
    return (should_store, change_types, distance_moved)


def detect_station_changes(station_id, current_data, last_event):
    """
    Detect if there are significant changes in station status
    Returns: (should_store, change_types, bikes_change)
    """
    if not last_event:
        return (True, ['first_seen'], None)

    change_types = []
    bikes_change = None

    # Check if number of available bikes changed
    old_bikes = last_event.get('num_bikes_available')
    new_bikes = current_data.get('num_bikes_available')

    if old_bikes is not None and new_bikes is not None and old_bikes != new_bikes:
        change_types.append('occupancy')
        bikes_change = new_bikes - old_bikes

    # Check if station status changed
    if current_data.get('is_installed') != last_event.get('is_installed'):
        change_types.append('installation_status')

    if current_data.get('is_renting') != last_event.get('is_renting'):
        change_types.append('renting_status')

    if current_data.get('is_returning') != last_event.get('is_returning'):
        change_types.append('returning_status')

    should_store = len(change_types) > 0
    return (should_store, change_types, bikes_change)


def collect_vehicles():
    """Collect vehicle data from MobiData-BW API"""
    vehicles = []

    try:
        # Get all e-scooter systems
        gbfs_base_url = 'https://api.mobidata-bw.de/sharing/gbfs'
        response = requests.get(gbfs_base_url, timeout=30)
        gbfs_data = response.json()

        # Query ALL providers matching keywords: dott, bolt, voi, zeus, hopp, yoio, lime
        provider_keywords = ['dott', 'bolt', 'voi', 'zeus', 'hopp', 'yoio', 'lime']

        systems = gbfs_data['systems']
        scooter_systems = [s for s in systems
                          if any(keyword in s['id'].lower() for keyword in provider_keywords)]

        logger.info(f"Querying {len(scooter_systems)} providers matching keywords: {', '.join(provider_keywords)}")

        # Collect from each provider
        for system in scooter_systems:
            provider_id = system['id']
            provider_url = system['url']

            try:
                gbfs_response = requests.get(provider_url, timeout=30)
                gbfs_info = gbfs_response.json()

                if 'data' not in gbfs_info:
                    continue

                lang_key = list(gbfs_info['data'].keys())[0]
                feeds = gbfs_info['data'][lang_key]['feeds']
                feed_dict = {feed['name']: feed['url'] for feed in feeds}

                if 'free_bike_status' not in feed_dict:
                    continue

                bike_response = requests.get(feed_dict['free_bike_status'], timeout=30)
                bike_data = bike_response.json()

                if 'data' not in bike_data or 'bikes' not in bike_data['data']:
                    continue

                bikes = bike_data['data']['bikes']

                # Add all vehicles (no bbox filtering)
                for bike in bikes:
                    bike['provider'] = provider_id
                    vehicles.append(bike)

                logger.info(f"Collected {len(bikes)} vehicles from {provider_id}")

            except Exception as e:
                logger.error(f"Error collecting from {provider_id}: {e}")
                continue

        return vehicles

    except Exception as e:
        logger.error(f"Error in collect_vehicles: {e}")
        raise


def collect_stations():
    """Collect station data from MobiData-BW API"""
    stations = []
    station_info_map = {}

    try:
        # Get all e-scooter systems
        gbfs_base_url = 'https://api.mobidata-bw.de/sharing/gbfs'
        response = requests.get(gbfs_base_url, timeout=30)
        gbfs_data = response.json()

        # Query ALL providers matching keywords: dott, bolt, voi, zeus, hopp, yoio, lime
        provider_keywords = ['dott', 'bolt', 'voi', 'zeus', 'hopp', 'yoio', 'lime']

        systems = gbfs_data['systems']
        scooter_systems = [s for s in systems
                          if any(keyword in s['id'].lower() for keyword in provider_keywords)]

        # Collect from each provider
        for system in scooter_systems:
            provider_id = system['id']
            provider_url = system['url']

            try:
                gbfs_response = requests.get(provider_url, timeout=30)
                gbfs_info = gbfs_response.json()

                if 'data' not in gbfs_info:
                    continue

                lang_key = list(gbfs_info['data'].keys())[0]
                feeds = gbfs_info['data'][lang_key]['feeds']
                feed_dict = {feed['name']: feed['url'] for feed in feeds}

                # Check if provider has station feeds
                if 'station_information' not in feed_dict or 'station_status' not in feed_dict:
                    continue

                # Get station information (static data)
                info_response = requests.get(feed_dict['station_information'], timeout=30)
                info_data = info_response.json()

                if 'data' in info_data and 'stations' in info_data['data']:
                    for station in info_data['data']['stations']:
                        station_id = station.get('station_id')
                        station_info_map[station_id] = {
                            'name': station.get('name'),
                            'lat': station.get('lat'),
                            'lon': station.get('lon'),
                            'capacity': station.get('capacity'),
                            'provider': provider_id
                        }

                # Get station status (dynamic data)
                status_response = requests.get(feed_dict['station_status'], timeout=30)
                status_data = status_response.json()

                if 'data' not in status_data or 'stations' not in status_data['data']:
                    continue

                for station_status in status_data['data']['stations']:
                    station_id = station_status.get('station_id')

                    if station_id in station_info_map:
                        # Merge static info with dynamic status
                        station = {**station_info_map[station_id], **station_status}
                        stations.append(station)

                logger.info(f"Collected {len([s for s in stations if s.get('provider') == provider_id])} stations from {provider_id}")

            except Exception as e:
                logger.error(f"Error collecting stations from {provider_id}: {e}")
                continue

        return stations

    except Exception as e:
        logger.error(f"Error in collect_stations: {e}")
        return []


def main():
    """Main collection function"""
    timestamp = datetime.utcnow().isoformat() + 'Z'

    logger.info(f"=== Starting collection at {timestamp} ===")

    try:
        # Initialize database
        init_database()

        # Load previous fetch cache
        last_fetch_cache = load_last_fetch_cache()
        logger.info(f"Loaded cache with {len(last_fetch_cache)} vehicles from previous fetch")

        # Collect vehicles
        vehicles = collect_vehicles()
        logger.info(f"Collected {len(vehicles)} vehicles total")

        if not vehicles:
            logger.warning("No vehicles found")
            return

        # Detect disappeared vehicles (in cache but not in current fetch)
        current_vehicle_ids = {v.get('bike_id') for v in vehicles if v.get('bike_id')}
        disappeared_vehicle_ids = set(last_fetch_cache.keys()) - current_vehicle_ids

        logger.info(f"Detected {len(disappeared_vehicle_ids)} disappeared vehicles (likely rented)")

        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        events_written = 0
        no_change_count = 0
        disappeared_events_written = 0

        try:
            # First, record disappeared vehicles
            for disappeared_id in disappeared_vehicle_ids:
                cached_data = last_fetch_cache[disappeared_id]

                cursor.execute('''
                    INSERT OR IGNORE INTO vehicle_events (
                        vehicle_id, timestamp, provider, lat, lon,
                        is_reserved, is_disabled, vehicle_type_id,
                        current_range_meters, current_fuel_percent,
                        last_reported, pricing_plan_id, change_types,
                        distance_moved_meters, disappeared
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    disappeared_id,
                    timestamp,
                    cached_data.get('provider'),
                    cached_data.get('lat'),
                    cached_data.get('lon'),
                    int(cached_data.get('is_reserved', False)),
                    int(cached_data.get('is_disabled', False)),
                    None,  # vehicle_type_id not in cache
                    cached_data.get('current_range_meters'),
                    cached_data.get('current_fuel_percent'),
                    None,  # last_reported not in cache
                    None,  # pricing_plan_id not in cache
                    json.dumps(['disappeared']),
                    None,
                    1  # disappeared flag
                ))
                disappeared_events_written += 1

            # Process current vehicles
            for vehicle in vehicles:
                vehicle_id = vehicle.get('bike_id')
                if not vehicle_id:
                    continue

                # Prepare current data
                current_data = {
                    'lat': vehicle.get('lat'),
                    'lon': vehicle.get('lon'),
                    'is_reserved': vehicle.get('is_reserved', False),
                    'is_disabled': vehicle.get('is_disabled', False),
                    'current_range_meters': vehicle.get('current_range_meters'),
                    'current_fuel_percent': vehicle.get('current_fuel_percent'),
                }

                # Get last event
                last_event = get_last_event(conn, vehicle_id)

                # Check if this is a reappearance (was in cache, disappeared, now back)
                was_in_cache = vehicle_id in last_fetch_cache
                is_reappearance = last_event and last_event.get('disappeared') == 1

                # Detect changes
                should_store, change_types, distance_moved = detect_changes(
                    vehicle_id, current_data, last_event
                )

                # If vehicle reappeared, add 'reappeared' to change types
                if is_reappearance:
                    change_types.append('reappeared')
                    should_store = True

                if should_store:
                    # Insert new event
                    cursor.execute('''
                        INSERT OR IGNORE INTO vehicle_events (
                            vehicle_id, timestamp, provider, lat, lon,
                            is_reserved, is_disabled, vehicle_type_id,
                            current_range_meters, current_fuel_percent,
                            last_reported, pricing_plan_id, change_types,
                            distance_moved_meters, disappeared
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        vehicle_id,
                        timestamp,
                        vehicle.get('provider'),
                        vehicle.get('lat'),
                        vehicle.get('lon'),
                        int(vehicle.get('is_reserved', False)),
                        int(vehicle.get('is_disabled', False)),
                        vehicle.get('vehicle_type_id'),
                        vehicle.get('current_range_meters'),
                        vehicle.get('current_fuel_percent'),
                        vehicle.get('last_reported'),
                        vehicle.get('pricing_plan_id'),
                        json.dumps(change_types),
                        distance_moved,
                        0  # disappeared = 0 (vehicle is present)
                    ))
                    events_written += 1
                else:
                    no_change_count += 1

            # Collect and process stations
            stations = collect_stations()
            station_events_written = 0
            station_no_change_count = 0

            if stations:
                logger.info(f"Collected {len(stations)} stations total")

                for station in stations:
                    station_id = station.get('station_id')
                    if not station_id:
                        continue

                    # Prepare current station data
                    current_station_data = {
                        'num_bikes_available': station.get('num_bikes_available'),
                        'num_docks_available': station.get('num_docks_available'),
                        'is_installed': station.get('is_installed', 1),
                        'is_renting': station.get('is_renting', 1),
                        'is_returning': station.get('is_returning', 1),
                    }

                    # Get last station event
                    last_station_event = get_last_station_event(conn, station_id)

                    # Detect station changes
                    should_store_station, station_change_types, bikes_change = detect_station_changes(
                        station_id, current_station_data, last_station_event
                    )

                    if should_store_station:
                        # Insert new station event
                        cursor.execute('''
                            INSERT OR IGNORE INTO station_events (
                                station_id, timestamp, provider, name, lat, lon,
                                capacity, num_bikes_available, num_docks_available,
                                is_installed, is_renting, is_returning,
                                change_types, bikes_change
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            station_id,
                            timestamp,
                            station.get('provider'),
                            station.get('name'),
                            station.get('lat'),
                            station.get('lon'),
                            station.get('capacity'),
                            station.get('num_bikes_available'),
                            station.get('num_docks_available'),
                            int(station.get('is_installed', 1)),
                            int(station.get('is_renting', 1)),
                            int(station.get('is_returning', 1)),
                            json.dumps(station_change_types),
                            bikes_change
                        ))
                        station_events_written += 1
                    else:
                        station_no_change_count += 1

                logger.info(f"✓ Stations: {station_events_written} events written, {station_no_change_count} no changes")

            # Store metadata
            storage_rate = (events_written / len(vehicles) * 100) if vehicles else 0
            cursor.execute('''
                INSERT INTO collection_metadata (
                    timestamp, vehicles_collected, events_written,
                    no_change_count, storage_rate, stations_collected,
                    station_events_written, errors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, len(vehicles), events_written, no_change_count, storage_rate,
                  len(stations), station_events_written, None))

            conn.commit()

            logger.info(f"✓ Completed: {events_written} events written, {disappeared_events_written} disappeared, {no_change_count} no changes ({storage_rate:.1f}% storage rate)")

            # Save current fetch to cache for next iteration
            save_fetch_cache(vehicles)
            logger.info(f"Saved {len(vehicles)} vehicles to cache")

        finally:
            conn.close()

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

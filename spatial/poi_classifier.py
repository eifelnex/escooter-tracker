"""OSM POI classification with 3-level hierarchy (key -> class -> subclass)."""

import yaml
import requests
import pandas as pd
import psycopg2
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_CONFIG_PATH = Path(__file__).parent / "osm_poi_config.yaml"

POI_KEYS = [
    'amenity', 'shop', 'tourism', 'leisure', 'building', 
    'public_transport', 'railway', 'highway', 'office'
]


def fetch_taginfo_values(key: str, min_count: int = 100, limit: int = 500) -> pd.DataFrame:
    """Fetch common values for a key from Taginfo API."""
    url = f"https://taginfo.openstreetmap.org/api/4/key/values"
    params = {
        'key': key,
        'page': 1,
        'rp': limit,
        'sortname': 'count',
        'sortorder': 'desc',
        'format': 'json'
    }

    headers = {'User-Agent': 'OSM-POI-Classifier/1.0 (escooter-analysis)'}

    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data['data'])
    if not df.empty:
        df = df[df['count'] >= min_count]
        df = df[['value', 'count', 'fraction']]
    return df


def fetch_all_common_values(keys: list = POI_KEYS, min_count: int = 100) -> Dict[str, list]:
    """Fetch common values for all POI keys."""
    result = {}
    for key in tqdm(keys, desc="Fetching from Taginfo"):
        df = fetch_taginfo_values(key, min_count=min_count)
        if not df.empty:
            result[key] = df['value'].tolist()
    return result





def build_value_to_class_map(config: dict) -> Dict[Tuple[str, str], str]:
    """Build a flat lookup dict from hierarchical config."""
    lookup = {}
    for osm_key, class_mapping in config.items():
        for class_name, values in class_mapping.items():
            for value in values:
                lookup[(osm_key, value)] = class_name
    return lookup


def save_config(config: dict, path: Path = DEFAULT_CONFIG_PATH):
    """Save classification config to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Config saved to {path}")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load classification config from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


class OSMPOIClassifier:
    """Classifies OSM POIs into a 3-level hierarchy."""

    def __init__(self, config: dict = None, config_path: Path = None):
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            self.config = DEFAULT_CLASS_MAPPING
        
        self._lookup = build_value_to_class_map(self.config)
        self._unmapped = defaultdict(lambda: defaultdict(int))
    
    def classify(self, key: str, value: str) -> str:
        """Get class for a (key, value) pair."""
        result = self._lookup.get((key, value))
        if result:
            return result

        self._unmapped[key][value] += 1
        return key
    
    def classify_df(self, df: pd.DataFrame,
                    key_col: str = 'mapping_key',
                    value_col: str = 'subclass') -> pd.Series:
        """Classify all rows in a DataFrame."""
        return df.apply(
            lambda r: self.classify(r[key_col], r[value_col]),
            axis=1
        )
    
    def get_unmapped(self, min_count: int = 10) -> pd.DataFrame:
        """Get unmapped (key, value) pairs that fell back to defaults."""
        rows = []
        for key, values in self._unmapped.items():
            for value, count in values.items():
                if count >= min_count:
                    rows.append({'key': key, 'value': value, 'count': count})

        return pd.DataFrame(rows).sort_values('count', ascending=False)
    
    def add_mapping(self, key: str, class_name: str, values: list):
        """Add mappings to the config."""
        if key not in self.config:
            self.config[key] = {}
        
        if class_name not in self.config[key]:
            self.config[key][class_name] = []
        
        self.config[key][class_name].extend(values)
        
        for value in values:
            self._lookup[(key, value)] = class_name

    def save(self, path: Path = DEFAULT_CONFIG_PATH):
        """Save current config to file."""
        save_config(self.config, path)


def query_nearby_pois(conn, coords_df: pd.DataFrame, radius_m: int = 50,
                      batch_size: int = 500, classifier: OSMPOIClassifier = None,
                      id_col: str = None) -> pd.DataFrame:
    """Query nearby POIs with clean mapping_key/subclass pattern."""

    coords_df = coords_df.reset_index(drop=True)

    cur = conn.cursor()
    df = None

    for start in tqdm(range(0, len(coords_df), batch_size), desc="Querying POIs"):
        batch = coords_df.iloc[start:start + batch_size]

        if id_col and id_col in batch.columns:
            values = ", ".join([
                f"({row[id_col]}, {row['lon']}, {row['lat']})"
                for _, row in batch.iterrows()
            ])
        else:
            values = ", ".join([
                f"({start + i}, {row['lon']}, {row['lat']})"
                for i, row in batch.iterrows()
            ])

        query = f"""
        WITH points AS (
            SELECT id, ST_Transform(ST_SetSRID(ST_MakePoint(lon, lat), 4326), 3857) as geom
            FROM (VALUES {values}) AS t(id, lon, lat)
        ),
        buildings AS (
            SELECT
                p.id as event_id,
                COALESCE(
                    CASE WHEN b.public_transport IS NOT NULL THEN 'public_transport' END,
                    CASE WHEN b.railway IS NOT NULL THEN 'railway' END,
                    CASE WHEN b.office IS NOT NULL THEN 'office' END,
                    CASE WHEN b.amenity IS NOT NULL THEN 'amenity' END,
                    CASE WHEN b.tourism IS NOT NULL THEN 'tourism' END,
                    CASE WHEN b.shop IS NOT NULL THEN 'shop' END,
                    CASE WHEN b.leisure IS NOT NULL THEN 'leisure' END,
                    CASE WHEN b.building IS NOT NULL THEN 'building' END
                ) as key,
                COALESCE(
                    b.public_transport, b.railway, b.office, b.amenity,
                    b.tourism, b.shop, b.leisure, b.building
                ) as subclass,
                b.name,
                ST_Distance(b.way, p.geom) as distance_m,
                'polygon' as source
            FROM points p
            JOIN planet_osm_polygon b ON ST_DWithin(b.way, p.geom, {radius_m})
            WHERE b.building IS NOT NULL OR b.amenity IS NOT NULL OR b.tourism IS NOT NULL
               OR b.shop IS NOT NULL OR b.leisure IS NOT NULL OR b.public_transport IS NOT NULL
               OR b.railway IS NOT NULL OR b.office IS NOT NULL
        ),
        pois AS (
            SELECT
                p.id as event_id,
                COALESCE(
                    CASE WHEN pt.public_transport IS NOT NULL THEN 'public_transport' END,
                    CASE WHEN pt.railway IS NOT NULL THEN 'railway' END,
                    CASE WHEN pt.highway IN ('bus_stop', 'platform') THEN 'highway' END,
                    CASE WHEN pt.office IS NOT NULL THEN 'office' END,
                    CASE WHEN pt.amenity IS NOT NULL THEN 'amenity' END,
                    CASE WHEN pt.tourism IS NOT NULL THEN 'tourism' END,
                    CASE WHEN pt.shop IS NOT NULL THEN 'shop' END,
                    CASE WHEN pt.leisure IS NOT NULL THEN 'leisure' END
                ) as key,
                COALESCE(
                    pt.public_transport, pt.railway,
                    CASE WHEN pt.highway IN ('bus_stop', 'platform') THEN pt.highway END,
                    pt.office, pt.amenity, pt.tourism, pt.shop, pt.leisure
                ) as subclass,
                pt.name,
                ST_Distance(pt.way, p.geom) as distance_m,
                'point' as source
            FROM points p
            JOIN planet_osm_point pt ON ST_DWithin(pt.way, p.geom, {radius_m})
            WHERE pt.amenity IS NOT NULL OR pt.tourism IS NOT NULL OR pt.shop IS NOT NULL
               OR pt.leisure IS NOT NULL OR pt.public_transport IS NOT NULL
               OR pt.railway IS NOT NULL OR pt.highway IN ('bus_stop', 'platform')
               OR pt.office IS NOT NULL
        )
        SELECT * FROM buildings
        UNION ALL
        SELECT * FROM pois
        """

        cur.execute(query)
        results = cur.fetchall()

        if results:
            batch_df = pd.DataFrame(results, columns=[
                'event_id', 'key', 'subclass', 'name', 'distance_m', 'source'
            ])
            del results

            batch_df['event_id'] = batch_df['event_id'].astype(int)
            batch_df['class'] = classifier.classify_df(batch_df, key_col='key', value_col='subclass')

            for col in ['key', 'subclass', 'source', 'class']:
                batch_df[col] = batch_df[col].astype('category')

            if df is None:
                df = batch_df
            else:
                df = pd.concat([df, batch_df], ignore_index=True)
            del batch_df

    cur.close()

    return df


def query_closest_transit(conn, coords_df: pd.DataFrame, radius_m: int = 500,
                          batch_size: int = 5000, id_col: Optional[str] = None) -> pd.DataFrame:
    """Find the closest public transport stop for each point."""
    from psycopg2.extras import execute_values

    coords_df = coords_df.reset_index(drop=True)

    cur = conn.cursor()
    all_results = []

    for start in tqdm(range(0, len(coords_df), batch_size), desc="Querying transit"):
        batch = coords_df.iloc[start:start + batch_size]

        if id_col and id_col in batch.columns:
            data = [(int(row[id_col]), float(row['lon']), float(row['lat']))
                    for _, row in batch.iterrows()]
        else:
            data = [(start + i, float(row['lon']), float(row['lat']))
                    for i, (_, row) in enumerate(batch.iterrows())]

        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS tmp_points (
                id BIGINT,
                lon DOUBLE PRECISION,
                lat DOUBLE PRECISION
            ) ON COMMIT DROP
        """)
        cur.execute("TRUNCATE tmp_points")

        execute_values(cur, "INSERT INTO tmp_points (id, lon, lat) VALUES %s", data)

        query = f"""
        WITH points AS (
            SELECT id, ST_Transform(ST_SetSRID(ST_MakePoint(lon, lat), 4326), 3857) as geom
            FROM tmp_points
        ),
        transit AS (
            SELECT
                p.id as event_id,
                pt.public_transport,
                pt.railway,
                pt.highway,
                pt.name,
                ST_Distance(pt.way, p.geom) as distance_m,
                pt.tags->'train' as tag_train,
                pt.tags->'subway' as tag_subway,
                pt.tags->'light_rail' as tag_light_rail,
                pt.tags->'tram' as tag_tram,
                pt.tags->'bus' as tag_bus
            FROM points p
            JOIN planet_osm_point pt ON ST_DWithin(pt.way, p.geom, {radius_m})
            WHERE pt.public_transport IS NOT NULL
               OR pt.railway IS NOT NULL
               OR pt.highway IN ('bus_stop', 'platform')

            UNION ALL

            SELECT
                p.id as event_id,
                b.public_transport,
                b.railway,
                NULL as highway,
                b.name,
                ST_Distance(b.way, p.geom) as distance_m,
                b.tags->'train' as tag_train,
                b.tags->'subway' as tag_subway,
                b.tags->'light_rail' as tag_light_rail,
                b.tags->'tram' as tag_tram,
                b.tags->'bus' as tag_bus
            FROM points p
            JOIN planet_osm_polygon b ON ST_DWithin(b.way, p.geom, {radius_m})
            WHERE b.public_transport IS NOT NULL OR b.railway IS NOT NULL
        ),
        closest AS (
            SELECT DISTINCT ON (event_id) *
            FROM transit
            ORDER BY event_id, distance_m
        )
        SELECT * FROM closest
        """

        cur.execute(query)
        all_results.extend(cur.fetchall())

    cur.close()

    df = pd.DataFrame(all_results, columns=[
        'event_id', 'public_transport', 'railway', 'highway', 'name', 'distance_m',
        'tag_train', 'tag_subway', 'tag_light_rail', 'tag_tram', 'tag_bus'
    ])

    df['event_id'] = df['event_id'].astype(int)
    df['transit_mode'] = df.apply(_classify_transit_mode, axis=1)

    return df


def _classify_transit_mode(row) -> str:
    """Classify transit mode from OSM columns and tags."""
    railway = row.get('railway')
    if pd.notna(railway):
        if railway == 'tram_stop':
            return 'tram'
        if railway == 'subway_entrance':
            return 'subway'
        if railway in ('station', 'halt', 'stop'):
            return 'train'
        if railway == 'light_rail':
            return 'light_rail'

    if row.get('highway') == 'bus_stop':
        return 'bus'

    for mode in ['train', 'subway', 'light_rail', 'tram', 'bus']:
        tag_val = row.get(f'tag_{mode}')
        if pd.notna(tag_val) and tag_val != 'no':
            return mode

    return None


def query_closest_transit_parallel(db_params: dict, coords_df: pd.DataFrame, radius_m: int = 500,
                                    batch_size: int = 500, id_col: Optional[str] = None,
                                    n_workers: int = 4) -> pd.DataFrame:
    """Parallel version of query_closest_transit using multiple DB connections."""
    coords_df = coords_df.reset_index(drop=True)

    batches = []
    for start in range(0, len(coords_df), batch_size):
        batch = coords_df.iloc[start:start + batch_size]
        batches.append((start, batch))

    all_results = []
    pbar = tqdm(total=len(batches), desc="Querying transit (parallel)")

    def process_batch(args):
        start, batch = args
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()

        if id_col and id_col in batch.columns:
            values = ", ".join([
                f"({row[id_col]}, {row['lon']}, {row['lat']})"
                for _, row in batch.iterrows()
            ])
        else:
            values = ", ".join([
                f"({start + i}, {row['lon']}, {row['lat']})"
                for i, row in batch.iterrows()
            ])

        query = f"""
        WITH points AS (
            SELECT id, ST_Transform(ST_SetSRID(ST_MakePoint(lon, lat), 4326), 3857) as geom
            FROM (VALUES {values}) AS t(id, lon, lat)
        ),
        transit AS (
            SELECT
                p.id as event_id,
                pt.public_transport,
                pt.railway,
                pt.highway,
                pt.name,
                ST_Distance(pt.way, p.geom) as distance_m,
                pt.tags->'train' as tag_train,
                pt.tags->'subway' as tag_subway,
                pt.tags->'light_rail' as tag_light_rail,
                pt.tags->'tram' as tag_tram,
                pt.tags->'bus' as tag_bus
            FROM points p
            JOIN planet_osm_point pt ON ST_DWithin(pt.way, p.geom, {radius_m})
            WHERE pt.public_transport IS NOT NULL
               OR pt.railway IS NOT NULL
               OR pt.highway IN ('bus_stop', 'platform')

            UNION ALL

            SELECT
                p.id as event_id,
                b.public_transport,
                b.railway,
                NULL as highway,
                b.name,
                ST_Distance(b.way, p.geom) as distance_m,
                b.tags->'train' as tag_train,
                b.tags->'subway' as tag_subway,
                b.tags->'light_rail' as tag_light_rail,
                b.tags->'tram' as tag_tram,
                b.tags->'bus' as tag_bus
            FROM points p
            JOIN planet_osm_polygon b ON ST_DWithin(b.way, p.geom, {radius_m})
            WHERE b.public_transport IS NOT NULL OR b.railway IS NOT NULL
        ),
        closest AS (
            SELECT DISTINCT ON (event_id) *
            FROM transit
            ORDER BY event_id, distance_m
        )
        SELECT * FROM closest
        """

        cur.execute(query)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            results = future.result()
            all_results.extend(results)
            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(all_results, columns=[
        'event_id', 'public_transport', 'railway', 'highway', 'name', 'distance_m',
        'tag_train', 'tag_subway', 'tag_light_rail', 'tag_tram', 'tag_bus'
    ])

    df['event_id'] = df['event_id'].astype(int)
    df['transit_mode'] = df.apply(_classify_transit_mode, axis=1)

    return df


def summarize_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to get counts per class per point."""
    return df.groupby(['event_id', 'class']).size().unstack(fill_value=0)


def summarize_by_key(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to get counts per key per point."""
    return df.groupby(['event_id', 'key']).size().unstack(fill_value=0)


def get_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Get overall distribution of classes."""
    return df['class'].value_counts().to_frame('count')


def get_subclass_distribution(df: pd.DataFrame, class_filter: str = None) -> pd.DataFrame:
    """Get distribution of subclasses, optionally filtered by class."""
    subset = df if class_filter is None else df[df['class'] == class_filter]
    return subset.groupby(['key', 'subclass']).size().reset_index(name='count').sort_values('count', ascending=False)



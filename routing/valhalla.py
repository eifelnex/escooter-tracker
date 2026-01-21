"""Valhalla routing client for e-scooter/bicycle routing."""

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from tqdm.auto import tqdm


def get_route(
    start: tuple[float, float],
    end: tuple[float, float],
    include_geometry: bool = False,
    base_url: str = "http://localhost:8002"
) -> dict:
    request_body = {
        "locations": [
            {"lat": start[0], "lon": start[1]},
            {"lat": end[0], "lon": end[1]}
        ],
        "costing": "bicycle",
        "costing_options": {
            "bicycle": {
                "bicycle_type": "hybrid",
                "cycling_speed": 17,
                "use_roads": 0.5,
                "use_hills": 0.8,
                "avoid_bad_surfaces": 0.9,
                "use_ferry": 0
            }
        },
        "units": "kilometers"
    }

    response = requests.post(f"{base_url}/route", json=request_body)
    response.raise_for_status()
    data = response.json()

    summary = data["trip"]["summary"]
    result = {
        "distance_km": round(summary["length"], 2),
        "duration_min": round(summary["time"] / 60, 1)
    }

    if include_geometry:
        result["polyline"] = data["trip"]["legs"][0]["shape"]

    return result


def get_routes_batch(
    pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    base_url: str = "http://localhost:8002",
    max_workers: int = 10,
    show_progress: bool = True
) -> List[dict]:
    """Batch routing with parallel requests and connection pooling."""
    results = [None] * len(pairs)

    from requests.adapters import HTTPAdapter

    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    def route_single_with_session(args):
        idx, (start, end) = args
        request_body = {
            "locations": [
                {"lat": start[0], "lon": start[1]},
                {"lat": end[0], "lon": end[1]}
            ],
            "costing": "bicycle",
            "costing_options": {
                "bicycle": {
                    "bicycle_type": "hybrid",
                    "cycling_speed": 17,
                    "use_roads": 0.5,
                    "use_hills": 0.8,
                    "avoid_bad_surfaces": 0.9,
                    "use_ferry": 0
                }
            },
            "units": "kilometers"
        }

        response = session.post(f"{base_url}/route", json=request_body)
        response.raise_for_status()
        data = response.json()

        summary = data["trip"]["summary"]
        return idx, {
            "distance_km": round(summary["length"], 2),
            "duration_min": round(summary["time"] / 60, 1)
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(route_single_with_session, (i, pair)) for i, pair in enumerate(pairs)]

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(pairs), desc="Routing")

        for future in iterator:
            idx, result = future.result()
            results[idx] = result

    session.close()

    return results


def get_routes_batch_matrix(
    pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    base_url: str = "http://localhost:8002",
    batch_size: int = 50,
    max_workers: int = 10,
    show_progress: bool = True
) -> List[dict]:
    """Batch routing using Valhalla's matrix API for better performance."""
    results = [None] * len(pairs)

    def create_batches(pairs, batch_size):
        batches = []
        for i in range(0, len(pairs), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(pairs))))
            batches.append(batch_indices)
        return batches

    batches = create_batches(pairs, batch_size)

    from requests.adapters import HTTPAdapter

    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    def process_batch(batch_indices):
        batch_pairs = [pairs[i] for i in batch_indices]

        sources = []
        targets = []
        source_map = {}
        target_map = {}
        pair_to_matrix = []

        for start, end in batch_pairs:
            if start not in source_map:
                source_map[start] = len(sources)
                sources.append({"lat": start[0], "lon": start[1]})

            if end not in target_map:
                target_map[end] = len(targets)
                targets.append({"lat": end[0], "lon": end[1]})

            pair_to_matrix.append((source_map[start], target_map[end]))

        request_body = {
            "sources": sources,
            "targets": targets,
            "costing": "bicycle",
            "costing_options": {
                "bicycle": {
                    "bicycle_type": "hybrid",
                    "cycling_speed": 17,
                    "use_roads": 0.5,
                    "use_hills": 0.8,
                    "avoid_bad_surfaces": 0.9,
                    "use_ferry": 0
                }
            },
            "units": "kilometers"
        }

        response = session.post(f"{base_url}/sources_to_targets", json=request_body)
        response.raise_for_status()
        data = response.json()

        batch_results = []
        matrix = data["sources_to_targets"]

        for src_idx, tgt_idx in pair_to_matrix:
            cell = matrix[src_idx][tgt_idx]
            batch_results.append({
                "distance_km": round(cell["distance"], 2),
                "duration_min": round(cell["time"] / 60, 1)
            })

        return batch_indices, batch_results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(batches), desc="Routing (matrix)")

        for future in iterator:
            batch_indices, batch_results = future.result()
            for idx, result in zip(batch_indices, batch_results):
                results[idx] = result

    session.close()

    return results


def get_routes_by_source(
    candidates_df: "pd.DataFrame",
    base_url: str = "http://localhost:8002",
    max_workers: int = 10,
    show_progress: bool = True
) -> "pd.DataFrame":
    """Route candidates grouped by source using matrix API (1 source to N targets)."""
    import pandas as pd

    from requests.adapters import HTTPAdapter

    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    groups = []
    for d_idx, group in candidates_df.groupby('d_idx', sort=False):
        original_indices = group.index.tolist()
        source = (group['d_lat'].iloc[0], group['d_lon'].iloc[0])
        targets = list(zip(group['f_lat'], group['f_lon']))
        groups.append((original_indices, source, targets))

    results_dist = [None] * len(candidates_df)
    results_time = [None] * len(candidates_df)

    def route_group(args):
        indices, source, targets = args

        request_body = {
            "sources": [{"lat": source[0], "lon": source[1]}],
            "targets": [{"lat": t[0], "lon": t[1]} for t in targets],
            "costing": "bicycle",
            "costing_options": {
                "bicycle": {
                    "bicycle_type": "hybrid",
                    "cycling_speed": 17,
                    "use_roads": 0.5,
                    "use_hills": 0.8,
                    "avoid_bad_surfaces": 0.9,
                    "use_ferry": 0
                }
            },
            "units": "kilometers"
        }

        response = session.post(f"{base_url}/sources_to_targets", json=request_body)
        response.raise_for_status()
        data = response.json()

        matrix = data["sources_to_targets"]
        row = matrix[0]

        results = []
        for cell in row:
            results.append({
                "distance_km": round(cell["distance"], 2),
                "duration_min": round(cell["time"] / 60, 1)
            })

        return indices, results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(route_group, g) for g in groups]

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(groups), desc="Routing (by source)")

        for future in iterator:
            indices, group_results = future.result()

            for idx, result in zip(indices, group_results):
                results_dist[idx] = result["distance_km"]
                results_time[idx] = result["duration_min"]

    session.close()

    return pd.DataFrame({
        'distance_km': results_dist,
        'duration_min': results_time
    })


def visualize_route(polyline: str, start: tuple[float, float], end: tuple[float, float]) -> "folium.Map":
    """Create a folium map with the route visualized."""
    import folium
    import polyline as pl

    coords = pl.decode(polyline, 6)
    center_lat = (start[0] + end[0]) / 2
    center_lon = (start[1] + end[1]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    folium.PolyLine(coords, weight=4, color="blue", opacity=0.8).add_to(m)

    folium.Marker(start, popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(end, popup="End", icon=folium.Icon(color="red")).add_to(m)

    return m

"""MOTIS public transport routing client."""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm


MOTIS_API = "http://localhost:8080/api/v1"
_MOTIS_REFERENCE_DATE = datetime.now() + timedelta(days=7)


def _map_to_motis_window(original_time: datetime) -> datetime:
    """Map historical timestamp to equivalent time within MOTIS's timetable window."""
    original_dow = original_time.weekday()
    ref_dow = _MOTIS_REFERENCE_DATE.weekday()
    days_diff = original_dow - ref_dow
    target_date = _MOTIS_REFERENCE_DATE + timedelta(days=days_diff)
    return datetime(
        target_date.year, target_date.month, target_date.day,
        original_time.hour, original_time.minute, original_time.second
    )


WALKING_SPEED_KMH = 5.0


def get_pt_route(
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
    departure_time: datetime,
    distance_km: Optional[float] = None,
    timeout: int = 30
) -> Dict:
    """Get the best public transport route between two points."""
    walk_only_min = round(distance_km / WALKING_SPEED_KMH * 60, 1) if distance_km else None
    mapped_time = _map_to_motis_window(departure_time)
    query_time_str = mapped_time.strftime("%Y-%m-%dT%H:%M:%S")

    params = {
        "fromPlace": f"{from_lat},{from_lon}",
        "toPlace": f"{to_lat},{to_lon}",
        "time": query_time_str + "Z"
    }

    resp = requests.get(f"{MOTIS_API}/plan", params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    itineraries = data.get("itineraries", [])
    if not itineraries:
        return {
            "success": False,
            "journey_duration_min": walk_only_min,
            "walking_min": walk_only_min,
            "transit_min": 0,
            "wait_min": 0,
            "total_min": walk_only_min,
            "transfers": 0,
            "modes": ["WALK"],
            "walk_only_min": walk_only_min,
            "original_time": departure_time.isoformat(),
            "query_time": query_time_str,
            "journey_start": None,
        }

    # Pick itinerary with earliest arrival time
    itin = min(itineraries, key=lambda x: x["endTime"])
    legs = itin["legs"]

    total_duration = itin["duration"]
    walking_time = sum(leg["duration"] for leg in legs if leg["mode"] == "WALK")
    transit_time = sum(leg["duration"] for leg in legs if leg["mode"] != "WALK")
    transit_legs = [l for l in legs if l["mode"] != "WALK"]
    transfers = len(transit_legs) - 1
    modes = list(set(l["mode"] for l in legs if l["mode"] != "WALK"))

    # Calculate wait time: difference between query time and when journey actually starts
    # Both times use the same format from MOTIS, so we parse them consistently
    journey_start_str = itin["startTime"].replace("Z", "")  # e.g., "2026-02-02T05:34:00"
    journey_start_dt = datetime.fromisoformat(journey_start_str)
    query_dt = datetime.fromisoformat(query_time_str)  # same format, no Z
    wait_seconds = (journey_start_dt - query_dt).total_seconds()
    wait_min = round(max(0, wait_seconds) / 60, 1)

    return {
        "success": True,
        "journey_duration_min": round(total_duration / 60, 1),
        "walking_min": round(walking_time / 60, 1),
        "transit_min": round(transit_time / 60, 1),
        "wait_min": wait_min,
        "total_min": round(total_duration / 60 + wait_min, 1),
        "transfers": transfers,
        "modes": modes,
        "walk_only_min": walk_only_min,
        "original_time": departure_time.isoformat(),
        "query_time": query_time_str,
        "journey_start": itin["startTime"],
    }


def batch_pt_routes(trips_df, max_workers=50, show_progress=True, checkpoint_path=None, chunk_size=5000):
    """Query PT routes for all trips in parallel with checkpointing."""
    import time
    print(f"[DEBUG] Starting batch_pt_routes...")
    print(f"[DEBUG] DataFrame shape: {trips_df.shape}")

    n_total = len(trips_df)
    completed_indices: set = set()
    results_df: Optional[pd.DataFrame] = None

    # Resume from checkpoint if exists
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[DEBUG] Loading checkpoint from {checkpoint_path}...")
        t0 = time.time()
        results_df = pd.read_parquet(checkpoint_path)
        print(f"[DEBUG] Checkpoint loaded in {time.time()-t0:.1f}s, shape: {results_df.shape}")

        t0 = time.time()
        completed_indices = set(results_df.index.tolist())
        print(f"[DEBUG] Built completed_indices set in {time.time()-t0:.1f}s")

        print(f"Resuming from checkpoint: {len(completed_indices):,} / {n_total:,} already done")

    # Find indices still to process
    print(f"[DEBUG] Building remaining indices list...")
    t0 = time.time()
    remaining = [i for i in range(n_total) if i not in completed_indices]
    print(f"[DEBUG] Built remaining list in {time.time()-t0:.1f}s, {len(remaining):,} items to process")

    if not remaining:
        return results_df if results_df is not None else pd.DataFrame()

    pbar = tqdm(total=n_total, initial=len(completed_indices), desc="Querying PT routes", disable=not show_progress, mininterval=0.5)

    # Process in chunks
    chunk_num = 0
    for chunk_start in range(0, len(remaining), chunk_size):
        chunk_num += 1
        chunk_indices = remaining[chunk_start:chunk_start + chunk_size]
        chunk_results = {}
        print(f"[DEBUG] Starting chunk {chunk_num}, indices {chunk_start} to {chunk_start + len(chunk_indices)}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            print(f"[DEBUG] Submitting {len(chunk_indices)} tasks to executor...")
            t0 = time.time()
            futures = {}
            for i in chunk_indices:
                row = trips_df.iloc[i]
                futures[executor.submit(get_pt_route, row.d_lat, row.d_lon, row.f_lat, row.f_lon, row.d_time, row.opt_route_km)] = i
            print(f"[DEBUG] Submitted all tasks in {time.time()-t0:.1f}s, waiting for results...")

            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                chunk_results[idx] = result
                pbar.update(1)

        print(f"[DEBUG] Chunk {chunk_num} complete, processing results...")

        # Convert chunk results to DataFrame
        chunk_df = pd.DataFrame.from_records([
            {**result, '_idx': idx} for idx, result in chunk_results.items()
        ])
        chunk_df.index = chunk_df['_idx']
        chunk_df = chunk_df.drop(columns=['_idx'])

        # Concat with existing results
        if results_df is None:
            results_df = chunk_df
        else:
            results_df = pd.concat([results_df, chunk_df])

        completed_indices.update(chunk_results.keys())

        # Checkpoint after each chunk
        if checkpoint_path:
            print(f"[DEBUG] Saving checkpoint...")
            t0 = time.time()
            results_df.to_parquet(checkpoint_path)
            print(f"[DEBUG] Checkpoint saved in {time.time()-t0:.1f}s")
            pbar.set_postfix({"saved": f"{len(results_df):,}"})

    pbar.close()

    # Sort by index and save final checkpoint
    results_df = results_df.sort_index()

    if checkpoint_path:
        results_df.to_parquet(checkpoint_path)

    return results_df

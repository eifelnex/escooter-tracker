"""MOTIS public transport routing client."""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from zoneinfo import ZoneInfo


MOTIS_API = "http://localhost:8080/api/v1"
LOCAL_TZ = ZoneInfo("Europe/Berlin")
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


def get_pt_route(
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
    departure_time: datetime,
    timeout: int = 30
) -> Dict:
    """Get the best public transport route between two points."""
    mapped_time = _map_to_motis_window(departure_time)
    utc_time = mapped_time.replace(tzinfo=timezone.utc)
    local_time = utc_time.astimezone(LOCAL_TZ)
    query_time_str = local_time.strftime("%Y-%m-%dT%H:%M:%S")

    params = {
        "fromPlace": f"{from_lat},{from_lon}",
        "toPlace": f"{to_lat},{to_lon}",
        "time": query_time_str + "Z"
    }

    resp = requests.get(f"{MOTIS_API}/plan", params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    itineraries = data["itineraries"]
    itin = itineraries[0]
    legs = itin["legs"]

    total_duration = itin["duration"]
    walking_time = sum(leg["duration"] for leg in legs if leg["mode"] == "WALK")
    transit_time = sum(leg["duration"] for leg in legs if leg["mode"] != "WALK")
    transit_legs = [l for l in legs if l["mode"] != "WALK"]
    transfers = len(transit_legs) - 1
    modes = list(set(l["mode"] for l in legs if l["mode"] != "WALK"))

    return {
        "duration_min": round(total_duration / 60, 1),
        "walking_min": round(walking_time / 60, 1),
        "transit_min": round(transit_time / 60, 1),
        "transfers": transfers,
        "modes": modes,
        "original_time": departure_time.isoformat(),
        "query_time": query_time_str,
    }


def batch_pt_routes(trips_df, max_workers=50, show_progress=True):
    """Query PT routes for all trips in parallel."""
    n_total = len(trips_df)
    results: list = [None] * n_total

    rows = list(trips_df.itertuples())

    pbar = tqdm(total=n_total, desc="Querying PT routes", disable=not show_progress)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_pt_route, row.d_lat, row.d_lon, row.f_lat, row.f_lon, row.d_time): i
            for i, row in enumerate(rows)
        }

        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result
            pbar.update(1)

    pbar.close()

    return pd.DataFrame(results)

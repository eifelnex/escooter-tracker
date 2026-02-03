import pandas as pd
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _recalibrate_numba(
    vehicle_ids: np.ndarray,
    timestamps: np.ndarray,
    ranges: np.ndarray,
    first_seen: np.ndarray,
    lookahead_ns: np.int64
) -> np.ndarray:
    n = len(vehicle_ids)
    result = np.full(n, np.nan)

    for i in prange(n):
        if not first_seen[i]:
            continue
        if np.isnan(ranges[i]):
            continue

        vid = vehicle_ids[i]
        t_start = timestamps[i]
        t_end = t_start + lookahead_ns
        max_range = ranges[i]

        # Look forward within same vehicle
        for j in range(i, n):
            if vehicle_ids[j] != vid:
                break
            if timestamps[j] > t_end:
                break
            if not np.isnan(ranges[j]) and ranges[j] > max_range:
                max_range = ranges[j]

        result[i] = max_range

    return result


def recalibrate_first_seen_range(df: pd.DataFrame, lookahead_minutes: int = 60) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.sort_values(['vehicle_id', 'timestamp'])

    lookahead_ns = np.int64(lookahead_minutes * 60 * 1_000_000_000)

    vehicle_ids = df['vehicle_id'].astype('category').cat.codes.values
    timestamps = df['timestamp'].values.astype('int64')
    ranges = df['current_range_meters'].values.astype('float64')
    first_seen = df['first_seen'].values.astype('bool')

    result = _recalibrate_numba(vehicle_ids, timestamps, ranges, first_seen, lookahead_ns)
    df['recalibrated_current_range_meters'] = result
    df = df.sort_index()

    return df

import gc
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from matching.matcher import TripMatcher, MatcherParams, prepare_events
from geo_utils import split_by_city_clusters, visualize_clusters
from routing import get_routes_by_source

# Providers that don't rotate vehicle IDs - use direct same-ID matching
NON_COMPLIANT_PROVIDERS = ['lime_zurich', 'lime_stuttgart', 'lime_basel', 'lime_uster', 'lime_opfikon']


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def match_non_compliant_provider(
    df: pd.DataFrame,
    provider: str,
    min_distance_m: float = 100,
    min_range_drain_m: float = 300,
    max_time_hours: float = 2.0,
    routing_base_url: str = "http://localhost:8002",
    show_progress: bool = True
) -> pd.DataFrame:
    """Match trips for providers with stable IDs by tracking same vehicle_id."""
    prov_df = df[df['provider'] == provider].sort_values(['vehicle_id', 'timestamp'])
    prov_df = prov_df[(prov_df['is_maintenance'] == False) & (prov_df['is_id_reset'] == False)]

    # Get next event index for each row (same vehicle)
    prov_df = prov_df.copy()
    prov_df['next_idx'] = prov_df.groupby('vehicle_id').apply(
        lambda g: g.index.to_series().shift(-1)
    ).droplevel(0)

    # Link disappearances to their next appearance
    disappeared = prov_df[prov_df['disappeared'] == True].copy()
    disappeared = disappeared[disappeared['next_idx'].notna()]

    next_indices = disappeared['next_idx'].astype(int).values
    next_rows = prov_df.loc[next_indices]

    candidates_df = pd.DataFrame({
        'd_idx': disappeared.index,
        'f_idx': next_indices,
        'provider': provider,
        'vehicle_type_id': disappeared['vehicle_type_id'].values,
        'd_lat': disappeared['lat'].values,
        'd_lon': disappeared['lon'].values,
        'f_lat': next_rows['lat'].values,
        'f_lon': next_rows['lon'].values,
        'd_time': disappeared['timestamp'].values,
        'f_time': next_rows['timestamp'].values,
        'd_range_km': disappeared['current_range_meters'].values / 1000,
        'f_range_km': next_rows['current_range_meters'].values / 1000,
    })

    delta_t_seconds = (candidates_df['f_time'].values - candidates_df['d_time'].values) / np.timedelta64(1, 's')
    candidates_df['delta_t_hours'] = delta_t_seconds / 3600

    lat1 = np.radians(candidates_df['d_lat'].values)
    lon1 = np.radians(candidates_df['d_lon'].values)
    lat2 = np.radians(candidates_df['f_lat'].values)
    lon2 = np.radians(candidates_df['f_lon'].values)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    candidates_df['haversine_km'] = 2 * 6371 * np.arcsin(np.sqrt(a))

    candidates_df['range_consumed'] = candidates_df['d_range_km'] - candidates_df['f_range_km']

    candidates_df = candidates_df[
        (candidates_df['delta_t_hours'] > 0) &
        (candidates_df['delta_t_hours'] <= max_time_hours) &
        (candidates_df['haversine_km'] >= min_distance_m / 1000) &
        (candidates_df['d_range_km'].notna()) &
        (candidates_df['f_range_km'].notna()) &
        (candidates_df['range_consumed'] >= min_range_drain_m / 1000)
    ]

    if len(candidates_df) == 0:
        return pd.DataFrame()

    candidates_df = candidates_df.reset_index(drop=True)

    route_results = get_routes_by_source(
        candidates_df,
        base_url=routing_base_url,
        max_workers=30,
        show_progress=show_progress
    )

    candidates_df['opt_route_km'] = route_results['distance_km']
    candidates_df['opt_route_min'] = route_results['duration_min']

    candidates_df = candidates_df[candidates_df['opt_route_km'].notna()]
    candidates_df = candidates_df[candidates_df['opt_route_km'] >= 0.1]

    candidates_df['speed'] = candidates_df['opt_route_km'] / candidates_df['delta_t_hours']

    # For stable-ID providers, matches are deterministic (no ambiguity)
    candidates_df['prob'] = 1.0
    candidates_df['prob_null'] = 0.0
    candidates_df['score'] = 0.0
    candidates_df['prob_forward'] = 1.0

    return candidates_df


def restart_valhalla(
    container_name: str = "valhalla",
    wait_seconds: int = 10,
    tiles_path: str = "C:\\valhalla_tiles",
    port: int = 8002,
    memory: str = "16g"
):
    """Restart Valhalla routing container to free memory between providers."""
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)

    subprocess.run([
        "docker", "run", "-dt",
        "--name", container_name,
        "-p", f"{port}:{port}",
        "-v", f"{tiles_path}:/custom_files",
        "-e", "use_tiles_ignore_pbf=True",
        f"--memory={memory}",
        "ghcr.io/gis-ops/docker-valhalla/valhalla:latest"
    ], check=True, capture_output=True)

    time.sleep(wait_seconds)


# Providers spanning multiple cities - split by geographic clusters
REGIONAL_PROVIDERS = ['voi_de', 'voi_ch']


def run_pipeline(
    events_path: str = "events_with_flags.parquet",
    output_dir: str = "matching_output",
    use_recalibrated_range: bool = True,
    providers: list = None,
    restart_valhalla_between: bool = True,
    valhalla_container: str = "valhalla",
    **matcher_kwargs
):
    """Run trip matching pipeline for all providers."""
    output_path = Path(output_dir)
    candidates_dir = output_path / "candidates"
    output_path.mkdir(exist_ok=True)
    candidates_dir.mkdir(exist_ok=True)

    df = pd.read_parquet(events_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    all_providers = df['provider'].unique().tolist()
    providers = providers or all_providers

    disappeared_all, first_seen_all = prepare_events(df, use_recalibrated_range=use_recalibrated_range)

    summary = []

    for provider in tqdm(providers, desc="Providers"):
        d_mask = disappeared_all['provider'] == provider
        f_mask = first_seen_all['provider'] == provider

        n_disappeared = d_mask.sum()
        n_first_seen = f_mask.sum()

        if provider in NON_COMPLIANT_PROVIDERS:
            candidates_path = candidates_dir / f"{provider}_candidates.parquet"

            if candidates_path.exists():
                continue

            candidates_df = match_non_compliant_provider(
                df, provider,
                min_distance_m=100,
                min_range_drain_m=300,
                max_time_hours=2.0,
                show_progress=True
            )

            if len(candidates_df) > 0:
                candidates_df.to_parquet(candidates_path)

            summary.append({
                'provider': provider,
                'n_disappeared': n_disappeared,
                'n_first_seen': n_first_seen,
                'n_candidates': len(candidates_df),
            })

            del candidates_df
            gc.collect()

            if restart_valhalla_between:
                restart_valhalla(container_name=valhalla_container)

            continue

        if provider in REGIONAL_PROVIDERS:
            cluster_cache_path = output_path / f"{provider}_clusters.pkl"

            if cluster_cache_path.exists():
                import pickle
                with open(cluster_cache_path, 'rb') as f:
                    clusters = pickle.load(f)
            else:
                coords_df = pd.concat([
                    disappeared_all.loc[d_mask, ['lat', 'lon']],
                    first_seen_all.loc[f_mask, ['lat', 'lon']]
                ])

                clusters, cluster_metadata = split_by_city_clusters(coords_df, cluster_radius_km=10.0, min_samples=30)

                del coords_df
                gc.collect()

                import pickle
                with open(cluster_cache_path, 'wb') as f:
                    pickle.dump(clusters, f)

                cluster_map = visualize_clusters(cluster_metadata, title=f"{provider} City Clusters")
                if cluster_map:
                    map_path = output_path / f"{provider}_clusters.html"
                    cluster_map.save(str(map_path))

            for cluster_name, cluster_indices in clusters.items():
                if cluster_name == 'noise':
                    cluster_name = 'other'

                sub_provider = f"{provider}_{cluster_name}"
                candidates_path = candidates_dir / f"{sub_provider}_candidates.parquet"

                if candidates_path.exists():
                    continue

                cluster_set = set(cluster_indices)

                d_cluster_mask = d_mask & disappeared_all.index.isin(cluster_set)
                f_cluster_mask = f_mask & first_seen_all.index.isin(cluster_set)

                disappeared = disappeared_all.loc[d_cluster_mask].copy()
                first_seen = first_seen_all.loc[f_cluster_mask].copy()

                matcher = TripMatcher(MatcherParams())
                result = matcher.fit(
                    disappeared,
                    first_seen,
                    candidates_cache=str(candidates_path),
                    **matcher_kwargs
                )

                if len(result.candidates) > 0:
                    result.candidates.to_parquet(candidates_path)

                summary.append({
                    'provider': sub_provider,
                    'n_disappeared': d_cluster_mask.sum(),
                    'n_first_seen': f_cluster_mask.sum(),
                    'n_candidates': len(result.candidates),
                })

                del disappeared, first_seen, matcher, result, cluster_set
                gc.collect()

            del clusters
            gc.collect()

            if restart_valhalla_between:
                restart_valhalla(container_name=valhalla_container)
        else:
            candidates_path = candidates_dir / f"{provider}_candidates.parquet"

            if candidates_path.exists():
                continue

            disappeared = disappeared_all.loc[d_mask].copy()
            first_seen = first_seen_all.loc[f_mask].copy()

            matcher = TripMatcher(MatcherParams())
            result = matcher.fit(
                disappeared,
                first_seen,
                candidates_cache=str(candidates_path),
                **matcher_kwargs
            )

            if len(result.candidates) > 0:
                result.candidates.to_parquet(candidates_path)

            summary.append({
                'provider': provider,
                'n_disappeared': n_disappeared,
                'n_first_seen': n_first_seen,
                'n_candidates': len(result.candidates),
            })

            del disappeared, first_seen, matcher, result
            gc.collect()

            if restart_valhalla_between:
                restart_valhalla(container_name=valhalla_container)

    summary_df = pd.DataFrame(summary)
    summary_df.to_parquet(output_path / "summary.parquet")

    return summary_df


def load_all_candidates(output_dir: str = "matching_output") -> pd.DataFrame:
    files = list((Path(output_dir) / "candidates").glob("*_candidates.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


if __name__ == "__main__":
    run_pipeline()

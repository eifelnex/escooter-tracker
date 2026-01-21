"""Pipeline for processing trip matching per provider."""

import gc
import subprocess
import time
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from matching.matcher import TripMatcher, MatcherParams, prepare_events
from geo_utils import split_by_city_clusters, visualize_clusters


def restart_valhalla(
    container_name: str = "valhalla",
    wait_seconds: int = 10,
    tiles_path: str = "C:\\valhalla_tiles",
    port: int = 8002,
    memory: str = "16g"
):
    print(f"\n  [valhalla] Stopping and removing container '{container_name}'...")
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)

    print(f"  [valhalla] Starting fresh container...")
    subprocess.run([
        "docker", "run", "-dt",
        "--name", container_name,
        "-p", f"{port}:{port}",
        "-v", f"{tiles_path}:/custom_files",
        "-e", "use_tiles_ignore_pbf=True",
        f"--memory={memory}",
        "ghcr.io/gis-ops/docker-valhalla/valhalla:latest"
    ], check=True, capture_output=True)

    print(f"  [valhalla] Waiting {wait_seconds}s for startup...")
    time.sleep(wait_seconds)
    print(f"  [valhalla] Container ready")

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
    output_path = Path(output_dir)
    candidates_dir = output_path / "candidates"
    output_path.mkdir(exist_ok=True)
    candidates_dir.mkdir(exist_ok=True)

    print(f"Loading {events_path}...")
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

        if provider in REGIONAL_PROVIDERS:
            print(f"\n{provider}: Splitting by city clusters...")

            cluster_cache_path = output_path / f"{provider}_clusters.pkl"

            if cluster_cache_path.exists():
                print(f"  [pipeline] Loading cached clusters from {cluster_cache_path}")
                import pickle
                with open(cluster_cache_path, 'rb') as f:
                    clusters = pickle.load(f)
                print(f"  [pipeline] Loaded {len(clusters)} clusters")
            else:
                print(f"  [pipeline] Creating coords_df with concat...")
                coords_df = pd.concat([
                    disappeared_all.loc[d_mask, ['lat', 'lon']],
                    first_seen_all.loc[f_mask, ['lat', 'lon']]
                ])
                print(f"  [pipeline] coords_df created: {len(coords_df):,} rows")

                print(f"  [pipeline] Calling split_by_city_clusters...")
                clusters, cluster_metadata = split_by_city_clusters(coords_df, cluster_radius_km=10.0, min_samples=30)
                print(f"  [pipeline] split_by_city_clusters returned")

                print(f"  [pipeline] Deleting coords_df...")
                del coords_df
                gc.collect()
                print(f"  [pipeline] coords_df deleted")

                import pickle
                with open(cluster_cache_path, 'wb') as f:
                    pickle.dump(clusters, f)
                print(f"  [pipeline] Saved clusters to {cluster_cache_path}")

                cluster_map = visualize_clusters(cluster_metadata, title=f"{provider} City Clusters")
                if cluster_map:
                    map_path = output_path / f"{provider}_clusters.html"
                    cluster_map.save(str(map_path))
                    print(f"  [pipeline] Saved cluster map to {map_path}")

            for cluster_name, cluster_indices in clusters.items():
                if cluster_name == 'noise':
                    cluster_name = 'other'

                sub_provider = f"{provider}_{cluster_name}"
                candidates_path = candidates_dir / f"{sub_provider}_candidates.parquet"

                if candidates_path.exists():
                    print(f"\nSkipping {sub_provider} (cached)")
                    continue

                cluster_set = set(cluster_indices)

                d_cluster_mask = d_mask & disappeared_all.index.isin(cluster_set)
                f_cluster_mask = f_mask & first_seen_all.index.isin(cluster_set)

                n_d = d_cluster_mask.sum()
                n_f = f_cluster_mask.sum()

                print(f"\n{sub_provider}: {n_d:,} disappeared, {n_f:,} first_seen")

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
                    'n_disappeared': n_d,
                    'n_first_seen': n_f,
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
                print(f"\nSkipping {provider} (cached)")
                continue

            print(f"\n{provider}: {n_disappeared:,} disappeared, {n_first_seen:,} first_seen")

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

    total_candidates = summary_df['n_candidates'].sum()
    total_disappeared = summary_df['n_disappeared'].sum()
    print(f"\nTotal: {total_candidates:,} candidates for {total_disappeared:,} disappearances")
    print(summary_df.to_string())

    return summary_df


def load_all_candidates(output_dir: str = "matching_output") -> pd.DataFrame:
    files = list((Path(output_dir) / "candidates").glob("*_candidates.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


if __name__ == "__main__":
    run_pipeline()

"""Geographic utilities for splitting datasets by city/region."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from typing import Dict, Tuple


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in meters."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    r = 6371000  # Radius of earth in meters
    return c * r


def split_by_city_clusters(
    df: pd.DataFrame,
    cluster_radius_km: float = 10.0,
    min_samples: int = 30,
    max_coords_for_dbscan: int = 100000,
    merge_clusters_within_km: float = 20.0,
    assign_all_to_nearest: bool = True
) -> Tuple[Dict[str, pd.Index], dict]:
    """Split DataFrame into geographic clusters using DBSCAN, return indices and metadata."""
    if len(df) == 0:
        return {}, {}

    print(f"  [cluster] Step 1: Getting unique coords from {len(df):,} rows...")
    unique_coords = df[['lat', 'lon']].drop_duplicates()

    if len(unique_coords) > max_coords_for_dbscan:
        print(f"  [cluster] Step 1a: Sampling {max_coords_for_dbscan:,} from {len(unique_coords):,} unique coords...")
        sample_coords = unique_coords.sample(n=max_coords_for_dbscan, random_state=42).values
    else:
        sample_coords = unique_coords.values
    print(f"  [cluster] Step 1 done: {len(sample_coords):,} coords for DBSCAN")

    eps_rad = cluster_radius_km / 6371.0

    print(f"  [cluster] Step 2: Running DBSCAN (eps={cluster_radius_km}km, min_samples={min_samples})...")
    clustering = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric='haversine',
        algorithm='ball_tree'
    ).fit(np.radians(sample_coords))

    labels = clustering.labels_
    unique_labels = set(labels) - {-1}
    print(f"  [cluster] Step 2 done: {len(unique_labels)} clusters found")

    if len(unique_labels) == 0:
        print("  [cluster] No clusters found, returning all as noise")
        return {'noise': df.index}, {'centers': [], 'noise_sample': sample_coords[:1000]}

    print(f"  [cluster] Step 3: Computing cluster centers...")
    cluster_centers = []
    cluster_ids = []
    for label in sorted(unique_labels):
        mask = labels == label
        center_lat = sample_coords[mask, 0].mean()
        center_lon = sample_coords[mask, 1].mean()
        cluster_centers.append([center_lat, center_lon])
        cluster_ids.append(label)

    cluster_centers = np.array(cluster_centers)
    print(f"  [cluster] Step 3 done: {len(cluster_centers)} cluster centers")

    if merge_clusters_within_km > 0 and len(cluster_centers) > 1:
        print(f"  [cluster] Step 3b: Merging clusters within {merge_clusters_within_km}km...")
        from sklearn.metrics.pairwise import haversine_distances
        center_dists = haversine_distances(np.radians(cluster_centers)) * 6371.0

        merged = [False] * len(cluster_centers)
        new_centers = []
        new_ids = []
        cluster_id_mapping = {}  # old_id -> new_id

        for i in range(len(cluster_centers)):
            if merged[i]:
                continue
            to_merge = [i]
            for j in range(i + 1, len(cluster_centers)):
                if not merged[j] and center_dists[i, j] < merge_clusters_within_km:
                    to_merge.append(j)
                    merged[j] = True

            merged_center = cluster_centers[to_merge].mean(axis=0)
            new_id = len(new_centers)
            new_centers.append(merged_center)
            new_ids.append(new_id)

            for idx in to_merge:
                cluster_id_mapping[cluster_ids[idx]] = new_id
            merged[i] = True

        if len(new_centers) < len(cluster_centers):
            print(f"  [cluster] Step 3b done: Merged {len(cluster_centers)} -> {len(new_centers)} clusters")
            cluster_centers = np.array(new_centers)
            cluster_ids = new_ids
        else:
            print(f"  [cluster] Step 3b done: No clusters merged")
            cluster_id_mapping = {cid: cid for cid in cluster_ids}
    else:
        cluster_id_mapping = {cid: cid for cid in cluster_ids}

    print(f"  [cluster] Step 4: Building BallTree with cluster centers...")
    tree = BallTree(np.radians(cluster_centers), metric='haversine')

    print(f"  [cluster] Step 4a: Querying nearest cluster for all {len(df):,} events...")
    all_coords = df[['lat', 'lon']].values
    distances, indices = tree.query(np.radians(all_coords), k=1)

    distances_km = distances.flatten() * 6371.0
    nearest_cluster_idx = indices.flatten()

    cluster_labels = np.array([cluster_ids[i] for i in nearest_cluster_idx])

    if not assign_all_to_nearest:
        max_assign_dist_km = cluster_radius_km * 1.5
        cluster_labels[distances_km > max_assign_dist_km] = -1
        n_noise = (cluster_labels == -1).sum()
        print(f"  [cluster] Step 4 done: {n_noise:,} events too far from clusters (noise)")
    else:
        print(f"  [cluster] Step 4 done: All events assigned to nearest cluster (no noise)")

    noise_mask = distances_km > (cluster_radius_km * 1.5)
    noise_coords = all_coords[noise_mask]
    noise_sample = noise_coords[np.random.choice(len(noise_coords), min(1000, len(noise_coords)), replace=False)] if len(noise_coords) > 0 else np.array([])

    print(f"  [cluster] Step 5: Cleaning up...")
    del sample_coords, unique_coords, tree, distances, indices
    import gc
    gc.collect()
    print(f"  [cluster] Step 5 done")

    print(f"  [cluster] Step 6: Building result dict...")
    result = {}
    cluster_labels_series = pd.Series(cluster_labels, index=df.index)

    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1 and assign_all_to_nearest:
            continue

        cluster_indices = df.index[cluster_labels_series == cluster_id]

        if cluster_id == -1:
            name = 'noise'
        else:
            center_idx = cluster_ids.index(cluster_id)
            center_lat, center_lon = cluster_centers[center_idx]
            name = f'cluster_{cluster_id}_({center_lat:.2f},{center_lon:.2f})'

        result[name] = cluster_indices

    print(f"Split into {len(result)} clusters:")
    for name, indices in result.items():
        print(f"  {name}: {len(indices):,} events")

    metadata = {
        'centers': cluster_centers,
        'cluster_ids': cluster_ids,
        'noise_sample': noise_sample,
        'cluster_radius_km': cluster_radius_km
    }

    return result, metadata


def visualize_clusters(metadata: dict, title: str = "City Clusters") -> "folium.Map":
    """Visualize cluster centers and noise samples on a folium map."""
    import folium
    from folium.plugins import MarkerCluster

    centers = metadata.get('centers', [])
    noise_sample = metadata.get('noise_sample', [])
    cluster_radius_km = metadata.get('cluster_radius_km', 10)

    if len(centers) == 0:
        print("No clusters to visualize")
        return None

    center_lat = np.mean([c[0] for c in centers])
    center_lon = np.mean([c[1] for c in centers])

    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue']

    for i, (lat, lon) in enumerate(centers):
        color = colors[i % len(colors)]

        folium.Marker(
            [lat, lon],
            popup=f"Cluster {i}: ({lat:.2f}, {lon:.2f})",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)

        folium.Circle(
            [lat, lon],
            radius=cluster_radius_km * 1000,
            color=color,
            fill=True,
            fill_opacity=0.1,
            popup=f"Cluster {i} radius: {cluster_radius_km}km"
        ).add_to(m)

    if len(noise_sample) > 0:
        noise_group = folium.FeatureGroup(name="Noise samples")
        for lat, lon in noise_sample[:500]:
            folium.CircleMarker(
                [lat, lon],
                radius=2,
                color='gray',
                fill=True,
                fill_opacity=0.5
            ).add_to(noise_group)
        noise_group.add_to(m)

    folium.LayerControl().add_to(m)

    return m

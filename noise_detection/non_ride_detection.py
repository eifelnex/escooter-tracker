import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.neighbors import BallTree


def add_first_seen_column(df: pd.DataFrame):
    df['first_seen'] = df['change_types'].apply(lambda x: 'first_seen' in str(x) if pd.notna(x) else False)
    return df


def flag_temporary_disappearances(df: pd.DataFrame):
    """Flags all disappearances except the last one per vehicle as temporary."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_temporary_disappearance'] = False

    disappeared_df = df[df['disappeared'] == True].copy()
    last_disappearance_idx = disappeared_df.groupby('vehicle_id')['timestamp'].idxmax()
    all_disappeared_idx = disappeared_df.index
    temp_idx = all_disappeared_idx.difference(last_disappearance_idx.tolist())

    df.loc[temp_idx, 'is_temporary_disappearance'] = True

    return df


def add_last_seen_column(df: pd.DataFrame):
    df['last_seen'] = df['change_types'].apply(lambda x: 'last_seen' in str(x) if pd.notna(x) else False)
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two points on earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371000


def check_vehicle_type_ranges(df: pd.DataFrame):
    return df.groupby('vehicle_type_id')['current_range_meters'].max().to_dict()


def flag_maintenance_events(df: pd.DataFrame,
                            time_window_minutes=30,
                            distance_threshold_meters=50,
                            battery_threshold=0.85,
                            battery_increase=0.1,
                            range_threshold=0.8,
                            range_increase_meters=5000,
                            vehicle_type_max_ranges=None,
                            time_bin_minutes=10,
                            show_progress=True):
    """Detects maintenance/recharge events where scooters reappear with higher battery."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_maintenance'] = False

    vehicle_type_max_ranges = vehicle_type_max_ranges or check_vehicle_type_ranges(df)

    vehicle_type_thresholds = {
        vtype: max_range * range_threshold
        for vtype, max_range in vehicle_type_max_ranges.items()
    }

    maintenance_indices = set()
    distance_threshold_radians = distance_threshold_meters / 6371000
    bins_to_check = int(np.ceil(time_window_minutes / time_bin_minutes)) + 1

    providers = df['provider'].unique()
    provider_iterator = tqdm(providers, desc="Processing providers", disable=not show_progress)

    for provider in provider_iterator:
        provider_df = df[df['provider'] == provider]

        has_battery = provider_df['current_fuel_percent'].notna().sum() > 0
        has_range = provider_df['current_range_meters'].notna().sum() > 0

        disappeared_mask = provider_df['disappeared'] == True

        if has_battery:
            first_seen_mask = (provider_df['first_seen'] == True) & \
                              (provider_df['current_fuel_percent'] >= battery_threshold)
            use_battery = True
        elif has_range:
            def check_range_threshold(row):
                vtype = row['vehicle_type_id']
                threshold = vehicle_type_thresholds.get(vtype, 0)
                return row['current_range_meters'] >= threshold

            first_seen_mask = (provider_df['first_seen'] == True) & \
                              provider_df.apply(check_range_threshold, axis=1)
            use_battery = False
        else:
            first_seen_mask = provider_df['first_seen'] == True
            use_battery = None

        disappeared_events = provider_df[disappeared_mask].copy()
        first_seen_events = provider_df[first_seen_mask].copy()

        if len(disappeared_events) == 0 or len(first_seen_events) == 0:
            continue

        matched_first_seen_provider = set()

        min_time = min(disappeared_events['timestamp'].min(),
                       first_seen_events['timestamp'].min())

        disappeared_events['time_bin'] = (
            (disappeared_events['timestamp'] - min_time).dt.total_seconds() // (time_bin_minutes * 60)
        ).astype(int)

        first_seen_events['time_bin'] = (
            (first_seen_events['timestamp'] - min_time).dt.total_seconds() // (time_bin_minutes * 60)
        ).astype(int)

        fs_by_bin = {}
        for bin_id, group in first_seen_events.groupby('time_bin'):
            coords = np.radians(group[['lat', 'lon']].values)
            bin_data = {
                'tree': BallTree(coords, metric='haversine'),
                'times': group['timestamp'].values,
                'indices': group.index.values,
            }
            if use_battery == True:
                bin_data['energy'] = group['current_fuel_percent'].values
                bin_data['vehicle_types'] = None
            elif use_battery == False:
                bin_data['energy'] = group['current_range_meters'].values
                bin_data['vehicle_types'] = group['vehicle_type_id'].values
            else:
                bin_data['energy'] = None
                bin_data['vehicle_types'] = None
            fs_by_bin[bin_id] = bin_data

        time_window_ns = np.timedelta64(time_window_minutes, 'm')

        for bin_id, d_group in disappeared_events.groupby('time_bin'):
            relevant_bins = [bin_id + offset for offset in range(bins_to_check + 1)
                            if (bin_id + offset) in fs_by_bin]

            if not relevant_bins:
                continue

            d_coords = np.radians(d_group[['lat', 'lon']].values)
            d_times = d_group['timestamp'].values
            d_indices = d_group.index.values

            if use_battery == True:
                d_energy = d_group['current_fuel_percent'].values
            elif use_battery == False:
                d_energy = d_group['current_range_meters'].values
            else:
                d_energy = None

            for fs_bin_id in relevant_bins:
                fs_data = fs_by_bin[fs_bin_id]
                neighbors_list = fs_data['tree'].query_radius(d_coords, r=distance_threshold_radians)

                for i, neighbor_indices in enumerate(neighbors_list):
                    if len(neighbor_indices) == 0:
                        continue

                    d_time = d_times[i]
                    d_idx = d_indices[i]
                    d_enrg = d_energy[i] if d_energy is not None else None

                    fs_times = fs_data['times'][neighbor_indices]
                    fs_energy = fs_data['energy'][neighbor_indices] if fs_data['energy'] is not None else None
                    fs_indices_local = fs_data['indices'][neighbor_indices]

                    time_valid = (fs_times > d_time) & (fs_times <= d_time + time_window_ns)

                    if fs_energy is not None and d_enrg is not None:
                        if use_battery == True:
                            energy_increased = fs_energy >= (d_enrg + battery_increase)
                        else:
                            energy_increased = fs_energy >= (d_enrg + range_increase_meters)
                    else:
                        energy_increased = np.ones(len(neighbor_indices), dtype=bool)

                    not_matched = np.array([idx not in matched_first_seen_provider for idx in fs_indices_local])
                    valid = time_valid & energy_increased & not_matched

                    if valid.any():
                        valid_indices = np.where(valid)[0]
                        closest_idx = valid_indices[np.argmin(fs_times[valid_indices])]
                        fs_match_idx = fs_indices_local[closest_idx]

                        maintenance_indices.add(d_idx)
                        maintenance_indices.add(fs_match_idx)
                        matched_first_seen_provider.add(fs_match_idx)

    if maintenance_indices:
        df.loc[list(maintenance_indices), 'is_maintenance'] = True

    return df


def flag_id_reset_events(df: pd.DataFrame,
                         time_windows_minutes=[(0, 45), (115, 125)],
                         distance_threshold_meters=30,
                         battery_decrease_max=0.01,
                         range_decrease_max_meters=100,
                         time_bin_minutes=30,
                         show_progress=True):
    """Detects vehicle ID resets where scooters reappear with new ID but similar battery."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_id_reset'] = False

    time_windows = list(time_windows_minutes)

    max_time_minutes = max(end for _, end in time_windows)

    id_reset_indices = set()
    distance_threshold_radians = distance_threshold_meters / 6371000
    bins_to_check = int(np.ceil(max_time_minutes / time_bin_minutes)) + 1

    providers = df['provider'].unique()
    provider_iterator = tqdm(providers, desc="Processing providers for ID resets", disable=not show_progress)

    for provider in provider_iterator:
        provider_df = df[df['provider'] == provider]

        has_battery = provider_df['current_fuel_percent'].notna().sum() > 0
        has_range = provider_df['current_range_meters'].notna().sum() > 0

        disappeared_mask = provider_df['disappeared'] == True
        first_seen_mask = provider_df['first_seen'] == True

        disappeared_events = provider_df[disappeared_mask].copy()
        first_seen_events = provider_df[first_seen_mask].copy()

        if len(disappeared_events) == 0 or len(first_seen_events) == 0:
            continue

        matched_first_seen_provider = set()
        matched_disappeared_provider = set()

        if has_battery:
            use_battery = True
        elif has_range:
            use_battery = False
        else:
            use_battery = None

        min_time = min(disappeared_events['timestamp'].min(),
                       first_seen_events['timestamp'].min())

        disappeared_events['time_bin'] = (
            (disappeared_events['timestamp'] - min_time).dt.total_seconds() // (time_bin_minutes * 60)
        ).astype(int)

        first_seen_events['time_bin'] = (
            (first_seen_events['timestamp'] - min_time).dt.total_seconds() // (time_bin_minutes * 60)
        ).astype(int)

        fs_by_bin = {}
        for bin_id, group in first_seen_events.groupby('time_bin'):
            coords = np.radians(group[['lat', 'lon']].values)
            bin_data = {
                'tree': BallTree(coords, metric='haversine'),
                'times': group['timestamp'].values,
                'indices': group.index.values,
                'vehicle_ids': group['vehicle_id'].values,
            }
            if use_battery == True:
                bin_data['energy'] = group['current_fuel_percent'].values
            elif use_battery == False:
                bin_data['energy'] = group['current_range_meters'].values
            else:
                bin_data['energy'] = None
            fs_by_bin[bin_id] = bin_data

        time_window_bounds_ns = [
            (np.timedelta64(start, 'm'), np.timedelta64(end, 'm'))
            for start, end in time_windows
        ]

        for bin_id, d_group in disappeared_events.groupby('time_bin'):
            relevant_bins = [bin_id + offset for offset in range(bins_to_check + 1)
                            if (bin_id + offset) in fs_by_bin]

            if not relevant_bins:
                continue

            d_coords = np.radians(d_group[['lat', 'lon']].values)
            d_times = d_group['timestamp'].values
            d_indices = d_group.index.values
            d_vehicle_ids = d_group['vehicle_id'].values

            if use_battery == True:
                d_energy = d_group['current_fuel_percent'].values
            elif use_battery == False:
                d_energy = d_group['current_range_meters'].values
            else:
                d_energy = None

            for fs_bin_id in relevant_bins:
                fs_data = fs_by_bin[fs_bin_id]
                neighbors_list = fs_data['tree'].query_radius(d_coords, r=distance_threshold_radians)

                for i, neighbor_indices in enumerate(neighbors_list):
                    if len(neighbor_indices) == 0:
                        continue

                    d_idx = d_indices[i]

                    if d_idx in matched_disappeared_provider:
                        continue

                    d_time = d_times[i]
                    d_enrg = d_energy[i] if d_energy is not None else None
                    d_vehicle_id = d_vehicle_ids[i]

                    fs_times = fs_data['times'][neighbor_indices]
                    fs_energy = fs_data['energy'][neighbor_indices] if fs_data['energy'] is not None else None
                    fs_indices_local = fs_data['indices'][neighbor_indices]
                    fs_vehicle_ids = fs_data['vehicle_ids'][neighbor_indices]

                    time_diffs = fs_times - d_time
                    time_valid = np.zeros(len(neighbor_indices), dtype=bool)
                    for window_start_ns, window_end_ns in time_window_bounds_ns:
                        in_window = (time_diffs > window_start_ns) & (time_diffs <= window_end_ns)
                        time_valid |= in_window

                    different_id = fs_vehicle_ids != d_vehicle_id

                    if fs_energy is not None and d_enrg is not None:
                        if use_battery == True:
                            energy_valid = (fs_energy >= (d_enrg - battery_decrease_max)) & (fs_energy <= d_enrg)
                        else:
                            energy_valid = (fs_energy >= (d_enrg - range_decrease_max_meters)) & (fs_energy <= d_enrg)
                    else:
                        energy_valid = np.ones(len(neighbor_indices), dtype=bool)

                    not_matched = np.array([idx not in matched_first_seen_provider for idx in fs_indices_local])
                    valid = time_valid & different_id & energy_valid & not_matched

                    if valid.any():
                        valid_indices = np.where(valid)[0]
                        closest_idx = valid_indices[np.argmin(fs_times[valid_indices])]
                        fs_match_idx = fs_indices_local[closest_idx]

                        id_reset_indices.add(d_idx)
                        id_reset_indices.add(fs_match_idx)
                        matched_disappeared_provider.add(d_idx)
                        matched_first_seen_provider.add(fs_match_idx)

    if id_reset_indices:
        df.loc[list(id_reset_indices), 'is_id_reset'] = True

    return df


def detect_stationary_id_resets(df: pd.DataFrame,
                                max_time_minutes=180,
                                distance_threshold_meters=50,
                                battery_decrease_max=0.01,
                                range_decrease_max_meters=500,
                                show_progress=True):
    """Detects stationary ID resets and returns detailed match information."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    distance_threshold_radians = distance_threshold_meters / 6371000

    disappeared = df[df['disappeared'] == True].copy().reset_index(drop=True)
    first_seen = df[df['first_seen'] == True].copy().reset_index(drop=True)

    matched_pairs = []

    providers = df['provider'].unique()
    provider_iterator = tqdm(providers, desc="Detecting stationary ID resets", disable=not show_progress)

    for provider in provider_iterator:
        d_prov = disappeared[disappeared['provider'] == provider].copy()
        f_prov = first_seen[first_seen['provider'] == provider].copy()

        if len(d_prov) == 0 or len(f_prov) == 0:
            continue

        d_prov = d_prov.sort_values('timestamp').reset_index(drop=True)
        f_prov = f_prov.sort_values('timestamp').reset_index(drop=True)

        matched_fs_indices = set()

        f_coords = np.radians(f_prov[['lat', 'lon']].values)
        tree = BallTree(f_coords, metric='haversine')

        for d_idx, d_row in d_prov.iterrows():
            d_coord = np.radians([[d_row['lat'], d_row['lon']]])
            d_time = d_row['timestamp']
            d_battery = d_row['current_fuel_percent']
            d_range = d_row['current_range_meters']
            d_vehicle = d_row['vehicle_id']

            indices, distances = tree.query_radius(d_coord, r=distance_threshold_radians, return_distance=True)
            indices = indices[0]
            distances = distances[0]

            if len(indices) == 0:
                continue

            best_match = None
            best_score = float('inf')

            for idx, dist in zip(indices, distances):
                if idx in matched_fs_indices:
                    continue

                f_row = f_prov.iloc[idx]
                f_time = f_row['timestamp']
                f_battery = f_row['current_fuel_percent']
                f_range = f_row['current_range_meters']

                time_diff = (f_time - d_time).total_seconds() / 60
                if time_diff <= 0 or time_diff > max_time_minutes:
                    continue

                has_battery = pd.notna(d_battery) and pd.notna(f_battery)
                has_range = pd.notna(d_range) and pd.notna(f_range)

                if has_battery:
                    if not (f_battery <= d_battery and f_battery >= d_battery - battery_decrease_max):
                        continue
                elif has_range:
                    if not (f_range <= d_range and f_range >= d_range - range_decrease_max_meters):
                        continue

                score = time_diff + (dist * 6371000)

                if score < best_score:
                    best_score = score
                    best_match = {
                        'disappeared_id': d_vehicle,
                        'appeared_id': f_row['vehicle_id'],
                        'provider': provider,
                        'time_diff_min': time_diff,
                        'distance_m': dist * 6371000,
                        'd_time': d_time,
                        'fs_time': f_time,
                        'd_battery': d_battery,
                        'fs_battery': f_battery,
                        'fs_idx': idx,
                    }

            if best_match:
                matched_fs_indices.add(best_match['fs_idx'])
                del best_match['fs_idx']
                matched_pairs.append(best_match)

    return pd.DataFrame(matched_pairs)


def estimate_fleet_size(df: pd.DataFrame, spike_window_minutes=30, use_spikes=True):
    """Estimates scooter count per provider using collection start spikes or unique vehicles."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    providers = df['provider'].unique()
    fleet_estimates = {}

    if use_spikes:
        first_seen_df = df[df['first_seen'] == True].copy()
        first_seen_df['hour_bin'] = first_seen_df['timestamp'].dt.floor('H')
        hourly_counts = first_seen_df.groupby('hour_bin').size()

        mean_count = hourly_counts.mean()
        std_count = hourly_counts.std()
        threshold = mean_count + 2 * std_count
        spike_hours = hourly_counts[hourly_counts > threshold].index

        for provider in tqdm(providers, desc="Estimating fleet sizes"):
            provider_df = df[df['provider'] == provider]
            max_count = 0

            for spike_time in spike_hours:
                window_start = spike_time
                window_end = spike_time + pd.Timedelta(minutes=spike_window_minutes)

                window_df = provider_df[
                    (provider_df['timestamp'] >= window_start) &
                    (provider_df['timestamp'] < window_end)
                ]
                unique_count = window_df['vehicle_id'].nunique()
                max_count = max(max_count, unique_count)

            fleet_estimates[provider] = max_count
    else:
        for provider in tqdm(providers, desc="Estimating fleet sizes"):
            provider_df = df[df['provider'] == provider]
            fleet_estimates[provider] = provider_df['vehicle_id'].nunique()

    fleet_df = pd.DataFrame.from_dict(fleet_estimates, orient='index', columns=['Estimated Fleet Size'])
    fleet_df.index.name = 'Provider'
    fleet_df = fleet_df.sort_values('Estimated Fleet Size', ascending=False)

    return fleet_df


def show_maintenance_distribution(df: pd.DataFrame):
    total_events = len(df)
    maintenance_events = df['is_maintenance'].sum()
    maintenance_pct = maintenance_events / total_events * 100

    print("=" * 60)
    print("MAINTENANCE EVENT STATISTICS")
    print("=" * 60)
    print(f"Total events:        {total_events:,}")
    print(f"Maintenance events:  {maintenance_events:,} ({maintenance_pct:.2f}%)")
    print(f"Regular events:      {total_events - maintenance_events:,} ({100-maintenance_pct:.2f}%)")
    print()

    maintenance_df = df[df['is_maintenance'] == True]

    print("Maintenance Events by Provider:")
    maint_by_provider = maintenance_df['provider'].value_counts()
    for provider, count in maint_by_provider.items():
        pct = (count / len(maintenance_df) * 100)
        print(f"  {provider}: {count:,} ({pct:.1f}%)")

    print("\nEvent Type Breakdown:")
    disappeared_maint = maintenance_df['disappeared'].sum()
    first_seen_maint = maintenance_df['first_seen'].sum()
    print(f"  Disappearances (before recharge): {disappeared_maint:,}")
    print(f"  First seen (after recharge):      {first_seen_maint:,}")

    print("\nBattery Statistics (Maintenance Events):")
    battery_stats = maintenance_df['current_fuel_percent'].describe()
    print(f"  Mean:   {battery_stats['mean']:.1f}%")
    print(f"  Median: {battery_stats['50%']:.1f}%")
    print(f"  Min:    {battery_stats['min']:.1f}%")
    print(f"  Max:    {battery_stats['max']:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    type_counts = pd.Series({
        'Disappeared': maintenance_df['disappeared'].sum(),
        'First Seen': maintenance_df['first_seen'].sum()
    })
    axes[0].pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
               colors=['#e74c3c', '#f39c12'], startangle=90)
    axes[0].set_title('Maintenance Events by Type', fontsize=14, fontweight='bold')

    maintenance_df = maintenance_df.copy()
    maintenance_df['date'] = pd.to_datetime(maintenance_df['timestamp']).dt.date
    daily_maintenance = maintenance_df.groupby('date').size()

    daily_maintenance.plot(kind='line', ax=axes[1], color='#e74c3c', marker='o')
    axes[1].set_title('Maintenance Events Over Time', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Number of Events')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def analyze_maintenance_by_provider(df: pd.DataFrame, spike_window_minutes=30, use_spikes=True):
    fleet_df = estimate_fleet_size(df, spike_window_minutes, use_spikes)

    maintenance_df = df[df['is_maintenance'] == True]
    maintenance_counts = maintenance_df['provider'].value_counts().to_frame('Maintenance Events')

    analysis_df = fleet_df.join(maintenance_counts, how='left')
    analysis_df['Maintenance Events'] = analysis_df['Maintenance Events'].fillna(0).astype(int)
    analysis_df['Events per Scooter'] = (analysis_df['Maintenance Events'] / analysis_df['Estimated Fleet Size']).round(2)

    total_events_per_provider = df.groupby('provider').size().to_frame('Total Events')
    analysis_df = analysis_df.join(total_events_per_provider)
    analysis_df['Maintenance %'] = (analysis_df['Maintenance Events'] / analysis_df['Total Events'] * 100).round(2)

    print("=" * 80)
    print("FLEET SIZE AND MAINTENANCE ANALYSIS BY PROVIDER")
    print("=" * 80)
    print(analysis_df.to_string())
    print()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    top_providers = analysis_df.head(15)
    axes[0, 0].barh(range(len(top_providers)), top_providers['Estimated Fleet Size'], color='#3498db')
    axes[0, 0].set_yticks(range(len(top_providers)))
    axes[0, 0].set_yticklabels(top_providers.index)
    axes[0, 0].set_title('Top 15 Providers by Estimated Fleet Size', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Estimated Number of Scooters')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    top_maintenance = analysis_df.nlargest(15, 'Maintenance Events')
    axes[0, 1].barh(range(len(top_maintenance)), top_maintenance['Maintenance Events'], color='#e74c3c')
    axes[0, 1].set_yticks(range(len(top_maintenance)))
    axes[0, 1].set_yticklabels(top_maintenance.index)
    axes[0, 1].set_title('Top 15 Providers by Maintenance Events', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Maintenance Events')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    top_ratio = analysis_df.nlargest(15, 'Events per Scooter')
    axes[1, 0].barh(range(len(top_ratio)), top_ratio['Events per Scooter'], color='#f39c12')
    axes[1, 0].set_yticks(range(len(top_ratio)))
    axes[1, 0].set_yticklabels(top_ratio.index)
    axes[1, 0].set_title('Top 15 Providers by Events per Scooter', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Maintenance Events per Scooter')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    axes[1, 1].scatter(analysis_df['Estimated Fleet Size'], analysis_df['Maintenance Events'],
                      alpha=0.6, s=100, color='#9b59b6')
    axes[1, 1].set_title('Fleet Size vs Maintenance Events', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Estimated Fleet Size')
    axes[1, 1].set_ylabel('Maintenance Events')
    axes[1, 1].grid(True, alpha=0.3)

    z = np.polyfit(analysis_df['Estimated Fleet Size'], analysis_df['Maintenance Events'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(analysis_df['Estimated Fleet Size'].min(),
                        analysis_df['Estimated Fleet Size'].max(), 100)
    axes[1, 1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    return analysis_df


def plot_maintenance_time_gaps(df: pd.DataFrame, time_window_minutes=30, distance_threshold_meters=50):
    maintenance_df = df[df['is_maintenance'] == True].copy()
    disappeared_events = maintenance_df[maintenance_df['disappeared'] == True].copy()
    first_seen_events = maintenance_df[maintenance_df['first_seen'] == True].copy()

    time_gaps = []
    distance_threshold_radians = distance_threshold_meters / 6371000

    for _, disappeared_row in tqdm(disappeared_events.iterrows(), total=len(disappeared_events), desc="Finding pairs"):
        disappeared_time = disappeared_row['timestamp']
        time_limit = disappeared_time + pd.Timedelta(minutes=time_window_minutes)

        nearby_fs = first_seen_events[
            (first_seen_events['timestamp'] > disappeared_time) &
            (first_seen_events['timestamp'] <= time_limit)
        ]

        if len(nearby_fs) == 0:
            continue

        disappeared_coords = np.radians([[disappeared_row['lat'], disappeared_row['lon']]])
        nearby_coords = np.radians(nearby_fs[['lat', 'lon']].values)

        tree = BallTree(nearby_coords, metric='haversine')
        distances, indices = tree.query(disappeared_coords, k=len(nearby_fs))

        for idx, dist in zip(indices[0], distances[0]):
            if dist <= distance_threshold_radians:
                fs_row = nearby_fs.iloc[idx]
                time_diff = (fs_row['timestamp'] - disappeared_time).total_seconds() / 60
                time_gaps.append(time_diff)
                break

    time_gaps = np.array(time_gaps)

    print(f"Found {len(time_gaps):,} matched pairs")
    print(f"Time gap statistics (minutes):")
    print(f"  Mean:   {time_gaps.mean():.2f}")
    print(f"  Median: {np.median(time_gaps):.2f}")
    print(f"  Min:    {time_gaps.min():.2f}")
    print(f"  Max:    {time_gaps.max():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(time_gaps, bins=50, edgecolor='black', alpha=0.7, color='#e74c3c')
    axes[0].axvline(time_gaps.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: {time_gaps.mean():.1f} min')
    axes[0].axvline(np.median(time_gaps), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(time_gaps):.1f} min')
    axes[0].set_title('Time Gap Between Disappeared and First Seen', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Difference (minutes)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    sorted_gaps = np.sort(time_gaps)
    cumulative = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps) * 100

    axes[1].plot(sorted_gaps, cumulative, linewidth=2, color='#e74c3c')
    axes[1].axhline(50, color='blue', linestyle='--', alpha=0.5, label='50th percentile')
    axes[1].axhline(90, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    axes[1].set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time Difference (minutes)')
    axes[1].set_ylabel('Cumulative Percentage (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

"""Probabilistic trip matching using distance, speed, and range consumption factors."""

import os
import numpy as np
import pandas as pd
from scipy.stats import truncnorm, lognorm, norm
from sklearn.neighbors import BallTree
from dataclasses import dataclass, field
from typing import Tuple, List
from tqdm.auto import tqdm

from routing import get_routes_by_source


@dataclass
class MatcherParams:
    """Parameters for the three-factor probabilistic model."""
    mu_speed: float = 14.0
    sigma_speed: float = 3.5
    speed_min: float = 5.0
    speed_max: float = 23.0

    mu_dist: float = 0.5
    sigma_dist: float = 0.8

    range_efficiency: float = 1.1
    range_sigma_base: float = 1.0
    range_sigma_scale: float = 0.3

    max_time_hours: float = 1.0
    max_distance_km: float = 8.0
    min_range_drain_km: float = 0.3
    min_route_km: float = 0.1

    routing_base_url: str = "http://localhost:8002"
    routing_max_workers: int = 30
    routing_batch_size: int = 100

    temperature: float = 1.0
    null_score_percentile: float = 25.0
    max_null_prob: float = 0.5

    a_speed: float = field(init=False)
    b_speed: float = field(init=False)

    def __post_init__(self):
        self.a_speed = (self.speed_min - self.mu_speed) / self.sigma_speed
        self.b_speed = (self.speed_max - self.mu_speed) / self.sigma_speed


@dataclass
class MatchResult:
    """Result of matching."""
    candidates: pd.DataFrame
    best_matches: pd.DataFrame
    params: MatcherParams
    unmatched_disappearances: pd.Index
    unmatched_first_seen: pd.Index


class TripMatcher:
    """Trip matcher using P(distance) * P(speed) * P(range_consumed | distance)."""

    def __init__(self, params: MatcherParams):
        self.params = params

    def build_candidates(
        self,
        disappeared: pd.DataFrame,
        first_seen: pd.DataFrame,
        time_bin_minutes: int = 30,
        show_progress: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        candidates = []
        max_dist_rad = self.params.max_distance_km / 6371.0
        bins_to_check = int(np.ceil(self.params.max_time_hours * 60 / time_bin_minutes))

        if verbose:
            print(f"\n=== Building Candidates ===")
            print(f"Hard filters:")
            print(f"  max_time_hours: {self.params.max_time_hours}")
            print(f"  max_distance_km: {self.params.max_distance_km}")
            print(f"  min_route_km: {self.params.min_route_km}")
            print(f"  min_range_drain_km: {self.params.min_range_drain_km}")

        providers = disappeared['provider'].unique()
        iterator = tqdm(providers, desc="Building candidates") if show_progress else providers

        for provider in iterator:
            d_prov = disappeared[disappeared['provider'] == provider]
            f_prov = first_seen[first_seen['provider'] == provider]

            if len(d_prov) == 0 or len(f_prov) == 0:
                continue

            for vtype in d_prov['vehicle_type_id'].unique():
                d_vtype = d_prov[d_prov['vehicle_type_id'] == vtype].copy()
                f_vtype = f_prov[f_prov['vehicle_type_id'] == vtype].copy()

                if len(f_vtype) == 0:
                    continue

                min_time = min(d_vtype['timestamp'].min(), f_vtype['timestamp'].min())
                d_vtype['time_bin'] = (
                    (d_vtype['timestamp'] - min_time).dt.total_seconds() // (time_bin_minutes * 60)
                ).astype(int)
                f_vtype['time_bin'] = (
                    (f_vtype['timestamp'] - min_time).dt.total_seconds() // (time_bin_minutes * 60)
                ).astype(int)

                fs_by_bin = {}
                for bin_id, group in f_vtype.groupby('time_bin'):
                    coords = np.radians(group[['lat', 'lon']].values)
                    ranges = group['current_range_meters'].values / 1000.0
                    fs_by_bin[bin_id] = {
                        'tree': BallTree(coords, metric='haversine'),
                        'indices': group.index.values,
                        'times': group['timestamp'].values,
                        'ranges_km': ranges,
                        'lats': group['lat'].values,
                        'lons': group['lon'].values,
                    }

                for d_bin_id, d_group in d_vtype.groupby('time_bin'):
                    relevant_bins = [
                        d_bin_id + offset
                        for offset in range(bins_to_check + 1)
                        if (d_bin_id + offset) in fs_by_bin
                    ]

                    if not relevant_bins:
                        continue

                    d_coords = np.radians(d_group[['lat', 'lon']].values)
                    d_times = d_group['timestamp'].values
                    d_indices = d_group.index.values
                    d_ranges_km = d_group['current_range_meters'].values / 1000.0
                    d_lats = d_group['lat'].values
                    d_lons = d_group['lon'].values

                    for fs_bin_id in relevant_bins:
                        fs_data = fs_by_bin[fs_bin_id]
                        neighbors_list = fs_data['tree'].query_radius(d_coords, r=max_dist_rad)

                        for i, neighbor_indices in enumerate(neighbors_list):
                            if len(neighbor_indices) == 0:
                                continue

                            d_idx = d_indices[i]
                            d_time = d_times[i]
                            d_range_km = d_ranges_km[i]
                            d_lat, d_lon = d_lats[i], d_lons[i]

                            for j in neighbor_indices:
                                f_idx = fs_data['indices'][j]
                                f_time = fs_data['times'][j]
                                f_range_km = fs_data['ranges_km'][j]
                                f_lat, f_lon = fs_data['lats'][j], fs_data['lons'][j]

                                delta_t_seconds = (f_time - d_time) / np.timedelta64(1, 's')
                                delta_t_hours = delta_t_seconds / 3600

                                if delta_t_hours <= 0 or delta_t_hours > self.params.max_time_hours:
                                    continue

                                range_consumed = d_range_km - f_range_km
                                if range_consumed < self.params.min_range_drain_km:
                                    continue

                                candidates.append({
                                    'd_idx': d_idx,
                                    'f_idx': f_idx,
                                    'provider': provider,
                                    'vehicle_type_id': vtype,
                                    'd_lat': d_lat,
                                    'd_lon': d_lon,
                                    'f_lat': f_lat,
                                    'f_lon': f_lon,
                                    'd_time': pd.Timestamp(d_time),
                                    'f_time': pd.Timestamp(f_time),
                                    'd_range_km': d_range_km,
                                    'f_range_km': f_range_km,
                                    'delta_t_hours': delta_t_hours,
                                    'range_consumed': range_consumed,
                                })

        candidates_df = pd.DataFrame(candidates)

        candidates_df = candidates_df.sort_values('d_idx').reset_index(drop=True)

        if verbose:
            print(f"\nPre-routing candidates: {len(candidates_df):,}")
            n_sources = candidates_df['d_idx'].nunique()
            print(f"Routing {n_sources:,} sources to their targets...")

        route_results = get_routes_by_source(
            candidates_df,
            base_url=self.params.routing_base_url,
            max_workers=self.params.routing_max_workers,
            show_progress=show_progress
        )

        candidates_df['opt_route_km'] = route_results['distance_km']
        candidates_df['opt_route_min'] = route_results['duration_min']

        n_before = len(candidates_df)
        candidates_df = candidates_df[candidates_df['opt_route_km'].notna()]
        candidates_df = candidates_df[candidates_df['opt_route_km'] >= self.params.min_route_km]
        candidates_df = candidates_df[candidates_df['d_range_km'] >= candidates_df['opt_route_km']]

        if verbose:
            print(f"Filtered: {n_before - len(candidates_df):,}")
            print(f"Final candidates: {len(candidates_df):,}")

        return candidates_df

    def score_candidates(self, candidates: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        candidates = candidates.copy()

        opt_route_km = candidates['opt_route_km'].values
        delta_t_hours = candidates['delta_t_hours'].values
        range_consumed = candidates['range_consumed'].values

        speed = opt_route_km / delta_t_hours
        candidates['speed'] = speed

        rv_dist = lognorm(s=self.params.sigma_dist, scale=np.exp(self.params.mu_dist))
        log_p_distance = rv_dist.logpdf(opt_route_km)

        log_p_speed = np.full_like(speed, -np.inf)
        valid_speed = (speed >= self.params.speed_min) & (speed <= self.params.speed_max)
        if np.any(valid_speed):
            rv_speed = truncnorm(
                self.params.a_speed, self.params.b_speed,
                loc=self.params.mu_speed, scale=self.params.sigma_speed
            )
            log_p_speed[valid_speed] = rv_speed.logpdf(speed[valid_speed])

        expected_range = opt_route_km * self.params.range_efficiency
        range_sigma = self.params.range_sigma_base + self.params.range_sigma_scale * opt_route_km
        log_p_range = norm.logpdf(range_consumed, loc=expected_range, scale=range_sigma)

        scores = log_p_distance + log_p_speed + log_p_range

        candidates['log_p_distance'] = log_p_distance
        candidates['log_p_speed'] = log_p_speed
        candidates['log_p_range'] = log_p_range
        candidates['score'] = scores

        valid_scores = np.isfinite(scores)
        n_invalid = (~valid_scores).sum()
        if n_invalid > 0 and verbose:
            print(f"  Filtered {n_invalid:,} candidates with invalid scores")

        return candidates[valid_scores]

    def compute_assignments(
        self,
        candidates: pd.DataFrame,
        null_score: float = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        candidates = candidates.copy()

        if null_score is None:
            counts = candidates.groupby('d_idx').size()
            unambiguous_idx = counts[counts == 1].index
            unambiguous = candidates[candidates['d_idx'].isin(unambiguous_idx)]

            if len(unambiguous) >= 20:
                valid_scores = unambiguous['score'].values
                valid_scores = valid_scores[np.isfinite(valid_scores)]
                null_score = np.percentile(valid_scores, self.params.null_score_percentile)
                if verbose:
                    print(f"  Auto null_score: {null_score:.2f} (p{self.params.null_score_percentile:.0f})")
            else:
                null_score = -5.0
                if verbose:
                    print(f"  Default null_score: {null_score:.2f}")

        def softmax_with_null(group):
            scores = group['score'].values
            all_scores = np.append(scores, null_score)
            all_scores_shifted = (all_scores - all_scores.max()) / self.params.temperature
            exp_scores = np.exp(all_scores_shifted)
            all_probs = exp_scores / exp_scores.sum()

            group = group.copy()
            group['prob'] = all_probs[:-1]
            group['prob_null'] = all_probs[-1]
            return group

        candidates = candidates.groupby('d_idx', group_keys=False).apply(softmax_with_null)

        return candidates

    def get_best_matches(self, candidates: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        idx = candidates.groupby('d_idx')['prob'].idxmax()
        best_matches = candidates.loc[idx].copy()

        n_before = len(best_matches)
        best_matches = best_matches[best_matches['prob_null'] <= self.params.max_null_prob]
        n_filtered = n_before - len(best_matches)

        if n_filtered > 0 and verbose:
            print(f"  Filtered {n_filtered:,} matches with prob_null > {self.params.max_null_prob:.2f}")

        return best_matches

    def fit(
        self,
        disappeared: pd.DataFrame,
        first_seen: pd.DataFrame,
        time_bin_minutes: int = 30,
        show_progress: bool = True,
        candidates_cache: str = None
    ) -> MatchResult:
        if candidates_cache and os.path.exists(candidates_cache):
            print(f"Loading cached candidates from {candidates_cache}...")
            candidates = pd.read_parquet(candidates_cache)
            candidates['d_time'] = pd.to_datetime(candidates['d_time'])
            candidates['f_time'] = pd.to_datetime(candidates['f_time'])
            print(f"Loaded {len(candidates):,} candidates")
        else:
            candidates = self.build_candidates(
                disappeared, first_seen,
                time_bin_minutes=time_bin_minutes,
                show_progress=show_progress,
                verbose=True
            )

        if 'score' not in candidates.columns:
            print(f"\n=== Scoring Candidates ===")
            print(f"Model: P(distance) * P(speed) * P(range|distance)")
            print(f"  Distance prior: LogNormal(mu={self.params.mu_dist:.2f}, sigma={self.params.sigma_dist:.2f})")
            print(f"  Speed prior: TruncNormal(mu={self.params.mu_speed:.1f}, sigma={self.params.sigma_speed:.1f}, [{self.params.speed_min}-{self.params.speed_max}])")
            print(f"  Range: Normal(distance*{self.params.range_efficiency:.1f}, {self.params.range_sigma_base:.1f} + {self.params.range_sigma_scale:.1f}*distance)")

            candidates = self.score_candidates(candidates, verbose=True)

        print(f"Valid candidates: {len(candidates):,}")

        n_d = candidates['d_idx'].nunique()
        print(f"Disappearances with candidates: {n_d:,}")

        print(f"\nCandidate statistics:")
        print(f"  Distance: {candidates['opt_route_km'].median():.2f} km (median)")
        print(f"  Speed: {candidates['speed'].median():.1f} km/h (median)")
        print(f"  Score: {candidates['score'].median():.2f} (median)")

        return MatchResult(
            candidates=candidates,
            best_matches=pd.DataFrame(),  # No assignment
            params=self.params,
            unmatched_disappearances=disappeared.index[~disappeared.index.isin(candidates['d_idx'])],
            unmatched_first_seen=first_seen.index[~first_seen.index.isin(candidates['f_idx'])]
        )


def prepare_events(df: pd.DataFrame, use_recalibrated_range: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[(df['is_maintenance'] == False) & (df['is_id_reset'] == False)]

    disappeared = df[df['disappeared'] == True].copy()
    first_seen = df[df['first_seen'] == True].copy()

    disappeared = disappeared[disappeared['current_range_meters'].notna()]
    first_seen = first_seen[first_seen['current_range_meters'].notna()]

    if use_recalibrated_range:
        has_recalibrated = first_seen['recalibrated_current_range_meters'].notna()
        first_seen.loc[has_recalibrated, 'current_range_meters'] = (
            first_seen.loc[has_recalibrated, 'recalibrated_current_range_meters']
        )
        print(f"Using recalibrated range for {has_recalibrated.sum():,} first_seen events")

    print(f"Prepared {len(disappeared):,} disappearance events, {len(first_seen):,} first_seen events")
    return disappeared, first_seen


def analyze_matches(result: MatchResult):
    import matplotlib.pyplot as plt

    candidates = result.candidates
    best_matches = result.best_matches

    print(f"Candidates: {len(candidates):,}, Matches: {len(best_matches):,}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    dist_p95 = np.percentile(best_matches['opt_route_km'], 95)
    ax.hist(best_matches['opt_route_km'].clip(upper=dist_p95), bins=50, edgecolor='black', alpha=0.7, density=True)
    x_dist = np.linspace(0.1, dist_p95, 100)
    rv_dist = lognorm(s=result.params.sigma_dist, scale=np.exp(result.params.mu_dist))
    ax.plot(x_dist, rv_dist.pdf(x_dist), 'r-', linewidth=2, label='Prior')
    ax.set_xlabel('Route Distance (km)')
    ax.set_title('Distance Distribution')
    ax.legend()

    ax = axes[0, 1]
    ax.hist(best_matches['speed'], bins=50, edgecolor='black', alpha=0.7, density=True)
    x_speed = np.linspace(result.params.speed_min, result.params.speed_max, 100)
    rv_speed = truncnorm(result.params.a_speed, result.params.b_speed,
                         loc=result.params.mu_speed, scale=result.params.sigma_speed)
    ax.plot(x_speed, rv_speed.pdf(x_speed), 'r-', linewidth=2, label='Prior')
    ax.set_xlabel('Speed (km/h)')
    ax.set_title('Speed Distribution')
    ax.legend()

    ax = axes[0, 2]
    max_val = max(np.percentile(best_matches['range_consumed'], 95),
                  np.percentile(best_matches['opt_route_km'], 95))
    mask = (best_matches['range_consumed'] <= max_val) & (best_matches['opt_route_km'] <= max_val)
    ax.scatter(best_matches.loc[mask, 'opt_route_km'], best_matches.loc[mask, 'range_consumed'],
               c=best_matches.loc[mask, 'prob'], cmap='viridis', alpha=0.4, s=10)
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1')
    ax.plot([0, max_val], [0, max_val * result.params.range_efficiency], 'r--', linewidth=1,
            label=f'{result.params.range_efficiency}:1')
    ax.set_xlabel('Route Distance (km)')
    ax.set_ylabel('Range Consumed (km)')
    ax.set_title('Range vs Route')
    ax.legend()

    ax = axes[1, 0]
    ax.hist(best_matches['prob'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='orange', linestyle='--', label='0.5 threshold')
    ax.set_xlabel('Match Probability')
    ax.set_title('Match Probabilities')
    ax.legend()

    ax = axes[1, 1]
    components = ['log_p_distance', 'log_p_speed', 'log_p_range']
    data = [best_matches[c].values for c in components]
    ax.boxplot(data, labels=['P(dist)', 'P(speed)', 'P(range)'])
    ax.set_ylabel('Log Probability')
    ax.set_title('Score Components')

    ax = axes[1, 2]
    ax.scatter(best_matches['delta_t_hours'] * 60, best_matches['opt_route_km'],
               c=best_matches['prob'], cmap='viridis', alpha=0.4, s=10)
    ax.set_xlabel('Trip Duration (min)')
    ax.set_ylabel('Route Distance (km)')
    ax.set_title('Duration vs Distance')

    plt.tight_layout()
    plt.show()

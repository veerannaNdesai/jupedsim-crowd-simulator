"""
feature_extractor.py
---------------------
Converts raw trajectory records (from ScenarioEngine) into the
ML-ready 34-column CSV with all engineered features, aggregated
hour-by-zone.

Output columns (exactly as spec):
    Zone_name, Venue_type, count, density, occupancy_ratio,
    avg_speed, flow_in, avg_dwelltime, flow_out, flow_imbalance,
    flow_ratio, count_t-1, density_t-1, flow_in_t-1,
    count_trend, density_change_rate, rolling_count_3h,
    neighbor_density_avg, neighbor_flow_in, event_type,
    event_scale, time_to_event_start, time_to_event_end,
    hour_of_day, day_of_week, is_weekend, pressure_score,
    exit_blocked_flag, exits_open_count, zone_capacity,
    zone_area_sqm, num_exits, weather, temperature_c
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from configs.config_loader import ScenarioConfig
from geometry.mall_geometry import ZoneMeta

logger = logging.getLogger(__name__)

EPSILON = 1e-6   # avoid division by zero


class FeatureExtractor:
    """
    Takes raw trajectory records + scenario config + zone metadata
    and produces a fully-featured ML DataFrame (one row per zone per hour).
    """

    def __init__(
        self,
        config: ScenarioConfig,
        zones: Dict[str, ZoneMeta],
    ) -> None:
        self.config = config
        self.zones = zones

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def extract(self, trajectory_records: List[dict]) -> pd.DataFrame:
        """
        Main entry point.

        Args:
            trajectory_records: raw dicts from ScenarioEngine.run()
        Returns:
            pd.DataFrame with all 34 ML feature columns
        """
        if not trajectory_records:
            logger.warning("No trajectory records — returning empty DataFrame")
            return pd.DataFrame()

        raw_df = pd.DataFrame(trajectory_records)

        # Step 1: Hourly zone aggregation
        hourly = self._aggregate_hourly(raw_df)

        # Step 2: Compute per-zone flow (in / out approximation)
        hourly = self._compute_flows(hourly, raw_df)

        # Step 3: Lag & temporal features
        hourly = self._compute_lag_features(hourly)

        # Step 4: Rolling features
        hourly = self._compute_rolling_features(hourly)

        # Step 5: Spatial features (neighbor density)
        hourly = self._compute_spatial_features(hourly)

        # Step 6: Event features
        hourly = self._compute_event_features(hourly)

        # Step 7: Time features
        hourly = self._compute_time_features(hourly)

        # Step 8: Safety & infra features
        hourly = self._compute_safety_features(hourly)

        # Step 9: Zone metadata
        hourly = self._attach_zone_meta(hourly)

        # Step 10: External features
        hourly["weather"] = self.config.external.weather
        hourly["temperature_c"] = self.config.external.temperature_c

        # Step 11: Venue info
        hourly["Venue_type"] = self.config.meta.venue_type

        # Rename and select final columns
        final = self._finalize_columns(hourly)
        logger.info("  Features extracted: %d rows × %d cols", *final.shape)
        return final

    # ------------------------------------------------------------------ #
    # Step 1: Hourly zone aggregation                                      #
    # ------------------------------------------------------------------ #

    def _aggregate_hourly(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Aggregate trajectory snapshots → one row per (zone, hour)."""

        # Exclude empty-hour sentinels
        df = raw[raw["agent_id"] != -1].copy()

        if df.empty:
            # Build zero rows for all zones/hours
            return self._zero_rows()

        grouped = (
            df.groupby(["zone_name", "sim_hour"])
            .agg(
                count=("agent_id", "nunique"),
                avg_speed=("speed", "mean"),
                avg_dwelltime=("dwell_time_min", "mean"),
            )
            .reset_index()
        )

        # Ensure all zones × all hours exist (fill zeros for missing)
        all_zones = list(self.zones.keys())
        all_hours = list(range(self.config.meta.duration_hours))
        idx = pd.MultiIndex.from_product(
            [all_zones, all_hours], names=["zone_name", "sim_hour"]
        )
        grouped = (
            grouped.set_index(["zone_name", "sim_hour"])
            .reindex(idx, fill_value=0)
            .reset_index()
        )
        # Fill NaN speeds / dwell
        grouped["avg_speed"] = grouped["avg_speed"].fillna(0.0)
        grouped["avg_dwelltime"] = grouped["avg_dwelltime"].fillna(0.0)

        return grouped

    def _zero_rows(self) -> pd.DataFrame:
        all_zones = list(self.zones.keys())
        all_hours = list(range(self.config.meta.duration_hours))
        rows = [
            {"zone_name": z, "sim_hour": h, "count": 0,
             "avg_speed": 0.0, "avg_dwelltime": 0.0}
            for z in all_zones for h in all_hours
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Step 2: Flow estimation                                              #
    # ------------------------------------------------------------------ #

    def _compute_flows(self, hourly: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate flow_in / flow_out per zone per hour.
        Flow_in  ≈ agents first seen in zone at hour h
        Flow_out ≈ agents last seen in zone at hour h (left next hour)
        """
        df = raw[raw["agent_id"] != -1].copy()

        if df.empty:
            hourly["flow_in"] = 0
            hourly["flow_out"] = 0
            return hourly

        # First appearance of each agent in each zone per hour
        first_seen = (
            df.groupby(["agent_id", "zone_name", "sim_hour"])["sim_time_s"]
            .min()
            .reset_index()
        )
        flow_in = (
            first_seen.groupby(["zone_name", "sim_hour"])["agent_id"]
            .count()
            .reset_index()
            .rename(columns={"agent_id": "flow_in"})
        )

        # Last appearance
        last_seen = (
            df.groupby(["agent_id", "zone_name", "sim_hour"])["sim_time_s"]
            .max()
            .reset_index()
        )
        flow_out = (
            last_seen.groupby(["zone_name", "sim_hour"])["agent_id"]
            .count()
            .reset_index()
            .rename(columns={"agent_id": "flow_out"})
        )

        hourly = hourly.merge(flow_in, on=["zone_name", "sim_hour"], how="left")
        hourly = hourly.merge(flow_out, on=["zone_name", "sim_hour"], how="left")
        hourly["flow_in"] = hourly["flow_in"].fillna(0).astype(int)
        hourly["flow_out"] = hourly["flow_out"].fillna(0).astype(int)

        hourly["flow_imbalance"] = hourly["flow_in"] - hourly["flow_out"]
        hourly["flow_ratio"] = hourly["flow_in"] / (hourly["flow_out"] + EPSILON)

        return hourly

    # ------------------------------------------------------------------ #
    # Step 3: Lag features                                                 #
    # ------------------------------------------------------------------ #

    def _compute_lag_features(self, hourly: pd.DataFrame) -> pd.DataFrame:
        """Add count_t-1, flow_in_t-1, count_trend."""
        hourly = hourly.sort_values(["zone_name", "sim_hour"]).reset_index(drop=True)
        hourly["count_t-1"]   = hourly.groupby("zone_name")["count"].shift(1).fillna(0)
        hourly["flow_in_t-1"] = hourly.groupby("zone_name")["flow_in"].shift(1).fillna(0)
        hourly["count_trend"]  = hourly["count"] - hourly["count_t-1"]
        return hourly

    # ------------------------------------------------------------------ #
    # Step 4: Rolling features                                             #
    # ------------------------------------------------------------------ #

    def _compute_rolling_features(self, hourly: pd.DataFrame) -> pd.DataFrame:
        """rolling_count_3h = mean(count over last 3 hours)."""
        hourly = hourly.sort_values(["zone_name", "sim_hour"]).reset_index(drop=True)
        hourly["rolling_count_3h"] = (
            hourly.groupby("zone_name")["count"]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )
        return hourly

    # ------------------------------------------------------------------ #
    # Step 5: Spatial features                                             #
    # ------------------------------------------------------------------ #

    def _compute_spatial_features(self, hourly: pd.DataFrame) -> pd.DataFrame:
        """neighbor_density_avg, neighbor_flow_in — using zone graph."""

        # Build a density lookup: {zone_name: {hour: density}}
        # Density = count / area  (area filled in step 9 later, use capacity as proxy here)
        cap_map = {zn: zm.capacity for zn, zm in self.zones.items()}
        area_map = {zn: zm.area_sqm for zn, zm in self.zones.items()}
        count_pivot = hourly.pivot(index="sim_hour", columns="zone_name", values="count").fillna(0)
        flow_pivot  = hourly.pivot(index="sim_hour", columns="zone_name", values="flow_in").fillna(0)

        def _get_neighbor_stats(row):
            zone = row["zone_name"]
            hour = row["sim_hour"]
            neighbors = self.zones.get(zone, None)
            if neighbors is None or not neighbors.neighbors:
                return pd.Series({"neighbor_density_avg": 0.0, "neighbor_flow_in": 0.0})
            nb_densities = []
            nb_flows = []
            for nb_name in neighbors.neighbors:
                if nb_name not in count_pivot.columns:
                    continue
                nb_count = count_pivot.loc[hour, nb_name] if hour in count_pivot.index else 0
                nb_area  = area_map.get(nb_name, 1.0)
                nb_densities.append(nb_count / (nb_area + EPSILON))
                nb_flows.append(
                    flow_pivot.loc[hour, nb_name] if hour in flow_pivot.index else 0
                )
            return pd.Series({
                "neighbor_density_avg": float(np.mean(nb_densities)) if nb_densities else 0.0,
                "neighbor_flow_in": float(np.sum(nb_flows)) if nb_flows else 0.0,
            })

        spatial = hourly.apply(_get_neighbor_stats, axis=1)
        hourly["neighbor_density_avg"] = spatial["neighbor_density_avg"].round(6)
        hourly["neighbor_flow_in"] = spatial["neighbor_flow_in"]
        return hourly

    # ------------------------------------------------------------------ #
    # Step 6: Event features                                               #
    # ------------------------------------------------------------------ #

    def _compute_event_features(self, hourly: pd.DataFrame) -> pd.DataFrame:
        ev = self.config.event

        def _event_row(hour):
            active = self.config.is_event_active(hour)
            t_start = (ev.start_hour - hour) if ev.start_hour is not None else 99
            t_end   = (ev.end_hour   - hour) if ev.end_hour   is not None else 99
            return pd.Series({
                "event_type":         ev.type if active else "none",
                "event_scale":        ev.scale if active else "none",
                "time_to_event_start": max(0, t_start),
                "time_to_event_end":   max(0, t_end),
            })

        event_df = hourly["sim_hour"].apply(_event_row)
        for col in event_df.columns:
            hourly[col] = event_df[col]

        return hourly

    # ------------------------------------------------------------------ #
    # Step 7: Time features                                                #
    # ------------------------------------------------------------------ #

    def _compute_time_features(self, hourly: pd.DataFrame) -> pd.DataFrame:
        from datetime import datetime, timedelta
        sim_date = datetime.strptime(self.config.meta.simulation_date, "%Y-%m-%d")

        hourly["hour_of_day"] = hourly["sim_hour"]
        hourly["day_of_week"] = sim_date.weekday()   # 0=Mon, 6=Sun
        hourly["is_weekend"]  = int(sim_date.weekday() >= 5)
        return hourly

    # ------------------------------------------------------------------ #
    # Step 8: Safety & infra features                                      #
    # ------------------------------------------------------------------ #

    def _compute_safety_features(self, hourly: pd.DataFrame) -> pd.DataFrame:
        # Pressure score: high density + high speed = crowd pressure risk
        # Will be refined after density is computed — placeholder here
        hourly["pressure_score"] = 0.0   # updated in finalize

        blocked = set(self.config.exits.blocked)
        hourly["exit_blocked_flag"] = int(len(blocked) > 0)
        hourly["exits_open_count"]  = self.config.exits.open_count
        return hourly

    # ------------------------------------------------------------------ #
    # Step 9: Zone metadata                                                #
    # ------------------------------------------------------------------ #

    def _attach_zone_meta(self, hourly: pd.DataFrame) -> pd.DataFrame:
        hourly["zone_capacity"] = hourly["zone_name"].map(
            {zn: zm.capacity for zn, zm in self.zones.items()}
        )
        hourly["zone_area_sqm"] = hourly["zone_name"].map(
            {zn: zm.area_sqm for zn, zm in self.zones.items()}
        )
        hourly["num_exits"] = hourly["zone_name"].map(
            {zn: zm.num_exits for zn, zm in self.zones.items()}
        )

        # Now compute density-dependent columns
        hourly["density"] = hourly["count"] / (hourly["zone_area_sqm"] + EPSILON)
        hourly["occupancy_ratio"] = hourly["count"] / (hourly["zone_capacity"] + EPSILON)

        # density_t-1 needs zone_area_sqm
        hourly["density_t-1"] = hourly["count_t-1"] / (hourly["zone_area_sqm"] + EPSILON)
        change_rate = (
            (hourly["density"] - hourly["density_t-1"])
            / (hourly["density_t-1"] + EPSILON)
        )

        clip_val = self.config.congestion.density_change_rate_clip
        hourly["density_change_rate"] = change_rate.clip(-clip_val, clip_val)

        # Pressure score = density × avg_speed (crowd momentum proxy)
        hourly["pressure_score"] = (hourly["density"] * hourly["avg_speed"]).round(4)

        return hourly

    # ------------------------------------------------------------------ #
    # Step 10: Finalize columns                                            #
    # ------------------------------------------------------------------ #

    def _finalize_columns(self, hourly: pd.DataFrame) -> pd.DataFrame:
        """Select and order the 34 output columns exactly per spec."""
        col_order = [
            "Zone_name", "Venue_type",
            "count", "density", "occupancy_ratio",
            "avg_speed", "flow_in", "avg_dwelltime", "flow_out",
            "flow_imbalance", "flow_ratio",
            "count_t-1", "density_t-1", "flow_in_t-1",
            "count_trend", "density_change_rate", "rolling_count_3h",
            "neighbor_density_avg", "neighbor_flow_in",
            "event_type", "event_scale",
            "time_to_event_start", "time_to_event_end",
            "hour_of_day", "day_of_week", "is_weekend",
            "pressure_score",
            "exit_blocked_flag", "exits_open_count",
            "zone_capacity", "zone_area_sqm", "num_exits",
            "weather", "temperature_c",
        ]

        # Rename zone_name → Zone_name
        hourly = hourly.rename(columns={"zone_name": "Zone_name"})

        # Round numeric columns
        numeric_cols = [
            "count", "density", "occupancy_ratio", "avg_speed",
            "flow_in", "avg_dwelltime", "flow_out", "flow_imbalance",
            "flow_ratio", "count_t-1", "density_t-1", "flow_in_t-1",
            "count_trend", "density_change_rate", "rolling_count_3h",
            "neighbor_density_avg", "neighbor_flow_in",
            "pressure_score", "zone_area_sqm",
        ]
        for col in numeric_cols:
            if col in hourly.columns:
                hourly[col] = pd.to_numeric(hourly[col], errors="coerce").round(4)

        # Integer columns
        int_cols = [
            "exits_open_count", "zone_capacity", "num_exits",
            "exit_blocked_flag", "is_weekend", "day_of_week",
            "hour_of_day", "time_to_event_start", "time_to_event_end",
        ]
        for col in int_cols:
            if col in hourly.columns:
                hourly[col] = hourly[col].fillna(0).astype(int)

        # Ensure all expected columns exist
        for col in col_order:
            if col not in hourly.columns:
                hourly[col] = 0

        return hourly[col_order].sort_values(
            ["hour_of_day", "Zone_name"]
        ).reset_index(drop=True)

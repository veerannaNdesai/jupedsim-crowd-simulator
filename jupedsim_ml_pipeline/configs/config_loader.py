"""
config_loader.py
----------------
Loads and validates scenario YAML configs into typed dataclasses.
"""

from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class AgentProfile:
    type: str
    fraction: float
    desired_speed_mean: float
    desired_speed_std: float
    radius_mean: float
    radius_std: float
    time_gap: float
    dwell_time_mean_min: float
    dwell_time_std_min: float


@dataclass
class EventConfig:
    type: str
    scale: str
    start_hour: Optional[int]
    end_hour: Optional[int]
    zone: Optional[str]
    crowd_multiplier: float


@dataclass
class ExitConfig:
    open_count: int
    blocked: List[str]


@dataclass
class ExternalConfig:
    weather: str
    temperature_c: float


@dataclass
class CongestionConfig:
    density_change_rate_clip: float = 100.0
    flow_imbalance_enabled: bool = True
    overcapacity_spillback: bool = False


@dataclass
class ScenarioMeta:
    name: str
    venue: str
    venue_type: str
    simulation_date: str
    duration_hours: int
    dt: float
    seed: int


@dataclass
class ScenarioConfig:
    meta: ScenarioMeta
    agent_profiles: List[AgentProfile]
    peak_footfall: int
    zone_attractors: Dict[str, float]
    time_of_day_multipliers: Dict[int, float]
    event: EventConfig
    exits: ExitConfig
    external: ExternalConfig
    congestion: CongestionConfig

    def get_hour_multiplier(self, hour: int) -> float:
        return self.time_of_day_multipliers.get(hour, 0.0)

    def is_event_active(self, hour: int) -> bool:
        if self.event.type == "none" or self.event.start_hour is None:
            return False
        return self.event.start_hour <= hour < self.event.end_hour

    def get_event_multiplier(self, hour: int) -> float:
        if self.is_event_active(hour):
            return self.event.crowd_multiplier
        return 1.0

    def is_exit_blocked(self, exit_name: str) -> bool:
        return exit_name in self.exits.blocked


def load_config(config_path: str | Path) -> ScenarioConfig:
    """Load and parse a scenario YAML file into a ScenarioConfig."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    meta = ScenarioMeta(**raw["scenario"])

    agent_profiles = [AgentProfile(**ap) for ap in raw["agent_profiles"]]

    # Normalize fraction sum
    total = sum(ap.fraction for ap in agent_profiles)
    for ap in agent_profiles:
        ap.fraction = ap.fraction / total

    tdm_raw = raw["time_of_day_multipliers"]
    time_of_day_multipliers = {int(k): float(v) for k, v in tdm_raw.items()}

    event_raw = raw["event"]
    event = EventConfig(**event_raw)

    exit_raw = raw["exits"]
    exits = ExitConfig(**exit_raw)

    external = ExternalConfig(**raw["external"])

    congestion_raw = raw.get("congestion", {})
    congestion = CongestionConfig(**congestion_raw)

    return ScenarioConfig(
        meta=meta,
        agent_profiles=agent_profiles,
        peak_footfall=raw["peak_footfall"],
        zone_attractors=raw["zone_attractors"],
        time_of_day_multipliers=time_of_day_multipliers,
        event=event,
        exits=exits,
        external=external,
        congestion=congestion,
    )

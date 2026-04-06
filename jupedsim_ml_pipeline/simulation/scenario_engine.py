"""
scenario_engine.py
------------------
Runs a single JuPedSim simulation for a given ScenarioConfig.
Simulates 24 hours by running each hour as an independent micro-simulation.
Returns a flat list of trajectory record dicts for downstream processing.
"""

from __future__ import annotations
import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

import jupedsim as jps

from configs.config_loader import AgentProfile, ScenarioConfig
from geometry.mall_geometry import (
    ExitPoint,
    ZoneMeta,
    build_mall_geometry,
    get_zone_for_position,
)

logger = logging.getLogger(__name__)

SIM_SECONDS_PER_HOUR: int = 240
SNAPSHOT_EVERY_N_SECONDS: float = 15.0
MAX_CONCURRENT_AGENTS: int = 250


class ScenarioEngine:
    """Orchestrates a full 24-hour JuPedSim pedestrian simulation."""

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self.rng = random.Random(config.meta.seed)
        self.np_rng = np.random.default_rng(config.meta.seed)
        self.walkable_area, self.zones, self.exit_points = build_mall_geometry()
        self.open_exits: List[ExitPoint] = self._filter_exits()
        logger.info(
            "Engine ready | scenario=%s | exits=%s",
            config.meta.name,
            [e.name for e in self.open_exits],
        )

    def run(self) -> List[dict]:
        """Run all 24 hours. Returns flat list of trajectory records."""
        all_records: List[dict] = []
        for hour in range(self.config.meta.duration_hours):
            logger.info("  Hour %02d:00 ...", hour)
            records = self._simulate_hour(hour)
            all_records.extend(records)
        logger.info("  Total records: %d", len(all_records))
        return all_records

    def _simulate_hour(self, hour: int) -> List[dict]:
        n_agents = min(self._agent_count_for_hour(hour), MAX_CONCURRENT_AGENTS)
        if n_agents == 0:
            return self._empty_records(hour)
        sim = self._make_simulation()
        stage_ids, journey_id = self._build_journey(sim, hour)
        agent_meta = self._spawn_agents(sim, n_agents, journey_id, stage_ids, hour)
        return self._run_and_record(sim, agent_meta, hour)

    def _make_simulation(self) -> jps.Simulation:
        model = jps.CollisionFreeSpeedModel(
            strength_neighbor_repulsion=8.0,
            range_neighbor_repulsion=0.1,
            strength_geometry_repulsion=5.0,
            range_geometry_repulsion=0.02,
        )
        return jps.Simulation(model=model, geometry=self.walkable_area, dt=self.config.meta.dt)

    def _build_journey(self, sim: jps.Simulation, hour: int) -> Tuple[List[int], int]:
        is_closing = (hour >= self.config.meta.duration_hours - 2)

        if is_closing:
            if self.open_exits:
                exit_id = sim.add_exit_stage(self.open_exits[0].polygon)
            else:
                lobby = self.zones["Lobby"]
                exit_id = sim.add_waypoint_stage(
                    (lobby.polygon.centroid.x, lobby.polygon.centroid.y), 3.0
                )
            stage_ids = [exit_id]
            jd = jps.JourneyDescription(stage_ids)
            journey_id = sim.add_journey(jd)
            return stage_ids, journey_id

        sorted_zones = sorted(
            [(zn, zm) for zn, zm in self.zones.items() if zm.zone_type != "EntryExitPoint"],
            key=lambda kv: self.config.zone_attractors.get(kv[1].zone_type, 0.0),
            reverse=True,
        )
        stage_ids: List[int] = []
        for zname, zmeta in sorted_zones:
            wp_id = sim.add_waypoint_stage(
                (zmeta.polygon.centroid.x, zmeta.polygon.centroid.y), 2.5
            )
            stage_ids.append(wp_id)

        if self.open_exits:
            exit_id = sim.add_exit_stage(self.open_exits[0].polygon)
        else:
            lobby = self.zones["Lobby"]
            exit_id = sim.add_waypoint_stage(
                (lobby.polygon.centroid.x, lobby.polygon.centroid.y), 3.0
            )
        stage_ids.append(exit_id)

        jd = jps.JourneyDescription(stage_ids)
        if len(stage_ids) > 1:
            if self.config.congestion.overcapacity_spillback:
                # Distribution based on least crowded (targeted) waypoint
                trans = jps.Transition.create_least_targeted_transition(
                    stage_ids[1:]
                )
            else:
                trans = jps.Transition.create_round_robin_transition(
                    [(sid, 1) for sid in stage_ids[1:]]
                )

            jd.set_transition_for_stage(stage_ids[0], trans)

            for sid in stage_ids[1:-1]:
                jd.set_transition_for_stage(
                    sid, jps.Transition.create_fixed_transition(stage_ids[-1])
                )
        journey_id = sim.add_journey(jd)
        return stage_ids, journey_id

    def _spawn_agents(
        self, sim: jps.Simulation, n_agents: int,
        journey_id: int, stage_ids: List[int], hour: int
    ) -> Dict[int, dict]:
        agent_meta: Dict[int, dict] = {}
        is_closing = (hour >= self.config.meta.duration_hours - 2)
        if is_closing:
            valid_zones = [z for name, z in self.zones.items() if z.zone_type not in ("EntryExitPoint", "Lobby")]
            if not valid_zones:
                valid_zones = [self.zones["Lobby"]]
            positions = []
            for _ in range(n_agents):
                zone = self.rng.choice(valid_zones)
                positions.extend(self._sample_positions(zone.polygon, 1))
        else:
            positions = self._sample_positions(self.zones["Lobby"].polygon, n_agents)
        for pos in positions:
            profile = self._pick_profile()
            speed = float(np.clip(
                self.np_rng.normal(profile.desired_speed_mean, profile.desired_speed_std), 0.3, 3.0
            ))
            radius = float(np.clip(
                self.np_rng.normal(profile.radius_mean, profile.radius_std), 0.1, 0.4
            ))
            dwell = float(max(1.0, self.np_rng.normal(
                profile.dwell_time_mean_min, profile.dwell_time_std_min
            )))
            try:
                aid = sim.add_agent(jps.CollisionFreeSpeedModelAgentParameters(
                    position=pos,
                    time_gap=profile.time_gap,
                    desired_speed=speed,
                    radius=radius,
                    journey_id=journey_id,
                    stage_id=stage_ids[0],
                ))
                agent_meta[aid] = {
                    "type": profile.type,
                    "desired_speed": speed,
                    "radius": radius,
                    "dwell_time_min": dwell,
                }
            except Exception:
                continue
        logger.debug("    Spawned %d agents (hour %02d)", len(agent_meta), hour)
        return agent_meta

    def _sample_positions(self, polygon: Polygon, n: int) -> List[Tuple[float, float]]:
        positions: List[Tuple[float, float]] = []
        minx, miny, maxx, maxy = polygon.bounds
        for _ in range(n * 200):
            if len(positions) >= n:
                break
            x = self.rng.uniform(minx, maxx)
            y = self.rng.uniform(miny, maxy)
            if polygon.contains(Point(x, y)):
                positions.append((x, y))
        return positions

    def _pick_profile(self) -> AgentProfile:
        r = self.rng.random()
        cumsum = 0.0
        for ap in self.config.agent_profiles:
            cumsum += ap.fraction
            if r <= cumsum:
                return ap
        return self.config.agent_profiles[-1]

    def _run_and_record(
        self, sim: jps.Simulation, agent_meta: Dict[int, dict], hour: int
    ) -> List[dict]:
        records: List[dict] = []
        total_steps = int(SIM_SECONDS_PER_HOUR / self.config.meta.dt)
        next_snap = SNAPSHOT_EVERY_N_SECONDS
        sim_time = 0.0
        for _ in range(total_steps):
            try:
                sim.iterate()
            except Exception as e:
                logger.warning("Step error: %s", e)
                break
            sim_time += self.config.meta.dt
            if sim_time >= next_snap:
                for agent in sim.agents():
                    x, y = agent.position
                    meta = agent_meta.get(agent.id, {})
                    zone = get_zone_for_position(x, y, self.zones) or "Outside"
                    records.append({
                        "agent_id": agent.id,
                        "x": round(x, 3),
                        "y": round(y, 3),
                        "speed": round(meta.get("desired_speed", 1.2), 3),
                        "sim_hour": hour,
                        "sim_time_s": round(sim_time, 1),
                        "agent_type": meta.get("type", "shopper"),
                        "dwell_time_min": meta.get("dwell_time_min", 30.0),
                        "zone_name": zone,
                    })
                next_snap += SNAPSHOT_EVERY_N_SECONDS
        return records

    def _agent_count_for_hour(self, hour: int) -> int:
        base = self.config.peak_footfall * self.config.get_hour_multiplier(hour)
        return max(0, int(base * self.config.get_event_multiplier(hour)))

    def _filter_exits(self) -> List[ExitPoint]:
        blocked = set(self.config.exits.blocked)
        return [ep for ep in self.exit_points if ep.name not in blocked]

    def _empty_records(self, hour: int) -> List[dict]:
        return [{"agent_id": -1, "x": 0.0, "y": 0.0, "speed": 0.0, "sim_hour": hour,
                 "sim_time_s": 0.0, "agent_type": "none", "dwell_time_min": 0.0, "zone_name": "Lobby"}]

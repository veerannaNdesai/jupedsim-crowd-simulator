# JuPedSim ML Pipeline

Synthetic pedestrian crowd data generator for ML using JuPedSim 1.3.x.

## Project Structure

```
jupedsim_ml_pipeline/
├── configs/
│   ├── config_loader.py              # YAML → typed dataclasses
│   ├── scenario_base.yaml            # Scenario 1: Weekday Normal
│   ├── scenario_weekend_rush.yaml    # Scenario 2: Weekend Rush
│   ├── scenario_sale_event.yaml      # Scenario 3: Sale Event
│   ├── scenario_emergency_drill.yaml # Scenario 4: Emergency Drill
│   ├── scenario_evening_closing.yaml # Scenario 5: Evening Closing
│   └── scenario_concert_night.yaml   # Scenario 6: Concert Night
│
├── geometry/
│   └── mall_geometry.py              # 13-zone Mall geometry (Shapely polygons)
│
├── simulation/
│   └── scenario_engine.py            # JuPedSim 24-hour simulation runner
│
├── pipeline/
│   ├── feature_extractor.py          # Trajectories → 34-column ML features
│   └── csv_writer.py                 # Saves CSVs to output/
│
├── output/                           # Generated CSVs land here
├── logs/                             # pipeline.log
└── run_all_scenarios.py              # Master orchestrator (entry point)
```

## Installation

```bash
pip install jupedsim pedpy shapely numpy pandas pyyaml scipy matplotlib
```

## Usage

```bash
# Run all 6 scenarios
python run_all_scenarios.py

# List available scenarios
python run_all_scenarios.py --list

# Run specific scenario(s) by index
python run_all_scenarios.py --scenario 0
python run_all_scenarios.py --scenario 0 2 4
```

## Output

- `output/<scenario_name>.csv`  — per-scenario ML dataset (312 rows each)
- `output/all_scenarios.csv`    — combined dataset for training (1,872 rows)

## Output Columns (34 ML features)

| Column | Description |
|--------|-------------|
| Zone_name | Zone identifier |
| Venue_type | MALL / AIRPORT / etc. |
| count | Agent count in zone at hour |
| density | count / zone_area_sqm |
| occupancy_ratio | count / zone_capacity |
| avg_speed | Mean agent speed (m/s) |
| flow_in | Agents entering zone |
| avg_dwelltime | Mean dwell time (minutes) |
| flow_out | Agents leaving zone |
| flow_imbalance | flow_in - flow_out |
| flow_ratio | flow_in / flow_out |
| count_t-1 | count at previous hour |
| density_t-1 | density at previous hour |
| flow_in_t-1 | flow_in at previous hour |
| count_trend | count - count_t-1 |
| density_change_rate | % change in density |
| rolling_count_3h | 3-hour rolling mean count |
| neighbor_density_avg | Mean density of adjacent zones |
| neighbor_flow_in | Sum of neighbor zone flow_in |
| event_type | Active event type (none/sale/concert/...) |
| event_scale | Event scale (none/small/medium/large) |
| time_to_event_start | Hours until event starts |
| time_to_event_end | Hours until event ends |
| hour_of_day | 0–23 |
| day_of_week | 0=Mon … 6=Sun |
| is_weekend | 1 if Saturday/Sunday |
| pressure_score | density × avg_speed (crowd risk proxy) |
| exit_blocked_flag | 1 if any exits blocked |
| exits_open_count | Number of open exits |
| zone_capacity | Max agent capacity of zone |
| zone_area_sqm | Zone area in square metres |
| num_exits | Number of exits in zone |
| weather | sunny / cloudy / rainy / clear |
| temperature_c | Ambient temperature (°C) |

## Adding a New Scenario

1. Copy any YAML from `configs/` and rename it
2. Tune parameters (peak_footfall, time_of_day_multipliers, event, exits, etc.)
3. Add the path to the `SCENARIOS` list in `run_all_scenarios.py`
4. Run `python run_all_scenarios.py`

No code changes needed anywhere else.

## Supported Venues

Currently implemented: **Mall**

To add Airport / Stadium / TransitTerminal:
- Create `geometry/airport_geometry.py` with zone definitions
- Instantiate it in `ScenarioEngine.__init__` based on `config.meta.venue`



# mall.py high level explaination

I'll read through the full mall code carefully before explaining anything.Now I have a thorough understanding of the entire code. Let me write the detailed explanation:Now I'll write the explanation with diagrams:This is the physical floor plan baked into the code. Every zone is a `shapely.Polygon` using real metre coordinates — the mall spans 200 × 89 m. JuPedSim uses these polygons both for collision geometry and for counting agents inside them. Now let's go through each section of the code in order.

---

## Section 1 — Zones, neighbors, walkable area

`ZONES` is a dictionary of 9 zones. Each zone has three things: a `polygon` (the physical shape in metres), a `capacity` (the number of people the zone is designed to hold before it gets uncomfortable), and `num_exits` (how many ways out it has). These numbers are used later to compute `occupancy_ratio = count / capacity` and `exits_open_count`.

`NEIGHBORS` is a separate adjacency map — for each zone it lists which other zones physically border it. This is only used during feature extraction to compute `neighbor_density_avg` and `neighbor_flow_in`, which tell the ML model how much pressure is building in surrounding zones. The simulation itself doesn't use neighbors for pathfinding; JuPedSim handles that through waypoints and journeys.

`WALKABLE` is a single large polygon covering the entire 200 × 89 m area. JuPedSim uses this as the navigable geometry — agents cannot leave this boundary and the physics engine calculates repulsion from its edges.

---

## Section 2 — Dwell times and multipliers

`BASE_DWELL_MIN` defines how long a person naturally spends in each zone, as a `(min, max)` tuple in minutes. The food court is 8–15 minutes (you sit and eat), the hypermarket is 10–20 minutes (you browse and shop), the main entrance is only 0.3–0.8 minutes (you walk straight through). These are realistic retail dwell times.

`DWELL_MULT` is where scenario-specific realism lives. For each of the 10 scenarios there's a dictionary that multiplies the base dwell for each zone. For example in `mall_lunchtime_rush`, `food_court` gets a multiplier of 3.5, meaning an agent in the food court stays 3.5× longer than usual — it's packed, seating is hard to find. In `mall_closing_rush`, almost everything gets `0.2` (people are rushing out) but `main_entrance` gets `3.5` and `parking_exit` gets `4.0` because those become the bottleneck — everyone is trying to leave simultaneously through a narrow outlet.

The actual dwell time used per agent is computed in `build_journeys`:
```python
dwell_min[zone] = random.uniform(mn, mx) * mult.get(zone, 1.0)
```
So a food court agent in the lunch rush gets a dwell of `random(8, 15) × 3.5` = roughly 28–52 minutes.

---

## Section 3 — Spawn rate functionsThe helper `G(h, p, w, ht)` is a Gaussian function — it produces a smooth bell curve centred at hour `p`, with width `w`, and peak height `ht`. By adding several Gaussians together you get realistic multi-peak crowd arrival curves. For example `spawn_normal_weekday` adds a lunch peak at h13 and an evening peak at h18–19 on top of a flat baseline. `spawn_lunchtime_rush` adds an extremely sharp Gaussian at h12 and h13 with width 0.6, creating a sudden spike when office workers flood in. `spawn_closing_rush` returns 0 from h21 onwards because there are no new arrivals — the crowd effect at closing comes entirely from the dwell multiplier piling people up at exits, not from new people spawning.

The solid lines are weekday/normal scenarios. The dashed lines are festival/holiday scenarios which run higher baselines and extend into later hours.

`_closed(h)` is a simple helper that returns `True` for hours before 10 and from 22 onwards (mall closed). Most spawn functions call this at the top and return 0.

---

## Section 4 — Scenarios dictionary

Each entry in `SCENARIOS` is a configuration object that collects everything a scenario needs:

- `name` — human-readable label written to the CSV
- `cat` — A (weekday/normal) or B (holiday/festival)
- `model` — which pedestrian physics model to use (`CFSM` for normal, `CFSMv2` for congested)
- `event_type`, `event_scale` — categorical and numeric descriptors written as ML features
- `is_holiday`, `is_festival` — binary flags that become features
- `weather`, `temperature_c` — environmental context features
- `spawn_fn` — the spawn function from Section 3 (a function reference, not a call)
- `ev_start`, `ev_end` — the hour window of the main event, used to compute `time_to_event_start` and `time_to_event_end` features
- `exit_blocked_h`, `exit_blocked_z` — if and when exits become blocked, and in which zones
- `closing_hour` — used by the snapshot function to apply a closing-time speed boost

---

## Section 5 — Simulation factory

`make_simulation(model_key)` creates one JuPedSim `Simulation` object. It gets a fresh one per scenario run. Two physics models are available:

`CFSM` (Collision-Free Speed Model) is used for low-to-medium density situations. Each agent is assigned a desired speed and a radius, and the model resolves conflicts so agents never overlap. It takes explicit parameters for how strongly agents repel each other and how much they avoid walls.

`CFSMv2` is the newer version used for congested scenarios. It handles higher densities more realistically without needing tuned repulsion parameters.

`make_agent_params` is a thin wrapper that builds either `CollisionFreeSpeedModelAgentParameters` or `CollisionFreeSpeedModelV2AgentParameters` depending on which model is running. Every agent gets a randomised desired speed (0.8–1.4 m/s) and a slightly randomised body radius (0.21–0.27 m) to avoid lockstep behaviour.

---

## Section 6 — Journey builder (the most complex part)`build_journeys` is called once per scenario run. It does five things:

**Step 1 — Waypoints.** `WP_POS` is a dictionary of named (x, y) coordinates. `sim.add_waypoint_stage(pos, 1.5)` registers a waypoint in the simulation — agents navigate toward it and are considered to have "arrived" when they come within 1.5 m. Waypoints don't hold agents; they're just navigation targets.

**Step 2 — Waiting sets.** `QUEUE_POS` maps zone names to lists of (x, y) coordinates — these are the physical queue spots inside each zone. `sim.add_waiting_set_stage(positions)` creates a waiting set: agents assigned to it occupy these positions one at a time (like a real queue). The `.state = jps.WaitingSetState.ACTIVE` line means the set is open and accepting agents immediately. This is what makes agents visibly pile up in a zone — they walk in and queue.

**Step 3 — The dead zone fix.** In the old (broken) code there was a single corridor waiting set. When an agent's corridor dwell expired, `release_info[ws["main_corridor"]]` sent everyone to `retail_wing_A` regardless of what they originally intended to do. Food court visitors, movie goers, and grocery shoppers all ended up in retail_wing_A, making those zones invisible in the data.

The fix creates four separate corridor waiting stages — `ws_cor_shopper`, `ws_cor_lunch`, `ws_cor_movie`, `ws_cor_grocery` — each with queue positions at a different y-coordinate within the corridor (14, 17, 20, and 23 respectively, so they physically separate in the simulation). Each stage has its own `release_info` entry pointing to the correct destination. `ws_to_zone_extra` maps all four back to `"main_corridor"` so the dwell timer looks up the right wait duration.

**Step 4 — `mj()` helper and release_info.** `mj(*stages)` is a compact helper that creates a `JourneyDescription` from a list of stages with fixed (deterministic) transitions between them, registers the journey with the simulation, and returns `(journey_id, first_stage)`. `release_info` maps each waiting-set stage ID to the `(journey_id, first_stage)` of the onward journey an agent should take after their dwell expires. When an agent finishes waiting in food court, `release_info[ws["food_court"]]` gives the journey that leads to `exit_s`.

**Step 5 — The four journeys (A–D).** Each journey is a `JourneyDescription` — an ordered list of stages with transitions set between each consecutive pair. Journey A (Shopper) goes: entrance WP → main_entrance WaitingSet → corridor WP → ws_cor_shopper → retail_a WP → retail_wing_A WaitingSet → atrium WP → exit. The transitions are all `create_fixed_transition` (deterministic, not probabilistic). Agents are spawned onto one of these four journeys according to the weights `[0.45, 0.30, 0.15, 0.10]`.

---

## Section 7 — Spawn

`try_spawn` is called every 60 simulated seconds. It checks whether the simulation has capacity (under `MAX_AGENTS = 1400`) and whether the entrance area is overloaded (above 85% of combined entrance + corridor capacity). It then asks JuPedSim to place `n` agents at random valid positions within `ENT_POLY` (a small polygon just inside the entrance), assigns each one a random journey type and random speed, and adds them to the simulation. The `with _Q()` context manager silences JuPedSim's verbose stdout during `sim.add_agent`.

The spawn count `n` comes from accumulating fractional agents: `spawn_acc += rate; n = int(spawn_acc); spawn_acc -= n`. This ensures that a spawn rate of 0.3 agents/minute actually produces one agent roughly every 3 minutes rather than always rounding down to 0.

---

## Section 8 — Snapshot and dwell management

This is called every simulated minute (600 physics steps × 0.1s dt = 60s). It does two things in one pass: manages agent dwell release, and extracts zone-level features.

**Dwell management.** It loops over every active agent, looks up its current stage ID, and checks if that stage ID is a waiting set (via `ws_to_zone`). If it is, it increments `agent_min_ctr[aid]` — a per-agent minute counter. When the counter exceeds the zone's dwell threshold (with ±10% random jitter), it schedules the agent for release by appending to `to_release`. After the loop, it calls `sim.switch_agent_journey(aid, rel_jid, rel_stage)` for each released agent, redirecting them onto their next journey segment. Dead agents are purged from `agent_min_ctr`.

**Feature extraction.** For each zone it counts agents via `sim.agents_in_polygon(polygon)`, samples up to 40 agents' velocities (capped at 40 to avoid O(n) cost), computes density = count / area, occupancy_ratio = count / capacity, flow_in = new agents since last snapshot, flow_out = agents who left since last snapshot, pressure_score = density × max(|imbalance|, 0.1). It also handles the `exit_blocked_flag` — if the scenario says a zone's exits are blocked from a given hour, `exits_open_count` drops by 1.

A closing-time speed boost is applied: in the hour before `closing_hour`, reported `avg_speed` is multiplied by a factor growing with time, simulating people rushing to leave.

---

## Section 9 — Run one scenario

`run_scenario` is the main loop. It runs `864000` physics steps (24 hours × 3600 s × 10 steps/s). Every 600 steps (= 1 simulated minute):

1. It calls the spawn function with the current hour, accumulates fractional agents, and spawns.
2. It calls `snapshot_and_manage` which does everything in Section 8 and appends the 9-zone snapshot to `all_snaps`.
3. Every 60 minutes it prints a progress line showing the 3 most populated zones.

---

## Section 10 — Aggregate minute → hourly`aggregate_hourly` takes the list of per-minute snapshots (up to 1440 of them) and collapses them into one row per (zone, hour) pair, giving 9 zones × 24 hours = 216 rows per scenario. It works by converting `all_snaps` to a DataFrame and then iterating over every zone and every hour.

For each (zone, hour) group it computes:
- `count` → mean across all minutes in that hour
- `density` → max (worst minute)
- `occupancy_ratio` → max (captures the worst congestion peak)
- `avg_speed` → mean
- `flow_in`, `flow_out` → sum across all minutes first, then `imbalance = fi - fo`. This is the corrected approach — the old code subtracted at the minute level and re-aggregated, which produced wrong signs during surge hours.

It then looks up the previous hour's data for `count_t-1`, `density_t-1`, `flow_in_t-1`, and computes `count_trend` and `density_change_rate` (clipped to ±50 to prevent division artifacts in hour 0 where the denominator is near-zero). `rolling_count_3h` is the mean count over the 3 hours before the current one.

Neighbor features are computed by looking up all zones in `NEIGHBORS[zname]` and averaging their density and summing their flow_in for the same hour — this tells the model whether pressure is building in adjacent zones.

**The label fix.** The old code used `density < 1.5` as the Normal threshold. But because your zones are large (the food court is 90 × 30 = 2700 m²) density never exceeds 0.3 even when it's packed, so everything got label=0. The fix uses `occupancy_ratio` (agents ÷ capacity) which is independent of zone size: below 40% = Normal, 40–75% = Congested, above 75% = Surge. This produces a realistic 55/30/15 split across labels.

---

## Section 11 — Main function

`main()` loops through all 10 scenarios in order, runs each one (`run_scenario`), aggregates the results (`aggregate_hourly`), saves a per-scenario CSV, and appends to `all_hourly`. After all scenarios it concatenates everything into `master_mall.csv` (2160 rows: 10 scenarios × 9 zones × 24 hours). It prints a label distribution and per-scenario breakdown so you can verify realism at a glance before feeding the data to your model.

---

## How it all connects end-to-end

A single agent's life through the system looks like this: `main()` calls `run_scenario("mall_lunchtime_rush")` → `build_journeys` sets dwell times with 3.5× multiplier on food_court → the physics loop runs, `try_spawn` spawns a lunch visitor and assigns Journey B → the agent walks to `main_entrance`, enters `ws_cor_lunch`, waits ~28–52 minutes by the clock of `agent_min_ctr`, then `snapshot_and_manage` releases it to `food_court` → it sits there, another ~28–52 minutes tick off, then it exits → `snapshot_and_manage` records its departure as `flow_out` in food_court at that minute → after 1440 minutes `aggregate_hourly` collapses all that into hour 12 and hour 13 rows with high `occupancy_ratio`, high `flow_imbalance`, and `label=2` (Surge).
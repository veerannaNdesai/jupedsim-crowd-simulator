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

"""
run_all_scenarios.py
--------------------
Master orchestrator for the JuPedSim ML Pipeline.

Usage:
    python run_all_scenarios.py                  # run all 6 scenarios
    python run_all_scenarios.py --scenario 1     # run single scenario by index
    python run_all_scenarios.py --list           # list available scenarios

Output:
    output/<scenario_name>.csv     per-scenario ML dataset
    output/all_scenarios.csv       combined dataset for training
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config, ScenarioConfig
from geometry.mall_geometry import build_mall_geometry
from simulation.scenario_engine import ScenarioEngine
from pipeline.feature_extractor import FeatureExtractor
from pipeline.csv_writer import CSVWriter

import pandas as pd

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ── scenario registry ─────────────────────────────────────────────────────────
SCENARIOS = [
    "configs/scenario_base.yaml",
    "configs/scenario_weekend_rush.yaml",
    "configs/scenario_sale_event.yaml",
    "configs/scenario_emergency_drill.yaml",
    "configs/scenario_evening_closing.yaml",
    "configs/scenario_concert_night.yaml",
    "configs/scenario_superstar_movieshow.yaml",
]


# ── pipeline runner ───────────────────────────────────────────────────────────

def run_scenario(config_path: str, writer: CSVWriter) -> pd.DataFrame:
    """Full pipeline for one scenario. Returns ML DataFrame."""
    config: ScenarioConfig = load_config(PROJECT_ROOT / config_path)
    logger.info("=" * 60)
    logger.info("SCENARIO : %s", config.meta.name)
    logger.info("Date     : %s | Venue: %s", config.meta.simulation_date, config.meta.venue_type)
    logger.info("Footfall : peak=%d | dt=%.2fs", config.peak_footfall, config.meta.dt)
    logger.info("=" * 60)

    t0 = time.time()

    # 1. Simulate
    engine = ScenarioEngine(config)
    trajectory_records = engine.run()

    t1 = time.time()
    logger.info("  Simulation done in %.1fs | records=%d", t1 - t0, len(trajectory_records))

    # 2. Extract features
    _, zones, _ = build_mall_geometry()
    extractor = FeatureExtractor(config, zones)
    ml_df = extractor.extract(trajectory_records)

    t2 = time.time()
    logger.info("  Feature extraction done in %.1fs", t2 - t1)

    # 3. Add scenario identifier column
    ml_df.insert(0, "scenario_name", config.meta.name)
    ml_df.insert(1, "simulation_date", config.meta.simulation_date)

    # 4. Write per-scenario CSV
    writer.write(ml_df, config.meta.name)

    logger.info("  Total: %.1fs\n", time.time() - t0)
    return ml_df


def run_all(indices: list[int] | None = None) -> None:
    writer = CSVWriter(PROJECT_ROOT / "output")
    all_dfs: list[pd.DataFrame] = []

    scenarios_to_run = (
        [SCENARIOS[i] for i in indices] if indices else SCENARIOS
    )

    total_start = time.time()
    for cfg_path in scenarios_to_run:
        try:
            df = run_scenario(cfg_path, writer)
            all_dfs.append(df)
        except Exception as e:
            logger.error("FAILED: %s — %s", cfg_path, e, exc_info=True)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        writer.write_combined(combined)

        logger.info("=" * 60)
        logger.info("ALL DONE in %.1fs", time.time() - total_start)
        logger.info("Scenarios  : %d", len(all_dfs))
        logger.info("Total rows : %d", len(combined))
        logger.info("Columns    : %d", len(combined.columns))
        logger.info("Output dir : %s", writer.output_dir)
        logger.info("=" * 60)

        # Print quick column summary
        print("\n── Column Summary ──────────────────────────────────────────")
        print(combined.dtypes.to_string())
        print("\n── Sample Row ──────────────────────────────────────────────")
        print(combined.head(2).T.to_string())
    else:
        logger.error("No scenarios completed successfully.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="JuPedSim ML Pipeline")
    parser.add_argument("--scenario", type=int, nargs="+",
                        help="Zero-based indices of scenarios to run (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List available scenarios and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable scenarios:")
        for i, s in enumerate(SCENARIOS):
            cfg = load_config(PROJECT_ROOT / s)
            print(f"  [{i}] {cfg.meta.name}  ({s})")
        return

    run_all(indices=args.scenario)


if __name__ == "__main__":
    main()

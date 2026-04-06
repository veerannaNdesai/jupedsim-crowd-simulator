"""
csv_writer.py
-------------
Saves the ML-ready DataFrame to CSV output directory.
"""

from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class CSVWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, df: pd.DataFrame, scenario_name: str) -> Path:
        safe_name = scenario_name.lower().replace(" ", "_").replace("/", "_")
        out_path = self.output_dir / f"{safe_name}.csv"
        df.to_csv(out_path, index=False)
        logger.info("  Saved: %s  (%d rows)", out_path.name, len(df))
        return out_path

    def write_combined(self, df: pd.DataFrame, filename: str = "all_scenarios.csv") -> Path:
        out_path = self.output_dir / filename
        df.to_csv(out_path, index=False)
        logger.info("  Combined dataset saved: %s  (%d rows)", out_path.name, len(df))
        return out_path

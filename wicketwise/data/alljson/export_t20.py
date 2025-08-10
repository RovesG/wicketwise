# Purpose: Export T20-only events from all_json corpus for model training (optional path)
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


TRAIN_COLUMNS = [
    # Identity/context for joins (train-safe)
    "date_utc",
    "venue",
    "team_batting",
    "team_bowling",
    "batter_id",
    "bowler_id",
    # Indexing
    "innings_index",
    "over_number",
    "delivery_index",
    # Outcomes
    "runs_batter",
    "runs_extras",
    "runs_total",
    "dismissal_kind",
]


def export_t20_events(events_df: pd.DataFrame, output_dir: str, partition_by: Optional[list] = None) -> str:
    df = events_df.copy()
    df = df[df["match_type"].str.lower() == "t20"].reset_index(drop=True)
    cols = [c for c in TRAIN_COLUMNS if c in df.columns]
    out = df[cols]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Partition by year (from date_utc) and team_batting by default
    if partition_by is None:
        partition_by = []
        if "date_utc" in out:
            out["year"] = out["date_utc"].str.slice(0, 4)
            partition_by.append("year")
        if "team_batting" in out:
            partition_by.append("team_batting")

    out.to_parquet(out_dir / "t20_events.parquet", index=False)

    return str(out_dir / "t20_events.parquet")

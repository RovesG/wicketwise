# Purpose: Orchestrate chunked, cached, and configurable KG building at scale
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List
import os
import logging
import pandas as pd

from .schema_resolver import resolve_schema
from .kg_aggregator import aggregate_core, compute_partnerships

logger = logging.getLogger(__name__)


@dataclass
class PipelineSettings:
    chunk_size: int = 500_000  # rows per chunk
    cache_dir: str = "models/aggregates"
    use_llm_schema_hint: bool = False
    compute_heavy_metrics: bool = False  # reserved for future
    normalize_ids: bool = False  # keep names as IDs by default to avoid breaking callers


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _merge_numeric_frames(left: pd.DataFrame, right: pd.DataFrame, on: List[str]) -> pd.DataFrame:
    if left is None or len(left) == 0:
        return right.copy()
    if right is None or len(right) == 0:
        return left.copy()
    # Sum numeric columns; keep keys in 'on'
    l = left.copy()
    r = right.copy()
    merged = l.merge(r, on=on, how="outer", suffixes=("_l", "_r"))
    for col in merged.columns:
        if col.endswith("_l"):
            base = col[:-2]
            rc = base + "_r"
            if rc in merged.columns:
                merged[base] = merged[col].fillna(0) + merged[rc].fillna(0)
    # Drop old suffixed columns
    drop_cols = [c for c in merged.columns if c.endswith("_l") or c.endswith("_r")]
    merged = merged.drop(columns=drop_cols)
    return merged


def build_aggregates_from_csv(csv_path: str, settings: PipelineSettings) -> Dict[str, pd.DataFrame]:
    """
    Chunked aggregation over a large CSV; caches intermediate and final aggregate tables.
    Returns a dict of aggregate DataFrames.
    """
    _ensure_dir(settings.cache_dir)
    cache_key = os.path.splitext(os.path.basename(csv_path))[0]
    final_cache = os.path.join(settings.cache_dir, f"{cache_key}_aggregates.pkl")

    # Reuse cache if available and newer than source
    try:
        if os.path.exists(final_cache) and os.path.getmtime(final_cache) >= os.path.getmtime(csv_path):
            logger.info(f"Loading cached aggregates: {final_cache}")
            return pd.read_pickle(final_cache)
    except Exception:
        pass

    # Streaming over chunks
    aggregates: Dict[str, Optional[pd.DataFrame]] = {
        "batter_bowler": None,
        "batter_venue": None,
        "bowler_venue": None,
        "batter_phase": None,
        "bowler_phase": None,
        "batter_stats": None,
        "bowler_stats": None,
        "venue_stats": None,
        "team_stats": None,
        "match_stats": None,
        "partnerships": None,
    }

    reader = pd.read_csv(csv_path, chunksize=settings.chunk_size)
    mapping: Optional[Dict[str, str]] = None

    for i, chunk in enumerate(reader):
        if mapping is None:
            mapping = resolve_schema(chunk, use_llm=settings.use_llm_schema_hint)
            logger.info(f"Resolved schema mapping: {mapping}")
        # Compute chunk aggregates
        aggs = aggregate_core(chunk, mapping)
        parts = compute_partnerships(chunk, mapping)
        if parts is not None and not parts.empty:
            aggs["partnerships"] = parts

        # Merge into running totals
        for key, df in aggs.items():
            if df is None or len(df) == 0:
                continue
            if aggregates[key] is None:
                aggregates[key] = df
            else:
                # Determine key columns (first two for edges; first for nodes; use heuristics)
                if key in ("batter_bowler", "batter_venue", "bowler_venue", "batter_phase", "bowler_phase", "partnerships"):
                    on = list(df.columns[:2]) if key != "batter_phase" and key != "bowler_phase" else [df.columns[0], "phase"]
                elif key in ("batter_stats", "bowler_stats", "venue_stats", "team_stats", "match_stats"):
                    on = [df.columns[0]]
                else:
                    on = [df.columns[0]]
                aggregates[key] = _merge_numeric_frames(aggregates[key], df, on=on)

        if (i + 1) % 10 == 0:
            logger.info(f"Aggregated { (i + 1) * settings.chunk_size:,} rows...")

    # Finalize
    out: Dict[str, pd.DataFrame] = {k: (v if v is not None else pd.DataFrame()) for k, v in aggregates.items()}

    # Cache results
    try:
        pd.to_pickle(out, final_cache)
        logger.info(f"Cached aggregates to {final_cache}")
    except Exception as e:
        logger.warning(f"Failed to cache aggregates: {e}")

    return out



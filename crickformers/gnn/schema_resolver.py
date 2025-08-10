# Purpose: Resolve schema/column names across heterogeneous cricket datasets
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _candidate_lists() -> Dict[str, List[str]]:
    return {
        "batter": ["batter", "batsman", "Batter", "Batter Name", "striker"],
        "bowler": ["bowler", "Bowler", "bowler_name"],
        "venue": ["venue", "ground", "stadium", "Venue"],
        "runs_scored": ["runs_scored", "runs", "Runs", "runs_off_bat"],
        "is_wicket": ["is_wicket", "wicket", "Wicket", "dismissal", "dismissal_type"],
        "team_batting": ["team_batting", "batting_team", "battingteam", "Batting Team"],
        "team_bowling": ["team_bowling", "bowling_team", "Bowling Team"],
        "match_id": ["match_id", "Match ID", "match", "game_id"],
        "innings": ["innings", "Innings"],
        "over": ["over", "Over"],
    }


def resolve_schema(df: pd.DataFrame, use_llm: bool = False) -> Dict[str, str]:
    """
    Resolve canonical column names from a DataFrame with flexible aliases.

    Priority: heuristics > (optional) LLM hints. Will not call external APIs unless use_llm is True
    and OPENAI_API_KEY is set; even then, heuristics remain primary.
    """
    candidates = _candidate_lists()
    mapping: Dict[str, str] = {}

    # Heuristic pass
    lower_cols = {c.lower(): c for c in df.columns}
    for canonical, names in candidates.items():
        chosen: Optional[str] = None
        for n in names:
            if n in df.columns:
                chosen = n
                break
            if n.lower() in lower_cols:
                chosen = lower_cols[n.lower()]
                break
        if chosen is None:
            # Fallback: try fuzzy simple contains
            chosen = _fuzzy_pick(df.columns, names)
        if chosen:
            mapping[canonical] = chosen

    # Derive/normalize fields
    if "is_wicket" not in mapping and "dismissal" in df.columns:
        mapping["is_wicket"] = "dismissal"

    # Optional LLM augmentation (non-blocking)
    if use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            mapping = _augmented_by_llm(df, mapping)
        except Exception as e:
            logger.warning(f"LLM schema hint failed, using heuristics only: {e}")

    # Validate criticals
    critical = ["batter", "bowler", "runs_scored", "match_id", "innings", "over"]
    missing = [k for k in critical if k not in mapping]
    if missing:
        logger.error(f"Missing critical columns: {missing}")
    return mapping


def _fuzzy_pick(columns: List[str], names: List[str]) -> Optional[str]:
    low_cols = [c.lower() for c in columns]
    low_names = [n.lower() for n in names]
    for i, c in enumerate(low_cols):
        for n in low_names:
            if n in c:
                return columns[i]
    return None


def _augmented_by_llm(df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Use an LLM to suggest column alignments for any unresolved canonical keys.
    This function is best-effort and never overrides existing heuristic choices.
    """
    unresolved = [k for k in _candidate_lists().keys() if k not in mapping]
    if not unresolved:
        return mapping

    # Prepare prompt (limited preview of columns)
    preview = {c: str(df[c].dtype) for c in df.columns[:50]}
    prompt = (
        "You are mapping cricket ball-by-ball CSV columns to canonical names.\n"
        f"Unresolved targets: {unresolved}.\n"
        f"Available columns (name:dtype): {preview}.\n"
        "Suggest a JSON mapping subset for unresolved targets only."
    )

    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = resp.choices[0].message.content or "{}"
        import json
        llm_map = json.loads(content)
        if isinstance(llm_map, dict):
            for k, v in llm_map.items():
                if k in unresolved and v in df.columns and k not in mapping:
                    mapping[k] = v
    except Exception as e:
        logger.debug(f"LLM augmentation skipped: {e}")
    return mapping



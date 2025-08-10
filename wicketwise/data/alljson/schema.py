# Purpose: Canonical schema and helpers for ingesting all_json corpus into BallEventV1
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass(frozen=True)
class BallEventV1:
    # identity
    source_file: str
    source_match_id: str
    source_format: str
    match_type: str
    competition: Optional[str]
    season: Optional[str]
    date_utc: Optional[str]
    city: Optional[str]
    venue: Optional[str]
    # participants
    team_batting: str
    team_bowling: str
    batter_id: Optional[str]
    bowler_id: Optional[str]
    non_striker_id: Optional[str]
    batter_name: str
    bowler_name: str
    non_striker_name: Optional[str]
    # indexing
    innings_index: int
    over_number: int
    delivery_index: int
    legal_ball: bool
    # event values
    runs_batter: int
    runs_extras: int
    runs_total: int
    extras_byes: Optional[int]
    extras_legbyes: Optional[int]
    extras_wides: Optional[int]
    extras_noballs: Optional[int]
    dismissal_kind: Optional[str]
    player_out_id: Optional[str]
    fielder_ids: Optional[List[str]]
    # provenance
    id_confidence: float
    anomalies: Optional[List[str]]
    parse_version: str = "v1.0"


CRICSHEET_SOURCE_FORMAT = "cricsheet_json"

# Extras considered not consuming legal balls
NON_LEGAL_EXTRAS_KEYS = {"wides", "noballs"}

# JSON keys we expect for extras
EXTRAS_KEYS = ("byes", "legbyes", "wides", "noballs")

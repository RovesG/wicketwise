# Purpose: Ingest cricsheet-like JSON files to BallEventV1 parquet events
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional

import pandas as pd

from .schema import BallEventV1, CRICSHEET_SOURCE_FORMAT, EXTRAS_KEYS, NON_LEGAL_EXTRAS_KEYS


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _name_to_id(name: Optional[str], people: Dict[str, str]) -> Optional[str]:
    if not name:
        return None
    if name in people:
        return people[name]
    # Fallback deterministic placeholder id
    return f"name::{name.strip().lower()}"


def iter_events_from_json(json_obj: Dict[str, Any], source_file: str) -> Iterator[BallEventV1]:
    info = json_obj.get("info", {})
    people = _safe_get(info, ["registry", "people"], {}) or {}

    teams = info.get("teams", []) or []
    season = info.get("season")
    competition = _safe_get(info, ["event", "name"], None)
    date_utc = (info.get("dates") or [None])[0]

    innings = json_obj.get("innings", []) or []

    for i_idx, inn in enumerate(innings):
        team_batting = inn.get("team")
        if teams and team_batting in teams and len(teams) == 2:
            team_bowling = teams[0] if teams[1] == team_batting else teams[1]
        else:
            team_bowling = None or ""

        overs = inn.get("overs", []) or []
        for over in overs:
            over_number = int(over.get("over", 0))
            deliveries = over.get("deliveries", []) or []
            for d_idx, delivery in enumerate(deliveries):
                batter_name = delivery.get("batter", "")
                bowler_name = delivery.get("bowler", "")
                non_striker_name = delivery.get("non_striker")

                batter_id = _name_to_id(batter_name, people)
                bowler_id = _name_to_id(bowler_name, people)
                non_striker_id = _name_to_id(non_striker_name, people) if non_striker_name else None

                runs = delivery.get("runs", {}) or {}
                runs_batter = int(runs.get("batter", 0))
                runs_extras = int(runs.get("extras", 0))
                runs_total = int(runs.get("total", runs_batter + runs_extras))

                extras_obj = delivery.get("extras", {}) or {}
                extras_byes = extras_obj.get("byes")
                extras_legbyes = extras_obj.get("legbyes")
                extras_wides = extras_obj.get("wides")
                extras_noballs = extras_obj.get("noballs")

                # legal ball detection
                legal_ball = not (
                    (isinstance(extras_wides, int) and extras_wides > 0)
                    or (isinstance(extras_noballs, int) and extras_noballs > 0)
                )

                # wickets
                dismissal_kind = None
                player_out_id = None
                fielder_ids: List[str] = []
                if "wickets" in delivery and delivery["wickets"]:
                    wk = delivery["wickets"][0]
                    dismissal_kind = wk.get("kind")
                    player_out_name = wk.get("player_out")
                    player_out_id = _name_to_id(player_out_name, people) if player_out_name else None
                    for f in wk.get("fielders", []) or []:
                        fid = _name_to_id(f.get("name"), people)
                        if fid:
                            fielder_ids.append(fid)

                anomalies: List[str] = []
                if runs_total != runs_batter + runs_extras:
                    anomalies.append("runs_mismatch")
                if not team_bowling:
                    anomalies.append("missing_team_bowling")

                yield BallEventV1(
                    source_file=source_file,
                    source_match_id=Path(source_file).stem,
                    source_format=CRICSHEET_SOURCE_FORMAT,
                    match_type=str(info.get("match_type", "")),
                    competition=competition,
                    season=season,
                    date_utc=date_utc,
                    city=info.get("city"),
                    venue=info.get("venue"),
                    team_batting=team_batting or "",
                    team_bowling=team_bowling or "",
                    batter_id=batter_id,
                    bowler_id=bowler_id,
                    non_striker_id=non_striker_id,
                    batter_name=batter_name,
                    bowler_name=bowler_name,
                    non_striker_name=non_striker_name,
                    innings_index=i_idx,
                    over_number=over_number,
                    delivery_index=d_idx,
                    legal_ball=bool(legal_ball),
                    runs_batter=runs_batter,
                    runs_extras=runs_extras,
                    runs_total=runs_total,
                    extras_byes=extras_byes,
                    extras_legbyes=extras_legbyes,
                    extras_wides=extras_wides,
                    extras_noballs=extras_noballs,
                    dismissal_kind=dismissal_kind,
                    player_out_id=player_out_id,
                    fielder_ids=fielder_ids or None,
                    id_confidence=1.0 if people else 0.8,
                    anomalies=anomalies or None,
                )


def flatten_file_to_dataframe(json_path: str) -> pd.DataFrame:
    p = Path(json_path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    rows = [e.__dict__ for e in iter_events_from_json(obj, p.name)]
    return pd.DataFrame(rows)

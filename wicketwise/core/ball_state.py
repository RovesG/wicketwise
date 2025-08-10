# Purpose: Canonical BallState v1 and loader utilities
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import pandas as pd


@dataclass
class BallStateV1:
    match_id: str
    innings: int
    over: float
    ball: int
    striker_id: str
    non_striker_id: Optional[str]
    bowler_id: str
    score: int
    wickets: int
    balls_remaining: int
    run_rate: float
    req_rate: Optional[float]
    target: Optional[int]
    phase: str
    venue_id: Optional[str] = None
    # Optionals
    pre_ball_odds: Optional[float] = None
    overround: Optional[float] = None
    liq_bucket: Optional[int] = None
    # Reserved for future extensions
    extras: Dict[str, float] = field(default_factory=dict)


def derive_phase_from_over(over: float) -> str:
    if over < 6:
        return "PP"
    if over < 16:
        return "MID"
    return "DEATH"


def build_ball_states(df: pd.DataFrame, mapping: Dict[str, str]) -> List[BallStateV1]:
    """
    Construct BallStateV1 list from a normalized DataFrame and schema mapping.
    Invariants: (match_id, innings, over, ball) unique; no lookahead features included.
    """
    # Column accessors
    mid = mapping.get("match_id", "match_id")
    inn = mapping.get("innings", "innings")
    over_col = mapping.get("over", "over")
    batter = mapping.get("batter", "batter")
    bowler = mapping.get("bowler", "bowler")
    venue = mapping.get("venue", "venue")
    runs = mapping.get("runs_scored", "runs_scored")
    wicket = mapping.get("is_wicket", "is_wicket")

    # Coerce dtypes
    work = df.copy()
    work[runs] = pd.to_numeric(work[runs], errors="coerce").fillna(0).astype(int)
    if wicket in work.columns:
        if work[wicket].dtype != int:
            work[wicket] = ((~work[wicket].isna()) & (work[wicket] != 0) & (work[wicket] != "")).astype(int)
    else:
        work[wicket] = 0

    # Derive running totals per (match_id, innings)
    work.sort_values([mid, inn, over_col], inplace=True)
    work["__score__"] = work.groupby([mid, inn])[runs].cumsum().shift(fill_value=0)
    work["__wickets__"] = work.groupby([mid, inn])[wicket].cumsum().shift(fill_value=0)

    # Basic balls remaining (T20 assumption: 120 balls per innings)
    # over may be fractional (e.g., 4.3). Convert to ball index approx: floor(over)*6 + remainder*10
    o = pd.to_numeric(work[over_col], errors="coerce").fillna(0.0)
    whole = (o // 1).astype(int)
    rem = ((o - whole) * 10).round().astype(int)
    ball_num = (whole * 6 + rem).clip(lower=0, upper=120)
    balls_remaining = (120 - ball_num).clip(lower=0)

    # Current run rate using prior totals only (no lookahead): score_so_far / overs_bowled_so_far
    overs_bowled = (ball_num / 6).replace(0, 0.0001)
    run_rate = (work["__score__"] / overs_bowled).astype(float)

    states: List[BallStateV1] = []
    for i, row in work.iterrows():
        states.append(
            BallStateV1(
                match_id=str(row[mid]),
                innings=int(row[inn]),
                over=float(row[over_col]),
                ball=int(ball_num.loc[i]),
                striker_id=str(row.get(batter, "unknown")),
                non_striker_id=None,
                bowler_id=str(row.get(bowler, "unknown")),
                score=int(row["__score__"]),
                wickets=int(row["__wickets__"]),
                balls_remaining=int(balls_remaining.loc[i]),
                run_rate=float(run_rate.loc[i]),
                req_rate=None,
                target=None,
                phase=derive_phase_from_over(float(row[over_col])),
                venue_id=str(row.get(venue)) if venue in row and pd.notna(row.get(venue)) else None,
            )
        )
    return states



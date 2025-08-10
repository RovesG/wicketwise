# Purpose: Vectorized aggregations for scalable cricket knowledge graph building
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

from __future__ import annotations

from typing import Dict, Tuple, List
import pandas as pd
import numpy as np


def derive_phase(over_series: pd.Series) -> pd.Series:
    over = pd.to_numeric(over_series, errors="coerce").fillna(0)
    return pd.cut(over, bins=[-np.inf, 6, 16, np.inf], labels=["powerplay", "middle_overs", "death_overs"])


def aggregate_core(df: pd.DataFrame, m: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Compute vectorized aggregates for edge tables and node statistics.
    Returns a dict of DataFrames keyed by aggregate name.
    """
    # Ensure required columns exist minimally
    req = [m[k] for k in ["batter", "bowler", "runs_scored", "match_id", "innings", "over"] if k in m]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    batter = m.get("batter")
    bowler = m.get("bowler")
    runs = m.get("runs_scored")
    is_wicket = m.get("is_wicket")
    venue = m.get("venue")
    team_bat = m.get("team_batting")
    team_bowl = m.get("team_bowling")
    match_id = m.get("match_id")
    innings = m.get("innings")
    over = m.get("over")

    work = df.copy()
    work[runs] = pd.to_numeric(work[runs], errors="coerce").fillna(0).astype(int)
    if is_wicket:
        # Normalize to boolean-like 0/1
        w = work[is_wicket]
        if w.dtype == bool:
            work[is_wicket] = w.astype(int)
        else:
            work[is_wicket] = (~work[is_wicket].isna()) & (work[is_wicket] != 0) & (work[is_wicket] != "")
            work[is_wicket] = work[is_wicket].astype(int)

    # Phase
    work["__phase__"] = derive_phase(work[over])

    # Batter vs Bowler
    bb = (
        work.groupby([batter, bowler], dropna=True)
        .agg(balls_faced=(runs, "size"), runs_scored=(runs, "sum"), dismissals=(is_wicket, "sum") if is_wicket else (runs, "size"))
        .reset_index()
    )

    # Batter ↔ Venue
    bv = pd.DataFrame()
    if venue and batter and venue in work.columns:
        bv = (
            work.dropna(subset=[venue, batter])
            .groupby([batter, venue])
            .agg(balls_faced=(runs, "size"), runs_scored=(runs, "sum"))
            .reset_index()
        )

    # Bowler ↔ Venue
    bowv = pd.DataFrame()
    if venue and bowler and venue in work.columns:
        bowv = (
            work.dropna(subset=[venue, bowler])
            .groupby([bowler, venue])
            .agg(balls_bowled=(runs, "size"), runs_conceded=(runs, "sum"), wickets=(is_wicket, "sum") if is_wicket else (runs, "size"))
            .reset_index()
        )

    # Phase edges
    bphase = (
        work.dropna(subset=[batter])
        .groupby([batter, "__phase__"])  # type: ignore[arg-type]
        .agg(balls_faced=(runs, "size"), runs_scored=(runs, "sum"))
        .reset_index()
        .rename(columns={"__phase__": "phase"})
    )
    bowphase = (
        work.dropna(subset=[bowler])
        .groupby([bowler, "__phase__"])  # type: ignore[arg-type]
        .agg(balls_bowled=(runs, "size"), runs_conceded=(runs, "sum"), wickets=(is_wicket, "sum") if is_wicket else (runs, "size"))
        .reset_index()
        .rename(columns={"__phase__": "phase"})
    )

    # Node stats
    batter_stats = (
        work.groupby(batter)
        .agg(total_runs=(runs, "sum"), balls_faced=(runs, "size"), dismissals=(is_wicket, "sum") if is_wicket else (runs, "size"), matches_played=(match_id, pd.Series.nunique))
        .reset_index()
    )
    batter_stats["average"] = batter_stats["total_runs"] / batter_stats["dismissals"].clip(lower=1)
    batter_stats["strike_rate"] = (batter_stats["total_runs"] / batter_stats["balls_faced"]).fillna(0) * 100

    bowler_stats = (
        work.groupby(bowler)
        .agg(runs_conceded=(runs, "sum"), balls_bowled=(runs, "size"), wickets=(is_wicket, "sum") if is_wicket else (runs, "size"), matches_played=(match_id, pd.Series.nunique))
        .reset_index()
    )
    bowler_stats["average"] = bowler_stats["runs_conceded"] / bowler_stats["wickets"].clip(lower=1)
    bowler_stats["economy"] = (bowler_stats["runs_conceded"] / bowler_stats["balls_bowled"]).fillna(0) * 6
    bowler_stats["strike_rate"] = bowler_stats["balls_bowled"] / bowler_stats["wickets"].clip(lower=1)
    bowler_stats["wicket_rate"] = (bowler_stats["wickets"] / bowler_stats["balls_bowled"]).fillna(0)

    venue_stats = pd.DataFrame()
    if venue and venue in work.columns:
        inn_totals = work.groupby([venue, match_id, innings])[runs].sum().reset_index()
        venue_stats = (
            work.groupby(venue)
            .agg(matches_played=(match_id, pd.Series.nunique), balls_played=(runs, "size"))
            .reset_index()
        )
        v_mean = inn_totals.groupby(venue)[runs].mean().rename("avg_score")
        v_max = inn_totals.groupby(venue)[runs].max().rename("highest_score")
        v_min = inn_totals.groupby(venue)[runs].min().rename("lowest_score")
        venue_stats = venue_stats.merge(v_mean, on=venue, how="left").merge(v_max, on=venue, how="left").merge(v_min, on=venue, how="left")

    team_stats = pd.DataFrame()
    if team_bat and team_bowl and team_bat in work.columns and team_bowl in work.columns:
        team_union = pd.concat([
            work[[team_bat, match_id, runs]].rename(columns={team_bat: "team"}).assign(_role="bat"),
            work[[team_bowl, match_id, runs]].rename(columns={team_bowl: "team"}).assign(_role="bowl"),
        ], ignore_index=True)
        team_stats = (
            team_union.groupby("team")
            .agg(matches_played=(match_id, pd.Series.nunique))
            .reset_index()
        )
        bat = work.groupby(team_bat)[runs].sum().rename("runs_scored").reset_index().rename(columns={team_bat: "team"})
        bowl = work.groupby(team_bowl)[runs].sum().rename("runs_conceded").reset_index().rename(columns={team_bowl: "team"})
        w_lost = work.groupby(team_bat)[is_wicket].sum().rename("wickets_lost").reset_index().rename(columns={team_bat: "team"}) if is_wicket else None
        w_taken = work.groupby(team_bowl)[is_wicket].sum().rename("wickets_taken").reset_index().rename(columns={team_bowl: "team"}) if is_wicket else None
        team_stats = team_stats.merge(bat, on="team", how="left").merge(bowl, on="team", how="left")
        if w_lost is not None:
            team_stats = team_stats.merge(w_lost, on="team", how="left").merge(w_taken, on="team", how="left")

    match_stats = (
        work.groupby(match_id)
        .agg(total_balls=(runs, "size"), total_runs=(runs, "sum"), total_wickets=(is_wicket, "sum") if is_wicket else (runs, "size"), innings_count=(innings, pd.Series.nunique))
        .reset_index()
    )

    return {
        "batter_bowler": bb,
        "batter_venue": bv,
        "bowler_venue": bowv,
        "batter_phase": bphase,
        "bowler_phase": bowphase,
        "batter_stats": batter_stats,
        "bowler_stats": bowler_stats,
        "venue_stats": venue_stats,
        "team_stats": team_stats,
        "match_stats": match_stats,
    }


def compute_partnerships(df: pd.DataFrame, m: Dict[str, str]) -> pd.DataFrame:
    """
    Approximate partnerships per (match_id, innings) using available columns.
    If non-striker is present, use exact pairs; otherwise, pair last two distinct batters seen since last wicket.
    Returns DataFrame with columns: batter_a, batter_b, runs, partnerships
    """
    batter = m.get("batter")
    match_id = m.get("match_id")
    innings = m.get("innings")
    over = m.get("over")
    runs = m.get("runs_scored")
    non_striker = None
    for cand in ["non_striker", "non_striker_id", "Non-Striker", "nonStriker"]:
        if cand in df.columns:
            non_striker = cand
            break

    work = df[[match_id, innings, over, batter, runs] + ([non_striker] if non_striker else [])].copy()
    work[runs] = pd.to_numeric(work[runs], errors="coerce").fillna(0).astype(int)
    work["__order__"] = pd.to_numeric(work[over], errors="coerce").fillna(0)
    pairs: Dict[Tuple[str, str], Dict[str, int]] = {}

    for (mid, inn), sub in work.sort_values([match_id, innings, "__order__"]).groupby([match_id, innings]):
        active_pair: List[str] = []
        run_sum = 0
        for _, row in sub.iterrows():
            b = row[batter]
            ns = row[non_striker] if non_striker else None
            run_sum += row[runs]

            if non_striker and pd.notna(ns):
                pair = tuple(sorted([str(b), str(ns)]))
                key = (pair[0], pair[1])
                acc = pairs.setdefault(key, {"runs": 0, "partnerships": 0})
                acc["runs"] += row[runs]
            else:
                # Fallback: maintain last two batters encountered as the active pair
                if b not in active_pair:
                    active_pair.append(str(b))
                    if len(active_pair) > 2:
                        active_pair = active_pair[-2:]
                if len(active_pair) == 2:
                    key = tuple(sorted(active_pair))
                    acc = pairs.setdefault((key[0], key[1]), {"runs": 0, "partnerships": 0})
                    acc["runs"] += row[runs]

        # Count one partnership occurrence for the innings for each seen pair
        # (approximate; exact needs wicket events and striker rotation)
        for k in list(pairs.keys()):
            pairs[k]["partnerships"] += 1

    if not pairs:
        return pd.DataFrame(columns=["batter_a", "batter_b", "runs", "partnerships"])
    out = pd.DataFrame(
        [(a, b, v["runs"], v["partnerships"]) for (a, b), v in pairs.items()],
        columns=["batter_a", "batter_b", "runs", "partnerships"],
    )
    return out



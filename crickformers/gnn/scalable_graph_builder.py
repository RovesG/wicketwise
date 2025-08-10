# Purpose: Scalable KG assembly from aggregated tables with optional lightweight metrics
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

from __future__ import annotations

from typing import Dict
import networkx as nx
import pandas as pd


def build_graph_from_aggregates(aggs: Dict[str, pd.DataFrame], thresholds: Dict[str, int] | None = None) -> nx.DiGraph:
    G = nx.DiGraph()

    # Nodes: batters
    if "batter_stats" in aggs:
        for _, r in aggs["batter_stats"].iterrows():
            G.add_node(r.iloc[0], type="batter", total_runs=int(r.get("total_runs", 0)), balls_faced=int(r.get("balls_faced", 0)),
                       dismissals=int(r.get("dismissals", 0)), average=float(r.get("average", 0.0)), strike_rate=float(r.get("strike_rate", 0.0)),
                       matches_played=int(r.get("matches_played", 0)))

    # Nodes: bowlers
    if "bowler_stats" in aggs:
        for _, r in aggs["bowler_stats"].iterrows():
            G.add_node(r.iloc[0], type="bowler", runs_conceded=int(r.get("runs_conceded", 0)), balls_bowled=int(r.get("balls_bowled", 0)),
                       wickets=int(r.get("wickets", 0)), average=float(r.get("average", 0.0)), economy=float(r.get("economy", 0.0)),
                       strike_rate=float(r.get("strike_rate", 0.0)), wicket_rate=float(r.get("wicket_rate", 0.0)),
                       matches_played=int(r.get("matches_played", 0)))

    # Nodes: venues (if present)
    if "venue_stats" in aggs and not aggs["venue_stats"].empty:
        vdf = aggs["venue_stats"]
        name_col = vdf.columns[0]
        for _, r in vdf.iterrows():
            G.add_node(r[name_col], type="venue", matches_played=int(r.get("matches_played", 0)), balls_played=int(r.get("balls_played", 0)),
                       avg_score=float(r.get("avg_score", 0.0)), highest_score=float(r.get("highest_score", 0.0)), lowest_score=float(r.get("lowest_score", 0.0)))

    # Nodes: phases
    for p in ["powerplay", "middle_overs", "death_overs"]:
        G.add_node(p, type="phase")

    # Edges: batter_vs_bowler
    if "batter_bowler" in aggs:
        bbd = aggs["batter_bowler"]
        a, b = bbd.columns[0], bbd.columns[1]
        for _, r in bbd.iterrows():
            G.add_edge(r[a], r[b], edge_type="batter_vs_bowler", balls_faced=int(r.get("balls_faced", 0)),
                       runs_scored=int(r.get("runs_scored", 0)), dismissals=int(r.get("dismissals", 0)))

    # Edges: batter↔venue
    if "batter_venue" in aggs and not aggs["batter_venue"].empty:
        dv = aggs["batter_venue"]; a, v = dv.columns[0], dv.columns[1]
        for _, r in dv.iterrows():
            G.add_edge(r[a], r[v], edge_type="plays_at_venue", balls_faced=int(r.get("balls_faced", 0)), runs_scored=int(r.get("runs_scored", 0)))

    # Edges: bowler↔venue
    if "bowler_venue" in aggs and not aggs["bowler_venue"].empty:
        dv = aggs["bowler_venue"]; a, v = dv.columns[0], dv.columns[1]
        for _, r in dv.iterrows():
            G.add_edge(r[a], r[v], edge_type="bowls_at_venue", balls_bowled=int(r.get("balls_bowled", 0)),
                       runs_conceded=int(r.get("runs_conceded", 0)), wickets=int(r.get("wickets", 0)))

    # Edges: phase associations
    if "batter_phase" in aggs:
        dfp = aggs["batter_phase"]
        a, p = dfp.columns[0], "phase"
        for _, r in dfp.iterrows():
            G.add_edge(r[a], r[p], edge_type="performs_in_phase", balls_faced=int(r.get("balls_faced", 0)), runs_scored=int(r.get("runs_scored", 0)))

    if "bowler_phase" in aggs:
        dfp = aggs["bowler_phase"]
        a, p = dfp.columns[0], "phase"
        for _, r in dfp.iterrows():
            G.add_edge(r[a], r[p], edge_type="bowls_in_phase", balls_bowled=int(r.get("balls_bowled", 0)),
                       runs_conceded=int(r.get("runs_conceded", 0)), wickets=int(r.get("wickets", 0)))

    # Optional lightweight metrics (degree centrality only by default)
    deg = nx.degree_centrality(G)
    for n, d in deg.items():
        if n in G.nodes:
            G.nodes[n]["degree_centrality"] = float(d)

    return G



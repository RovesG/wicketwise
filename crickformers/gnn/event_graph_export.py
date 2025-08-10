# Purpose: Event-centric knowledge graph export and invariants
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd
import networkx as nx


def _event_id(match_id: str, innings: int, over: float) -> str:
    return f"event:{match_id}:{innings}:{over}"


def build_event_graph_sample(df: pd.DataFrame, mapping: Dict[str, str], max_events: int = 100_000) -> nx.DiGraph:
    """
    Build an event-centric KG sample with BallEvent nodes and typed relations.
    Creates nodes: Match, Innings, Over, Player(role), Venue, BallEvent
    Relations per event: IN_MATCH, IN_INNINGS, IN_OVER, FACED_BY, BOWLED_BY, AT_VENUE
    """
    mid = mapping.get("match_id", "match_id")
    inn = mapping.get("innings", "innings")
    over_col = mapping.get("over", "over")
    batter = mapping.get("batter", "batter")
    bowler = mapping.get("bowler", "bowler")
    venue = mapping.get("venue", "venue")

    G = nx.DiGraph()
    work = df[[mid, inn, over_col, batter, bowler] + ([venue] if venue in df.columns else [])].head(max_events).copy()

    for _, row in work.iterrows():
        m = str(row[mid]); i = int(row[inn]); o = float(row[over_col])
        ev = _event_id(m, i, o)
        bat = str(row.get(batter, "unknown_batter"))
        bow = str(row.get(bowler, "unknown_bowler"))
        ven = str(row.get(venue)) if venue in row and pd.notna(row.get(venue)) else None

        # BallEvent node
        G.add_node(ev, type="BallEvent", match_id=m, innings=i, over=o)

        # Match/Innings/Over nodes
        match_node = f"match:{m}"; G.add_node(match_node, type="Match")
        innings_node = f"innings:{m}:{i}"; G.add_node(innings_node, type="Innings")
        over_node = f"over:{m}:{i}:{o}"; G.add_node(over_node, type="Over")

        # Player nodes
        G.add_node(bat, type="Player", role="batter")
        G.add_node(bow, type="Player", role="bowler")

        # Venue node (optional)
        if ven:
            G.add_node(ven, type="Venue")

        # Edges per event
        G.add_edge(ev, match_node, edge_type="IN_MATCH")
        G.add_edge(ev, innings_node, edge_type="IN_INNINGS")
        G.add_edge(ev, over_node, edge_type="IN_OVER")
        G.add_edge(ev, bat, edge_type="FACED_BY")
        G.add_edge(ev, bow, edge_type="BOWLED_BY")
        if ven:
            G.add_edge(ev, ven, edge_type="AT_VENUE")

    return G


def validate_event_graph(G: nx.DiGraph) -> Tuple[bool, str]:
    """
    Validate invariants: each BallEvent has exactly one of IN_MATCH/IN_INNINGS/IN_OVER,
    and exactly one FACED_BY and BOWLED_BY; optional AT_VENUE.
    """
    for n, data in G.nodes(data=True):
        if data.get("type") != "BallEvent":
            continue
        outs = list(G.out_edges(n, data=True))
        types = [d.get("edge_type") for (_, _, d) in outs]
        if types.count("IN_MATCH") != 1:
            return False, f"BallEvent {n} missing/duplicate IN_MATCH"
        if types.count("IN_INNINGS") != 1:
            return False, f"BallEvent {n} missing/duplicate IN_INNINGS"
        if types.count("IN_OVER") != 1:
            return False, f"BallEvent {n} missing/duplicate IN_OVER"
        if types.count("FACED_BY") != 1:
            return False, f"BallEvent {n} missing/duplicate FACED_BY"
        if types.count("BOWLED_BY") != 1:
            return False, f"BallEvent {n} missing/duplicate BOWLED_BY"
    return True, "ok"



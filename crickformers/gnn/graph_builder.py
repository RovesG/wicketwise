# Purpose: Constructs a cricket knowledge graph using NetworkX.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains the logic for building a NetworkX DiGraph from
structured cricket match data. The graph represents relationships
between players, teams, venues, and match events with timestamped context.
"""

import networkx as nx
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def _determine_phase(over: int) -> str:
    """Determine match phase based on over number."""
    if over < 6:
        return "powerplay"
    elif over < 16:
        return "middle_overs"
    else:
        return "death_overs"

def _parse_datetime(date_str: str) -> datetime:
    """Parse datetime string with fallback for different formats."""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # If all formats fail, return current time and log warning
    logger.warning(f"Could not parse date: {date_str}, using current time")
    return datetime.now()

def build_cricket_graph(match_data: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Constructs a directed graph from a list of structured match events.
    Enhanced to support heterogeneous edges with timestamped context.

    Args:
        match_data: A list of dictionaries, where each dictionary represents
                    a ball-by-ball event with structured data.

    Returns:
        A NetworkX DiGraph object representing the cricket knowledge graph.
    """
    G = nx.DiGraph()

    for ball in match_data:
        # Extract entities from the ball data
        batter = ball.get("batter_id")
        bowler = ball.get("bowler_id")
        venue = ball.get("venue_name")
        team = ball.get("batting_team_name")
        bowler_type = ball.get("bowler_style")
        
        # Extract timestamped context
        match_date_str = ball.get("match_date", "")
        match_date = _parse_datetime(match_date_str) if match_date_str else datetime.now()
        over = ball.get("over", 0)
        phase = _determine_phase(over)
        runs = ball.get("runs", 0)
        dismissal_type = ball.get("dismissal_type", "")
        
        # Add nodes with a 'type' attribute for easy filtering
        if batter: G.add_node(batter, type="batter")
        if bowler: G.add_node(bowler, type="bowler")
        if venue: G.add_node(venue, type="venue")
        if team: G.add_node(team, type="team")
        if bowler_type: G.add_node(bowler_type, type="bowler_type")

        # Add heterogeneous edges with timestamped attributes
        
        # 1. "faced" edge - batter faced bowler
        if batter and bowler:
            edge_attrs = {
                "edge_type": "faced",
                "match_date": match_date,
                "phase": phase,
                "venue": venue,
                "runs": runs,
                "dismissal_type": dismissal_type if dismissal_type else "none"
            }
            
            # Update existing edge or create new one
            if G.has_edge(batter, bowler):
                existing_attrs = G[batter][bowler]
                existing_attrs["runs"] = existing_attrs.get("runs", 0) + runs
                existing_attrs["balls_faced"] = existing_attrs.get("balls_faced", 0) + 1
                # Keep most recent timestamp and phase
                existing_attrs["match_date"] = match_date
                existing_attrs["phase"] = phase
                existing_attrs["venue"] = venue
                if dismissal_type:
                    existing_attrs["dismissal_type"] = dismissal_type
            else:
                edge_attrs["balls_faced"] = 1
                G.add_edge(batter, bowler, **edge_attrs)

        # 2. "dismissed_by" edge - bowler dismissed batter
        if dismissal_type and batter and bowler:
            edge_attrs = {
                "edge_type": "dismissed_by",
                "match_date": match_date,
                "phase": phase,
                "venue": venue,
                "runs": runs,
                "dismissal_type": dismissal_type
            }
            
            # Update existing edge or create new one
            if G.has_edge(bowler, batter):
                existing_attrs = G[bowler][batter]
                existing_attrs["dismissals"] = existing_attrs.get("dismissals", 0) + 1
                # Keep most recent timestamp and phase
                existing_attrs["match_date"] = match_date
                existing_attrs["phase"] = phase
                existing_attrs["venue"] = venue
                existing_attrs["dismissal_type"] = dismissal_type
            else:
                edge_attrs["dismissals"] = 1
                G.add_edge(bowler, batter, **edge_attrs)

        # 3. "plays_for" edge - player plays for team
        if batter and team:
            edge_attrs = {
                "edge_type": "plays_for",
                "match_date": match_date,
                "phase": phase,
                "venue": venue,
                "runs": runs,
                "dismissal_type": "none"
            }
            
            if G.has_edge(batter, team):
                existing_attrs = G[batter][team]
                existing_attrs["runs"] = existing_attrs.get("runs", 0) + runs
                existing_attrs["balls_played"] = existing_attrs.get("balls_played", 0) + 1
                existing_attrs["match_date"] = match_date
                existing_attrs["phase"] = phase
                existing_attrs["venue"] = venue
            else:
                edge_attrs["balls_played"] = 1
                G.add_edge(batter, team, **edge_attrs)
            
        if bowler and team:
            edge_attrs = {
                "edge_type": "plays_for",
                "match_date": match_date,
                "phase": phase,
                "venue": venue,
                "runs": runs,
                "dismissal_type": "none"
            }
            
            if G.has_edge(bowler, team):
                existing_attrs = G[bowler][team]
                existing_attrs["runs_conceded"] = existing_attrs.get("runs_conceded", 0) + runs
                existing_attrs["balls_bowled"] = existing_attrs.get("balls_bowled", 0) + 1
                existing_attrs["match_date"] = match_date
                existing_attrs["phase"] = phase
                existing_attrs["venue"] = venue
                if dismissal_type:
                    existing_attrs["wickets"] = existing_attrs.get("wickets", 0) + 1
            else:
                edge_attrs["runs_conceded"] = runs
                edge_attrs["balls_bowled"] = 1
                edge_attrs["wickets"] = 1 if dismissal_type else 0
                G.add_edge(bowler, team, **edge_attrs)

        # 4. "match_played_at" edge - team played at venue
        if team and venue:
            edge_attrs = {
                "edge_type": "match_played_at",
                "match_date": match_date,
                "phase": phase,
                "venue": venue,
                "runs": runs,
                "dismissal_type": "none"
            }
            
            if G.has_edge(team, venue):
                existing_attrs = G[team][venue]
                existing_attrs["runs"] = existing_attrs.get("runs", 0) + runs
                existing_attrs["balls_played"] = existing_attrs.get("balls_played", 0) + 1
                existing_attrs["match_date"] = match_date
                existing_attrs["phase"] = phase
                if dismissal_type:
                    existing_attrs["wickets"] = existing_attrs.get("wickets", 0) + 1
            else:
                edge_attrs["balls_played"] = 1
                edge_attrs["wickets"] = 1 if dismissal_type else 0
                G.add_edge(team, venue, **edge_attrs)
            
        # 5. "excels_against" edge - batter excels against bowler type
        if runs >= 4 and batter and bowler_type:
            edge_attrs = {
                "edge_type": "excels_against",
                "match_date": match_date,
                "phase": phase,
                "venue": venue,
                "runs": runs,
                "dismissal_type": dismissal_type if dismissal_type else "none"
            }
            
            # Add or update the weight of the "excels_against" edge
            if G.has_edge(batter, bowler_type):
                existing_attrs = G[batter][bowler_type]
                existing_attrs["weight"] = existing_attrs.get("weight", 0) + 1
                existing_attrs["runs"] = existing_attrs.get("runs", 0) + runs
                existing_attrs["balls_faced"] = existing_attrs.get("balls_faced", 0) + 1
                existing_attrs["match_date"] = match_date
                existing_attrs["phase"] = phase
                existing_attrs["venue"] = venue
                if dismissal_type:
                    existing_attrs["dismissal_type"] = dismissal_type
            else:
                edge_attrs["weight"] = 1
                edge_attrs["balls_faced"] = 1
                G.add_edge(batter, bowler_type, **edge_attrs)

    return G 
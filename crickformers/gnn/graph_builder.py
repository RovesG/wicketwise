# Purpose: Constructs a cricket knowledge graph using NetworkX.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains the logic for building a NetworkX DiGraph from
structured cricket match data. The graph represents relationships
between players, teams, venues, and match events with timestamped context.
"""

import networkx as nx
from typing import List, Dict, Any, Optional
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
        
        # Determine and add event node
        event_type = _determine_event_type(runs, dismissal_type)
        if event_type:
            G.add_node(event_type, type="event")
            
            # Create batter → event edge
            if batter:
                _add_player_event_edge(G, batter, event_type, "batter_event", 
                                     match_date, phase, venue, runs, dismissal_type)
            
            # Create bowler → event edge  
            if bowler:
                _add_player_event_edge(G, bowler, event_type, "bowler_event",
                                     match_date, phase, venue, runs, dismissal_type)

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

    # Add new edge types after processing all balls
    _add_partnership_edges(G, match_data)
    _add_teammate_edges(G, match_data)
    _add_bowler_phase_edges(G, match_data)
    
    # Add role embeddings to player nodes
    from .role_embeddings import add_role_embeddings_to_graph
    G = add_role_embeddings_to_graph(G)

    return G


def build_cricket_graph_with_style_embeddings(match_data: List[Dict[str, Any]], 
                                             style_embeddings_path: Optional[str] = None,
                                             style_embeddings_dict: Optional[Dict[str, List[float]]] = None) -> nx.DiGraph:
    """
    Build a cricket knowledge graph with video-based style embeddings.
    
    This function builds the standard cricket graph and then adds style embeddings
    to player nodes from either a JSON file or a provided dictionary.
    
    Args:
        match_data: List of ball-by-ball match data dictionaries
        style_embeddings_path: Optional path to JSON file with style embeddings
        style_embeddings_dict: Optional dictionary with style embeddings
        
    Returns:
        NetworkX DiGraph with style embeddings added to player nodes
    """
    # Build the standard graph
    G = build_cricket_graph(match_data)
    
    # Load and add style embeddings
    from .style_embeddings import load_style_embeddings_from_json, add_style_embeddings_to_graph
    
    # Determine style embeddings source
    if style_embeddings_dict is not None:
        style_embeddings = style_embeddings_dict
    elif style_embeddings_path is not None:
        style_embeddings = load_style_embeddings_from_json(style_embeddings_path)
    else:
        # No style embeddings provided, use empty dict (will use default embeddings)
        style_embeddings = {}
    
    # Add style embeddings to the graph
    G = add_style_embeddings_to_graph(G, style_embeddings)
    
    return G


def build_cricket_hetero_graph(match_data: List[Dict[str, Any]]):
    """
    Build a cricket knowledge graph in PyTorch Geometric HeteroData format.
    
    This is the main entry point for creating HeteroData graphs.
    
    Args:
        match_data: List of ball-by-ball match data dictionaries
        
    Returns:
        HeteroData object ready for GNN training
    """
    # First build NetworkX graph
    nx_graph = build_cricket_graph(match_data)
    
    # Convert to HeteroData
    from .hetero_graph_builder import networkx_to_hetero_data
    hetero_data = networkx_to_hetero_data(nx_graph)
    
    return hetero_data


def _add_partnership_edges(G: nx.DiGraph, match_data: List[Dict[str, Any]]) -> None:
    """
    Add 'partnered_with' edges between batters who batted in the same partnership.
    
    Args:
        G: The NetworkX graph to add edges to
        match_data: List of ball-by-ball match data
    """
    # Group balls by match, innings, and partnership
    partnerships = {}
    
    for ball in match_data:
        match_id = ball.get("match_id", "unknown")
        innings = ball.get("innings", 1)
        batter = ball.get("batter_id")
        non_striker = ball.get("non_striker_id")
        
        # Extract temporal info
        match_date_str = ball.get("match_date", "")
        match_date = _parse_datetime(match_date_str) if match_date_str else datetime.now()
        over = ball.get("over", 0)
        phase = _determine_phase(over)
        venue = ball.get("venue_name")
        runs = ball.get("runs", 0)
        
        if batter and non_striker:
            # Create partnership key
            partnership_key = (match_id, innings)
            
            if partnership_key not in partnerships:
                partnerships[partnership_key] = {
                    'batters': set(),
                    'match_date': match_date,
                    'phase': phase,
                    'venue': venue,
                    'total_runs': 0,
                    'balls': 0
                }
            
            # Add batters to partnership
            partnerships[partnership_key]['batters'].add(batter)
            partnerships[partnership_key]['batters'].add(non_striker)
            partnerships[partnership_key]['total_runs'] += runs
            partnerships[partnership_key]['balls'] += 1
            
            # Update with most recent info
            partnerships[partnership_key]['match_date'] = match_date
            partnerships[partnership_key]['phase'] = phase
            partnerships[partnership_key]['venue'] = venue
    
    # Create partnership edges
    for partnership_info in partnerships.values():
        batters = list(partnership_info['batters'])
        
        # Create edges between all pairs of batters in the partnership
        for i, batter1 in enumerate(batters):
            for batter2 in batters[i+1:]:
                if batter1 != batter2:
                    edge_attrs = {
                        "edge_type": "partnered_with",
                        "match_date": partnership_info['match_date'],
                        "phase": partnership_info['phase'],
                        "venue": partnership_info['venue'],
                        "runs": partnership_info['total_runs'],
                        "dismissal_type": "none",
                        "weight": 1.0,
                        "balls_together": partnership_info['balls']
                    }
                    
                    # Add bidirectional edges (undirected partnership)
                    if not G.has_edge(batter1, batter2):
                        G.add_edge(batter1, batter2, **edge_attrs)
                    else:
                        # Update existing partnership
                        existing = G[batter1][batter2]
                        existing["runs"] = existing.get("runs", 0) + partnership_info['total_runs']
                        existing["balls_together"] = existing.get("balls_together", 0) + partnership_info['balls']
                        existing["weight"] = existing.get("weight", 0) + 1.0
                        existing["match_date"] = partnership_info['match_date']
                        existing["phase"] = partnership_info['phase']
                        existing["venue"] = partnership_info['venue']
                    
                    if not G.has_edge(batter2, batter1):
                        G.add_edge(batter2, batter1, **edge_attrs)
                    else:
                        # Update existing partnership (reverse direction)
                        existing = G[batter2][batter1]
                        existing["runs"] = existing.get("runs", 0) + partnership_info['total_runs']
                        existing["balls_together"] = existing.get("balls_together", 0) + partnership_info['balls']
                        existing["weight"] = existing.get("weight", 0) + 1.0
                        existing["match_date"] = partnership_info['match_date']
                        existing["phase"] = partnership_info['phase']
                        existing["venue"] = partnership_info['venue']


def _add_teammate_edges(G: nx.DiGraph, match_data: List[Dict[str, Any]]) -> None:
    """
    Add 'teammate_of' edges between players who appeared in the same match and team.
    
    Args:
        G: The NetworkX graph to add edges to
        match_data: List of ball-by-ball match data
    """
    # Group players by match and team
    team_matches = {}
    
    for ball in match_data:
        match_id = ball.get("match_id", "unknown")
        batting_team = ball.get("batting_team_name")
        bowling_team = ball.get("bowling_team_name")
        batter = ball.get("batter_id")
        bowler = ball.get("bowler_id")
        non_striker = ball.get("non_striker_id")
        
        # Extract temporal info
        match_date_str = ball.get("match_date", "")
        match_date = _parse_datetime(match_date_str) if match_date_str else datetime.now()
        over = ball.get("over", 0)
        phase = _determine_phase(over)
        venue = ball.get("venue_name")
        runs = ball.get("runs", 0)
        
        # Process batting team players
        if batting_team and batter:
            team_key = (match_id, batting_team)
            if team_key not in team_matches:
                team_matches[team_key] = {
                    'players': set(),
                    'match_date': match_date,
                    'phase': phase,
                    'venue': venue,
                    'total_runs': 0,
                    'balls': 0
                }
            
            team_matches[team_key]['players'].add(batter)
            if non_striker:
                team_matches[team_key]['players'].add(non_striker)
            team_matches[team_key]['total_runs'] += runs
            team_matches[team_key]['balls'] += 1
            team_matches[team_key]['match_date'] = match_date
            team_matches[team_key]['phase'] = phase
            team_matches[team_key]['venue'] = venue
        
        # Process bowling team players
        if bowling_team and bowler:
            team_key = (match_id, bowling_team)
            if team_key not in team_matches:
                team_matches[team_key] = {
                    'players': set(),
                    'match_date': match_date,
                    'phase': phase,
                    'venue': venue,
                    'total_runs': 0,
                    'balls': 0
                }
            
            team_matches[team_key]['players'].add(bowler)
            # Note: runs are against the bowling team, so we don't add them
            team_matches[team_key]['balls'] += 1
            team_matches[team_key]['match_date'] = match_date
            team_matches[team_key]['phase'] = phase
            team_matches[team_key]['venue'] = venue
    
    # Create teammate edges
    for team_info in team_matches.values():
        players = list(team_info['players'])
        
        # Create edges between all pairs of teammates
        for i, player1 in enumerate(players):
            for player2 in players[i+1:]:
                if player1 != player2:
                    edge_attrs = {
                        "edge_type": "teammate_of",
                        "match_date": team_info['match_date'],
                        "phase": team_info['phase'],
                        "venue": team_info['venue'],
                        "runs": team_info['total_runs'],
                        "dismissal_type": "none",
                        "weight": 1.0,
                        "balls_together": team_info['balls']
                    }
                    
                    # Add bidirectional edges (undirected teammate relationship)
                    if not G.has_edge(player1, player2):
                        G.add_edge(player1, player2, **edge_attrs)
                    else:
                        # Update existing teammate relationship
                        existing = G[player1][player2]
                        existing["weight"] = existing.get("weight", 0) + 1.0
                        existing["balls_together"] = existing.get("balls_together", 0) + team_info['balls']
                        existing["match_date"] = team_info['match_date']
                        existing["phase"] = team_info['phase']
                        existing["venue"] = team_info['venue']
                    
                    if not G.has_edge(player2, player1):
                        G.add_edge(player2, player1, **edge_attrs)
                    else:
                        # Update existing teammate relationship (reverse direction)
                        existing = G[player2][player1]
                        existing["weight"] = existing.get("weight", 0) + 1.0
                        existing["balls_together"] = existing.get("balls_together", 0) + team_info['balls']
                        existing["match_date"] = team_info['match_date']
                        existing["phase"] = team_info['phase']
                        existing["venue"] = team_info['venue']


def _add_bowler_phase_edges(G: nx.DiGraph, match_data: List[Dict[str, Any]]) -> None:
    """
    Add 'bowled_at' edges from bowlers to match phase nodes.
    
    Args:
        G: The NetworkX graph to add edges to
        match_data: List of ball-by-ball match data
    """
    # Track bowler-phase combinations
    bowler_phases = {}
    
    for ball in match_data:
        bowler = ball.get("bowler_id")
        over = ball.get("over", 0)
        phase = _determine_phase(over)
        
        # Extract temporal info
        match_date_str = ball.get("match_date", "")
        match_date = _parse_datetime(match_date_str) if match_date_str else datetime.now()
        venue = ball.get("venue_name")
        runs = ball.get("runs", 0)
        dismissal_type = ball.get("dismissal_type", "")
        
        if bowler and phase:
            # Add phase node if it doesn't exist
            G.add_node(phase, type="phase")
            
            # Create bowler-phase key
            bp_key = (bowler, phase)
            
            if bp_key not in bowler_phases:
                bowler_phases[bp_key] = {
                    'match_date': match_date,
                    'venue': venue,
                    'total_runs': 0,
                    'balls_bowled': 0,
                    'wickets': 0,
                    'overs': set()
                }
            
            # Update bowler-phase stats
            bowler_phases[bp_key]['total_runs'] += runs
            bowler_phases[bp_key]['balls_bowled'] += 1
            bowler_phases[bp_key]['overs'].add(over)
            if dismissal_type:
                bowler_phases[bp_key]['wickets'] += 1
            bowler_phases[bp_key]['match_date'] = match_date
            bowler_phases[bp_key]['venue'] = venue
    
    # Create bowler-phase edges
    for (bowler, phase), phase_info in bowler_phases.items():
        edge_attrs = {
            "edge_type": "bowled_at",
            "match_date": phase_info['match_date'],
            "phase": phase,
            "venue": phase_info['venue'],
            "runs": phase_info['total_runs'],
            "dismissal_type": "none",
            "weight": 1.0,
            "balls_bowled": phase_info['balls_bowled'],
            "wickets": phase_info['wickets'],
            "overs_bowled": len(phase_info['overs'])
        }
        
        if not G.has_edge(bowler, phase):
            G.add_edge(bowler, phase, **edge_attrs)
        else:
            # Update existing bowler-phase relationship
            existing = G[bowler][phase]
            existing["runs"] = existing.get("runs", 0) + phase_info['total_runs']
            existing["balls_bowled"] = existing.get("balls_bowled", 0) + phase_info['balls_bowled']
            existing["wickets"] = existing.get("wickets", 0) + phase_info['wickets']
            existing["overs_bowled"] = existing.get("overs_bowled", 0) + len(phase_info['overs'])
            existing["weight"] = existing.get("weight", 0) + 1.0
            existing["match_date"] = phase_info['match_date']
            existing["venue"] = phase_info['venue']


def _determine_event_type(runs: int, dismissal_type: str) -> str:
    """
    Determine the event type based on runs scored and dismissal information.
    
    Args:
        runs: Number of runs scored on the ball
        dismissal_type: Type of dismissal (empty string if no dismissal)
    
    Returns:
        Event type string: "four", "six", "dot", "wicket", or None
    """
    # Wicket takes priority over runs
    if dismissal_type and dismissal_type.strip():
        return "wicket"
    
    # Determine event based on runs
    if runs == 4:
        return "four"
    elif runs == 6:
        return "six"
    elif runs == 0:
        return "dot"
    else:
        # For 1, 2, 3, 5 runs, we could add more event types
        # For now, we'll treat them as "single", "double", etc.
        # But the requirement only asks for four, six, dot, wicket
        return None


def _add_player_event_edge(G: nx.DiGraph, player: str, event_type: str, edge_type: str,
                          match_date: datetime, phase: str, venue: str, 
                          runs: int, dismissal_type: str) -> None:
    """
    Add an edge from a player to an event node with appropriate attributes.
    
    Args:
        G: The NetworkX graph
        player: Player ID
        event_type: Type of event (four, six, dot, wicket)
        edge_type: Type of edge (batter_event, bowler_event)
        match_date: Date of the match
        phase: Phase of the match
        venue: Venue name
        runs: Runs scored
        dismissal_type: Type of dismissal
    """
    edge_attrs = {
        "edge_type": edge_type,
        "match_date": match_date,
        "phase": phase,
        "venue": venue,
        "runs": runs,
        "dismissal_type": dismissal_type if dismissal_type else "none",
        "weight": 1.0
    }
    
    # Add or update edge
    if G.has_edge(player, event_type):
        existing_attrs = G[player][event_type]
        # Aggregate statistics
        existing_attrs["weight"] = existing_attrs.get("weight", 0) + 1.0
        existing_attrs["total_runs"] = existing_attrs.get("total_runs", 0) + runs
        existing_attrs["event_count"] = existing_attrs.get("event_count", 0) + 1
        # Keep most recent temporal info
        existing_attrs["match_date"] = match_date
        existing_attrs["phase"] = phase
        existing_attrs["venue"] = venue
        if dismissal_type:
            existing_attrs["dismissal_type"] = dismissal_type
    else:
        edge_attrs["total_runs"] = runs
        edge_attrs["event_count"] = 1
        G.add_edge(player, event_type, **edge_attrs) 
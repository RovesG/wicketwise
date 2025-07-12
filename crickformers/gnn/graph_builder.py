# Purpose: Constructs a cricket knowledge graph using NetworkX.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains the logic for building a NetworkX DiGraph from
structured cricket match data. The graph represents relationships
between players, teams, venues, and match events.
"""

import networkx as nx
from typing import List, Dict, Any

def build_cricket_graph(match_data: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Constructs a directed graph from a list of structured match events.

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
        
        # Add nodes with a 'type' attribute for easy filtering
        if batter: G.add_node(batter, type="batter")
        if bowler: G.add_node(bowler, type="bowler")
        if venue: G.add_node(venue, type="venue")
        if team: G.add_node(team, type="team")
        if bowler_type: G.add_node(bowler_type, type="bowler_type")

        # Add edges based on the event
        if batter and bowler:
            G.add_edge(batter, bowler, edge_type="faced", runs=ball.get("runs", 0))

        if ball.get("dismissal_type"):
            G.add_edge(bowler, batter, edge_type="dismissed_by", 
                         dismissal_type=ball.get("dismissal_type"))

        if batter and team:
            G.add_edge(batter, team, edge_type="plays_for")
            
        if bowler and team:
            G.add_edge(bowler, team, edge_type="plays_for")
            
        if team and venue:
            G.add_edge(team, venue, edge_type="match_played_at", overs=ball.get("over"))
            
        # Example of a more complex relationship: "excels_against"
        if ball.get("runs", 0) >= 4 and batter and bowler_type:
            # Add or update the weight of the "excels_against" edge
            if G.has_edge(batter, bowler_type):
                G[batter][bowler_type]['weight'] += 1
            else:
                G.add_edge(batter, bowler_type, edge_type="excels_against", weight=1)

    return G 
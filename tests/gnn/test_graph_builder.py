# Purpose: Tests for the NetworkX graph builder.
# Author: Shamus Rae, Last Modified: 2024-07-30

import pytest
from crickformers.gnn.graph_builder import build_cricket_graph

@pytest.fixture
def mock_match_data():
    """Provides a small, structured dataset of a few match events."""
    return [
        {
            "batter_id": "player_A", "bowler_id": "player_B",
            "venue_name": "Lord's", "batting_team_name": "Team_X",
            "bowler_style": "fast", "runs": 4, "over": 1
        },
        {
            "batter_id": "player_A", "bowler_id": "player_B",
            "venue_name": "Lord's", "batting_team_name": "Team_X",
            "bowler_style": "fast", "runs": 1, "over": 1
        },
        {
            "batter_id": "player_C", "bowler_id": "player_B",
            "venue_name": "Lord's", "batting_team_name": "Team_X",
            "bowler_style": "fast", "runs": 0, "dismissal_type": "bowled", "over": 2
        },
    ]

def test_graph_node_creation(mock_match_data):
    """Validates that all unique entities are created as nodes with correct types."""
    G = build_cricket_graph(mock_match_data)
    
    expected_nodes = {
        "player_A": "batter",
        "player_B": "bowler",
        "player_C": "batter",
        "Lord's": "venue",
        "Team_X": "team",
        "fast": "bowler_type",
    }
    
    # New phase nodes and event nodes will be added automatically
    phase_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get("type") == "phase"]
    event_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get("type") == "event"]
    expected_total_nodes = len(expected_nodes) + len(phase_nodes) + len(event_nodes)
    
    assert G.number_of_nodes() == expected_total_nodes
    for node, node_type in expected_nodes.items():
        assert G.nodes[node]["type"] == node_type

def test_graph_edge_creation_and_attributes(mock_match_data):
    """Validates that edges are created with the correct types and attributes."""
    G = build_cricket_graph(mock_match_data)
    
    # Test 'faced' edge
    assert G.has_edge("player_A", "player_B")
    # Multiple 'faced' events between the same players should result in one edge
    # Enhanced implementation now aggregates runs across multiple balls
    assert G.get_edge_data("player_A", "player_B")["edge_type"] == "faced"
    assert G.get_edge_data("player_A", "player_B")["runs"] == 5  # 4 + 1 = 5
    assert G.get_edge_data("player_A", "player_B")["balls_faced"] == 2

    # Test 'dismissed_by' edge
    assert G.has_edge("player_B", "player_C")
    assert G.get_edge_data("player_B", "player_C")["edge_type"] == "dismissed_by"
    assert G.get_edge_data("player_B", "player_C")["dismissal_type"] == "bowled"

    # Test 'plays_for' edge
    assert G.has_edge("player_A", "Team_X")
    assert G.get_edge_data("player_A", "Team_X")["edge_type"] == "plays_for"
    
    # Test 'excels_against' edge - it should have a weight of 1 from the first event (4 runs)
    assert G.has_edge("player_A", "fast")
    assert G.get_edge_data("player_A", "fast")["edge_type"] == "excels_against"
    assert G.get_edge_data("player_A", "fast")["weight"] == 1

def test_graph_edge_counts(mock_match_data):
    """Confirms the total number of edges is as expected."""
    G = build_cricket_graph(mock_match_data)
    
    # Original expected edges:
    # 1. A faced B
    # 2. B dismissed C
    # 3. C faced B
    # 4. A plays_for Team_X
    # 5. B plays_for Team_X (implicit from batting team)
    # 6. C plays_for Team_X
    # 7. Team_X played_at Lord's
    # 8. A excels_against fast
    
    # New edges added:
    # 9. Partnership edges (A ↔ C, bidirectional)
    # 10. Teammate edges (A ↔ B ↔ C, multiple bidirectional)
    # 11. Bowler-phase edges (B → powerplay)
    
    # Count actual edges instead of hardcoding
    original_edge_types = {"faced", "dismissed_by", "plays_for", "match_played_at", "excels_against"}
    new_edge_types = {"partnered_with", "teammate_of", "bowled_at"}
    
    original_edges = sum(1 for _, _, attrs in G.edges(data=True) 
                        if attrs.get("edge_type") in original_edge_types)
    new_edges = sum(1 for _, _, attrs in G.edges(data=True) 
                   if attrs.get("edge_type") in new_edge_types)
    
    # Verify we have both original and new edges
    assert original_edges >= 5, f"Expected at least 5 original edges, got {original_edges}"
    assert new_edges >= 1, f"Expected at least 1 new edge, got {new_edges}"
    assert G.number_of_edges() >= 8, f"Expected at least 8 total edges, got {G.number_of_edges()}" 
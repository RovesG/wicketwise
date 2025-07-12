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
    
    assert G.number_of_nodes() == len(expected_nodes)
    for node, node_type in expected_nodes.items():
        assert G.nodes[node]["type"] == node_type

def test_graph_edge_creation_and_attributes(mock_match_data):
    """Validates that edges are created with the correct types and attributes."""
    G = build_cricket_graph(mock_match_data)
    
    # Test 'faced' edge
    assert G.has_edge("player_A", "player_B")
    # Multiple 'faced' events between the same players should result in one edge
    # NetworkX updates edge attributes on addition, so the last 'runs' value will be stored.
    assert G.get_edge_data("player_A", "player_B")["edge_type"] == "faced"
    assert G.get_edge_data("player_A", "player_B")["runs"] == 1

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
    
    # Expected edges:
    # 1. A faced B
    # 2. B dismissed C
    # 3. C faced B
    # 4. A plays_for Team_X
    # 5. B plays_for Team_X (implicit from batting team)
    # 6. C plays_for Team_X
    # 7. Team_X played_at Lord's
    # 8. A excels_against fast
    # Total unique edges will be less due to overwriting/updates
    
    # Let's count them from the graph:
    # A -> B (faced)
    # C -> B (faced)
    # B -> C (dismissed_by)
    # A -> Team_X (plays_for)
    # B -> Team_X (plays_for)
    # C -> Team_X (plays_for)
    # Team_X -> Lord's (match_played_at)
    # A -> fast (excels_against)
    assert G.number_of_edges() == 8 
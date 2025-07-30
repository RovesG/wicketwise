# Purpose: Tests for enhanced graph builder with multi-type, timestamped edges
# Author: Assistant, Last Modified: 2024

import pytest
from datetime import datetime
from crickformers.gnn.graph_builder import build_cricket_graph, _determine_phase, _parse_datetime

@pytest.fixture
def match_data_two_matches():
    """
    Provides structured dataset with 2 matches containing multiple edge types.
    Each match has different venues, dates, and phases to test timestamped attributes.
    """
    return [
        # Match 1 - Powerplay phase
        {
            "batter_id": "batter_A", "bowler_id": "bowler_X",
            "venue_name": "Lord's", "batting_team_name": "Team_Alpha",
            "bowler_style": "fast", "runs": 4, "over": 2,
            "match_date": "2024-07-15 14:30:00", "dismissal_type": ""
        },
        {
            "batter_id": "batter_A", "bowler_id": "bowler_X",
            "venue_name": "Lord's", "batting_team_name": "Team_Alpha",
            "bowler_style": "fast", "runs": 1, "over": 3,
            "match_date": "2024-07-15 14:35:00", "dismissal_type": ""
        },
        {
            "batter_id": "batter_B", "bowler_id": "bowler_Y",
            "venue_name": "Lord's", "batting_team_name": "Team_Alpha",
            "bowler_style": "spin", "runs": 0, "over": 4,
            "match_date": "2024-07-15 14:40:00", "dismissal_type": "bowled"
        },
        {
            "batter_id": "batter_C", "bowler_id": "bowler_X",
            "venue_name": "Lord's", "batting_team_name": "Team_Alpha", 
            "bowler_style": "fast", "runs": 6, "over": 5,
            "match_date": "2024-07-15 14:45:00", "dismissal_type": ""
        },
        
        # Match 2 - Middle overs and death overs
        {
            "batter_id": "batter_D", "bowler_id": "bowler_Z",
            "venue_name": "The Oval", "batting_team_name": "Team_Beta",
            "bowler_style": "medium", "runs": 2, "over": 8,
            "match_date": "2024-07-20 19:00:00", "dismissal_type": ""
        },
        {
            "batter_id": "batter_E", "bowler_id": "bowler_Z",
            "venue_name": "The Oval", "batting_team_name": "Team_Beta",
            "bowler_style": "medium", "runs": 4, "over": 12,
            "match_date": "2024-07-20 19:30:00", "dismissal_type": ""
        },
        {
            "batter_id": "batter_F", "bowler_id": "bowler_W",
            "venue_name": "The Oval", "batting_team_name": "Team_Beta",
            "bowler_style": "spin", "runs": 1, "over": 18,
            "match_date": "2024-07-20 20:15:00", "dismissal_type": "caught"
        },
        {
            "batter_id": "batter_G", "bowler_id": "bowler_W",
            "venue_name": "The Oval", "batting_team_name": "Team_Beta",
            "bowler_style": "spin", "runs": 6, "over": 19,
            "match_date": "2024-07-20 20:20:00", "dismissal_type": ""
        }
    ]

def test_datetime_parsing():
    """Test datetime parsing functionality with various formats."""
    
    # Test standard format
    dt1 = _parse_datetime("2024-07-15 14:30:00")
    assert dt1 == datetime(2024, 7, 15, 14, 30, 0)
    
    # Test date only
    dt2 = _parse_datetime("2024-07-15")
    assert dt2 == datetime(2024, 7, 15, 0, 0, 0)
    
    # Test ISO format
    dt3 = _parse_datetime("2024-07-15T14:30:00")
    assert dt3 == datetime(2024, 7, 15, 14, 30, 0)
    
    # Test invalid format (should not raise error)
    dt4 = _parse_datetime("invalid-date")
    assert isinstance(dt4, datetime)  # Should return current time

def test_phase_determination():
    """Test phase determination based on over number."""
    
    # Test powerplay phase
    assert _determine_phase(1) == "powerplay"
    assert _determine_phase(5) == "powerplay"
    
    # Test middle overs phase  
    assert _determine_phase(6) == "middle_overs"
    assert _determine_phase(10) == "middle_overs"
    assert _determine_phase(15) == "middle_overs"
    
    # Test death overs phase
    assert _determine_phase(16) == "death_overs"
    assert _determine_phase(20) == "death_overs"

def test_heterogeneous_edge_types(match_data_two_matches):
    """Test that all 5 edge types are created correctly."""
    
    G = build_cricket_graph(match_data_two_matches)
    
    # Collect all edge types
    edge_types = set()
    for u, v, data in G.edges(data=True):
        edge_types.add(data.get("edge_type"))
    
    # Assert all original edge types are present, plus new ones
    original_edge_types = {"faced", "dismissed_by", "plays_for", "match_played_at", "excels_against"}
    new_edge_types = {"partnered_with", "teammate_of", "bowled_at"}
    
    # Check that all original edge types are present
    for edge_type in original_edge_types:
        assert edge_type in edge_types, f"Missing original edge type: {edge_type}"
    
    # Check that at least some new edge types are present
    new_types_found = edge_types.intersection(new_edge_types)
    assert len(new_types_found) > 0, f"No new edge types found. Expected: {new_edge_types}, Found: {edge_types}"

def test_faced_edge_attributes(match_data_two_matches):
    """Test 'faced' edge attributes and aggregation."""
    
    G = build_cricket_graph(match_data_two_matches)
    
    # Test batter_A faced bowler_X (2 balls, 5 runs total)
    assert G.has_edge("batter_A", "bowler_X")
    edge_data = G.get_edge_data("batter_A", "bowler_X")
    
    assert edge_data["edge_type"] == "faced"
    assert edge_data["runs"] == 5  # 4 + 1
    assert edge_data["balls_faced"] == 2
    assert edge_data["phase"] == "powerplay"  # Most recent phase
    assert edge_data["venue"] == "Lord's"
    assert edge_data["dismissal_type"] == "none"
    assert isinstance(edge_data["match_date"], datetime)

def test_dismissed_by_edge_attributes(match_data_two_matches):
    """Test 'dismissed_by' edge attributes."""
    
    G = build_cricket_graph(match_data_two_matches)
    
    # Test bowler_Y dismissed batter_B
    assert G.has_edge("bowler_Y", "batter_B")
    edge_data = G.get_edge_data("bowler_Y", "batter_B")
    
    assert edge_data["edge_type"] == "dismissed_by"
    assert edge_data["dismissals"] == 1
    assert edge_data["phase"] == "powerplay"
    assert edge_data["venue"] == "Lord's"
    assert edge_data["dismissal_type"] == "bowled"
    assert isinstance(edge_data["match_date"], datetime)
    
    # Test bowler_W dismissed batter_F  
    assert G.has_edge("bowler_W", "batter_F")
    edge_data = G.get_edge_data("bowler_W", "batter_F")
    
    assert edge_data["edge_type"] == "dismissed_by"
    assert edge_data["dismissals"] == 1
    assert edge_data["phase"] == "death_overs"
    assert edge_data["venue"] == "The Oval"
    assert edge_data["dismissal_type"] == "caught"

def test_plays_for_edge_attributes(match_data_two_matches):
    """Test 'plays_for' edge attributes for batters and bowlers."""
    
    G = build_cricket_graph(match_data_two_matches)
    
    # Test batter plays_for team
    assert G.has_edge("batter_A", "Team_Alpha")
    edge_data = G.get_edge_data("batter_A", "Team_Alpha")
    
    assert edge_data["edge_type"] == "plays_for"
    assert edge_data["runs"] == 5  # Total runs scored by batter_A
    assert edge_data["balls_played"] == 2
    assert edge_data["phase"] == "powerplay"
    assert edge_data["venue"] == "Lord's"
    assert isinstance(edge_data["match_date"], datetime)
    
    # Test bowler plays_for team
    assert G.has_edge("bowler_X", "Team_Alpha")
    edge_data = G.get_edge_data("bowler_X", "Team_Alpha")
    
    assert edge_data["edge_type"] == "plays_for"
    assert edge_data["runs_conceded"] == 11  # 5 (batter_A) + 6 (batter_C)
    assert edge_data["balls_bowled"] == 3
    assert edge_data["wickets"] == 0  # No dismissals by bowler_X
    assert edge_data["phase"] == "powerplay"
    assert edge_data["venue"] == "Lord's"

def test_match_played_at_edge_attributes(match_data_two_matches):
    """Test 'match_played_at' edge attributes."""
    
    G = build_cricket_graph(match_data_two_matches)
    
    # Test Team_Alpha played at Lord's
    assert G.has_edge("Team_Alpha", "Lord's")
    edge_data = G.get_edge_data("Team_Alpha", "Lord's")
    
    assert edge_data["edge_type"] == "match_played_at"
    assert edge_data["runs"] == 11  # Total runs in match 1
    assert edge_data["balls_played"] == 4
    assert edge_data["wickets"] == 1  # One dismissal in match 1
    assert edge_data["phase"] == "powerplay"
    assert edge_data["venue"] == "Lord's"
    
    # Test Team_Beta played at The Oval
    assert G.has_edge("Team_Beta", "The Oval")
    edge_data = G.get_edge_data("Team_Beta", "The Oval")
    
    assert edge_data["edge_type"] == "match_played_at"
    assert edge_data["runs"] == 13  # Total runs in match 2
    assert edge_data["balls_played"] == 4
    assert edge_data["wickets"] == 1  # One dismissal in match 2
    assert edge_data["phase"] == "death_overs"  # Most recent phase
    assert edge_data["venue"] == "The Oval"

def test_excels_against_edge_attributes(match_data_two_matches):
    """Test 'excels_against' edge attributes."""
    
    G = build_cricket_graph(match_data_two_matches)
    
    # Test batter_A excels against fast bowling (scored 4 runs)
    assert G.has_edge("batter_A", "fast")
    edge_data = G.get_edge_data("batter_A", "fast")
    
    assert edge_data["edge_type"] == "excels_against"
    assert edge_data["weight"] == 1  # One boundary
    assert edge_data["runs"] == 4
    assert edge_data["balls_faced"] == 1
    assert edge_data["phase"] == "powerplay"
    assert edge_data["venue"] == "Lord's"
    assert edge_data["dismissal_type"] == "none"
    
    # Test batter_C excels against fast bowling (scored 6 runs)
    assert G.has_edge("batter_C", "fast")
    edge_data = G.get_edge_data("batter_C", "fast")
    
    assert edge_data["edge_type"] == "excels_against"
    assert edge_data["weight"] == 1  # One six
    assert edge_data["runs"] == 6
    assert edge_data["balls_faced"] == 1
    assert edge_data["phase"] == "powerplay"
    assert edge_data["venue"] == "Lord's"
    
    # Test batter_E excels against medium bowling (scored 4 runs)
    assert G.has_edge("batter_E", "medium")
    edge_data = G.get_edge_data("batter_E", "medium")
    
    assert edge_data["edge_type"] == "excels_against"
    assert edge_data["weight"] == 1
    assert edge_data["runs"] == 4
    assert edge_data["balls_faced"] == 1
    assert edge_data["phase"] == "middle_overs"
    assert edge_data["venue"] == "The Oval"

def test_timestamp_consistency_across_matches(match_data_two_matches):
    """Test that timestamps are correctly preserved across different matches."""
    
    G = build_cricket_graph(match_data_two_matches)
    
    # Check that edges from different matches have different timestamps
    lord_edges = [(u, v, data) for u, v, data in G.edges(data=True) if data.get("venue") == "Lord's"]
    oval_edges = [(u, v, data) for u, v, data in G.edges(data=True) if data.get("venue") == "The Oval"]
    
    # All Lord's edges should have July 15th dates
    for u, v, data in lord_edges:
        assert data["match_date"].date() == datetime(2024, 7, 15).date()
    
    # All Oval edges should have July 20th dates
    for u, v, data in oval_edges:
        assert data["match_date"].date() == datetime(2024, 7, 20).date()

def test_phase_consistency_across_matches(match_data_two_matches):
    """Test that phases are correctly determined across different matches."""
    
    G = build_cricket_graph(match_data_two_matches)
    
    # Check powerplay edges from Match 1
    powerplay_edges = [(u, v, data) for u, v, data in G.edges(data=True) 
                      if data.get("phase") == "powerplay"]
    
    # Should have powerplay edges from Match 1
    assert len(powerplay_edges) > 0
    
    # Check middle_overs edges from Match 2
    middle_edges = [(u, v, data) for u, v, data in G.edges(data=True) 
                   if data.get("phase") == "middle_overs"]
    
    # Should have middle_overs edges from Match 2
    assert len(middle_edges) > 0
    
    # Check death_overs edges from Match 2
    death_edges = [(u, v, data) for u, v, data in G.edges(data=True) 
                  if data.get("phase") == "death_overs"]
    
    # Should have death_overs edges from Match 2
    assert len(death_edges) > 0

def test_node_types_preserved(match_data_two_matches):
    """Test that node types are correctly preserved."""
    
    G = build_cricket_graph(match_data_two_matches)
    
    # Check node types
    expected_node_types = {
        "batter_A": "batter", "batter_B": "batter", "batter_C": "batter",
        "batter_D": "batter", "batter_E": "batter", "batter_F": "batter", "batter_G": "batter",
        "bowler_X": "bowler", "bowler_Y": "bowler", "bowler_Z": "bowler", "bowler_W": "bowler",
        "Team_Alpha": "team", "Team_Beta": "team",
        "Lord's": "venue", "The Oval": "venue",
        "fast": "bowler_type", "spin": "bowler_type", "medium": "bowler_type"
    }
    
    for node, expected_type in expected_node_types.items():
        assert G.nodes[node]["type"] == expected_type

def test_edge_aggregation_across_balls(match_data_two_matches):
    """Test that edge attributes are correctly aggregated across multiple balls."""
    
    G = build_cricket_graph(match_data_two_matches)
    
    # bowler_X bowled to multiple batters
    # Should have separate edges but correct aggregation for team relationship
    team_edge = G.get_edge_data("bowler_X", "Team_Alpha")
    assert team_edge["balls_bowled"] == 3  # 2 to batter_A + 1 to batter_C
    assert team_edge["runs_conceded"] == 11  # 5 + 6
    assert team_edge["wickets"] == 0  # No dismissals
    
    # Team_Alpha played multiple balls at Lord's
    venue_edge = G.get_edge_data("Team_Alpha", "Lord's")
    assert venue_edge["balls_played"] == 4  # All 4 balls in match 1
    assert venue_edge["runs"] == 11  # Total runs
    assert venue_edge["wickets"] == 1  # One dismissal 
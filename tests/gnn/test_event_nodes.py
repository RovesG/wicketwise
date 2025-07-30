# Purpose: Unit tests for event nodes in cricket knowledge graph
# Author: WicketWise Team, Last Modified: 2024-07-19

import pytest
import networkx as nx
from datetime import datetime
from crickformers.gnn.graph_builder import (
    build_cricket_graph,
    _determine_event_type,
    _add_player_event_edge
)


class TestEventNodes:
    """Test suite for event nodes functionality."""
    
    @pytest.fixture
    def sample_event_data(self):
        """Create sample match data with various event types."""
        base_date = datetime(2024, 1, 15, 14, 30)
        
        return [
            # Four
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 1,
                "batter_id": "batter_1",
                "bowler_id": "bowler_1",
                "batting_team_name": "team_a",
                "bowling_team_name": "team_b",
                "venue_name": "venue_1",
                "match_date": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": 4,
                "dismissal_type": ""
            },
            # Six
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 2,
                "batter_id": "batter_1",
                "bowler_id": "bowler_2",
                "batting_team_name": "team_a",
                "bowling_team_name": "team_b",
                "venue_name": "venue_1",
                "match_date": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": 6,
                "dismissal_type": ""
            },
            # Dot ball
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 3,
                "batter_id": "batter_2",
                "bowler_id": "bowler_1",
                "batting_team_name": "team_a",
                "bowling_team_name": "team_b",
                "venue_name": "venue_1",
                "match_date": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": 0,
                "dismissal_type": ""
            },
            # Wicket
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 4,
                "batter_id": "batter_2",
                "bowler_id": "bowler_2",
                "batting_team_name": "team_a",
                "bowling_team_name": "team_b",
                "venue_name": "venue_1",
                "match_date": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": 0,
                "dismissal_type": "bowled"
            },
            # Single (should not create event node based on requirements)
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 5,
                "batter_id": "batter_3",
                "bowler_id": "bowler_1",
                "batting_team_name": "team_a",
                "bowling_team_name": "team_b",
                "venue_name": "venue_1",
                "match_date": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": 1,
                "dismissal_type": ""
            },
            # Another four (for aggregation testing)
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 6,
                "batter_id": "batter_1",
                "bowler_id": "bowler_1",
                "batting_team_name": "team_a",
                "bowling_team_name": "team_b",
                "venue_name": "venue_1",
                "match_date": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": 4,
                "dismissal_type": ""
            }
        ]
    
    def test_determine_event_type_four(self):
        """Test event type determination for four."""
        event_type = _determine_event_type(4, "")
        assert event_type == "four", "4 runs should be classified as 'four'"
    
    def test_determine_event_type_six(self):
        """Test event type determination for six."""
        event_type = _determine_event_type(6, "")
        assert event_type == "six", "6 runs should be classified as 'six'"
    
    def test_determine_event_type_dot(self):
        """Test event type determination for dot ball."""
        event_type = _determine_event_type(0, "")
        assert event_type == "dot", "0 runs should be classified as 'dot'"
    
    def test_determine_event_type_wicket(self):
        """Test event type determination for wicket."""
        event_type = _determine_event_type(0, "bowled")
        assert event_type == "wicket", "Dismissal should be classified as 'wicket'"
        
        # Wicket takes priority over runs
        event_type = _determine_event_type(4, "caught")
        assert event_type == "wicket", "Wicket should take priority over runs"
    
    def test_determine_event_type_other_runs(self):
        """Test event type determination for other run values."""
        for runs in [1, 2, 3, 5]:
            event_type = _determine_event_type(runs, "")
            assert event_type is None, f"{runs} runs should return None (not tracked)"
    
    def test_determine_event_type_empty_dismissal(self):
        """Test event type determination with empty dismissal."""
        event_type = _determine_event_type(0, "")
        assert event_type == "dot", "Empty dismissal with 0 runs should be dot"
        
        event_type = _determine_event_type(4, "   ")
        assert event_type == "four", "Whitespace dismissal should be ignored"
    
    def test_event_nodes_created(self, sample_event_data):
        """Test that event nodes are created correctly."""
        G = build_cricket_graph(sample_event_data)
        
        # Check that event nodes exist
        event_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get("type") == "event"]
        
        expected_events = {"four", "six", "dot", "wicket"}
        actual_events = set(event_nodes)
        
        assert expected_events.issubset(actual_events), f"Missing event nodes. Expected: {expected_events}, Found: {actual_events}"
        
        # Check node types
        for event in expected_events:
            assert G.nodes[event]["type"] == "event", f"Event node {event} should have type 'event'"
    
    def test_batter_event_edges(self, sample_event_data):
        """Test that batter → event edges are created correctly."""
        G = build_cricket_graph(sample_event_data)
        
        # Check specific batter-event connections
        assert G.has_edge("batter_1", "four"), "batter_1 should be connected to 'four'"
        assert G.has_edge("batter_1", "six"), "batter_1 should be connected to 'six'"
        assert G.has_edge("batter_2", "dot"), "batter_2 should be connected to 'dot'"
        assert G.has_edge("batter_2", "wicket"), "batter_2 should be connected to 'wicket'"
        
        # Check edge attributes
        four_edge = G["batter_1"]["four"]
        assert four_edge["edge_type"] == "batter_event", "Edge should have correct type"
        assert "match_date" in four_edge, "Edge should have match_date"
        assert "phase" in four_edge, "Edge should have phase"
        assert "venue" in four_edge, "Edge should have venue"
        assert four_edge["weight"] >= 1.0, "Edge should have weight >= 1.0"
        assert "event_count" in four_edge, "Edge should have event_count"
        assert "total_runs" in four_edge, "Edge should have total_runs"
    
    def test_bowler_event_edges(self, sample_event_data):
        """Test that bowler → event edges are created correctly."""
        G = build_cricket_graph(sample_event_data)
        
        # Check specific bowler-event connections
        assert G.has_edge("bowler_1", "four"), "bowler_1 should be connected to 'four'"
        assert G.has_edge("bowler_1", "dot"), "bowler_1 should be connected to 'dot'"
        assert G.has_edge("bowler_2", "six"), "bowler_2 should be connected to 'six'"
        assert G.has_edge("bowler_2", "wicket"), "bowler_2 should be connected to 'wicket'"
        
        # Check edge attributes
        wicket_edge = G["bowler_2"]["wicket"]
        assert wicket_edge["edge_type"] == "bowler_event", "Edge should have correct type"
        assert wicket_edge["dismissal_type"] == "bowled", "Edge should preserve dismissal type"
        assert wicket_edge["runs"] == 0, "Wicket edge should have correct runs"
    
    def test_event_aggregation(self, sample_event_data):
        """Test that multiple events of same type are aggregated correctly."""
        G = build_cricket_graph(sample_event_data)
        
        # batter_1 hits two fours, so the edge should be aggregated
        four_edge = G["batter_1"]["four"]
        assert four_edge["weight"] >= 2.0, "Multiple fours should increase weight"
        assert four_edge["event_count"] >= 2, "Event count should be >= 2"
        assert four_edge["total_runs"] >= 8, "Total runs should be >= 8 (2 * 4)"
        
        # bowler_1 concedes two fours, so the edge should be aggregated
        bowler_four_edge = G["bowler_1"]["four"]
        assert bowler_four_edge["weight"] >= 2.0, "Multiple fours conceded should increase weight"
        assert bowler_four_edge["event_count"] >= 2, "Bowler event count should be >= 2"
    
    def test_event_count_matches_input_data(self, sample_event_data):
        """Test that count of each event type matches input data."""
        G = build_cricket_graph(sample_event_data)
        
        # Count events in input data
        input_events = {}
        for ball in sample_event_data:
            event_type = _determine_event_type(ball["runs"], ball["dismissal_type"])
            if event_type:
                input_events[event_type] = input_events.get(event_type, 0) + 1
        
        # Count events from graph edges
        graph_events = {}
        for source, target, attrs in G.edges(data=True):
            if attrs.get("edge_type") == "batter_event":  # Count from batter perspective
                event_count = attrs.get("event_count", 1)
                graph_events[target] = graph_events.get(target, 0) + event_count
        
        # Compare counts
        for event_type, expected_count in input_events.items():
            actual_count = graph_events.get(event_type, 0)
            assert actual_count == expected_count, \
                f"Event {event_type}: expected {expected_count}, got {actual_count}"
    
    def test_no_single_event_nodes(self, sample_event_data):
        """Test that single runs don't create event nodes."""
        G = build_cricket_graph(sample_event_data)
        
        # Should not have nodes for singles, doubles, triples
        unwanted_events = ["single", "double", "triple", "1", "2", "3", "5"]
        event_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get("type") == "event"]
        
        for unwanted in unwanted_events:
            assert unwanted not in event_nodes, f"Should not create event node for {unwanted}"
    
    def test_edge_attributes_completeness(self, sample_event_data):
        """Test that all event edges have required attributes."""
        G = build_cricket_graph(sample_event_data)
        
        required_attrs = ["edge_type", "match_date", "phase", "venue", "runs", 
                         "dismissal_type", "weight", "event_count", "total_runs"]
        
        event_edge_types = ["batter_event", "bowler_event"]
        
        for source, target, attrs in G.edges(data=True):
            if attrs.get("edge_type") in event_edge_types:
                for attr in required_attrs:
                    assert attr in attrs, f"Edge {source}->{target} missing attribute: {attr}"
                
                # Check data types
                assert isinstance(attrs["weight"], (int, float)), "Weight should be numeric"
                assert isinstance(attrs["event_count"], int), "Event count should be integer"
                assert isinstance(attrs["total_runs"], (int, float)), "Total runs should be numeric"
                assert attrs["weight"] >= 1.0, "Weight should be >= 1.0"
                assert attrs["event_count"] >= 1, "Event count should be >= 1"
    
    def test_wicket_priority_over_runs(self):
        """Test that wickets take priority over runs in event determination."""
        test_cases = [
            {"runs": 0, "dismissal": "bowled", "expected": "wicket"},
            {"runs": 1, "dismissal": "caught", "expected": "wicket"},
            {"runs": 4, "dismissal": "run_out", "expected": "wicket"},
            {"runs": 6, "dismissal": "stumped", "expected": "wicket"},
        ]
        
        for case in test_cases:
            event_type = _determine_event_type(case["runs"], case["dismissal"])
            assert event_type == case["expected"], \
                f"Runs {case['runs']} with dismissal '{case['dismissal']}' should be '{case['expected']}'"
    
    def test_add_player_event_edge_function(self):
        """Test the _add_player_event_edge function directly."""
        G = nx.DiGraph()
        G.add_node("player_1", type="batter")
        G.add_node("four", type="event")
        
        match_date = datetime(2024, 1, 15, 14, 30)
        
        # Add first edge
        _add_player_event_edge(G, "player_1", "four", "batter_event", 
                              match_date, "powerplay", "venue_1", 4, "")
        
        assert G.has_edge("player_1", "four"), "Edge should be created"
        
        edge_attrs = G["player_1"]["four"]
        assert edge_attrs["edge_type"] == "batter_event"
        assert edge_attrs["weight"] == 1.0
        assert edge_attrs["event_count"] == 1
        assert edge_attrs["total_runs"] == 4
        
        # Add second edge (should aggregate)
        _add_player_event_edge(G, "player_1", "four", "batter_event",
                              match_date, "powerplay", "venue_1", 4, "")
        
        edge_attrs = G["player_1"]["four"]
        assert edge_attrs["weight"] == 2.0, "Weight should be aggregated"
        assert edge_attrs["event_count"] == 2, "Event count should be aggregated"
        assert edge_attrs["total_runs"] == 8, "Total runs should be aggregated"
    
    def test_mixed_event_scenarios(self):
        """Test complex scenarios with mixed events."""
        match_data = [
            # Wicket with runs (run out scenario)
            {
                "match_id": "match_1", "innings": 1, "over": 1,
                "batter_id": "batter_1", "bowler_id": "bowler_1",
                "batting_team_name": "team_a", "bowling_team_name": "team_b",
                "venue_name": "venue_1", "match_date": "2024-01-15 14:30:00",
                "runs": 2, "dismissal_type": "run_out"
            },
            # Four followed by wicket from same bowler
            {
                "match_id": "match_1", "innings": 1, "over": 2,
                "batter_id": "batter_2", "bowler_id": "bowler_1",
                "batting_team_name": "team_a", "bowling_team_name": "team_b",
                "venue_name": "venue_1", "match_date": "2024-01-15 14:30:00",
                "runs": 4, "dismissal_type": ""
            },
            {
                "match_id": "match_1", "innings": 1, "over": 3,
                "batter_id": "batter_3", "bowler_id": "bowler_1",
                "batting_team_name": "team_a", "bowling_team_name": "team_b",
                "venue_name": "venue_1", "match_date": "2024-01-15 14:30:00",
                "runs": 0, "dismissal_type": "bowled"
            }
        ]
        
        G = build_cricket_graph(match_data)
        
        # Check that wicket takes priority
        assert G.has_edge("batter_1", "wicket"), "Run out should create wicket edge"
        assert not G.has_edge("batter_1", "double"), "Should not create double edge for run out"
        
        # Check that bowler is connected to both four and wicket
        assert G.has_edge("bowler_1", "four"), "Bowler should be connected to four"
        assert G.has_edge("bowler_1", "wicket"), "Bowler should be connected to wicket"
        
        # Check aggregation for bowler_1 -> wicket (should have 2 wickets)
        wicket_edge = G["bowler_1"]["wicket"]
        assert wicket_edge["event_count"] == 2, "Bowler should have 2 wickets"
        assert wicket_edge["weight"] == 2.0, "Bowler wicket weight should be 2.0"
    
    def test_event_nodes_with_no_events(self):
        """Test graph building with data that creates no event nodes."""
        match_data = [
            {
                "match_id": "match_1", "innings": 1, "over": 1,
                "batter_id": "batter_1", "bowler_id": "bowler_1",
                "batting_team_name": "team_a", "bowling_team_name": "team_b",
                "venue_name": "venue_1", "match_date": "2024-01-15 14:30:00",
                "runs": 1, "dismissal_type": ""  # Single - no event node
            },
            {
                "match_id": "match_1", "innings": 1, "over": 2,
                "batter_id": "batter_1", "bowler_id": "bowler_1",
                "batting_team_name": "team_a", "bowling_team_name": "team_b",
                "venue_name": "venue_1", "match_date": "2024-01-15 14:30:00",
                "runs": 2, "dismissal_type": ""  # Double - no event node
            }
        ]
        
        G = build_cricket_graph(match_data)
        
        # Should have no event nodes
        event_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get("type") == "event"]
        assert len(event_nodes) == 0, "Should have no event nodes for singles and doubles"
        
        # Should still have player nodes and other edges
        assert "batter_1" in G.nodes(), "Should still have batter node"
        assert "bowler_1" in G.nodes(), "Should still have bowler node"


class TestEventNodeEdgeCases:
    """Test edge cases for event nodes."""
    
    def test_none_values_handling(self):
        """Test handling of None values."""
        event_type = _determine_event_type(None, None)
        assert event_type is None, "None values should return None"
        
        event_type = _determine_event_type(0, None)
        assert event_type == "dot", "None dismissal with 0 runs should be dot"
    
    def test_negative_runs(self):
        """Test handling of negative runs (edge case)."""
        event_type = _determine_event_type(-1, "")
        assert event_type is None, "Negative runs should return None"
    
    def test_large_runs(self):
        """Test handling of unusually large run values."""
        event_type = _determine_event_type(100, "")
        assert event_type is None, "Large run values should return None"
    
    def test_case_insensitive_dismissal(self):
        """Test case handling of dismissal types."""
        event_type = _determine_event_type(0, "BOWLED")
        assert event_type == "wicket", "Uppercase dismissal should work"
        
        event_type = _determine_event_type(0, "Caught")
        assert event_type == "wicket", "Mixed case dismissal should work"
    
    def test_empty_player_ids(self):
        """Test handling of empty player IDs."""
        match_data = [
            {
                "match_id": "match_1", "innings": 1, "over": 1,
                "batter_id": "", "bowler_id": None,  # Empty/None player IDs
                "batting_team_name": "team_a", "bowling_team_name": "team_b",
                "venue_name": "venue_1", "match_date": "2024-01-15 14:30:00",
                "runs": 4, "dismissal_type": ""
            }
        ]
        
        G = build_cricket_graph(match_data)
        
        # Should still create event node
        assert "four" in G.nodes(), "Event node should be created even with empty player IDs"
        
        # Should not create edges to empty players
        event_edges = [(s, t, a) for s, t, a in G.edges(data=True) 
                      if a.get("edge_type") in ["batter_event", "bowler_event"]]
        assert len(event_edges) == 0, "Should not create edges to empty player IDs"
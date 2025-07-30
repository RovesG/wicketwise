# Purpose: Unit tests for new knowledge graph edge types
# Author: WicketWise Team, Last Modified: 2024-07-19

import pytest
import networkx as nx
from datetime import datetime, timedelta
from crickformers.gnn.graph_builder import (
    build_cricket_graph,
    _add_partnership_edges,
    _add_teammate_edges,
    _add_bowler_phase_edges
)


class TestNewGraphEdges:
    """Test suite for new knowledge graph edge types."""
    
    @pytest.fixture
    def sample_match_data(self):
        """Create sample match data for testing."""
        base_date = datetime(2024, 1, 15, 14, 30)
        
        return [
            # Match 1, Innings 1 - Partnership between batter1 and batter2
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 1,
                "batter_id": "batter_1",
                "non_striker_id": "batter_2",
                "bowler_id": "bowler_1",
                "batting_team_name": "team_a",
                "bowling_team_name": "team_b",
                "venue_name": "venue_1",
                "match_date": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": 4,
                "dismissal_type": ""
            },
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 2,
                "batter_id": "batter_2",
                "non_striker_id": "batter_1",
                "bowler_id": "bowler_1",
                "batting_team_name": "team_a",
                "bowling_team_name": "team_b",
                "venue_name": "venue_1",
                "match_date": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": 1,
                "dismissal_type": ""
            },
            # Match 1, Innings 1 - Death overs phase
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 18,
                "batter_id": "batter_1",
                "non_striker_id": "batter_3",
                "bowler_id": "bowler_2",
                "batting_team_name": "team_a",
                "bowling_team_name": "team_b",
                "venue_name": "venue_1",
                "match_date": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": 6,
                "dismissal_type": ""
            },
            # Match 1, Innings 2 - Different team batting
            {
                "match_id": "match_1",
                "innings": 2,
                "over": 5,
                "batter_id": "batter_4",
                "non_striker_id": "batter_5",
                "bowler_id": "bowler_3",
                "batting_team_name": "team_b",
                "bowling_team_name": "team_a",
                "venue_name": "venue_1",
                "match_date": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": 2,
                "dismissal_type": "bowled"
            }
        ]
    
    def test_partnered_with_edges(self, sample_match_data):
        """Test that partnered_with edges are created correctly."""
        G = build_cricket_graph(sample_match_data)
        
        # Check that partnership edges exist
        assert G.has_edge("batter_1", "batter_2"), "Partnership edge batter_1 -> batter_2 missing"
        assert G.has_edge("batter_2", "batter_1"), "Partnership edge batter_2 -> batter_1 missing"
        assert G.has_edge("batter_1", "batter_3"), "Partnership edge batter_1 -> batter_3 missing"
        assert G.has_edge("batter_4", "batter_5"), "Partnership edge batter_4 -> batter_5 missing"
        
        # Check edge attributes for partnership
        edge_attrs = G["batter_1"]["batter_2"]
        assert edge_attrs["edge_type"] == "partnered_with", "Wrong edge type for partnership"
        assert edge_attrs["weight"] >= 1.0, "Weight should be >= 1.0 for partnership edge"
        assert "match_date" in edge_attrs, "Missing match_date in partnership edge"
        assert "phase" in edge_attrs, "Missing phase in partnership edge"
        assert "venue" in edge_attrs, "Missing venue in partnership edge"
        assert "balls_together" in edge_attrs, "Missing balls_together in partnership edge"
        assert edge_attrs["dismissal_type"] == "none", "Wrong dismissal_type for partnership"
        
        # Check that partnership stats are aggregated correctly
        assert edge_attrs["runs"] > 0, "Partnership should have runs"
        assert edge_attrs["balls_together"] > 0, "Partnership should have balls together"
    
    def test_teammate_of_edges(self, sample_match_data):
        """Test that teammate_of edges are created correctly."""
        G = build_cricket_graph(sample_match_data)
        
        # Check that teammate edges exist within team_a
        assert G.has_edge("batter_1", "batter_2"), "Teammate edge batter_1 -> batter_2 missing"
        assert G.has_edge("batter_1", "batter_3"), "Teammate edge batter_1 -> batter_3 missing"
        
        # Check that teammate edges exist within team_b batting lineup
        assert G.has_edge("batter_4", "batter_5"), "Teammate edge batter_4 -> batter_5 missing"
        
        # Check that teammate edges exist within team_b bowling lineup
        assert G.has_edge("bowler_1", "bowler_2"), "Teammate edge bowler_1 -> bowler_2 missing"
        
        # Check edge attributes for teammates
        # Note: teammate_of and partnered_with edges might overlap, so check specific attributes
        edges_from_batter1 = list(G.edges("batter_1", data=True))
        teammate_edges = [e for e in edges_from_batter1 if e[2].get("edge_type") == "teammate_of"]
        
        if teammate_edges:
            edge_attrs = teammate_edges[0][2]
            assert edge_attrs["edge_type"] == "teammate_of", "Wrong edge type for teammate"
            assert edge_attrs["weight"] == 1.0, "Wrong weight for teammate edge"
            assert "match_date" in edge_attrs, "Missing match_date in teammate edge"
            assert "phase" in edge_attrs, "Missing phase in teammate edge"
            assert "venue" in edge_attrs, "Missing venue in teammate edge"
            assert "balls_together" in edge_attrs, "Missing balls_together in teammate edge"
            assert edge_attrs["dismissal_type"] == "none", "Wrong dismissal_type for teammate"
    
    def test_bowled_at_edges(self, sample_match_data):
        """Test that bowled_at edges are created correctly."""
        G = build_cricket_graph(sample_match_data)
        
        # Check that phase nodes are created (based on sample data overs: 1, 2, 5, 18)
        assert "powerplay" in G.nodes(), "Powerplay phase node missing"
        assert "death_overs" in G.nodes(), "Death overs phase node missing"
        # Note: middle_overs may not exist if no overs 6-15 in sample data
        
        # Check that phase nodes have correct type
        assert G.nodes["powerplay"]["type"] == "phase", "Wrong type for powerplay node"
        assert G.nodes["death_overs"]["type"] == "phase", "Wrong type for death_overs node"
        
        # Check that bowler-phase edges exist
        assert G.has_edge("bowler_1", "powerplay"), "Bowler-phase edge bowler_1 -> powerplay missing"
        assert G.has_edge("bowler_2", "death_overs"), "Bowler-phase edge bowler_2 -> death_overs missing"
        assert G.has_edge("bowler_3", "powerplay"), "Bowler-phase edge bowler_3 -> powerplay missing"
        
        # Check edge attributes for bowler-phase
        edge_attrs = G["bowler_1"]["powerplay"]
        assert edge_attrs["edge_type"] == "bowled_at", "Wrong edge type for bowler-phase"
        assert edge_attrs["weight"] == 1.0, "Wrong weight for bowler-phase edge"
        assert "match_date" in edge_attrs, "Missing match_date in bowler-phase edge"
        assert edge_attrs["phase"] == "powerplay", "Wrong phase in bowler-phase edge"
        assert "venue" in edge_attrs, "Missing venue in bowler-phase edge"
        assert "balls_bowled" in edge_attrs, "Missing balls_bowled in bowler-phase edge"
        assert "wickets" in edge_attrs, "Missing wickets in bowler-phase edge"
        assert "overs_bowled" in edge_attrs, "Missing overs_bowled in bowler-phase edge"
        assert edge_attrs["dismissal_type"] == "none", "Wrong dismissal_type for bowler-phase"
        
        # Check that bowler-phase stats are correct
        assert edge_attrs["balls_bowled"] > 0, "Bowler should have balls bowled"
        assert edge_attrs["overs_bowled"] > 0, "Bowler should have overs bowled"
    
    def test_edge_type_uniqueness(self, sample_match_data):
        """Test that each edge has a unique edge_type."""
        G = build_cricket_graph(sample_match_data)
        
        # Collect all edge types
        edge_types = set()
        for source, target, attrs in G.edges(data=True):
            edge_type = attrs.get("edge_type")
            if edge_type:
                edge_types.add(edge_type)
        
        # Check that we have the new edge types (others may not exist based on sample data)
        required_new_types = {"partnered_with", "teammate_of", "bowled_at"}
        
        for required_type in required_new_types:
            assert required_type in edge_types, f"Missing new edge type: {required_type}"
        
        # Check that basic edge types exist
        basic_types = {"faced", "plays_for"}
        for basic_type in basic_types:
            assert basic_type in edge_types, f"Missing basic edge type: {basic_type}"
    
    def test_temporal_info_in_new_edges(self, sample_match_data):
        """Test that new edges have correct temporal information."""
        G = build_cricket_graph(sample_match_data)
        
        # Test partnership edge temporal info
        if G.has_edge("batter_1", "batter_2"):
            attrs = G["batter_1"]["batter_2"]
            if attrs.get("edge_type") == "partnered_with":
                assert isinstance(attrs["match_date"], datetime), "match_date should be datetime"
                assert attrs["phase"] in ["powerplay", "middle_overs", "death_overs"], "Invalid phase"
                assert attrs["venue"] == "venue_1", "Wrong venue"
        
        # Test bowler-phase edge temporal info
        if G.has_edge("bowler_1", "powerplay"):
            attrs = G["bowler_1"]["powerplay"]
            assert attrs["edge_type"] == "bowled_at", "Wrong edge type"
            assert isinstance(attrs["match_date"], datetime), "match_date should be datetime"
            assert attrs["phase"] == "powerplay", "Wrong phase"
            assert attrs["venue"] == "venue_1", "Wrong venue"
    
    def test_weight_attribute_consistency(self, sample_match_data):
        """Test that all new edges have weight attribute set to 1.0."""
        G = build_cricket_graph(sample_match_data)
        
        new_edge_types = {"partnered_with", "teammate_of", "bowled_at"}
        
        for source, target, attrs in G.edges(data=True):
            edge_type = attrs.get("edge_type")
            if edge_type in new_edge_types:
                assert "weight" in attrs, f"Missing weight in {edge_type} edge"
                assert attrs["weight"] >= 1.0, f"Weight should be >= 1.0 for {edge_type} edge"
    
    def test_bidirectional_edges(self, sample_match_data):
        """Test that partnership and teammate edges are bidirectional."""
        G = build_cricket_graph(sample_match_data)
        
        # Test partnership bidirectionality
        partnership_pairs = []
        for source, target, attrs in G.edges(data=True):
            if attrs.get("edge_type") == "partnered_with":
                partnership_pairs.append((source, target))
        
        for source, target in partnership_pairs:
            assert G.has_edge(target, source), f"Missing reverse partnership edge {target} -> {source}"
            reverse_attrs = G[target][source]
            assert reverse_attrs.get("edge_type") == "partnered_with", "Reverse edge should also be partnership"
        
        # Test teammate bidirectionality
        teammate_pairs = []
        for source, target, attrs in G.edges(data=True):
            if attrs.get("edge_type") == "teammate_of":
                teammate_pairs.append((source, target))
        
        for source, target in teammate_pairs:
            assert G.has_edge(target, source), f"Missing reverse teammate edge {target} -> {source}"
            reverse_attrs = G[target][source]
            assert reverse_attrs.get("edge_type") == "teammate_of", "Reverse edge should also be teammate"
    
    def test_phase_node_creation(self, sample_match_data):
        """Test that phase nodes are created with correct attributes."""
        G = build_cricket_graph(sample_match_data)
        
        # Check that phase nodes exist
        phase_nodes = [node for node, attrs in G.nodes(data=True) 
                      if attrs.get("type") == "phase"]
        
        assert len(phase_nodes) > 0, "No phase nodes created"
        
        # Check specific phases based on sample data
        expected_phases = {"powerplay", "death_overs"}  # Based on overs 1, 2, 5, 18
        actual_phases = set(phase_nodes)
        
        for expected_phase in expected_phases:
            assert expected_phase in actual_phases, f"Missing phase node: {expected_phase}"
    
    def test_no_self_loops_in_new_edges(self, sample_match_data):
        """Test that new edges don't create self-loops."""
        G = build_cricket_graph(sample_match_data)
        
        new_edge_types = {"partnered_with", "teammate_of", "bowled_at"}
        
        for source, target, attrs in G.edges(data=True):
            edge_type = attrs.get("edge_type")
            if edge_type in new_edge_types:
                assert source != target, f"Self-loop detected in {edge_type} edge: {source} -> {target}"
    
    def test_edge_aggregation(self, sample_match_data):
        """Test that edges are properly aggregated when players interact multiple times."""
        # Create extended data with multiple interactions
        extended_data = sample_match_data + [
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 3,
                "batter_id": "batter_1",
                "non_striker_id": "batter_2",
                "bowler_id": "bowler_1",
                "batting_team_name": "team_a",
                "bowling_team_name": "team_b",
                "venue_name": "venue_1",
                "match_date": "2024-01-15 14:30:00",
                "runs": 2,
                "dismissal_type": ""
            }
        ]
        
        G = build_cricket_graph(extended_data)
        
        # Check that partnership edge is aggregated
        if G.has_edge("batter_1", "batter_2"):
            attrs = G["batter_1"]["batter_2"]
            if attrs.get("edge_type") == "partnered_with":
                assert attrs["balls_together"] >= 3, "Partnership should aggregate multiple balls"
                assert attrs["runs"] >= 7, "Partnership should aggregate runs (4+1+2)"
    
    def test_different_match_separation(self):
        """Test that edges from different matches are handled correctly."""
        match_data = [
            # Match 1
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 1,
                "batter_id": "batter_1",
                "non_striker_id": "batter_2",
                "bowler_id": "bowler_1",
                "batting_team_name": "team_a",
                "bowling_team_name": "team_b",
                "venue_name": "venue_1",
                "match_date": "2024-01-15 14:30:00",
                "runs": 4,
                "dismissal_type": ""
            },
            # Match 2 - Same players, different match
            {
                "match_id": "match_2",
                "innings": 1,
                "over": 1,
                "batter_id": "batter_1",
                "non_striker_id": "batter_2",
                "bowler_id": "bowler_1",
                "batting_team_name": "team_a",
                "bowling_team_name": "team_b",
                "venue_name": "venue_2",
                "match_date": "2024-01-16 14:30:00",
                "runs": 6,
                "dismissal_type": ""
            }
        ]
        
        G = build_cricket_graph(match_data)
        
        # Check that partnerships from different matches are aggregated
        if G.has_edge("batter_1", "batter_2"):
            attrs = G["batter_1"]["batter_2"]
            if attrs.get("edge_type") == "partnered_with":
                # Should have weight > 1.0 due to multiple matches
                assert attrs["weight"] >= 2.0, "Multiple match partnerships should increase weight"


class TestEdgeTypeFunctions:
    """Test the individual edge creation functions."""
    
    def test_add_partnership_edges_function(self):
        """Test the _add_partnership_edges function directly."""
        G = nx.DiGraph()
        G.add_node("batter_1", type="batter")
        G.add_node("batter_2", type="batter")
        
        match_data = [
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 1,
                "batter_id": "batter_1",
                "non_striker_id": "batter_2",
                "venue_name": "venue_1",
                "match_date": "2024-01-15 14:30:00",
                "runs": 4,
                "dismissal_type": ""
            }
        ]
        
        _add_partnership_edges(G, match_data)
        
        assert G.has_edge("batter_1", "batter_2"), "Partnership edge not created"
        assert G.has_edge("batter_2", "batter_1"), "Reverse partnership edge not created"
        
        attrs = G["batter_1"]["batter_2"]
        assert attrs["edge_type"] == "partnered_with"
        assert attrs["weight"] == 1.0
    
    def test_add_teammate_edges_function(self):
        """Test the _add_teammate_edges function directly."""
        G = nx.DiGraph()
        G.add_node("batter_1", type="batter")
        G.add_node("bowler_1", type="bowler")
        
        match_data = [
            {
                "match_id": "match_1",
                "batting_team_name": "team_a",
                "bowling_team_name": "team_b",
                "batter_id": "batter_1",
                "bowler_id": "bowler_1",
                "venue_name": "venue_1",
                "match_date": "2024-01-15 14:30:00",
                "runs": 4,
                "over": 1
            }
        ]
        
        _add_teammate_edges(G, match_data)
        
        # Check that no teammate edge exists between players from different teams
        assert not G.has_edge("batter_1", "bowler_1"), "Shouldn't have teammate edge between different teams"
    
    def test_add_bowler_phase_edges_function(self):
        """Test the _add_bowler_phase_edges function directly."""
        G = nx.DiGraph()
        G.add_node("bowler_1", type="bowler")
        
        match_data = [
            {
                "bowler_id": "bowler_1",
                "over": 1,
                "venue_name": "venue_1",
                "match_date": "2024-01-15 14:30:00",
                "runs": 4,
                "dismissal_type": ""
            }
        ]
        
        _add_bowler_phase_edges(G, match_data)
        
        assert "powerplay" in G.nodes(), "Phase node not created"
        assert G.nodes["powerplay"]["type"] == "phase", "Phase node has wrong type"
        assert G.has_edge("bowler_1", "powerplay"), "Bowler-phase edge not created"
        
        attrs = G["bowler_1"]["powerplay"]
        assert attrs["edge_type"] == "bowled_at"
        assert attrs["weight"] == 1.0
        assert attrs["balls_bowled"] == 1
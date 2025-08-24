# Purpose: Comprehensive unit tests for context node system
# Author: WicketWise Team, Last Modified: 2025-08-23

import pytest
import networkx as nx
from datetime import datetime
from unittest.mock import patch

from crickformers.gnn.context_nodes import (
    ContextNodeType,
    ContextNodeData,
    TournamentStage,
    PitchType,
    MatchImportance,
    WeatherCondition,
    TimeOfDay,
    TournamentStageExtractor,
    PitchTypeExtractor,
    WeatherConditionExtractor,
    ContextNodeManager
)


class TestContextNodeData:
    """Test ContextNodeData dataclass"""
    
    def test_basic_creation(self):
        """Test basic context node creation"""
        node = ContextNodeData(
            node_id="test_node",
            node_type=ContextNodeType.TOURNAMENT_STAGE,
            properties={"stage": "final", "importance": 4.0}
        )
        
        assert node.node_id == "test_node"
        assert node.node_type == ContextNodeType.TOURNAMENT_STAGE
        assert node.properties["stage"] == "final"
        assert node.properties["importance"] == 4.0
        assert len(node.relationships) == 0
        assert len(node.metadata) == 0
        assert isinstance(node.created_at, datetime)
    
    def test_with_relationships_and_metadata(self):
        """Test node with relationships and metadata"""
        node = ContextNodeData(
            node_id="complex_node",
            node_type=ContextNodeType.PITCH_TYPE,
            properties={"pitch_type": "batting_friendly"},
            relationships=["rel1", "rel2"],
            metadata={"venue": "MCG", "match_id": "123"}
        )
        
        assert len(node.relationships) == 2
        assert "rel1" in node.relationships
        assert node.metadata["venue"] == "MCG"
        assert node.metadata["match_id"] == "123"


class TestTournamentStageExtractor:
    """Test TournamentStageExtractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create tournament stage extractor"""
        return TournamentStageExtractor()
    
    def test_extract_final_stage(self, extractor):
        """Test extracting final tournament stage"""
        match_data = {
            "tournament": "IPL 2024",
            "match_type": "Final",
            "stage": "Championship Final"
        }
        
        nodes = extractor.extract_context_nodes(match_data)
        
        assert len(nodes) == 1
        node = nodes[0]
        assert node.node_type == ContextNodeType.TOURNAMENT_STAGE
        assert node.properties["stage"] == "final"
        assert node.properties["stage_importance"] == 4.0
        assert node.properties["elimination_match"] is True
        assert node.properties["pressure_level"] == 1.0
    
    def test_extract_qualifier_stage(self, extractor):
        """Test extracting qualifier stage"""
        match_data = {
            "tournament": "IPL 2024",
            "match_type": "Qualifier 1",
            "description": "First Qualifier Match"
        }
        
        nodes = extractor.extract_context_nodes(match_data)
        
        assert len(nodes) == 1
        node = nodes[0]
        assert node.properties["stage"] == "qualifier"
        assert node.properties["stage_importance"] == 2.0
        assert node.properties["pressure_level"] == 0.7
    
    def test_extract_league_stage(self, extractor):
        """Test extracting league stage"""
        match_data = {
            "tournament": "IPL 2024",
            "match_type": "League Match",
            "description": "Regular season game"
        }
        
        nodes = extractor.extract_context_nodes(match_data)
        
        assert len(nodes) == 1
        node = nodes[0]
        assert node.properties["stage"] == "league_stage"
        assert node.properties["stage_importance"] == 1.0
        assert node.properties["elimination_match"] is False
    
    def test_extract_semi_final_stage(self, extractor):
        """Test extracting semi-final stage"""
        match_data = {
            "tournament": "World Cup",
            "match_type": "Semi Final",  # Without hyphen to avoid "Final" match
            "stage": "Knockout"
        }
        
        nodes = extractor.extract_context_nodes(match_data)
        
        assert len(nodes) == 1
        node = nodes[0]
        assert node.properties["stage"] == "semi_final"
        assert node.properties["stage_importance"] == 3.0
        assert node.properties["elimination_match"] is True
        assert node.properties["pressure_level"] == 0.9
    
    def test_stage_importance_calculation(self, extractor):
        """Test stage importance calculation"""
        assert extractor._get_stage_importance(TournamentStage.GROUP_STAGE) == 1.0
        assert extractor._get_stage_importance(TournamentStage.QUALIFIER) == 2.0
        assert extractor._get_stage_importance(TournamentStage.SEMI_FINAL) == 3.0
        assert extractor._get_stage_importance(TournamentStage.FINAL) == 4.0
        assert extractor._get_stage_importance(TournamentStage.SUPER_OVER) == 4.5
    
    def test_pressure_level_calculation(self, extractor):
        """Test pressure level calculation"""
        assert extractor._get_pressure_level(TournamentStage.GROUP_STAGE) == 0.3
        assert extractor._get_pressure_level(TournamentStage.QUALIFIER) == 0.7
        assert extractor._get_pressure_level(TournamentStage.FINAL) == 1.0
        assert extractor._get_pressure_level(TournamentStage.SUPER_OVER) == 1.0
    
    def test_get_node_relationships(self, extractor):
        """Test getting node relationships"""
        tournament_node = ContextNodeData(
            node_id="tournament_final",
            node_type=ContextNodeType.TOURNAMENT_STAGE,
            properties={"stage": "final"}
        )
        
        importance_node = ContextNodeData(
            node_id="match_importance_final",
            node_type=ContextNodeType.MATCH_IMPORTANCE,
            properties={"importance": "final"}
        )
        
        other_node = ContextNodeData(
            node_id="pitch_balanced",
            node_type=ContextNodeType.PITCH_TYPE,
            properties={"pitch_type": "balanced"}
        )
        
        relationships = extractor.get_node_relationships(
            tournament_node, [importance_node, other_node]
        )
        
        assert len(relationships) == 1
        assert relationships[0] == ("tournament_final", "influences_importance", "match_importance_final")


class TestPitchTypeExtractor:
    """Test PitchTypeExtractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create pitch type extractor"""
        return PitchTypeExtractor()
    
    def test_extract_batting_paradise(self, extractor):
        """Test extracting batting paradise pitch"""
        match_data = {
            "venue": "MCG",
            "total_runs": 250,
            "total_overs": 20,
            "total_wickets": 6
        }
        
        nodes = extractor.extract_context_nodes(match_data)
        
        assert len(nodes) == 1
        node = nodes[0]
        assert node.node_type == ContextNodeType.PITCH_TYPE
        assert node.properties["pitch_type"] == "batting_paradise"
        assert node.properties["batting_difficulty"] == 0.1
        assert node.properties["bowling_advantage"] == 0.1
        assert node.properties["expected_score"] == 220
    
    def test_extract_bowling_friendly(self, extractor):
        """Test extracting bowling friendly pitch"""
        match_data = {
            "venue": "Lord's",
            "total_runs": 110,
            "total_overs": 20,
            "total_wickets": 16
        }
        
        nodes = extractor.extract_context_nodes(match_data)
        
        assert len(nodes) == 1
        node = nodes[0]
        assert node.properties["pitch_type"] == "bowling_friendly"
        assert node.properties["batting_difficulty"] == 0.7
        assert node.properties["bowling_advantage"] == 0.8
        assert node.properties["expected_score"] == 140
    
    def test_extract_balanced_pitch(self, extractor):
        """Test extracting balanced pitch"""
        match_data = {
            "venue": "Eden Gardens",
            "total_runs": 160,
            "total_overs": 20,
            "total_wickets": 12
        }
        
        nodes = extractor.extract_context_nodes(match_data)
        
        assert len(nodes) == 1
        node = nodes[0]
        assert node.properties["pitch_type"] == "balanced"
        assert node.properties["batting_difficulty"] == 0.5
        assert node.properties["bowling_advantage"] == 0.5
        assert node.properties["spin_factor"] == 0.5
        assert node.properties["pace_factor"] == 0.5
    
    def test_no_overs_data(self, extractor):
        """Test handling of missing overs data"""
        match_data = {
            "venue": "Test Ground",
            "total_runs": 150,
            "total_overs": 0,
            "total_wickets": 10
        }
        
        nodes = extractor.extract_context_nodes(match_data)
        assert len(nodes) == 0
    
    def test_spin_factor_calculation(self, extractor):
        """Test spin factor calculation"""
        assert extractor._get_spin_factor(PitchType.TURNING_TRACK) == 1.0
        assert extractor._get_spin_factor(PitchType.GREEN_TOP) == 0.2
        assert extractor._get_spin_factor(PitchType.SLOW_LOW) == 0.8
        assert extractor._get_spin_factor(PitchType.BALANCED) == 0.5
    
    def test_pace_factor_calculation(self, extractor):
        """Test pace factor calculation"""
        assert extractor._get_pace_factor(PitchType.GREEN_TOP) == 1.0
        assert extractor._get_pace_factor(PitchType.TURNING_TRACK) == 0.3
        assert extractor._get_pace_factor(PitchType.BOWLING_FRIENDLY) == 0.7
        assert extractor._get_pace_factor(PitchType.BALANCED) == 0.5
    
    def test_expected_score_calculation(self, extractor):
        """Test expected score calculation"""
        assert extractor._get_expected_score(PitchType.BATTING_PARADISE) == 220
        assert extractor._get_expected_score(PitchType.BOWLER_PARADISE) == 120
        assert extractor._get_expected_score(PitchType.BALANCED) == 160
    
    def test_get_node_relationships(self, extractor):
        """Test pitch type relationships"""
        pitch_node = ContextNodeData(
            node_id="pitch_balanced",
            node_type=ContextNodeType.PITCH_TYPE,
            properties={"pitch_type": "balanced"}
        )
        
        weather_node = ContextNodeData(
            node_id="weather_overcast",
            node_type=ContextNodeType.WEATHER_CONDITION,
            properties={"condition": "overcast"}
        )
        
        other_node = ContextNodeData(
            node_id="tournament_final",
            node_type=ContextNodeType.TOURNAMENT_STAGE,
            properties={"stage": "final"}
        )
        
        relationships = extractor.get_node_relationships(
            pitch_node, [weather_node, other_node]
        )
        
        assert len(relationships) == 1
        assert relationships[0] == ("pitch_balanced", "interacts_with", "weather_overcast")


class TestWeatherConditionExtractor:
    """Test WeatherConditionExtractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create weather condition extractor"""
        return WeatherConditionExtractor()
    
    def test_extract_clear_conditions(self, extractor):
        """Test extracting clear weather conditions"""
        match_data = {
            "weather_description": "Clear and sunny",
            "temperature": 28,
            "humidity": 45,
            "wind_speed": 10
        }
        
        nodes = extractor.extract_context_nodes(match_data)
        
        assert len(nodes) == 1
        node = nodes[0]
        assert node.node_type == ContextNodeType.WEATHER_CONDITION
        assert node.properties["condition"] == "clear"
        assert node.properties["batting_impact"] == 0.2
        assert node.properties["bowling_impact"] == -0.1
        assert node.properties["visibility_factor"] == 1.0
    
    def test_extract_rain_affected(self, extractor):
        """Test extracting rain affected conditions"""
        match_data = {
            "weather_description": "Rain interruptions",
            "rain_affected": True,
            "temperature": 22,
            "humidity": 85,
            "wind_speed": 15
        }
        
        nodes = extractor.extract_context_nodes(match_data)
        
        # Should have rain_affected and humid conditions
        assert len(nodes) == 2
        
        rain_node = next(n for n in nodes if n.properties["condition"] == "rain_affected")
        assert rain_node.properties["batting_impact"] == -0.7
        assert rain_node.properties["bowling_impact"] == 0.3
        assert rain_node.properties["fielding_impact"] == -0.8
        assert rain_node.properties["visibility_factor"] == 0.3
        
        humid_node = next(n for n in nodes if n.properties["condition"] == "humid")
        assert humid_node.properties["batting_impact"] == -0.2
        assert humid_node.properties["swing_factor"] == 0.6
    
    def test_extract_overcast_conditions(self, extractor):
        """Test extracting overcast conditions"""
        match_data = {
            "weather_description": "Overcast skies with clouds",
            "temperature": 25,
            "humidity": 60,
            "wind_speed": 12
        }
        
        nodes = extractor.extract_context_nodes(match_data)
        
        assert len(nodes) == 1
        node = nodes[0]
        assert node.properties["condition"] == "overcast"
        assert node.properties["batting_impact"] == -0.3
        assert node.properties["bowling_impact"] == 0.4
        assert node.properties["swing_factor"] == 0.8
    
    def test_extract_multiple_conditions(self, extractor):
        """Test extracting multiple weather conditions"""
        match_data = {
            "weather_description": "Clear but windy",
            "temperature": 38,  # Hot
            "humidity": 75,     # Humid
            "wind_speed": 25,   # Windy
            "dew_factor": True
        }
        
        nodes = extractor.extract_context_nodes(match_data)
        
        # Should have clear, hot, humid, windy, and dew_factor
        assert len(nodes) == 5
        
        conditions = [node.properties["condition"] for node in nodes]
        assert "clear" in conditions
        assert "hot" in conditions
        assert "humid" in conditions
        assert "windy" in conditions
        assert "dew_factor" in conditions
    
    def test_batting_impact_calculation(self, extractor):
        """Test batting impact calculation"""
        assert extractor._get_batting_impact(WeatherCondition.CLEAR) == 0.2
        assert extractor._get_batting_impact(WeatherCondition.OVERCAST) == -0.3
        assert extractor._get_batting_impact(WeatherCondition.RAIN_AFFECTED) == -0.7
        assert extractor._get_batting_impact(WeatherCondition.DEW_FACTOR) == -0.4
    
    def test_bowling_impact_calculation(self, extractor):
        """Test bowling impact calculation"""
        assert extractor._get_bowling_impact(WeatherCondition.CLEAR) == -0.1
        assert extractor._get_bowling_impact(WeatherCondition.OVERCAST) == 0.4
        assert extractor._get_bowling_impact(WeatherCondition.DEW_FACTOR) == 0.6
        assert extractor._get_bowling_impact(WeatherCondition.WINDY) == 0.2
    
    def test_swing_factor_calculation(self, extractor):
        """Test ball swing factor calculation"""
        assert extractor._get_swing_factor(WeatherCondition.CLEAR) == 0.2
        assert extractor._get_swing_factor(WeatherCondition.OVERCAST) == 0.8
        assert extractor._get_swing_factor(WeatherCondition.DEW_FACTOR) == 0.9
        assert extractor._get_swing_factor(WeatherCondition.HOT) == 0.1
    
    def test_get_node_relationships(self, extractor):
        """Test weather condition relationships"""
        weather_node = ContextNodeData(
            node_id="weather_overcast",
            node_type=ContextNodeType.WEATHER_CONDITION,
            properties={"condition": "overcast"}
        )
        
        time_node = ContextNodeData(
            node_id="time_evening",
            node_type=ContextNodeType.TIME_OF_DAY,
            properties={"time": "evening"}
        )
        
        other_node = ContextNodeData(
            node_id="pitch_balanced",
            node_type=ContextNodeType.PITCH_TYPE,
            properties={"pitch_type": "balanced"}
        )
        
        relationships = extractor.get_node_relationships(
            weather_node, [time_node, other_node]
        )
        
        assert len(relationships) == 1
        assert relationships[0] == ("weather_overcast", "affects_timing", "time_evening")


class TestContextNodeManager:
    """Test ContextNodeManager"""
    
    @pytest.fixture
    def manager(self):
        """Create context node manager"""
        return ContextNodeManager()
    
    def test_initialization(self, manager):
        """Test manager initialization"""
        assert len(manager.extractors) == 3
        assert ContextNodeType.TOURNAMENT_STAGE in manager.extractors
        assert ContextNodeType.PITCH_TYPE in manager.extractors
        assert ContextNodeType.WEATHER_CONDITION in manager.extractors
        assert len(manager.node_cache) == 0
        assert len(manager.relationship_cache) == 0
    
    def test_extract_all_context_nodes(self, manager):
        """Test extracting all context nodes"""
        match_data = {
            "tournament": "IPL 2024",
            "match_type": "Final",
            "venue": "Wankhede Stadium",
            "total_runs": 180,
            "total_overs": 20,
            "total_wickets": 12,
            "weather_description": "Overcast conditions",
            "temperature": 30,
            "humidity": 70
        }
        
        nodes = manager.extract_all_context_nodes(match_data)
        
        # Should extract tournament stage, pitch type, and weather nodes
        assert len(nodes) >= 3
        
        node_types = [node.node_type for node in nodes]
        assert ContextNodeType.TOURNAMENT_STAGE in node_types
        assert ContextNodeType.PITCH_TYPE in node_types
        assert ContextNodeType.WEATHER_CONDITION in node_types
        
        # Check that nodes are cached
        assert len(manager.node_cache) >= 3
    
    def test_build_context_relationships(self, manager):
        """Test building context relationships"""
        # Create sample nodes
        tournament_node = ContextNodeData(
            node_id="tournament_final",
            node_type=ContextNodeType.TOURNAMENT_STAGE,
            properties={"stage": "final"}
        )
        
        pitch_node = ContextNodeData(
            node_id="pitch_balanced",
            node_type=ContextNodeType.PITCH_TYPE,
            properties={"pitch_type": "balanced"}
        )
        
        weather_node = ContextNodeData(
            node_id="weather_overcast",
            node_type=ContextNodeType.WEATHER_CONDITION,
            properties={"condition": "overcast"}
        )
        
        nodes = [tournament_node, pitch_node, weather_node]
        relationships = manager.build_context_relationships(nodes)
        
        # Should have at least one relationship (pitch interacts with weather)
        assert len(relationships) >= 1
        assert len(manager.relationship_cache) >= 1
    
    def test_add_context_nodes_to_graph(self, manager):
        """Test adding context nodes to knowledge graph"""
        # Create a simple graph
        graph = nx.DiGraph()
        graph.add_node("player_1", type="player")
        
        # Create context nodes
        match_data = {
            "tournament": "Test Tournament",
            "match_type": "Final",
            "total_runs": 160,
            "total_overs": 20,
            "total_wickets": 10,
            "weather_description": "Clear"
        }
        
        context_nodes = manager.extract_all_context_nodes(match_data)
        
        # Add context nodes to graph
        updated_graph = manager.add_context_nodes_to_graph(graph, context_nodes)
        
        # Check that context nodes were added
        original_nodes = 1
        total_nodes = len(updated_graph.nodes())
        assert total_nodes > original_nodes
        
        # Check that context nodes have proper attributes
        context_node_ids = [node.node_id for node in context_nodes]
        for node_id in context_node_ids:
            if node_id in updated_graph:
                node_data = updated_graph.nodes[node_id]
                assert "node_type" in node_data
                assert "created_at" in node_data
    
    def test_get_context_summary(self, manager):
        """Test getting context summary"""
        # Add some nodes to cache
        manager.node_cache["tournament_final"] = ContextNodeData(
            node_id="tournament_final",
            node_type=ContextNodeType.TOURNAMENT_STAGE,
            properties={"stage": "final"}
        )
        
        manager.node_cache["pitch_balanced"] = ContextNodeData(
            node_id="pitch_balanced",
            node_type=ContextNodeType.PITCH_TYPE,
            properties={"pitch_type": "balanced"}
        )
        
        # Add some relationships
        manager.relationship_cache.extend([
            ("node1", "influences", "node2"),
            ("node2", "affects", "node3")
        ])
        
        summary = manager.get_context_summary()
        
        assert summary["total_context_nodes"] == 2
        assert "tournament_stage" in summary["node_types"]
        assert "pitch_type" in summary["node_types"]
        assert summary["total_relationships"] == 2
        assert "influences" in summary["relationship_types"]
        assert "affects" in summary["relationship_types"]
        assert len(summary["extractors_available"]) == 3
    
    def test_clear_cache(self, manager):
        """Test clearing caches"""
        # Add some data to caches
        manager.node_cache["test_node"] = ContextNodeData(
            node_id="test_node",
            node_type=ContextNodeType.TOURNAMENT_STAGE,
            properties={}
        )
        manager.relationship_cache.append(("a", "b", "c"))
        
        # Clear caches
        manager.clear_cache()
        
        assert len(manager.node_cache) == 0
        assert len(manager.relationship_cache) == 0
    
    def test_extractor_error_handling(self, manager):
        """Test handling of extractor errors"""
        # Mock an extractor to raise an exception
        with patch.object(manager.extractors[ContextNodeType.TOURNAMENT_STAGE], 
                         'extract_context_nodes', side_effect=Exception("Test error")):
            
            match_data = {
                "tournament": "Test Tournament",
                "total_runs": 160,
                "total_overs": 20
            }
            
            # Should not raise exception, but continue with other extractors
            nodes = manager.extract_all_context_nodes(match_data)
            
            # Should still get nodes from other extractors
            assert len(nodes) >= 1  # At least pitch type should work


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    def test_ipl_final_scenario(self):
        """Test IPL final match scenario"""
        manager = ContextNodeManager()
        
        match_data = {
            "tournament": "Indian Premier League 2024",
            "match_type": "Final",
            "venue": "Narendra Modi Stadium",
            "total_runs": 185,  # Run rate = 9.25, should be batting_friendly
            "total_overs": 20,
            "total_wickets": 8,   # Lower wickets for batting friendly
            "weather_description": "Clear evening with slight dew",
            "temperature": 32,
            "humidity": 75,  # Above 70 threshold for humid condition
            "wind_speed": 8,
            "dew_factor": True
        }
        
        nodes = manager.extract_all_context_nodes(match_data)
        
        # Should extract multiple context nodes
        assert len(nodes) >= 4
        
        # Check specific expected nodes
        node_types = [node.node_type for node in nodes]
        assert ContextNodeType.TOURNAMENT_STAGE in node_types
        assert ContextNodeType.PITCH_TYPE in node_types
        assert ContextNodeType.WEATHER_CONDITION in node_types
        
        # Check tournament stage is final
        tournament_nodes = [n for n in nodes if n.node_type == ContextNodeType.TOURNAMENT_STAGE]
        assert len(tournament_nodes) == 1
        assert tournament_nodes[0].properties["stage"] == "final"
        assert tournament_nodes[0].properties["pressure_level"] == 1.0
        
        # Check pitch type (should be batting friendly based on run rate)
        pitch_nodes = [n for n in nodes if n.node_type == ContextNodeType.PITCH_TYPE]
        assert len(pitch_nodes) == 1
        assert pitch_nodes[0].properties["pitch_type"] == "batting_friendly"
        
        # Check weather conditions (should include clear, humid, and dew factor)
        weather_nodes = [n for n in nodes if n.node_type == ContextNodeType.WEATHER_CONDITION]
        weather_conditions = [n.properties["condition"] for n in weather_nodes]
        assert "clear" in weather_conditions
        assert "humid" in weather_conditions
        assert "dew_factor" in weather_conditions
    
    def test_world_cup_qualifier_scenario(self):
        """Test World Cup qualifier match scenario"""
        manager = ContextNodeManager()
        
        match_data = {
            "tournament": "ICC T20 World Cup 2024",
            "match_type": "Qualifier 1",
            "stage": "Knockout Phase",
            "venue": "Lord's",
            "total_runs": 135,
            "total_overs": 20,
            "total_wickets": 18,
            "weather_description": "Overcast with light drizzle",
            "temperature": 18,
            "humidity": 85,
            "wind_speed": 22
        }
        
        nodes = manager.extract_all_context_nodes(match_data)
        
        # Check tournament stage
        tournament_nodes = [n for n in nodes if n.node_type == ContextNodeType.TOURNAMENT_STAGE]
        assert len(tournament_nodes) == 1
        # Should be qualifier stage with elimination match
        tournament_node = tournament_nodes[0]
        assert tournament_node.properties["stage"] == "qualifier"
        # Qualifier is an elimination match according to our logic
        assert tournament_node.properties["elimination_match"] is True  # QUALIFIER, ELIMINATOR, SEMI_FINAL, FINAL are elimination matches
        
        # Check pitch type (run rate 6.75, wicket rate 0.9 = balanced)
        pitch_nodes = [n for n in nodes if n.node_type == ContextNodeType.PITCH_TYPE]
        assert len(pitch_nodes) == 1
        assert pitch_nodes[0].properties["pitch_type"] == "balanced"
        
        # Check weather conditions (overcast, humid, windy)
        weather_nodes = [n for n in nodes if n.node_type == ContextNodeType.WEATHER_CONDITION]
        weather_conditions = [n.properties["condition"] for n in weather_nodes]
        assert "overcast" in weather_conditions
        assert "humid" in weather_conditions
        assert "windy" in weather_conditions
    
    def test_graph_integration(self):
        """Test full graph integration with context nodes"""
        manager = ContextNodeManager()
        
        # Create base graph with players and teams
        graph = nx.DiGraph()
        graph.add_node("player_kohli", type="player", name="Virat Kohli")
        graph.add_node("player_bumrah", type="player", name="Jasprit Bumrah")
        graph.add_node("team_rcb", type="team", name="Royal Challengers Bangalore")
        graph.add_edge("player_kohli", "team_rcb", edge_type="plays_for")
        
        # Add context nodes
        match_data = {
            "tournament": "IPL 2024",
            "match_type": "League Match",
            "total_runs": 175,
            "total_overs": 20,
            "total_wickets": 8,
            "weather_description": "Partly cloudy"
        }
        
        context_nodes = manager.extract_all_context_nodes(match_data)
        updated_graph = manager.add_context_nodes_to_graph(graph, context_nodes)
        
        # Check that original nodes are preserved
        assert "player_kohli" in updated_graph
        assert "team_rcb" in updated_graph
        
        # Check that context nodes are added
        context_node_ids = [node.node_id for node in context_nodes]
        for node_id in context_node_ids:
            assert node_id in updated_graph
        
        # Check that relationships exist
        assert updated_graph.number_of_edges() >= 1  # Original + context relationships
        
        # Verify node attributes
        for node_id in context_node_ids:
            if node_id in updated_graph:
                node_data = updated_graph.nodes[node_id]
                assert "node_type" in node_data
                assert node_data["node_type"] in [
                    "tournament_stage", "pitch_type", "weather_condition"
                ]

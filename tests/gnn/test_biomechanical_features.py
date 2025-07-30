# Purpose: Unit tests for biomechanical feature processing
# Author: Shamus Rae, Last Modified: 2024-01-15

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import os
from collections import deque
from unittest.mock import patch, mock_open
from datetime import datetime, timedelta
import networkx as nx

from crickformers.gnn.biomechanical_features import (
    BiomechanicalSignalSchema,
    BiomechanicalConfig,
    BiomechanicalSignalLoader,
    BiomechanicalAggregator,
    BiomechanicalEventMetadata,
    process_match_biomechanical_data,
    add_biomechanical_features_to_graph,
    create_biomechanical_event_nodes,
    get_biomechanical_events_for_delivery,
    get_biomechanical_events_by_match,
    analyze_biomechanical_event_patterns,
    get_biomechanical_feature_dimension,
    create_sample_biomechanical_data
)


class TestBiomechanicalSignalSchema:
    """Test biomechanical signal schema definitions."""
    
    def test_schema_signal_lists(self):
        """Test that signal lists are properly defined."""
        schema = BiomechanicalSignalSchema()
        
        # Check batter signals
        assert len(schema.BATTER_SIGNALS) == 4
        assert 'head_stability' in schema.BATTER_SIGNALS
        assert 'backlift_type' in schema.BATTER_SIGNALS
        assert 'footwork_direction' in schema.BATTER_SIGNALS
        assert 'shot_commitment' in schema.BATTER_SIGNALS
        
        # Check bowler signals
        assert len(schema.BOWLER_SIGNALS) == 3
        assert 'release_point_consistency' in schema.BOWLER_SIGNALS
        assert 'arm_path' in schema.BOWLER_SIGNALS
        assert 'follow_through_momentum' in schema.BOWLER_SIGNALS
        
        # Check fielder signals
        assert len(schema.FIELDER_SIGNALS) == 3
        assert 'closing_speed' in schema.FIELDER_SIGNALS
        assert 'reaction_time' in schema.FIELDER_SIGNALS
        assert 'interception_type' in schema.FIELDER_SIGNALS
    
    def test_get_all_signals(self):
        """Test getting all signal names."""
        all_signals = BiomechanicalSignalSchema.get_all_signals()
        assert len(all_signals) == 10  # 4 + 3 + 3
        assert 'head_stability' in all_signals
        assert 'release_point_consistency' in all_signals
        assert 'closing_speed' in all_signals
    
    def test_get_signals_for_role(self):
        """Test getting signals for specific roles."""
        schema = BiomechanicalSignalSchema()
        
        batter_signals = schema.get_signals_for_role('batter')
        assert batter_signals == schema.BATTER_SIGNALS
        
        bowler_signals = schema.get_signals_for_role('BOWLER')  # Test case insensitive
        assert bowler_signals == schema.BOWLER_SIGNALS
        
        fielder_signals = schema.get_signals_for_role('fielder')
        assert fielder_signals == schema.FIELDER_SIGNALS
        
        # Test unknown role
        unknown_signals = schema.get_signals_for_role('unknown')
        assert unknown_signals == []


class TestBiomechanicalConfig:
    """Test biomechanical configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BiomechanicalConfig()
        
        assert config.rolling_window == 100
        assert config.min_deliveries_required == 10
        assert config.missing_value_threshold == 0.3
        assert config.default_signal_value == 0.5
        assert config.feature_prefix == "biomech_"
        
        # Check signal ranges
        assert len(config.signal_ranges) == 10
        for signal, (min_val, max_val) in config.signal_ranges.items():
            assert min_val == 0.0
            assert max_val == 1.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BiomechanicalConfig(
            rolling_window=50,
            min_deliveries_required=5,
            missing_value_threshold=0.2,
            default_signal_value=0.6,
            feature_prefix="bio_"
        )
        
        assert config.rolling_window == 50
        assert config.min_deliveries_required == 5
        assert config.missing_value_threshold == 0.2
        assert config.default_signal_value == 0.6
        assert config.feature_prefix == "bio_"


class TestBiomechanicalSignalLoader:
    """Test biomechanical signal loading and validation."""
    
    @pytest.fixture
    def sample_signal_data(self):
        """Sample biomechanical signal data for testing."""
        return {
            "match1_1_1_1": {
                "head_stability": 0.8,
                "backlift_type": 0.7,
                "footwork_direction": 0.6,
                "shot_commitment": 0.9,
                "release_point_consistency": 0.85,
                "arm_path": 0.75,
                "follow_through_momentum": 0.8,
                "closing_speed": 0.7,
                "reaction_time": 0.9,
                "interception_type": 0.6
            },
            "match1_1_1_2": {
                "head_stability": 0.7,
                "backlift_type": 0.8,
                "release_point_consistency": 0.9
            }
        }
    
    def test_load_from_dict(self, sample_signal_data):
        """Test loading signals from dictionary."""
        loader = BiomechanicalSignalLoader()
        validated_data = loader.load_from_dict(sample_signal_data)
        
        assert len(validated_data) == 2
        assert "match1_1_1_1" in validated_data
        assert "match1_1_1_2" in validated_data
        
        # Check first delivery has all signals
        delivery1 = validated_data["match1_1_1_1"]
        assert len(delivery1) == 10
        assert delivery1["head_stability"] == 0.8
        
        # Check second delivery has missing signals filled with defaults
        delivery2 = validated_data["match1_1_1_2"]
        assert len(delivery2) == 10
        assert delivery2["head_stability"] == 0.7
        assert delivery2["shot_commitment"] == 0.5  # Default value
    
    def test_load_from_json_file(self, sample_signal_data):
        """Test loading signals from JSON file."""
        loader = BiomechanicalSignalLoader()
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_signal_data, f)
            temp_path = f.name
        
        try:
            validated_data = loader.load_from_json(temp_path)
            assert len(validated_data) == 2
            assert "match1_1_1_1" in validated_data
        finally:
            os.unlink(temp_path)
    
    def test_load_from_nonexistent_file(self):
        """Test loading from nonexistent file."""
        loader = BiomechanicalSignalLoader()
        result = loader.load_from_json("nonexistent_file.json")
        assert result == {}
    
    def test_signal_validation_and_clamping(self):
        """Test signal value validation and clamping."""
        loader = BiomechanicalSignalLoader()
        
        # Test data with out-of-range values
        test_data = {
            "delivery1": {
                "head_stability": 1.5,    # Too high
                "backlift_type": -0.2,    # Too low
                "shot_commitment": 0.7,   # Valid
                "invalid_signal": "text"  # Invalid type
            }
        }
        
        validated_data = loader.load_from_dict(test_data)
        delivery = validated_data["delivery1"]
        
        # Check clamping
        assert delivery["head_stability"] == 1.0  # Clamped to max
        assert delivery["backlift_type"] == 0.0   # Clamped to min
        assert delivery["shot_commitment"] == 0.7  # Unchanged
        
        # Check that all signals are present (missing ones get defaults)
        assert len(delivery) == 10
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        loader = BiomechanicalSignalLoader()
        
        result = loader.load_from_dict({})
        assert result == {}
        
        result = loader.load_from_dict({"delivery1": {}})
        assert len(result["delivery1"]) == 10  # All default values


class TestBiomechanicalAggregator:
    """Test biomechanical signal aggregation."""
    
    @pytest.fixture
    def aggregator(self):
        """Create aggregator with test configuration."""
        config = BiomechanicalConfig(rolling_window=20, min_deliveries_required=5)
        return BiomechanicalAggregator(config)
    
    @pytest.fixture
    def sample_delivery_signals(self):
        """Sample delivery signals for testing."""
        return {
            "head_stability": 0.8,
            "backlift_type": 0.7,
            "shot_commitment": 0.9,
            "release_point_consistency": 0.85
        }
    
    def test_add_delivery_signals(self, aggregator, sample_delivery_signals):
        """Test adding delivery signals for a player."""
        player_id = "player1"
        
        # Add multiple deliveries
        for i in range(10):
            signals = {k: v + i * 0.01 for k, v in sample_delivery_signals.items()}
            aggregator.add_delivery_signals(player_id, signals, "batter")
        
        # Check that signals were stored
        assert player_id in aggregator.player_signal_history
        player_signals = aggregator.player_signal_history[player_id]
        
        assert len(player_signals["head_stability"]) == 10
        assert len(player_signals["backlift_type"]) == 10
        
        # Check values are correct
        assert player_signals["head_stability"][0] == 0.8
        assert player_signals["head_stability"][-1] == 0.89  # 0.8 + 9 * 0.01
    
    def test_role_specific_signals(self, aggregator):
        """Test that only role-specific signals are added."""
        player_id = "bowler1"
        
        # Add signals with bowler role
        all_signals = {
            "head_stability": 0.8,          # Batter signal
            "release_point_consistency": 0.9, # Bowler signal
            "closing_speed": 0.7            # Fielder signal
        }
        
        aggregator.add_delivery_signals(player_id, all_signals, "bowler")
        
        player_signals = aggregator.player_signal_history[player_id]
        
        # Should only have bowler signals
        assert "release_point_consistency" in player_signals
        assert "arm_path" in player_signals  # Even if not provided (gets default)
        assert "head_stability" not in player_signals
        assert "closing_speed" not in player_signals
    
    def test_get_player_aggregated_features(self, aggregator):
        """Test getting aggregated features for a player."""
        player_id = "player1"
        
        # Add enough deliveries for aggregation
        base_values = {"head_stability": 0.8, "shot_commitment": 0.7}
        
        for i in range(15):
            # Add some variation
            signals = {
                "head_stability": 0.8 + np.random.normal(0, 0.1),
                "shot_commitment": 0.7 + np.random.normal(0, 0.05)
            }
            aggregator.add_delivery_signals(player_id, signals, "batter")
        
        features = aggregator.get_player_aggregated_features(player_id)
        
        # Check that aggregated features are generated
        assert "biomech_head_stability_mean" in features
        assert "biomech_head_stability_std" in features
        assert "biomech_head_stability_recent" in features
        assert "biomech_head_stability_trend" in features
        
        assert "biomech_shot_commitment_mean" in features
        assert "biomech_shot_commitment_std" in features
        
        # Check values are reasonable
        assert 0.7 <= features["biomech_head_stability_mean"] <= 0.9
        assert features["biomech_head_stability_std"] >= 0
    
    def test_insufficient_data_handling(self, aggregator):
        """Test handling of insufficient data for aggregation."""
        player_id = "player1"
        
        # Add only a few deliveries (less than min_deliveries_required)
        for i in range(3):
            signals = {"head_stability": 0.8}
            aggregator.add_delivery_signals(player_id, signals, "batter")
        
        features = aggregator.get_player_aggregated_features(player_id)
        
        # Should return empty features due to insufficient data
        assert features == {}
    
    def test_missing_value_threshold(self):
        """Test missing value threshold handling."""
        config = BiomechanicalConfig(
            rolling_window=20,
            min_deliveries_required=5,
            missing_value_threshold=0.3  # Allow up to 30% missing
        )
        aggregator = BiomechanicalAggregator(config)
        
        player_id = "player1"
        
        # Add deliveries with some missing values (NaN)
        for i in range(10):
            if i < 7:  # 70% valid data (within threshold)
                signals = {"head_stability": 0.8 + i * 0.01}
            else:
                signals = {"head_stability": float('nan')}
            
            aggregator.add_delivery_signals(player_id, signals, "batter")
        
        features = aggregator.get_player_aggregated_features(player_id)
        
        # Should generate features despite some missing values
        assert "biomech_head_stability_mean" in features
        assert not np.isnan(features["biomech_head_stability_mean"])
    
    def test_excessive_missing_values(self):
        """Test handling of excessive missing values."""
        config = BiomechanicalConfig(
            rolling_window=20,
            min_deliveries_required=5,
            missing_value_threshold=0.3
        )
        aggregator = BiomechanicalAggregator(config)
        
        player_id = "player1"
        
        # Add deliveries with too many missing values
        for i in range(10):
            if i < 3:  # Only 30% valid data (below threshold)
                signals = {"head_stability": 0.8}
            else:
                signals = {"head_stability": float('nan')}
            
            aggregator.add_delivery_signals(player_id, signals, "batter")
        
        features = aggregator.get_player_aggregated_features(player_id)
        
        # Should not generate features due to excessive missing values
        assert "biomech_head_stability_mean" not in features
    
    def test_trend_calculation(self, aggregator):
        """Test trend calculation in aggregated features."""
        player_id = "player1"
        
        # Add deliveries with clear upward trend
        for i in range(25):  # Need enough for trend calculation
            signals = {"head_stability": 0.5 + i * 0.01}  # Increasing trend
            aggregator.add_delivery_signals(player_id, signals, "batter")
        
        features = aggregator.get_player_aggregated_features(player_id)
        
        # Should detect positive trend
        assert "biomech_head_stability_trend" in features
        trend = features["biomech_head_stability_trend"]
        assert trend > 0  # Positive trend
    
    def test_get_all_players_features(self, aggregator):
        """Test getting features for all players."""
        # Add data for multiple players
        for player_num in range(3):
            player_id = f"player{player_num}"
            
            for i in range(10):
                signals = {"head_stability": 0.7 + player_num * 0.1}
                aggregator.add_delivery_signals(player_id, signals, "batter")
        
        all_features = aggregator.get_all_players_features()
        
        assert len(all_features) == 3
        assert "player0" in all_features
        assert "player1" in all_features
        assert "player2" in all_features
        
        # Check that different players have different feature values
        player0_mean = all_features["player0"]["biomech_head_stability_mean"]
        player1_mean = all_features["player1"]["biomech_head_stability_mean"]
        assert abs(player0_mean - player1_mean) > 0.05
    
    def test_feature_names_generation(self, aggregator):
        """Test generation of feature names."""
        feature_names = aggregator.get_feature_names()
        
        # Should have 4 features per signal (mean, std, recent, trend)
        expected_count = len(BiomechanicalSignalSchema.get_all_signals()) * 4
        assert len(feature_names) == expected_count
        
        # Check specific feature names
        assert "biomech_head_stability_mean" in feature_names
        assert "biomech_head_stability_std" in feature_names
        assert "biomech_head_stability_recent" in feature_names
        assert "biomech_head_stability_trend" in feature_names
    
    def test_rolling_window_limit(self):
        """Test that rolling window limits are respected."""
        config = BiomechanicalConfig(rolling_window=5)
        aggregator = BiomechanicalAggregator(config)
        
        player_id = "player1"
        
        # Add more deliveries than window size
        for i in range(10):
            signals = {"head_stability": i * 0.1}
            aggregator.add_delivery_signals(player_id, signals, "batter")
        
        # Check that only last 5 values are kept
        player_signals = aggregator.player_signal_history[player_id]
        head_stability_values = list(player_signals["head_stability"])
        
        assert len(head_stability_values) == 5
        assert head_stability_values[0] == 0.5  # i=5, 5*0.1
        assert head_stability_values[-1] == 0.9  # i=9, 9*0.1
    
    def test_player_delivery_count(self, aggregator):
        """Test getting delivery count for a player."""
        player_id = "player1"
        
        # Initially no deliveries
        assert aggregator.get_player_delivery_count(player_id) == 0
        
        # Add some deliveries
        for i in range(7):
            signals = {"head_stability": 0.8}
            aggregator.add_delivery_signals(player_id, signals, "batter")
        
        assert aggregator.get_player_delivery_count(player_id) == 7
    
    def test_reset_player_history(self, aggregator):
        """Test resetting player history."""
        player_id = "player1"
        
        # Add some data
        for i in range(5):
            signals = {"head_stability": 0.8}
            aggregator.add_delivery_signals(player_id, signals, "batter")
        
        assert aggregator.get_player_delivery_count(player_id) == 5
        
        # Reset and check
        aggregator.reset_player_history(player_id)
        assert aggregator.get_player_delivery_count(player_id) == 0


class TestMatchBiomechanicalProcessing:
    """Test match-level biomechanical data processing."""
    
    @pytest.fixture
    def sample_match_data(self):
        """Sample match data for testing."""
        return pd.DataFrame([
            {
                'match_id': 'test_match',
                'innings': 1,
                'over': 1,
                'ball': 1,
                'batter': 'player1',
                'bowler': 'player2',
                'fielder': 'player3',
                'runs_scored': 4,
                'wicket_type': None
            },
            {
                'match_id': 'test_match',
                'innings': 1,
                'over': 1,
                'ball': 2,
                'batter': 'player1',
                'bowler': 'player2',
                'fielder': None,
                'runs_scored': 1,
                'wicket_type': None
            },
            {
                'match_id': 'test_match',
                'innings': 1,
                'over': 1,
                'ball': 3,
                'batter': 'player4',
                'bowler': 'player2',
                'fielder': 'player3',
                'runs_scored': 0,
                'wicket_type': 'bowled'
            }
        ])
    
    @pytest.fixture
    def sample_biomech_data(self):
        """Sample biomechanical data for testing."""
        return {
            "test_match_1_1_1": {
                "head_stability": 0.9,
                "shot_commitment": 0.95,
                "release_point_consistency": 0.8,
                "closing_speed": 0.7
            },
            "test_match_1_1_2": {
                "head_stability": 0.7,
                "shot_commitment": 0.8,
                "release_point_consistency": 0.85
            },
            "test_match_1_1_3": {
                "head_stability": 0.5,
                "shot_commitment": 0.4,
                "release_point_consistency": 0.95,
                "closing_speed": 0.8
            }
        }
    
    def test_process_match_biomechanical_data(self, sample_match_data, sample_biomech_data):
        """Test processing biomechanical data for a match."""
        config = BiomechanicalConfig(min_deliveries_required=1)  # Lower threshold for testing
        
        result = process_match_biomechanical_data(
            sample_match_data,
            sample_biomech_data,
            config
        )
        
        # Should have features for all players
        assert 'player1' in result  # Batter in 2 deliveries
        assert 'player2' in result  # Bowler in 3 deliveries
        assert 'player3' in result  # Fielder in 2 deliveries
        assert 'player4' in result  # Batter in 1 delivery
        
        # Check that features are generated
        player1_features = result['player1']
        assert len(player1_features) > 0
        assert any('biomech_head_stability' in key for key in player1_features.keys())
    
    def test_process_match_missing_biomech_data(self, sample_match_data):
        """Test processing match with missing biomechanical data."""
        empty_biomech_data = {}
        
        result = process_match_biomechanical_data(
            sample_match_data,
            empty_biomech_data
        )
        
        # Should return empty result
        assert result == {}
    
    def test_process_match_partial_biomech_data(self, sample_match_data):
        """Test processing match with partial biomechanical data."""
        partial_biomech_data = {
            "test_match_1_1_1": {
                "head_stability": 0.8
            }
            # Missing data for other deliveries
        }
        
        config = BiomechanicalConfig(min_deliveries_required=1)
        result = process_match_biomechanical_data(
            sample_match_data,
            partial_biomech_data,
            config
        )
        
        # Should only have features for players in the first delivery
        assert 'player1' in result  # Batter
        assert 'player2' in result  # Bowler
        assert 'player3' in result  # Fielder


class TestGraphIntegration:
    """Test integration of biomechanical features into knowledge graphs."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create sample cricket knowledge graph."""
        G = nx.Graph()
        
        # Add player nodes with existing features
        G.add_node('player1', node_type='player', features=np.array([0.5, 0.6, 0.7]))
        G.add_node('player2', node_type='player', features=np.array([0.8, 0.9]))
        G.add_node('team1', node_type='team', features=np.array([0.3, 0.4]))
        
        return G
    
    @pytest.fixture
    def sample_biomech_features(self):
        """Sample biomechanical features for testing."""
        return {
            'player1': {
                'biomech_head_stability_mean': 0.85,
                'biomech_head_stability_std': 0.05,
                'biomech_shot_commitment_mean': 0.9,
                'biomech_shot_commitment_std': 0.03
            },
            'player2': {
                'biomech_release_point_consistency_mean': 0.92,
                'biomech_release_point_consistency_std': 0.02,
                'biomech_arm_path_mean': 0.88,
                'biomech_arm_path_std': 0.04
            }
        }
    
    def test_add_biomechanical_features_to_graph(self, sample_graph, sample_biomech_features):
        """Test adding biomechanical features to graph nodes."""
        updated_graph = add_biomechanical_features_to_graph(
            sample_graph,
            sample_biomech_features
        )
        
        # Check that player nodes were updated
        player1_data = updated_graph.nodes['player1']
        assert 'biomechanical_features' in player1_data
        assert 'feature_names' in player1_data
        
        # Check feature concatenation
        original_features = np.array([0.5, 0.6, 0.7])
        biomech_features = np.array([0.85, 0.05, 0.9, 0.03])
        expected_features = np.concatenate([original_features, biomech_features])
        
        np.testing.assert_array_almost_equal(
            player1_data['features'],
            expected_features
        )
        
        # Check that non-player nodes are unchanged
        team1_data = updated_graph.nodes['team1']
        assert 'biomechanical_features' not in team1_data
        np.testing.assert_array_equal(
            team1_data['features'],
            np.array([0.3, 0.4])
        )
    
    def test_add_features_to_node_without_existing_features(self, sample_biomech_features):
        """Test adding features to node without existing features."""
        G = nx.Graph()
        G.add_node('player1', node_type='player')  # No existing features
        
        updated_graph = add_biomechanical_features_to_graph(
            G,
            sample_biomech_features
        )
        
        player1_data = updated_graph.nodes['player1']
        
        # Should have only biomechanical features
        expected_features = np.array([0.85, 0.05, 0.9, 0.03])
        np.testing.assert_array_almost_equal(
            player1_data['features'],
            expected_features
        )
    
    def test_add_features_missing_player(self, sample_graph):
        """Test adding features for player not in graph."""
        biomech_features = {
            'nonexistent_player': {
                'biomech_head_stability_mean': 0.8
            }
        }
        
        # Should not raise error and should not modify graph
        updated_graph = add_biomechanical_features_to_graph(
            sample_graph,
            biomech_features
        )
        
        # Graph should be unchanged
        assert len(updated_graph.nodes) == len(sample_graph.nodes)
        for node_id in sample_graph.nodes:
            np.testing.assert_array_equal(
                updated_graph.nodes[node_id]['features'],
                sample_graph.nodes[node_id]['features']
            )


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_biomechanical_feature_dimension(self):
        """Test getting biomechanical feature dimension."""
        dimension = get_biomechanical_feature_dimension()
        
        # Should be number of signals * 4 features per signal
        num_signals = len(BiomechanicalSignalSchema.get_all_signals())
        expected_dimension = num_signals * 4
        
        assert dimension == expected_dimension
        assert dimension == 40  # 10 signals * 4 features
    
    def test_create_sample_biomechanical_data(self):
        """Test creating sample biomechanical data."""
        match_data = pd.DataFrame([
            {
                'match_id': 'test',
                'innings': 1,
                'over': 1,
                'ball': 1,
                'runs_scored': 4,
                'wicket_type': None
            },
            {
                'match_id': 'test',
                'innings': 1,
                'over': 1,
                'ball': 2,
                'runs_scored': 0,
                'wicket_type': 'bowled'
            }
        ])
        
        sample_data = create_sample_biomechanical_data(match_data)
        
        assert len(sample_data) == 2
        assert 'test_1_1_1' in sample_data
        assert 'test_1_1_2' in sample_data
        
        # Check that all signals are present
        for delivery_id, signals in sample_data.items():
            assert len(signals) == 10  # All biomechanical signals
            
            # Check that values are in valid range
            for signal_name, value in signals.items():
                assert 0.0 <= value <= 1.0
        
        # Check that boundary delivery has higher technique scores
        boundary_signals = sample_data['test_1_1_1']
        wicket_signals = sample_data['test_1_1_2']
        
        # Boundary should have better batter technique
        assert boundary_signals['head_stability'] > wicket_signals['head_stability']
        assert boundary_signals['shot_commitment'] > wicket_signals['shot_commitment']
        
        # Wicket should have better bowler technique
        assert wicket_signals['release_point_consistency'] > boundary_signals['release_point_consistency']


class TestBiomechanicalEventMetadata:
    """Test biomechanical event metadata functionality."""
    
    def test_metadata_creation(self):
        """Test creating biomechanical event metadata."""
        metadata = BiomechanicalEventMetadata(
            delivery_id="test_1_1_1",
            match_id="test_match",
            innings=1,
            over=1.1,
            ball=1,
            batter="player1",
            bowler="player2",
            runs_scored=4,
            confidence_score=0.95
        )
        
        assert metadata.delivery_id == "test_1_1_1"
        assert metadata.match_id == "test_match"
        assert metadata.innings == 1
        assert metadata.over == 1.1
        assert metadata.ball == 1
        assert metadata.batter == "player1"
        assert metadata.bowler == "player2"
        assert metadata.runs_scored == 4
        assert metadata.confidence_score == 0.95
        assert metadata.processing_version == "1.0"
    
    def test_metadata_with_timestamp(self):
        """Test metadata with timestamp information."""
        timestamp = datetime(2024, 1, 15, 14, 30, 0)
        
        metadata = BiomechanicalEventMetadata(
            delivery_id="test_1_1_1",
            match_id="test_match",
            innings=1,
            over=1.1,
            ball=1,
            timestamp=timestamp
        )
        
        assert metadata.timestamp == timestamp
    
    def test_metadata_optional_fields(self):
        """Test metadata with optional fields."""
        metadata = BiomechanicalEventMetadata(
            delivery_id="test_1_1_1",
            match_id="test_match",
            innings=1,
            over=1.1,
            ball=1,
            video_frame_start=1000,
            video_frame_end=1030,
            wicket_type="bowled"
        )
        
        assert metadata.video_frame_start == 1000
        assert metadata.video_frame_end == 1030
        assert metadata.wicket_type == "bowled"


class TestBiomechanicalEventNodes:
    """Test biomechanical event node creation and management."""
    
    @pytest.fixture
    def sample_graph_with_events(self):
        """Create sample graph with event nodes."""
        G = nx.Graph()
        
        # Add event nodes
        G.add_node('four', node_type='event')
        G.add_node('six', node_type='event')
        G.add_node('dot', node_type='event')
        G.add_node('wicket', node_type='event')
        
        return G
    
    @pytest.fixture
    def sample_match_data_with_dates(self):
        """Sample match data with date information."""
        return pd.DataFrame([
            {
                'match_id': 'test_match',
                'innings': 1,
                'over': 1,
                'ball': 1,
                'batter': 'player1',
                'bowler': 'player2',
                'fielder': 'player3',
                'runs_scored': 4,
                'wicket_type': None,
                'date': '2024-01-15T14:30:00'
            },
            {
                'match_id': 'test_match',
                'innings': 1,
                'over': 1,
                'ball': 2,
                'batter': 'player1',
                'bowler': 'player2',
                'fielder': None,
                'runs_scored': 0,
                'wicket_type': 'bowled',
                'date': '2024-01-15T14:30:30'
            },
            {
                'match_id': 'test_match',
                'innings': 1,
                'over': 1,
                'ball': 3,
                'batter': 'player4',
                'bowler': 'player2',
                'fielder': 'player3',
                'runs_scored': 6,
                'wicket_type': None,
                'date': '2024-01-15T14:31:00'
            }
        ])
    
    @pytest.fixture
    def sample_biomech_data_for_events(self):
        """Sample biomechanical data for event testing."""
        return {
            "test_match_1_1_1": {
                "head_stability": 0.9,
                "shot_commitment": 0.95,
                "backlift_type": 0.8,
                "footwork_direction": 0.85,
                "release_point_consistency": 0.7,
                "arm_path": 0.75,
                "follow_through_momentum": 0.8,
                "closing_speed": 0.8,
                "reaction_time": 0.9,
                "interception_type": 0.7
            },
            "test_match_1_1_2": {
                "head_stability": 0.5,
                "shot_commitment": 0.4,
                "release_point_consistency": 0.95,
                "arm_path": 0.9,
                "follow_through_momentum": 0.85
            },
            "test_match_1_1_3": {
                "head_stability": 0.95,
                "shot_commitment": 0.98,
                "backlift_type": 0.9,
                "footwork_direction": 0.92
            }
        }
    
    def test_create_biomechanical_event_nodes(self, sample_graph_with_events, sample_match_data_with_dates, sample_biomech_data_for_events):
        """Test creating biomechanical event nodes."""
        updated_graph = create_biomechanical_event_nodes(
            sample_graph_with_events,
            sample_match_data_with_dates,
            sample_biomech_data_for_events
        )
        
        # Check that biomechanical event nodes were created
        biomech_event_nodes = [
            node_id for node_id, node_data in updated_graph.nodes(data=True)
            if node_data.get('node_type') == 'biomechanical_event'
        ]
        
        assert len(biomech_event_nodes) == 3  # One for each delivery
        
        # Check node naming convention
        expected_nodes = [
            'biomech_event_test_match_1_1_1',
            'biomech_event_test_match_1_1_2',
            'biomech_event_test_match_1_1_3'
        ]
        
        for expected_node in expected_nodes:
            assert expected_node in biomech_event_nodes
    
    def test_biomechanical_event_node_attributes(self, sample_graph_with_events, sample_match_data_with_dates, sample_biomech_data_for_events):
        """Test biomechanical event node attributes."""
        updated_graph = create_biomechanical_event_nodes(
            sample_graph_with_events,
            sample_match_data_with_dates,
            sample_biomech_data_for_events
        )
        
        # Check first biomechanical event node
        biomech_node_id = 'biomech_event_test_match_1_1_1'
        node_data = updated_graph.nodes[biomech_node_id]
        
        assert node_data['node_type'] == 'biomechanical_event'
        assert node_data['delivery_id'] == 'test_match_1_1_1'
        assert node_data['match_id'] == 'test_match'
        assert node_data['innings'] == 1
        assert node_data['over'] == 1
        assert node_data['ball'] == 1
        assert node_data['signal_count'] == 10
        assert len(node_data['signal_names']) == 10
        assert len(node_data['features']) == 10
        
        # Check biomechanical signals
        signals = node_data['biomechanical_signals']
        assert signals['head_stability'] == 0.9
        assert signals['shot_commitment'] == 0.95
        
        # Check metadata
        metadata = node_data['metadata']
        assert isinstance(metadata, BiomechanicalEventMetadata)
        assert metadata.delivery_id == 'test_match_1_1_1'
        assert metadata.batter == 'player1'
        assert metadata.bowler == 'player2'
        assert metadata.runs_scored == 4
        assert metadata.timestamp is not None
    
    def test_biomechanical_event_edges(self, sample_graph_with_events, sample_match_data_with_dates, sample_biomech_data_for_events):
        """Test has_biomechanics edge creation."""
        updated_graph = create_biomechanical_event_nodes(
            sample_graph_with_events,
            sample_match_data_with_dates,
            sample_biomech_data_for_events
        )
        
        # Check that has_biomechanics edges were created
        has_biomechanics_edges = [
            (u, v, data) for u, v, data in updated_graph.edges(data=True)
            if data.get('edge_type') == 'has_biomechanics'
        ]
        
        assert len(has_biomechanics_edges) == 3  # One for each delivery
        
        # Check edge attributes
        edge_u, edge_v, edge_data = has_biomechanics_edges[0]
        
        assert edge_data['edge_type'] == 'has_biomechanics'
        assert edge_data['weight'] == 1.0
        assert 'delivery_id' in edge_data
        assert 'signal_count' in edge_data
        assert 'timestamp' in edge_data
        assert 'confidence' in edge_data
        assert 'days_ago' in edge_data
    
    def test_edge_source_target_types(self, sample_graph_with_events, sample_match_data_with_dates, sample_biomech_data_for_events):
        """Test that edges connect correct node types."""
        updated_graph = create_biomechanical_event_nodes(
            sample_graph_with_events,
            sample_match_data_with_dates,
            sample_biomech_data_for_events
        )
        
        # Check edge connections
        for u, v, data in updated_graph.edges(data=True):
            if data.get('edge_type') == 'has_biomechanics':
                u_type = updated_graph.nodes[u].get('node_type')
                v_type = updated_graph.nodes[v].get('node_type')
                
                # One should be event, other should be biomechanical_event
                assert (u_type == 'event' and v_type == 'biomechanical_event') or \
                       (u_type == 'biomechanical_event' and v_type == 'event')
    
    def test_timestamp_parsing(self, sample_graph_with_events):
        """Test timestamp parsing from different formats."""
        # Test with string timestamp
        match_data_str = pd.DataFrame([{
            'match_id': 'test',
            'innings': 1,
            'over': 1,
            'ball': 1,
            'batter': 'player1',
            'bowler': 'player2',
            'runs_scored': 4,
            'date': '2024-01-15T14:30:00'
        }])
        
        biomech_data = {
            "test_1_1_1": {"head_stability": 0.8}
        }
        
        updated_graph = create_biomechanical_event_nodes(
            sample_graph_with_events,
            match_data_str,
            biomech_data
        )
        
        biomech_node = 'biomech_event_test_1_1_1'
        metadata = updated_graph.nodes[biomech_node]['metadata']
        
        assert metadata.timestamp is not None
        assert isinstance(metadata.timestamp, datetime)
        assert metadata.timestamp.year == 2024
        assert metadata.timestamp.month == 1
        assert metadata.timestamp.day == 15
    
    def test_missing_biomech_data(self, sample_graph_with_events, sample_match_data_with_dates):
        """Test handling missing biomechanical data."""
        empty_biomech_data = {}
        
        updated_graph = create_biomechanical_event_nodes(
            sample_graph_with_events,
            sample_match_data_with_dates,
            empty_biomech_data
        )
        
        # Should not create any biomechanical event nodes
        biomech_event_nodes = [
            node_id for node_id, node_data in updated_graph.nodes(data=True)
            if node_data.get('node_type') == 'biomechanical_event'
        ]
        
        assert len(biomech_event_nodes) == 0
    
    def test_partial_biomech_data(self, sample_graph_with_events, sample_match_data_with_dates):
        """Test handling partial biomechanical data."""
        partial_biomech_data = {
            "test_match_1_1_1": {"head_stability": 0.8}
            # Missing data for other deliveries
        }
        
        updated_graph = create_biomechanical_event_nodes(
            sample_graph_with_events,
            sample_match_data_with_dates,
            partial_biomech_data
        )
        
        # Should create only one biomechanical event node
        biomech_event_nodes = [
            node_id for node_id, node_data in updated_graph.nodes(data=True)
            if node_data.get('node_type') == 'biomechanical_event'
        ]
        
        assert len(biomech_event_nodes) == 1
        assert 'biomech_event_test_match_1_1_1' in biomech_event_nodes


class TestBiomechanicalEventQueries:
    """Test querying biomechanical event nodes."""
    
    @pytest.fixture
    def sample_graph_with_biomech_events(self):
        """Create sample graph with biomechanical events."""
        G = nx.Graph()
        
        # Add event nodes
        G.add_node('four', node_type='event')
        G.add_node('wicket', node_type='event')
        
        # Add biomechanical event nodes
        metadata1 = BiomechanicalEventMetadata(
            delivery_id="match1_1_1_1",
            match_id="match1",
            innings=1,
            over=1.1,
            ball=1,
            batter="player1",
            timestamp=datetime(2024, 1, 15, 14, 30, 0)
        )
        
        metadata2 = BiomechanicalEventMetadata(
            delivery_id="match1_1_1_2",
            match_id="match1",
            innings=1,
            over=1.2,
            ball=2,
            batter="player1",
            timestamp=datetime(2024, 1, 15, 14, 30, 30)
        )
        
        metadata3 = BiomechanicalEventMetadata(
            delivery_id="match2_1_1_1",
            match_id="match2",
            innings=1,
            over=1.1,
            ball=1,
            batter="player2",
            timestamp=datetime(2024, 1, 16, 15, 0, 0)
        )
        
        G.add_node(
            'biomech_event_match1_1_1_1',
            node_type='biomechanical_event',
            delivery_id='match1_1_1_1',
            match_id='match1',
            metadata=metadata1,
            biomechanical_signals={'head_stability': 0.8, 'shot_commitment': 0.9}
        )
        
        G.add_node(
            'biomech_event_match1_1_1_2',
            node_type='biomechanical_event',
            delivery_id='match1_1_1_2',
            match_id='match1',
            metadata=metadata2,
            biomechanical_signals={'head_stability': 0.6, 'shot_commitment': 0.7}
        )
        
        G.add_node(
            'biomech_event_match2_1_1_1',
            node_type='biomechanical_event',
            delivery_id='match2_1_1_1',
            match_id='match2',
            metadata=metadata3,
            biomechanical_signals={'head_stability': 0.9, 'shot_commitment': 0.95}
        )
        
        # Add has_biomechanics edges
        G.add_edge('four', 'biomech_event_match1_1_1_1', edge_type='has_biomechanics')
        G.add_edge('wicket', 'biomech_event_match1_1_1_2', edge_type='has_biomechanics')
        G.add_edge('four', 'biomech_event_match2_1_1_1', edge_type='has_biomechanics')
        
        return G
    
    def test_get_biomechanical_events_for_delivery(self, sample_graph_with_biomech_events):
        """Test getting biomechanical events for specific delivery."""
        events = get_biomechanical_events_for_delivery(
            sample_graph_with_biomech_events,
            'match1_1_1_1'
        )
        
        assert len(events) == 1
        node_id, node_data = events[0]
        assert node_id == 'biomech_event_match1_1_1_1'
        assert node_data['delivery_id'] == 'match1_1_1_1'
    
    def test_get_biomechanical_events_for_nonexistent_delivery(self, sample_graph_with_biomech_events):
        """Test getting events for non-existent delivery."""
        events = get_biomechanical_events_for_delivery(
            sample_graph_with_biomech_events,
            'nonexistent_delivery'
        )
        
        assert len(events) == 0
    
    def test_get_biomechanical_events_by_match(self, sample_graph_with_biomech_events):
        """Test getting biomechanical events grouped by match."""
        match_events = get_biomechanical_events_by_match(
            sample_graph_with_biomech_events,
            'match1'
        )
        
        assert len(match_events) == 2
        assert 'match1_1_1_1' in match_events
        assert 'match1_1_1_2' in match_events
        
        # Check first delivery events
        delivery1_events = match_events['match1_1_1_1']
        assert len(delivery1_events) == 1
        assert delivery1_events[0][0] == 'biomech_event_match1_1_1_1'
    
    def test_get_biomechanical_events_by_nonexistent_match(self, sample_graph_with_biomech_events):
        """Test getting events for non-existent match."""
        match_events = get_biomechanical_events_by_match(
            sample_graph_with_biomech_events,
            'nonexistent_match'
        )
        
        assert len(match_events) == 0


class TestBiomechanicalEventPatternAnalysis:
    """Test biomechanical event pattern analysis."""
    
    @pytest.fixture
    def sample_graph_for_pattern_analysis(self):
        """Create sample graph for pattern analysis."""
        G = nx.Graph()
        
        # Add event nodes
        G.add_node('four', node_type='event')
        G.add_node('dot', node_type='event')
        G.add_node('wicket', node_type='event')
        
        # Create multiple biomechanical events for player1
        timestamps = [
            datetime(2024, 1, 15, 14, 30, 0),
            datetime(2024, 1, 15, 14, 31, 0),
            datetime(2024, 1, 15, 14, 32, 0),
            datetime(2024, 1, 16, 15, 0, 0)
        ]
        
        signal_values = [
            {'head_stability': 0.8, 'shot_commitment': 0.9},
            {'head_stability': 0.75, 'shot_commitment': 0.85},
            {'head_stability': 0.85, 'shot_commitment': 0.95},
            {'head_stability': 0.9, 'shot_commitment': 0.98}
        ]
        
        outcomes = [
            {'runs_scored': 4, 'wicket_type': None},
            {'runs_scored': 0, 'wicket_type': None},
            {'runs_scored': 1, 'wicket_type': None},
            {'runs_scored': 0, 'wicket_type': 'bowled'}
        ]
        
        for i, (timestamp, signals, outcome) in enumerate(zip(timestamps, signal_values, outcomes)):
            delivery_id = f"test_match_1_1_{i+1}"
            
            metadata = BiomechanicalEventMetadata(
                delivery_id=delivery_id,
                match_id="test_match",
                innings=1,
                over=1.0 + (i+1) * 0.1,
                ball=i+1,
                batter="player1",
                timestamp=timestamp,
                runs_scored=outcome['runs_scored'],
                wicket_type=outcome['wicket_type']
            )
            
            biomech_node_id = f"biomech_event_{delivery_id}"
            G.add_node(
                biomech_node_id,
                node_type='biomechanical_event',
                delivery_id=delivery_id,
                match_id="test_match",
                metadata=metadata,
                biomechanical_signals=signals
            )
            
            # Connect to appropriate event node
            if outcome['runs_scored'] >= 4:
                event_node = 'four'
            elif outcome['wicket_type']:
                event_node = 'wicket'
            else:
                event_node = 'dot'
            
            G.add_edge(event_node, biomech_node_id, edge_type='has_biomechanics')
        
        return G
    
    def test_analyze_biomechanical_event_patterns(self, sample_graph_for_pattern_analysis):
        """Test biomechanical pattern analysis for a player."""
        analysis = analyze_biomechanical_event_patterns(
            sample_graph_for_pattern_analysis,
            'player1'
        )
        
        assert analysis['player_id'] == 'player1'
        assert analysis['event_count'] == 4
        
        # Check pattern analysis
        patterns = analysis['patterns']
        assert 'head_stability' in patterns
        assert 'shot_commitment' in patterns
        
        head_stability_pattern = patterns['head_stability']
        assert 'mean' in head_stability_pattern
        assert 'std' in head_stability_pattern
        assert 'min' in head_stability_pattern
        assert 'max' in head_stability_pattern
        assert 'trend' in head_stability_pattern
        
        # Check values are reasonable
        assert 0.7 <= head_stability_pattern['mean'] <= 0.9
        assert head_stability_pattern['min'] == 0.75
        assert head_stability_pattern['max'] == 0.9
        assert head_stability_pattern['trend'] > 0  # Improving trend
    
    def test_analyze_patterns_with_event_type_filter(self, sample_graph_for_pattern_analysis):
        """Test pattern analysis with event type filtering."""
        analysis = analyze_biomechanical_event_patterns(
            sample_graph_for_pattern_analysis,
            'player1',
            'four'
        )
        
        # Should only analyze events connected to 'four' node
        assert analysis['event_count'] == 1
        
        # Check that only the boundary event is analyzed
        patterns = analysis['patterns']
        head_stability_pattern = patterns['head_stability']
        assert head_stability_pattern['mean'] == 0.8  # Only the boundary delivery
    
    def test_analyze_patterns_temporal_span(self, sample_graph_for_pattern_analysis):
        """Test temporal span analysis."""
        analysis = analyze_biomechanical_event_patterns(
            sample_graph_for_pattern_analysis,
            'player1'
        )
        
        temporal_span = analysis['temporal_span']
        assert temporal_span is not None
        assert 'start' in temporal_span
        assert 'end' in temporal_span
        assert 'duration_days' in temporal_span
        
        # Check duration
        assert temporal_span['duration_days'] == 1  # Events span 1 day
    
    def test_analyze_patterns_outcome_distribution(self, sample_graph_for_pattern_analysis):
        """Test outcome distribution analysis."""
        analysis = analyze_biomechanical_event_patterns(
            sample_graph_for_pattern_analysis,
            'player1'
        )
        
        outcome_dist = analysis['outcome_distribution']
        assert 'runs' in outcome_dist
        assert 'wickets' in outcome_dist
        
        runs_dist = outcome_dist['runs']
        assert runs_dist[0] == 2  # Two dot balls
        assert runs_dist[1] == 1  # One single
        assert runs_dist[4] == 1  # One boundary
        
        wickets_dist = outcome_dist['wickets']
        assert wickets_dist['bowled'] == 1  # One bowled dismissal
    
    def test_analyze_patterns_no_events(self, sample_graph_for_pattern_analysis):
        """Test pattern analysis for player with no events."""
        analysis = analyze_biomechanical_event_patterns(
            sample_graph_for_pattern_analysis,
            'nonexistent_player'
        )
        
        assert analysis['player_id'] == 'nonexistent_player'
        assert analysis['event_count'] == 0
        assert analysis['patterns'] == {}


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_match_data(self):
        """Test processing empty match data."""
        empty_match_data = pd.DataFrame()
        biomech_data = {}
        
        result = process_match_biomechanical_data(empty_match_data, biomech_data)
        assert result == {}
    
    def test_malformed_biomech_data(self):
        """Test handling malformed biomechanical data."""
        match_data = pd.DataFrame([{
            'match_id': 'test',
            'innings': 1,
            'over': 1,
            'ball': 1,
            'batter': 'player1',
            'bowler': 'player2'
        }])
        
        malformed_data = {
            "test_1_1_1": "not_a_dict"  # Should be dictionary
        }
        
        # Should handle gracefully
        result = process_match_biomechanical_data(match_data, malformed_data)
        assert result == {}
    
    def test_extreme_signal_values(self):
        """Test handling extreme signal values."""
        aggregator = BiomechanicalAggregator()
        
        # Add extreme values
        extreme_signals = {
            "head_stability": 999.0,    # Way too high
            "shot_commitment": -100.0   # Way too low
        }
        
        # Should be clamped during loading
        loader = BiomechanicalSignalLoader()
        validated_data = loader.load_from_dict({"delivery1": extreme_signals})
        
        delivery_signals = validated_data["delivery1"]
        assert delivery_signals["head_stability"] == 1.0  # Clamped to max
        assert delivery_signals["shot_commitment"] == 0.0  # Clamped to min
    
    def test_all_nan_values(self):
        """Test handling all NaN values for a signal."""
        config = BiomechanicalConfig(min_deliveries_required=5)
        aggregator = BiomechanicalAggregator(config)
        
        player_id = "player1"
        
        # Add deliveries with all NaN values
        for i in range(10):
            signals = {"head_stability": float('nan')}
            aggregator.add_delivery_signals(player_id, signals, "batter")
        
        features = aggregator.get_player_aggregated_features(player_id)
        
        # Should not generate features for all-NaN signal
        assert "biomech_head_stability_mean" not in features
    
    def test_single_value_std_calculation(self):
        """Test standard deviation calculation with single repeated value."""
        config = BiomechanicalConfig(min_deliveries_required=5)
        aggregator = BiomechanicalAggregator(config)
        
        player_id = "player1"
        
        # Add deliveries with same value
        for i in range(10):
            signals = {"head_stability": 0.8}  # Same value
            aggregator.add_delivery_signals(player_id, signals, "batter")
        
        features = aggregator.get_player_aggregated_features(player_id)
        
        # Should handle zero standard deviation
        assert "biomech_head_stability_std" in features
        assert features["biomech_head_stability_std"] == 0.0
    
    def test_biomech_event_invalid_date_format(self):
        """Test handling invalid date formats in biomechanical events."""
        G = nx.Graph()
        G.add_node('four', node_type='event')
        
        match_data = pd.DataFrame([{
            'match_id': 'test',
            'innings': 1,
            'over': 1,
            'ball': 1,
            'batter': 'player1',
            'bowler': 'player2',
            'runs_scored': 4,
            'date': 'invalid_date_format'
        }])
        
        biomech_data = {
            "test_1_1_1": {"head_stability": 0.8}
        }
        
        # Should handle gracefully without crashing
        updated_graph = create_biomechanical_event_nodes(G, match_data, biomech_data)
        
        # Node should still be created, but without timestamp
        biomech_node = 'biomech_event_test_1_1_1'
        assert biomech_node in updated_graph.nodes
        
        metadata = updated_graph.nodes[biomech_node]['metadata']
        assert metadata.timestamp is None
    
    def test_biomech_event_empty_graph(self):
        """Test creating biomechanical events on empty graph."""
        empty_graph = nx.Graph()
        
        match_data = pd.DataFrame([{
            'match_id': 'test',
            'innings': 1,
            'over': 1,
            'ball': 1,
            'batter': 'player1',
            'bowler': 'player2',
            'runs_scored': 4
        }])
        
        biomech_data = {
            "test_1_1_1": {"head_stability": 0.8}
        }
        
        # Should create biomechanical event node but no edges (no event nodes to connect to)
        updated_graph = create_biomechanical_event_nodes(empty_graph, match_data, biomech_data)
        
        # Should have the biomechanical event node
        biomech_nodes = [
            node_id for node_id, node_data in updated_graph.nodes(data=True)
            if node_data.get('node_type') == 'biomechanical_event'
        ]
        assert len(biomech_nodes) == 1
        
        # Should have no has_biomechanics edges
        biomech_edges = [
            (u, v, data) for u, v, data in updated_graph.edges(data=True)
            if data.get('edge_type') == 'has_biomechanics'
        ]
        assert len(biomech_edges) == 0
# Purpose: Unit tests for temporal graph builder with learnable decay
# Author: Shamus Rae, Last Modified: 2024-01-15

import pytest
import torch
import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from crickformers.gnn.temporal_graph_builder import (
    TemporalGraphBuilder,
    create_temporal_graph_builder
)
from crickformers.model.learnable_temporal_decay import LearnableTemporalDecay


class TestTemporalGraphBuilder:
    """Test temporal graph builder."""
    
    @pytest.fixture
    def feature_names(self):
        """Sample feature names."""
        return ['batting_avg', 'strike_rate', 'recent_form', 'pressure_score']
    
    @pytest.fixture
    def temporal_config(self):
        """Temporal decay configuration."""
        return {
            'initial_half_life': 30.0,
            'min_half_life': 1.0,
            'max_half_life': 365.0,
            'learnable': True
        }
    
    @pytest.fixture
    def graph_builder(self, feature_names, temporal_config):
        """Create temporal graph builder."""
        return TemporalGraphBuilder(
            feature_names=feature_names,
            temporal_config=temporal_config,
            use_adaptive_encoder=False
        )
    
    @pytest.fixture
    def adaptive_graph_builder(self, feature_names, temporal_config):
        """Create temporal graph builder with adaptive encoder."""
        encoder_config = {
            'embed_dim': 32,
            'max_days': 365,
            'use_positional_encoding': True
        }
        return TemporalGraphBuilder(
            feature_names=feature_names,
            temporal_config=temporal_config,
            use_adaptive_encoder=True,
            encoder_config=encoder_config
        )
    
    @pytest.fixture
    def sample_match_data(self):
        """Create sample match data."""
        np.random.seed(42)
        
        data = []
        base_date = datetime.now() - timedelta(days=30)
        
        for match_id in ['match1', 'match2', 'match3']:
            match_date = base_date + timedelta(days=np.random.randint(0, 60))
            
            for over in range(1, 21):  # 20 overs
                for ball in range(1, 7):  # 6 balls per over
                    data.append({
                        'match_id': match_id,
                        'match_date': match_date,
                        'over': over,
                        'ball': ball,
                        'batter': np.random.choice(['Player1', 'Player2', 'Player3']),
                        'non_striker': np.random.choice(['Player1', 'Player2', 'Player3']),
                        'bowler': np.random.choice(['Bowler1', 'Bowler2', 'Bowler3']),
                        'runs_scored': np.random.choice([0, 1, 2, 4, 6], p=[0.4, 0.3, 0.15, 0.1, 0.05]),
                        'wicket_type': np.random.choice([None, 'caught', 'bowled'], p=[0.9, 0.07, 0.03]),
                        'venue': np.random.choice(['Venue1', 'Venue2']),
                        'conditions': np.random.choice(['clear', 'cloudy'])
                    })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_player_history(self, feature_names):
        """Create sample player history data."""
        np.random.seed(42)
        
        history = {}
        base_date = datetime.now()
        
        for player in ['Player1', 'Player2', 'Player3']:
            player_data = []
            
            for i in range(20):  # 20 historical matches
                match_date = base_date - timedelta(days=i * 7)  # Weekly matches
                
                row = {'match_date': match_date}
                for feature_name in feature_names:
                    row[feature_name] = np.random.uniform(0.1, 1.0)
                
                player_data.append(row)
            
            history[player] = pd.DataFrame(player_data)
        
        return history
    
    @pytest.fixture
    def empty_graph(self):
        """Create empty graph with some nodes."""
        graph = nx.Graph()
        
        # Add player nodes
        for player in ['Player1', 'Player2', 'Player3']:
            graph.add_node(player, node_type='player')
        
        # Add bowler nodes
        for bowler in ['Bowler1', 'Bowler2', 'Bowler3']:
            graph.add_node(bowler, node_type='player')
        
        return graph
    
    def test_initialization(self, graph_builder, feature_names, temporal_config):
        """Test graph builder initialization."""
        assert graph_builder.feature_names == feature_names
        assert graph_builder.temporal_config == temporal_config
        assert not graph_builder.use_adaptive_encoder
        assert graph_builder.adaptive_encoder is None
        
        # Check temporal decay module
        assert isinstance(graph_builder.temporal_decay, LearnableTemporalDecay)
        assert graph_builder.temporal_decay.feature_names == feature_names
    
    def test_adaptive_encoder_initialization(self, adaptive_graph_builder):
        """Test initialization with adaptive encoder."""
        assert adaptive_graph_builder.use_adaptive_encoder
        assert adaptive_graph_builder.adaptive_encoder is not None
    
    def test_extract_match_date(self, graph_builder, sample_match_data):
        """Test match date extraction."""
        match_group = sample_match_data[sample_match_data['match_id'] == 'match1']
        match_date = graph_builder._extract_match_date(match_group)
        
        assert isinstance(match_date, datetime)
        assert match_date == match_group['match_date'].iloc[0]
    
    def test_extract_match_date_missing(self, graph_builder):
        """Test match date extraction with missing data."""
        empty_df = pd.DataFrame({'match_id': ['test']})
        match_date = graph_builder._extract_match_date(empty_df)
        
        assert match_date is None
    
    def test_extract_partnerships(self, graph_builder, sample_match_data):
        """Test partnership extraction."""
        match_group = sample_match_data[sample_match_data['match_id'] == 'match1']
        partnerships = graph_builder._extract_partnerships(match_group)
        
        assert isinstance(partnerships, list)
        assert len(partnerships) > 0
        
        for partnership in partnerships:
            assert 'players' in partnership
            assert 'runs' in partnership
            assert 'balls' in partnership
            assert 'boundaries' in partnership
            assert 'wickets' in partnership
            assert len(partnership['players']) == 2
    
    def test_extract_partnership_features(self, graph_builder):
        """Test partnership feature extraction."""
        partnership_data = pd.DataFrame({
            'runs_scored': [1, 4, 0, 2, 6, 1],
            'wicket_type': [None, None, None, None, None, 'caught']
        })
        
        partnership = {
            'players': ('Player1', 'Player2'),
            'runs': 14,
            'balls': 6,
            'boundaries': 2,
            'wickets': 1,
            'data': partnership_data
        }
        
        features = graph_builder._extract_partnership_features(partnership)
        
        assert isinstance(features, dict)
        assert 'partnership_runs' in features
        assert 'partnership_balls' in features
        assert 'partnership_run_rate' in features
        assert 'partnership_boundaries' in features
        assert 'strike_rotation' in features
        assert 'dot_ball_rate' in features
        
        assert features['partnership_runs'] == 14
        assert features['partnership_balls'] == 6
        assert features['partnership_boundaries'] == 2
    
    def test_extract_ball_features(self, graph_builder):
        """Test ball feature extraction."""
        ball_row = pd.Series({
            'runs_scored': 4,
            'wicket_type': None,
            'over': 15,
            'ball': 3
        })
        
        features = graph_builder._extract_ball_features(ball_row)
        
        assert isinstance(features, dict)
        assert features['runs_scored'] == 4.0
        assert features['is_boundary'] == 1.0
        assert features['is_wicket'] == 0.0
        assert features['is_dot'] == 0.0
        assert features['over'] == 15.0
        assert features['ball_in_over'] == 3.0
    
    def test_add_temporal_edges(self, graph_builder, empty_graph, sample_match_data):
        """Test adding temporal edges to graph."""
        reference_date = datetime.now()
        
        updated_graph = graph_builder.add_temporal_edges(
            empty_graph, sample_match_data, reference_date
        )
        
        assert isinstance(updated_graph, nx.Graph)
        
        # Should have added edges
        initial_edges = len(empty_graph.edges())
        final_edges = len(updated_graph.edges())
        assert final_edges > initial_edges
        
        # Check edge attributes
        for u, v, data in updated_graph.edges(data=True):
            if 'weight' in data:
                assert data['weight'] > 0
                assert 'temporal_weight' in data
                assert 'days_ago' in data
    
    def test_compute_form_vectors(self, graph_builder, sample_player_history):
        """Test form vector computation."""
        reference_date = datetime.now()
        
        form_vectors = graph_builder.compute_form_vectors(
            sample_player_history, reference_date
        )
        
        assert isinstance(form_vectors, dict)
        assert len(form_vectors) == len(sample_player_history)
        
        for player_id, form_vector in form_vectors.items():
            assert isinstance(form_vector, torch.Tensor)
            assert form_vector.shape == (len(graph_builder.feature_names),)
            assert torch.all(torch.isfinite(form_vector))
    
    def test_compute_form_vectors_empty_history(self, graph_builder):
        """Test form vector computation with empty history."""
        empty_history = {'Player1': pd.DataFrame()}
        
        form_vectors = graph_builder.compute_form_vectors(empty_history)
        
        assert len(form_vectors) == 0
    
    def test_update_node_features(self, graph_builder, empty_graph):
        """Test node feature updates."""
        # Create mock form vectors
        form_vectors = {
            'Player1': torch.randn(len(graph_builder.feature_names)),
            'Player2': torch.randn(len(graph_builder.feature_names)),
            'Player3': torch.randn(len(graph_builder.feature_names))
        }
        
        feature_dim = 64
        updated_graph = graph_builder.update_node_features(
            empty_graph, form_vectors, feature_dim
        )
        
        # Check updated nodes
        for node_id, node_data in updated_graph.nodes(data=True):
            if node_data.get('node_type') == 'player' and node_id in form_vectors:
                assert 'features' in node_data
                assert 'temporal_form' in node_data
                assert 'form_updated' in node_data
                
                assert len(node_data['features']) == feature_dim
                assert len(node_data['temporal_form']) == len(graph_builder.feature_names)
                assert node_data['form_updated'] is True
    
    def test_update_node_features_with_adaptive_encoder(self, adaptive_graph_builder, empty_graph):
        """Test node feature updates with adaptive encoder."""
        form_vectors = {
            'Player1': torch.randn(len(adaptive_graph_builder.feature_names))
        }
        
        updated_graph = adaptive_graph_builder.update_node_features(
            empty_graph, form_vectors, feature_dim=32
        )
        
        # Should use adaptive encoder
        player_node = updated_graph.nodes['Player1']
        assert 'features' in player_node
        assert len(player_node['features']) == 32
    
    def test_get_temporal_statistics(self, graph_builder, sample_player_history):
        """Test temporal statistics retrieval."""
        # Compute some form vectors to generate statistics
        form_vectors = graph_builder.compute_form_vectors(sample_player_history)
        
        stats = graph_builder.get_temporal_statistics()
        
        assert isinstance(stats, dict)
        assert 'num_players_with_features' in stats
        
        if form_vectors:
            assert stats['num_players_with_features'] > 0
            assert 'avg_form_vector_magnitude' in stats
    
    def test_save_and_load_temporal_state(self, graph_builder, tmp_path):
        """Test saving and loading temporal state."""
        filepath = tmp_path / "temporal_state.pt"
        
        # Save state
        graph_builder.save_temporal_state(str(filepath))
        assert filepath.exists()
        
        # Modify state
        original_half_lives = graph_builder.temporal_decay.get_half_lives().clone()
        with torch.no_grad():
            graph_builder.temporal_decay.log_half_lives.fill_(torch.log(torch.tensor(60.0)))
        
        # Load state
        graph_builder.load_temporal_state(str(filepath))
        loaded_half_lives = graph_builder.temporal_decay.get_half_lives()
        
        # Should restore original state
        assert torch.allclose(original_half_lives, loaded_half_lives, atol=1e-4)
    
    def test_integration_workflow(self, graph_builder, empty_graph, sample_match_data, sample_player_history):
        """Test complete integration workflow."""
        reference_date = datetime.now()
        
        # Step 1: Add temporal edges
        graph_with_edges = graph_builder.add_temporal_edges(
            empty_graph, sample_match_data, reference_date
        )
        
        # Step 2: Compute form vectors
        form_vectors = graph_builder.compute_form_vectors(
            sample_player_history, reference_date
        )
        
        # Step 3: Update node features
        final_graph = graph_builder.update_node_features(
            graph_with_edges, form_vectors, feature_dim=64
        )
        
        # Verify final graph
        initial_edges = len(empty_graph.edges())
        final_edges = len(final_graph.edges())
        assert final_edges >= initial_edges  # Should have at least as many edges
        
        updated_nodes = 0
        for node_id, node_data in final_graph.nodes(data=True):
            if node_data.get('form_updated'):
                updated_nodes += 1
        
        assert updated_nodes > 0
    
    def test_edge_weight_computation(self, graph_builder):
        """Test edge weight computation with temporal decay."""
        # Create test data
        days_ago = torch.tensor([0.0, 10.0, 30.0, 90.0])
        base_weights = torch.ones(4)
        
        # Test without edge features
        edge_weights = graph_builder.temporal_decay.compute_edge_weights(
            days_ago, base_weights
        )
        
        assert edge_weights.shape == base_weights.shape
        assert torch.all(edge_weights > 0)
        
        # Recent edges should have higher weights
        assert edge_weights[0] > edge_weights[1] > edge_weights[2] > edge_weights[3]
    
    def test_form_vector_aggregation(self, graph_builder, feature_names):
        """Test form vector aggregation with temporal decay."""
        # Create historical feature data
        history_length = 10
        feature_history = torch.randn(history_length, len(feature_names))
        days_ago_history = torch.linspace(0, 90, history_length)
        
        form_vector = graph_builder.temporal_decay.get_aggregated_form_vector(
            feature_history, days_ago_history, feature_names
        )
        
        assert form_vector.shape == (len(feature_names),)
        assert torch.all(torch.isfinite(form_vector))
        
        # Recent data should have more influence
        recent_form = graph_builder.temporal_decay.get_aggregated_form_vector(
            feature_history[:5], days_ago_history[:5], feature_names
        )
        old_form = graph_builder.temporal_decay.get_aggregated_form_vector(
            feature_history[5:], days_ago_history[5:], feature_names
        )
        
        # The form vectors should be different
        assert not torch.allclose(recent_form, old_form, atol=1e-3)
    
    def test_temporal_weight_consistency(self, graph_builder):
        """Test temporal weight consistency across different methods."""
        days_ago = torch.tensor([15.0, 30.0, 45.0])
        
        # Get weights from temporal decay directly
        direct_weights = graph_builder.temporal_decay.compute_temporal_weights(days_ago)
        
        # Get weights through edge computation
        base_weights = torch.ones_like(days_ago)
        edge_weights = graph_builder.temporal_decay.compute_edge_weights(days_ago, base_weights)
        
        # Should be consistent (edge_weights = base_weights * temporal_weights)
        assert torch.allclose(direct_weights, edge_weights, atol=1e-6)


class TestFactoryFunction:
    """Test factory function for temporal graph builder."""
    
    def test_create_with_defaults(self):
        """Test factory function with default configuration."""
        feature_names = ['feature1', 'feature2']
        builder = create_temporal_graph_builder(feature_names)
        
        assert isinstance(builder, TemporalGraphBuilder)
        assert builder.feature_names == feature_names
        assert not builder.use_adaptive_encoder
    
    def test_create_with_config(self):
        """Test factory function with custom configuration."""
        feature_names = ['feature1', 'feature2']
        config = {
            'temporal_config': {
                'initial_half_life': 45.0,
                'min_half_life': 5.0,
                'max_half_life': 180.0,
                'learnable': False
            },
            'use_adaptive_encoder': True,
            'encoder_config': {
                'embed_dim': 128,
                'max_days': 180
            }
        }
        
        builder = create_temporal_graph_builder(feature_names, config)
        
        assert builder.use_adaptive_encoder
        assert builder.adaptive_encoder is not None
        assert not builder.temporal_decay.learnable
        assert builder.temporal_decay.min_half_life == 5.0
    
    def test_config_deep_merge(self):
        """Test deep merging of configuration."""
        feature_names = ['feature1']
        config = {
            'temporal_config': {
                'initial_half_life': 60.0
                # Other defaults should be preserved
            }
        }
        
        builder = create_temporal_graph_builder(feature_names, config)
        
        # Custom value should be used
        half_lives = builder.temporal_decay.get_half_lives()
        assert torch.allclose(half_lives, torch.tensor(60.0), atol=5.0)
        
        # Defaults should be preserved
        assert builder.temporal_decay.min_half_life == 1.0
        assert builder.temporal_decay.max_half_life == 365.0
        assert builder.temporal_decay.learnable


class TestTemporalGraphBuilderEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def feature_names(self):
        return ['feature1', 'feature2']
    
    @pytest.fixture
    def graph_builder(self, feature_names):
        return TemporalGraphBuilder(feature_names)
    
    def test_empty_match_data(self, graph_builder):
        """Test handling of empty match data."""
        empty_graph = nx.Graph()
        empty_data = pd.DataFrame()
        
        result_graph = graph_builder.add_temporal_edges(empty_graph, empty_data)
        
        # Should return original graph unchanged
        assert len(result_graph.edges()) == 0
        assert len(result_graph.nodes()) == 0
    
    def test_malformed_match_data(self, graph_builder):
        """Test handling of malformed match data."""
        graph = nx.Graph()
        graph.add_node('Player1', node_type='player')
        
        # Data with missing columns
        bad_data = pd.DataFrame({
            'match_id': ['match1'],
            'some_other_column': ['value']
        })
        
        # Should not crash
        result_graph = graph_builder.add_temporal_edges(graph, bad_data)
        assert isinstance(result_graph, nx.Graph)
    
    def test_missing_player_nodes(self, graph_builder):
        """Test handling when player nodes don't exist in graph."""
        graph = nx.Graph()  # Empty graph
        
        match_data = pd.DataFrame({
            'match_id': ['match1'],
            'match_date': [datetime.now()],
            'batter': ['NonexistentPlayer'],
            'bowler': ['NonexistentBowler'],
            'runs_scored': [4]
        })
        
        # Should not crash
        result_graph = graph_builder.add_temporal_edges(graph, match_data)
        assert isinstance(result_graph, nx.Graph)
    
    def test_invalid_dates(self, graph_builder):
        """Test handling of invalid dates."""
        graph = nx.Graph()
        
        match_data = pd.DataFrame({
            'match_id': ['match1'],
            'match_date': [None],  # Invalid date
            'batter': ['Player1'],
            'runs_scored': [1]
        })
        
        # Should handle gracefully
        result_graph = graph_builder.add_temporal_edges(graph, match_data)
        assert isinstance(result_graph, nx.Graph)
    
    def test_extreme_temporal_values(self, graph_builder):
        """Test handling of extreme temporal values."""
        # Very old date
        old_date = datetime.now() - timedelta(days=10000)
        
        match_data = pd.DataFrame({
            'match_id': ['match1'],
            'match_date': [old_date],
            'batter': ['Player1'],
            'runs_scored': [1]
        })
        
        graph = nx.Graph()
        graph.add_node('Player1', node_type='player')
        
        result_graph = graph_builder.add_temporal_edges(graph, match_data)
        
        # Should handle extreme values without numerical issues
        assert isinstance(result_graph, nx.Graph)
        
        # Check that weights are still finite
        for u, v, data in result_graph.edges(data=True):
            if 'weight' in data:
                assert np.isfinite(data['weight'])
                assert data['weight'] >= 0
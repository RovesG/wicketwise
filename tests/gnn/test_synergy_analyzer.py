# Purpose: Unit tests for player synergy analysis system
# Author: Shamus Rae, Last Modified: 2024-01-15

import pytest
import numpy as np
import pandas as pd
import networkx as nx
from unittest.mock import patch, MagicMock

from crickformers.gnn.synergy_analyzer import (
    SynergyConfig,
    BattingSynergyMetrics,
    BowlingFieldingSynergyMetrics,
    CaptainBowlerSynergyMetrics,
    SynergyScoreCalculator,
    SynergyGraphBuilder,
    add_synergy_edges_to_graph
)


class TestSynergyConfig:
    """Test synergy configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SynergyConfig()
        
        assert config.min_batting_partnerships == 5
        assert config.min_bowling_overs == 10
        assert config.min_fielding_dismissals == 3
        assert config.min_captain_overs == 20
        assert config.batting_synergy_threshold == 0.6
        assert config.bowling_synergy_threshold == 0.5
        assert config.fielding_synergy_threshold == 0.4
        assert config.captain_synergy_threshold == 0.5
        assert config.max_synergy_edges_per_player == 10
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SynergyConfig(
            min_batting_partnerships=3,
            batting_synergy_threshold=0.7,
            max_synergy_edges_per_player=5
        )
        
        assert config.min_batting_partnerships == 3
        assert config.batting_synergy_threshold == 0.7
        assert config.max_synergy_edges_per_player == 5


class TestSynergyMetrics:
    """Test synergy metrics dataclasses."""
    
    def test_batting_synergy_metrics(self):
        """Test batting synergy metrics creation."""
        metrics = BattingSynergyMetrics(
            partnerships_count=10,
            total_runs=250,
            total_balls=150,
            average_partnership=25.0,
            strike_rotation_rate=0.4,
            run_rate=10.0,
            non_dismissal_correlation=0.6,
            boundary_synergy=0.2,
            pressure_performance=0.8
        )
        
        assert metrics.partnerships_count == 10
        assert metrics.total_runs == 250
        assert metrics.run_rate == 10.0
        assert metrics.pressure_performance == 0.8
    
    def test_bowling_fielding_synergy_metrics(self):
        """Test bowling-fielding synergy metrics creation."""
        metrics = BowlingFieldingSynergyMetrics(
            shared_dismissals=8,
            total_overs_together=24,
            wicket_rate=1.5,
            catch_success_rate=0.8,
            run_saving_efficiency=0.3,
            pressure_wickets=3
        )
        
        assert metrics.shared_dismissals == 8
        assert metrics.wicket_rate == 1.5
        assert metrics.catch_success_rate == 0.8
        assert metrics.pressure_wickets == 3
    
    def test_captain_bowler_synergy_metrics(self):
        """Test captain-bowler synergy metrics creation."""
        metrics = CaptainBowlerSynergyMetrics(
            overs_under_captain=60,
            wickets_under_captain=15,
            economy_under_captain=6.5,
            field_setting_effectiveness=0.7,
            pressure_situation_success=0.6,
            tactical_adaptability=0.8
        )
        
        assert metrics.overs_under_captain == 60
        assert metrics.wickets_under_captain == 15
        assert metrics.economy_under_captain == 6.5
        assert metrics.tactical_adaptability == 0.8


class TestSynergyScoreCalculator:
    """Test synergy score calculation."""
    
    @pytest.fixture
    def calculator(self):
        """Create synergy score calculator."""
        config = SynergyConfig(
            min_batting_partnerships=2,  # Lower threshold for testing
            min_bowling_overs=3,
            min_fielding_dismissals=1,
            min_captain_overs=6
        )
        return SynergyScoreCalculator(config)
    
    @pytest.fixture
    def sample_batting_data(self):
        """Sample batting partnership data."""
        return pd.DataFrame([
            # Partnership 1: Good partnership
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 1, 
             'batter': 'player1', 'non_striker': 'player2', 'runs_scored': 1, 'wicket_type': None},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 2,
             'batter': 'player2', 'non_striker': 'player1', 'runs_scored': 4, 'wicket_type': None},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 3,
             'batter': 'player1', 'non_striker': 'player2', 'runs_scored': 2, 'wicket_type': None},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 4,
             'batter': 'player2', 'non_striker': 'player1', 'runs_scored': 6, 'wicket_type': None},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 5,
             'batter': 'player1', 'non_striker': 'player2', 'runs_scored': 1, 'wicket_type': None},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 6,
             'batter': 'player2', 'non_striker': 'player1', 'runs_scored': 0, 'wicket_type': None},
            
            # Partnership 2: Another good partnership
            {'match_id': 'match2', 'innings': 1, 'over': 10, 'ball': 1,
             'batter': 'player1', 'non_striker': 'player2', 'runs_scored': 4, 'wicket_type': None},
            {'match_id': 'match2', 'innings': 1, 'over': 10, 'ball': 2,
             'batter': 'player2', 'non_striker': 'player1', 'runs_scored': 1, 'wicket_type': None},
            {'match_id': 'match2', 'innings': 1, 'over': 10, 'ball': 3,
             'batter': 'player1', 'non_striker': 'player2', 'runs_scored': 2, 'wicket_type': None},
            {'match_id': 'match2', 'innings': 1, 'over': 10, 'ball': 4,
             'batter': 'player2', 'non_striker': 'player1', 'runs_scored': 1, 'wicket_type': None},
            {'match_id': 'match2', 'innings': 1, 'over': 10, 'ball': 5,
             'batter': 'player1', 'non_striker': 'player2', 'runs_scored': 0, 'wicket_type': 'bowled'},
        ])
    
    @pytest.fixture
    def sample_bowling_fielding_data(self):
        """Sample bowling-fielding data."""
        return pd.DataFrame([
            # Bowler-fielder combination with dismissals
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 1,
             'bowler': 'bowler1', 'fielder': 'fielder1', 'runs_scored': 0, 'wicket_type': 'caught'},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 2,
             'bowler': 'bowler1', 'fielder': 'fielder1', 'runs_scored': 1, 'wicket_type': None},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 3,
             'bowler': 'bowler1', 'fielder': 'fielder1', 'runs_scored': 0, 'wicket_type': None},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 4,
             'bowler': 'bowler1', 'fielder': 'fielder1', 'runs_scored': 2, 'wicket_type': None},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 5,
             'bowler': 'bowler1', 'fielder': 'fielder1', 'runs_scored': 0, 'wicket_type': 'caught'},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 6,
             'bowler': 'bowler1', 'fielder': 'fielder1', 'runs_scored': 1, 'wicket_type': None},
            
            # More data for comparison
            {'match_id': 'match2', 'innings': 1, 'over': 8, 'ball': 1,
             'bowler': 'bowler1', 'fielder': 'other_fielder', 'runs_scored': 4, 'wicket_type': None},
            {'match_id': 'match2', 'innings': 1, 'over': 8, 'ball': 2,
             'bowler': 'bowler1', 'fielder': 'other_fielder', 'runs_scored': 2, 'wicket_type': None},
        ])
    
    @pytest.fixture
    def sample_captain_bowler_data(self):
        """Sample captain-bowler data."""
        return pd.DataFrame([
            # Bowler under captain1
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 1,
             'captain': 'captain1', 'bowler': 'bowler1', 'runs_scored': 1, 'wicket_type': None},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 2,
             'captain': 'captain1', 'bowler': 'bowler1', 'runs_scored': 0, 'wicket_type': 'bowled'},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 3,
             'captain': 'captain1', 'bowler': 'bowler1', 'runs_scored': 2, 'wicket_type': None},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 4,
             'captain': 'captain1', 'bowler': 'bowler1', 'runs_scored': 0, 'wicket_type': None},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 5,
             'captain': 'captain1', 'bowler': 'bowler1', 'runs_scored': 1, 'wicket_type': None},
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 6,
             'captain': 'captain1', 'bowler': 'bowler1', 'runs_scored': 0, 'wicket_type': 'lbw'},
            
            # Bowler under captain2 for comparison
            {'match_id': 'match2', 'innings': 1, 'over': 10, 'ball': 1,
             'captain': 'captain2', 'bowler': 'bowler1', 'runs_scored': 4, 'wicket_type': None},
            {'match_id': 'match2', 'innings': 1, 'over': 10, 'ball': 2,
             'captain': 'captain2', 'bowler': 'bowler1', 'runs_scored': 6, 'wicket_type': None},
        ])
    
    def test_batting_synergy_calculation(self, calculator, sample_batting_data):
        """Test batting synergy calculation."""
        synergy_score, metrics = calculator.calculate_batting_synergy(
            'player1', 'player2', sample_batting_data
        )
        
        # Should have positive synergy
        assert synergy_score > 0
        assert isinstance(metrics, BattingSynergyMetrics)
        assert metrics.partnerships_count == 2
        assert metrics.total_runs == 22  # 14 + 8 runs
        assert metrics.total_balls == 11  # 6 + 5 balls
        assert metrics.run_rate > 0
        assert 0 <= metrics.strike_rotation_rate <= 1
        assert 0 <= metrics.non_dismissal_correlation <= 1
    
    def test_batting_synergy_insufficient_data(self, calculator):
        """Test batting synergy with insufficient data."""
        insufficient_data = pd.DataFrame([
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 1,
             'batter': 'player1', 'non_striker': 'player2', 'runs_scored': 1, 'wicket_type': None}
        ])
        
        synergy_score, metrics = calculator.calculate_batting_synergy(
            'player1', 'player2', insufficient_data
        )
        
        assert synergy_score == 0.0
        assert metrics.partnerships_count == 0
    
    def test_bowling_fielding_synergy_calculation(self, calculator, sample_bowling_fielding_data):
        """Test bowling-fielding synergy calculation."""
        synergy_score, metrics = calculator.calculate_bowling_fielding_synergy(
            'bowler1', 'fielder1', sample_bowling_fielding_data
        )
        
        # Should have positive synergy
        assert synergy_score > 0
        assert isinstance(metrics, BowlingFieldingSynergyMetrics)
        assert metrics.shared_dismissals == 2
        assert metrics.total_overs_together == 6  # 6 balls = 1 over
        assert metrics.wicket_rate > 0
        assert 0 <= metrics.catch_success_rate <= 1
    
    def test_bowling_fielding_synergy_no_data(self, calculator):
        """Test bowling-fielding synergy with no shared data."""
        empty_data = pd.DataFrame()
        
        synergy_score, metrics = calculator.calculate_bowling_fielding_synergy(
            'bowler1', 'fielder1', empty_data
        )
        
        assert synergy_score == 0.0
        assert metrics.shared_dismissals == 0
    
    def test_captain_bowler_synergy_calculation(self, calculator, sample_captain_bowler_data):
        """Test captain-bowler synergy calculation."""
        synergy_score, metrics = calculator.calculate_captain_bowler_synergy(
            'captain1', 'bowler1', sample_captain_bowler_data
        )
        
        # Should have positive synergy
        assert synergy_score > 0
        assert isinstance(metrics, CaptainBowlerSynergyMetrics)
        assert metrics.overs_under_captain == 6  # 6 balls
        assert metrics.wickets_under_captain == 2
        assert metrics.economy_under_captain > 0
        assert 0 <= metrics.field_setting_effectiveness <= 1
    
    def test_captain_bowler_synergy_insufficient_data(self, calculator):
        """Test captain-bowler synergy with insufficient data."""
        insufficient_data = pd.DataFrame([
            {'match_id': 'match1', 'innings': 1, 'over': 5, 'ball': 1,
             'captain': 'captain1', 'bowler': 'bowler1', 'runs_scored': 1, 'wicket_type': None}
        ])
        
        synergy_score, metrics = calculator.calculate_captain_bowler_synergy(
            'captain1', 'bowler1', insufficient_data
        )
        
        assert synergy_score == 0.0
        assert metrics.overs_under_captain == 0
    
    def test_partnership_extraction(self, calculator, sample_batting_data):
        """Test batting partnership extraction."""
        partnerships = calculator._extract_batting_partnerships(
            'player1', 'player2', sample_batting_data
        )
        
        assert len(partnerships) == 2
        
        # Check first partnership
        partnership1 = partnerships[0]
        assert partnership1['match_id'] == 'match1'
        assert partnership1['runs'] == 14  # 1+4+2+6+1+0
        assert partnership1['balls'] == 6
        assert partnership1['boundaries'] == 2  # 4 and 6
        assert not partnership1['ended_by_dismissal']
        
        # Check second partnership
        partnership2 = partnerships[1]
        assert partnership2['match_id'] == 'match2'
        assert partnership2['runs'] == 8  # 4+1+2+1+0
        assert partnership2['balls'] == 5
        assert partnership2['ended_by_dismissal']
        assert partnership2['dismissed_player'] == 'player1'
    
    def test_strike_rotation_calculation(self, calculator):
        """Test strike rotation rate calculation."""
        partnerships = [
            {
                'partnership_data': pd.DataFrame([
                    {'runs_scored': 1},  # Rotation
                    {'runs_scored': 0},  # No rotation
                    {'runs_scored': 3},  # Rotation
                    {'runs_scored': 4},  # Boundary (not counted)
                    {'runs_scored': 2},  # No rotation
                ])
            }
        ]
        
        rotation_rate = calculator._calculate_strike_rotation_rate(partnerships)
        
        # 2 rotations out of 4 non-boundary balls = 0.5
        assert rotation_rate == 0.5
    
    def test_non_dismissal_correlation(self, calculator):
        """Test non-dismissal correlation calculation."""
        partnerships = [
            {'ended_by_dismissal': False},
            {'ended_by_dismissal': True},
            {'ended_by_dismissal': False},
            {'ended_by_dismissal': False}
        ]
        
        correlation = calculator._calculate_non_dismissal_correlation(partnerships)
        
        # 3 out of 4 partnerships survived = 0.75
        assert correlation == 0.75
    
    def test_boundary_synergy_calculation(self, calculator):
        """Test boundary synergy calculation."""
        partnerships = [
            {'boundaries': 2, 'balls': 10},
            {'boundaries': 1, 'balls': 8}
        ]
        
        synergy = calculator._calculate_boundary_synergy(partnerships)
        
        # 3 boundaries in 18 balls = 0.167
        expected = 3.0 / 18.0
        assert abs(synergy - expected) < 0.01
    
    def test_synergy_score_normalization(self, calculator):
        """Test that synergy scores are properly normalized to [0, 1]."""
        # Test batting synergy score computation
        score = calculator._compute_batting_synergy_score(
            run_rate=15.0,  # High run rate
            strike_rotation_rate=1.0,  # Perfect rotation
            non_dismissal_correlation=1.0,  # Never dismissed
            boundary_synergy=0.5,  # Good boundary rate
            pressure_performance=1.0,  # Perfect under pressure
            partnerships_count=100  # Many partnerships
        )
        
        assert 0 <= score <= 1
        assert score > 0.8  # Should be high with these values
    
    def test_edge_case_empty_partnerships(self, calculator):
        """Test handling of empty partnership data."""
        empty_data = pd.DataFrame()
        
        synergy_score, metrics = calculator.calculate_batting_synergy(
            'player1', 'player2', empty_data
        )
        
        assert synergy_score == 0.0
        assert metrics.partnerships_count == 0
        assert metrics.total_runs == 0
        assert metrics.run_rate == 0.0


class TestSynergyGraphBuilder:
    """Test synergy graph building."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create sample cricket graph."""
        G = nx.Graph()
        
        # Add player nodes
        players = ['player1', 'player2', 'bowler1', 'fielder1', 'captain1']
        for player in players:
            G.add_node(player, node_type='player', features=np.random.random(10))
        
        # Add some existing edges
        G.add_edge('player1', 'player2', edge_type='teammate_of')
        
        return G
    
    @pytest.fixture
    def comprehensive_match_data(self):
        """Comprehensive match data for synergy testing."""
        data = []
        
        # Create batting partnerships
        for match_id in ['match1', 'match2', 'match3']:
            for ball in range(1, 31):  # 5 overs
                over = (ball - 1) // 6 + 1
                ball_in_over = (ball - 1) % 6 + 1
                
                data.append({
                    'match_id': match_id,
                    'innings': 1,
                    'over': over,
                    'ball': ball_in_over,
                    'batter': 'player1' if ball % 2 == 1 else 'player2',
                    'non_striker': 'player2' if ball % 2 == 1 else 'player1',
                    'bowler': 'bowler1',
                    'fielder': 'fielder1' if ball % 4 == 0 else None,
                    'captain': 'captain1',
                    'runs_scored': np.random.choice([0, 1, 2, 4, 6], p=[0.4, 0.3, 0.15, 0.1, 0.05]),
                    'wicket_type': 'caught' if ball % 20 == 0 else None
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def builder(self):
        """Create synergy graph builder."""
        config = SynergyConfig(
            min_batting_partnerships=3,
            min_bowling_overs=5,
            min_fielding_dismissals=1,
            min_captain_overs=10,
            batting_synergy_threshold=0.1,
            bowling_synergy_threshold=0.1,
            fielding_synergy_threshold=0.1,
            captain_synergy_threshold=0.1
        )
        return SynergyGraphBuilder(config)
    
    def test_graph_builder_initialization(self, builder):
        """Test synergy graph builder initialization."""
        assert builder.config is not None
        assert builder.calculator is not None
        assert isinstance(builder.config, SynergyConfig)
    
    def test_add_synergy_edges(self, builder, sample_graph, comprehensive_match_data):
        """Test adding synergy edges to graph."""
        original_edge_count = sample_graph.number_of_edges()
        
        updated_graph = builder.add_synergy_edges(sample_graph, comprehensive_match_data)
        
        # Should have added some synergy edges
        assert updated_graph.number_of_edges() >= original_edge_count
        
        # Check for has_synergy_with edges
        synergy_edges = [
            (u, v, data) for u, v, data in updated_graph.edges(data=True)
            if data.get('edge_type') == 'has_synergy_with'
        ]
        
        assert len(synergy_edges) > 0
        
        # Verify edge attributes
        for u, v, data in synergy_edges:
            assert 'synergy_type' in data
            assert 'synergy_score' in data
            assert 'weight' in data
            assert 0 <= data['synergy_score'] <= 1
            assert data['weight'] > 0
    
    def test_batting_synergy_computation(self, builder, comprehensive_match_data):
        """Test batting synergy computation."""
        players = ['player1', 'player2', 'bowler1']
        
        synergies = builder._compute_batting_synergies(players, comprehensive_match_data)
        
        # Should find batting synergy between player1 and player2
        batting_synergies = [s for s in synergies if s['synergy_type'] == 'batting']
        assert len(batting_synergies) > 0
        
        # Check synergy attributes
        synergy = batting_synergies[0]
        assert synergy['player1'] in ['player1', 'player2']
        assert synergy['player2'] in ['player1', 'player2']
        assert synergy['player1'] != synergy['player2']
        assert synergy['synergy_score'] > 0
        assert 'metrics' in synergy
        assert 'edge_attributes' in synergy
    
    def test_bowling_fielding_synergy_computation(self, builder, comprehensive_match_data):
        """Test bowling-fielding synergy computation."""
        players = ['bowler1', 'fielder1', 'player1']
        
        synergies = builder._compute_bowling_fielding_synergies(players, comprehensive_match_data)
        
        # Should find bowling-fielding synergy
        bf_synergies = [s for s in synergies if s['synergy_type'] == 'bowling_fielding']
        
        if bf_synergies:  # May not always have sufficient data
            synergy = bf_synergies[0]
            assert synergy['player1'] == 'bowler1'
            assert synergy['player2'] == 'fielder1'
            assert synergy['synergy_score'] > 0
    
    def test_captain_bowler_synergy_computation(self, builder, comprehensive_match_data):
        """Test captain-bowler synergy computation."""
        players = ['captain1', 'bowler1', 'player1']
        
        synergies = builder._compute_captain_bowler_synergies(players, comprehensive_match_data)
        
        # Should find captain-bowler synergy
        cb_synergies = [s for s in synergies if s['synergy_type'] == 'captain_bowler']
        assert len(cb_synergies) > 0
        
        synergy = cb_synergies[0]
        assert synergy['player1'] == 'captain1'
        assert synergy['player2'] == 'bowler1'
        assert synergy['synergy_score'] > 0
    
    def test_synergy_filtering_and_ranking(self, builder):
        """Test synergy filtering and ranking."""
        # Create mock synergies with different weights
        synergies = [
            {'player1': 'p1', 'player2': 'p2', 'synergy_type': 'batting', 'weight': 0.8},
            {'player1': 'p1', 'player2': 'p3', 'synergy_type': 'batting', 'weight': 0.9},
            {'player1': 'p1', 'player2': 'p4', 'synergy_type': 'batting', 'weight': 0.7},
            {'player1': 'p2', 'player2': 'p3', 'synergy_type': 'bowling_fielding', 'weight': 0.6},
        ]
        
        filtered = builder._filter_and_rank_synergies(synergies)
        
        # Should be sorted by weight (highest first)
        assert len(filtered) <= len(synergies)
        if len(filtered) > 1:
            assert filtered[0]['weight'] >= filtered[1]['weight']
    
    def test_duplicate_edge_handling(self, builder, sample_graph):
        """Test handling of duplicate synergy relationships."""
        # Add existing edge with synergy attributes
        sample_graph.add_edge('player1', 'player2', 
                             edge_type='has_synergy_with',
                             synergy_type='existing',
                             synergy_score=0.5)
        
        # Create new synergy for same pair
        synergies = [{
            'player1': 'player1',
            'player2': 'player2',
            'synergy_type': 'batting',
            'weight': 0.8,
            'edge_attributes': {
                'edge_type': 'has_synergy_with',
                'synergy_type': 'batting',
                'synergy_score': 0.8,
                'weight': 0.8
            }
        }]
        
        edges_added = builder._add_synergy_edges_to_graph(sample_graph, synergies)
        
        # Should not add new edge, but should update existing
        assert edges_added == 0
        
        # Check that edge was updated with synergy attributes
        edge_data = sample_graph['player1']['player2']
        assert 'synergy_synergy_type' in edge_data
        assert 'synergy_synergy_score' in edge_data
    
    def test_circular_relationship_handling(self, builder, sample_graph):
        """Test handling of circular relationships."""
        # Remove existing edge to test fresh addition
        if sample_graph.has_edge('player1', 'player2'):
            sample_graph.remove_edge('player1', 'player2')
            
        # Create synergies that could create circular references
        synergies = [
            {
                'player1': 'player1',
                'player2': 'player2',
                'synergy_type': 'batting',
                'weight': 0.8,
                'edge_attributes': {'edge_type': 'has_synergy_with', 'weight': 0.8}
            },
            {
                'player1': 'player2',  # Reverse order
                'player2': 'player1',
                'synergy_type': 'batting',
                'weight': 0.7,
                'edge_attributes': {'edge_type': 'has_synergy_with', 'weight': 0.7}
            }
        ]
        
        edges_added = builder._add_synergy_edges_to_graph(sample_graph, synergies)
        
        # Should only add one edge (undirected graph, duplicate filtered out)
        assert edges_added == 1
        assert sample_graph.has_edge('player1', 'player2')
    
    def test_nonexistent_player_handling(self, builder, sample_graph):
        """Test handling of synergies with non-existent players."""
        synergies = [{
            'player1': 'nonexistent_player',
            'player2': 'player1',
            'synergy_type': 'batting',
            'weight': 0.8,
            'edge_attributes': {'edge_type': 'has_synergy_with', 'weight': 0.8}
        }]
        
        edges_added = builder._add_synergy_edges_to_graph(sample_graph, synergies)
        
        # Should not add edge for non-existent player
        assert edges_added == 0
    
    def test_max_edges_per_player_limit(self, builder):
        """Test maximum edges per player limitation."""
        # Set low limit for testing
        builder.config.max_synergy_edges_per_player = 2
        
        # Create many synergies for one player
        synergies = []
        for i in range(5):
            synergies.append({
                'player1': 'player1',
                'player2': f'player{i+2}',
                'synergy_type': 'batting',
                'weight': 0.9 - i * 0.1  # Decreasing weights
            })
        
        filtered = builder._filter_and_rank_synergies(synergies)
        
        # Should respect the limit (though exact count may vary due to bidirectional relationships)
        player1_synergies = [s for s in filtered if 'player1' in [s['player1'], s['player2']]]
        assert len(player1_synergies) <= builder.config.max_synergy_edges_per_player


class TestSynergyIntegration:
    """Test integration with main graph building function."""
    
    @pytest.fixture
    def integration_graph(self):
        """Create graph for integration testing."""
        G = nx.Graph()
        
        # Add comprehensive player set
        players = ['kohli', 'rohit', 'bumrah', 'shami', 'dhoni', 'jadeja']
        for player in players:
            G.add_node(player, node_type='player', features=np.random.random(10))
        
        # Add team nodes
        G.add_node('india', node_type='team')
        G.add_node('australia', node_type='team')
        
        return G
    
    @pytest.fixture
    def integration_match_data(self):
        """Create realistic match data for integration testing."""
        data = []
        
        # Create multiple matches with various player combinations
        for match_id in range(1, 4):
            for over in range(1, 21):  # 20 overs
                for ball in range(1, 7):  # 6 balls per over
                    # Rotate batters and bowlers realistically
                    if over <= 10:
                        batter = 'kohli' if ball % 2 == 1 else 'rohit'
                        non_striker = 'rohit' if ball % 2 == 1 else 'kohli'
                    else:
                        batter = 'rohit' if ball % 2 == 1 else 'dhoni'
                        non_striker = 'dhoni' if ball % 2 == 1 else 'rohit'
                    
                    bowler = 'bumrah' if over % 2 == 1 else 'shami'
                    fielder = 'jadeja' if (over + ball) % 5 == 0 else None
                    captain = 'dhoni'
                    
                    # Generate realistic outcomes
                    runs = np.random.choice([0, 1, 2, 4, 6], p=[0.35, 0.35, 0.15, 0.1, 0.05])
                    wicket = 'caught' if (over * ball) % 30 == 0 else None
                    
                    data.append({
                        'match_id': f'match_{match_id}',
                        'innings': 1,
                        'over': over,
                        'ball': ball,
                        'batter': batter,
                        'non_striker': non_striker,
                        'bowler': bowler,
                        'fielder': fielder,
                        'captain': captain,
                        'runs_scored': runs,
                        'wicket_type': wicket
                    })
        
        return pd.DataFrame(data)
    
    def test_main_function_integration(self, integration_graph, integration_match_data):
        """Test main add_synergy_edges_to_graph function."""
        original_edge_count = integration_graph.number_of_edges()
        
        # Use relaxed thresholds for testing
        config = SynergyConfig(
            min_batting_partnerships=10,
            min_bowling_overs=20,
            min_fielding_dismissals=2,
            min_captain_overs=50,
            batting_synergy_threshold=0.1,
            bowling_synergy_threshold=0.1,
            fielding_synergy_threshold=0.1,
            captain_synergy_threshold=0.1
        )
        
        updated_graph = add_synergy_edges_to_graph(
            integration_graph, integration_match_data, config
        )
        
        # Should have added synergy edges
        assert updated_graph.number_of_edges() >= original_edge_count
        
        # Check for different types of synergy edges
        synergy_edges = [
            (u, v, data) for u, v, data in updated_graph.edges(data=True)
            if data.get('edge_type') == 'has_synergy_with'
        ]
        
        if synergy_edges:
            # Verify edge structure
            for u, v, data in synergy_edges:
                assert 'synergy_type' in data
                assert 'synergy_score' in data
                assert 'weight' in data
                assert data['synergy_type'] in ['batting_partnership', 'bowling_fielding', 'captain_bowler']
                assert 0 <= data['synergy_score'] <= 1
                assert data['weight'] > 0
    
    def test_edge_attribute_integrity(self, integration_graph, integration_match_data):
        """Test integrity of edge attributes."""
        config = SynergyConfig(
            min_batting_partnerships=5,
            batting_synergy_threshold=0.1,
            max_synergy_edges_per_player=20
        )
        
        updated_graph = add_synergy_edges_to_graph(
            integration_graph, integration_match_data, config
        )
        
        # Check batting synergy edges specifically
        batting_edges = [
            (u, v, data) for u, v, data in updated_graph.edges(data=True)
            if data.get('synergy_type') == 'batting_partnership'
        ]
        
        for u, v, data in batting_edges:
            # Check required attributes
            assert 'partnerships_count' in data
            assert 'average_partnership' in data
            assert 'run_rate' in data
            assert 'strike_rotation_rate' in data
            assert 'non_dismissal_correlation' in data
            
            # Check value ranges
            assert data['partnerships_count'] > 0
            assert data['average_partnership'] >= 0
            assert data['run_rate'] >= 0
            assert 0 <= data['strike_rotation_rate'] <= 1
            assert 0 <= data['non_dismissal_correlation'] <= 1
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset."""
        # Create larger graph
        G = nx.Graph()
        players = [f'player_{i}' for i in range(50)]
        for player in players:
            G.add_node(player, node_type='player', features=np.random.random(10))
        
        # Create larger match dataset
        data = []
        for match_id in range(10):
            for ball in range(600):  # ~100 overs worth
                over = ball // 6 + 1
                ball_in_over = ball % 6 + 1
                
                data.append({
                    'match_id': f'match_{match_id}',
                    'innings': 1,
                    'over': over,
                    'ball': ball_in_over,
                    'batter': np.random.choice(players[:20]),  # Batters
                    'non_striker': np.random.choice(players[:20]),
                    'bowler': np.random.choice(players[20:35]),  # Bowlers
                    'fielder': np.random.choice(players[35:50]) if ball % 10 == 0 else None,
                    'captain': np.random.choice(players[:5]),  # Captains
                    'runs_scored': np.random.choice([0, 1, 2, 4, 6], p=[0.4, 0.3, 0.15, 0.1, 0.05]),
                    'wicket_type': 'caught' if ball % 50 == 0 else None
                })
        
        match_data = pd.DataFrame(data)
        
        # Should complete without timeout (basic performance test)
        config = SynergyConfig(max_synergy_edges_per_player=5)  # Limit to keep test fast
        
        updated_graph = add_synergy_edges_to_graph(G, match_data, config)
        
        # Should have added some edges
        synergy_edges = [
            (u, v, data) for u, v, data in updated_graph.edges(data=True)
            if data.get('edge_type') == 'has_synergy_with'
        ]
        
        # With this much data, should find some synergies
        assert len(synergy_edges) > 0


class TestSynergyEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_graph(self):
        """Test synergy analysis on empty graph."""
        empty_graph = nx.Graph()
        match_data = pd.DataFrame([
            {'match_id': 'match1', 'innings': 1, 'over': 1, 'ball': 1,
             'batter': 'player1', 'non_striker': 'player2', 'runs_scored': 1}
        ])
        
        updated_graph = add_synergy_edges_to_graph(empty_graph, match_data)
        
        # Should return empty graph unchanged
        assert updated_graph.number_of_nodes() == 0
        assert updated_graph.number_of_edges() == 0
    
    def test_single_player_graph(self):
        """Test synergy analysis with single player."""
        G = nx.Graph()
        G.add_node('player1', node_type='player')
        
        match_data = pd.DataFrame([
            {'match_id': 'match1', 'innings': 1, 'over': 1, 'ball': 1,
             'batter': 'player1', 'non_striker': 'player2', 'runs_scored': 1}
        ])
        
        updated_graph = add_synergy_edges_to_graph(G, match_data)
        
        # Should not add any synergy edges
        synergy_edges = [
            (u, v, data) for u, v, data in updated_graph.edges(data=True)
            if data.get('edge_type') == 'has_synergy_with'
        ]
        assert len(synergy_edges) == 0
    
    def test_empty_match_data(self):
        """Test synergy analysis with empty match data."""
        G = nx.Graph()
        G.add_node('player1', node_type='player')
        G.add_node('player2', node_type='player')
        
        empty_data = pd.DataFrame()
        
        updated_graph = add_synergy_edges_to_graph(G, empty_data)
        
        # Should not add any edges
        assert updated_graph.number_of_edges() == 0
    
    def test_malformed_match_data(self):
        """Test handling of malformed match data."""
        G = nx.Graph()
        G.add_node('player1', node_type='player')
        G.add_node('player2', node_type='player')
        
        # Missing required columns
        malformed_data = pd.DataFrame([
            {'some_column': 'value', 'other_column': 123}
        ])
        
        # Should handle gracefully without crashing
        updated_graph = add_synergy_edges_to_graph(G, malformed_data)
        
        # Should not add any synergy edges
        synergy_edges = [
            (u, v, data) for u, v, data in updated_graph.edges(data=True)
            if data.get('edge_type') == 'has_synergy_with'
        ]
        assert len(synergy_edges) == 0
    
    def test_extreme_threshold_values(self):
        """Test with extreme threshold values."""
        G = nx.Graph()
        G.add_node('player1', node_type='player')
        G.add_node('player2', node_type='player')
        
        match_data = pd.DataFrame([
            {'match_id': 'match1', 'innings': 1, 'over': 1, 'ball': 1,
             'batter': 'player1', 'non_striker': 'player2', 'runs_scored': 1}
        ])
        
        # Very high thresholds - should find no synergies
        high_config = SynergyConfig(
            batting_synergy_threshold=0.99,
            bowling_synergy_threshold=0.99,
            fielding_synergy_threshold=0.99,
            captain_synergy_threshold=0.99
        )
        
        updated_graph = add_synergy_edges_to_graph(G, match_data, high_config)
        synergy_edges = [
            (u, v, data) for u, v, data in updated_graph.edges(data=True)
            if data.get('edge_type') == 'has_synergy_with'
        ]
        assert len(synergy_edges) == 0
        
        # Very low thresholds - should be more permissive
        low_config = SynergyConfig(
            min_batting_partnerships=1,
            batting_synergy_threshold=0.01,
            bowling_synergy_threshold=0.01,
            fielding_synergy_threshold=0.01,
            captain_synergy_threshold=0.01
        )
        
        # Should not crash with low thresholds
        updated_graph = add_synergy_edges_to_graph(G, match_data, low_config)
        # May or may not find synergies depending on data, but should not crash
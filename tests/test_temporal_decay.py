# Purpose: Tests for temporal decay weighting in GNN training
# Author: Shamus Rae, Last Modified: 2024-12-19

"""
Test suite for temporal decay functionality in GNN training.
"""

import pytest
import torch
import networkx as nx
from datetime import datetime, timedelta
import numpy as np

from crickformers.gnn.gnn_trainer import CricketGNNTrainer


class TestTemporalDecayWeighting:
    """Test temporal decay weighting functionality."""
    
    def create_test_graph_with_dates(self, reference_date: datetime) -> nx.DiGraph:
        """Create a test graph with edges at different dates."""
        G = nx.DiGraph()
        
        # Add nodes
        G.add_node("Player1", type="batter")
        G.add_node("Player2", type="bowler")
        G.add_node("Player3", type="batter")
        G.add_node("Team1", type="team")
        G.add_node("Team2", type="team")
        G.add_node("Venue1", type="venue")
        
        # Add edges with different dates relative to reference
        edges_with_dates = [
            # Recent edges (within last 7 days)
            ("Player1", "Player2", {
                "edge_type": "faced",
                "match_date": reference_date - timedelta(days=1),
                "runs": 15,
                "phase": "powerplay"
            }),
            ("Player1", "Team1", {
                "edge_type": "plays_for",
                "match_date": reference_date - timedelta(days=2),
                "runs": 25,
                "phase": "middle_overs"
            }),
            
            # Medium-term edges (1-4 weeks ago)
            ("Player2", "Team2", {
                "edge_type": "plays_for",
                "match_date": reference_date - timedelta(days=14),
                "runs": 8,
                "phase": "death_overs"
            }),
            ("Player3", "Player2", {
                "edge_type": "faced",
                "match_date": reference_date - timedelta(days=21),
                "runs": 12,
                "phase": "powerplay"
            }),
            
            # Old edges (2+ months ago)
            ("Team1", "Venue1", {
                "edge_type": "match_played_at",
                "match_date": reference_date - timedelta(days=60),
                "runs": 180,
                "phase": "powerplay"
            }),
            ("Team2", "Venue1", {
                "edge_type": "match_played_at",
                "match_date": reference_date - timedelta(days=90),
                "runs": 165,
                "phase": "middle_overs"
            })
        ]
        
        for source, target, attrs in edges_with_dates:
            G.add_edge(source, target, **attrs)
        
        return G
    
    def test_basic_temporal_decay_computation(self):
        """Test basic temporal decay weight computation."""
        reference_date = datetime(2024, 12, 19)
        graph = self.create_test_graph_with_dates(reference_date)
        
        trainer = CricketGNNTrainer(
            graph=graph,
            temporal_decay_alpha=0.01,
            reference_date=reference_date
        )
        
        # Should have 6 edges
        assert len(trainer.edge_weights) == 6
        
        # All weights should be positive
        assert torch.all(trainer.edge_weights > 0)
        
        # All weights should be <= 1
        assert torch.all(trainer.edge_weights <= 1)
    
    def test_recent_edges_have_higher_weights(self):
        """Test that recent edges have higher weights than old ones."""
        reference_date = datetime(2024, 12, 19)
        graph = self.create_test_graph_with_dates(reference_date)
        
        trainer = CricketGNNTrainer(
            graph=graph,
            temporal_decay_alpha=0.01,
            reference_date=reference_date
        )
        
        weights = trainer.edge_weights.numpy()
        
        # Get edges and their dates
        edges = list(graph.edges(data=True))
        
        # Find recent and old edges
        recent_indices = []
        old_indices = []
        
        for i, (_, _, attrs) in enumerate(edges):
            match_date = attrs['match_date']
            days_ago = (reference_date - match_date).days
            
            if days_ago <= 7:
                recent_indices.append(i)
            elif days_ago >= 60:
                old_indices.append(i)
        
        # Recent edges should have higher weights
        if recent_indices and old_indices:
            recent_weights = weights[recent_indices]
            old_weights = weights[old_indices]
            
            assert np.mean(recent_weights) > np.mean(old_weights)
            assert np.min(recent_weights) > np.max(old_weights)
    
    def test_different_alpha_values(self):
        """Test different alpha values produce different decay rates."""
        reference_date = datetime(2024, 12, 19)
        graph = self.create_test_graph_with_dates(reference_date)
        
        # Test with low alpha (slow decay)
        trainer_low = CricketGNNTrainer(
            graph=graph,
            temporal_decay_alpha=0.001,
            reference_date=reference_date
        )
        
        # Test with high alpha (fast decay)
        trainer_high = CricketGNNTrainer(
            graph=graph,
            temporal_decay_alpha=0.05,
            reference_date=reference_date
        )
        
        weights_low = trainer_low.edge_weights.numpy()
        weights_high = trainer_high.edge_weights.numpy()
        
        # High alpha should produce more variation in weights
        assert np.std(weights_high) > np.std(weights_low)
        
        # For old edges, high alpha should produce lower weights
        edges = list(graph.edges(data=True))
        old_edge_indices = [
            i for i, (_, _, attrs) in enumerate(edges)
            if (reference_date - attrs['match_date']).days >= 60
        ]
        
        if old_edge_indices:
            old_weights_low = weights_low[old_edge_indices]
            old_weights_high = weights_high[old_edge_indices]
            assert np.mean(old_weights_high) < np.mean(old_weights_low)
    
    def test_edge_weight_ranges(self):
        """Test edge weight ranges for different time periods."""
        reference_date = datetime(2024, 12, 19)
        graph = self.create_test_graph_with_dates(reference_date)
        
        trainer = CricketGNNTrainer(
            graph=graph,
            temporal_decay_alpha=0.01,
            reference_date=reference_date
        )
        
        weights = trainer.edge_weights.numpy()
        edges = list(graph.edges(data=True))
        
        # Categorize edges by age
        same_day_weights = []
        week_old_weights = []
        month_old_weights = []
        very_old_weights = []
        
        for i, (_, _, attrs) in enumerate(edges):
            days_ago = (reference_date - attrs['match_date']).days
            weight = weights[i]
            
            if days_ago == 0:
                same_day_weights.append(weight)
            elif days_ago <= 7:
                week_old_weights.append(weight)
            elif days_ago <= 30:
                month_old_weights.append(weight)
            else:
                very_old_weights.append(weight)
        
        # Same day should have weight close to 1
        if same_day_weights:
            assert all(w > 0.99 for w in same_day_weights)
        
        # Week old should have reasonable weight
        if week_old_weights:
            assert all(0.9 < w <= 1.0 for w in week_old_weights)
        
        # Month old should have noticeably lower weight
        if month_old_weights:
            assert all(0.7 < w <= 0.95 for w in month_old_weights)
        
        # Very old should have much lower weight
        if very_old_weights:
            assert all(w < 0.6 for w in very_old_weights)
    
    def test_temporal_decay_stats(self):
        """Test temporal decay statistics computation."""
        reference_date = datetime(2024, 12, 19)
        graph = self.create_test_graph_with_dates(reference_date)
        
        trainer = CricketGNNTrainer(
            graph=graph,
            temporal_decay_alpha=0.02,
            reference_date=reference_date
        )
        
        stats = trainer.get_temporal_decay_stats()
        
        # Check required keys
        required_keys = ['min_weight', 'max_weight', 'mean_weight', 'std_weight', 'num_edges', 'alpha']
        for key in required_keys:
            assert key in stats
        
        # Check value ranges
        assert 0 < stats['min_weight'] <= stats['max_weight'] <= 1
        assert 0 < stats['mean_weight'] <= 1
        assert stats['std_weight'] >= 0
        assert stats['num_edges'] == 6
        assert stats['alpha'] == 0.02
    
    def test_missing_match_dates(self):
        """Test handling of missing match dates."""
        reference_date = datetime(2024, 12, 19)
        
        # Create graph with some edges missing match dates
        G = nx.DiGraph()
        G.add_node("Player1", type="batter")
        G.add_node("Player2", type="bowler")
        G.add_node("Team1", type="team")
        
        # Edge with match date
        G.add_edge("Player1", "Player2", 
            edge_type="faced",
            match_date=reference_date - timedelta(days=5),
            runs=10
        )
        
        # Edge without match date (use None as placeholder)
        G.add_edge("Player1", "Team1", 
            edge_type="plays_for",
            match_date=None,
            runs=20
        )
        
        trainer = CricketGNNTrainer(
            graph=G,
            temporal_decay_alpha=0.01,
            reference_date=reference_date
        )
        
        # Should handle missing dates gracefully
        assert len(trainer.edge_weights) == 2
        
        # Edge without date should use reference date (weight = 1)
        weights = trainer.edge_weights.numpy()
        assert any(w > 0.99 for w in weights)  # One edge should have weight ~1
    
    def test_invalid_date_formats(self):
        """Test handling of invalid date formats."""
        reference_date = datetime(2024, 12, 19)
        
        G = nx.DiGraph()
        G.add_node("Player1", type="batter")
        G.add_node("Player2", type="bowler")
        
        # Edge with invalid date format
        G.add_edge("Player1", "Player2", 
            edge_type="faced",
            match_date="invalid_date",
            runs=10
        )
        
        trainer = CricketGNNTrainer(
            graph=G,
            temporal_decay_alpha=0.01,
            reference_date=reference_date
        )
        
        # Should handle invalid dates gracefully
        assert len(trainer.edge_weights) == 1
        
        # Invalid date should be treated as reference date
        weight = trainer.edge_weights.item()
        assert weight > 0.99
    
    def test_zero_alpha_behavior(self):
        """Test behavior with zero alpha (no decay)."""
        reference_date = datetime(2024, 12, 19)
        graph = self.create_test_graph_with_dates(reference_date)
        
        trainer = CricketGNNTrainer(
            graph=graph,
            temporal_decay_alpha=0.0,
            reference_date=reference_date
        )
        
        weights = trainer.edge_weights.numpy()
        
        # All weights should be 1 (no decay)
        assert np.allclose(weights, 1.0)
    
    def test_training_with_temporal_decay(self):
        """Test that training works with temporal decay enabled."""
        reference_date = datetime(2024, 12, 19)
        graph = self.create_test_graph_with_dates(reference_date)
        
        trainer = CricketGNNTrainer(
            graph=graph,
            temporal_decay_alpha=0.01,
            reference_date=reference_date,
            model_type="gcn"
        )
        
        # Training should work without errors
        trainer.train(epochs=5)
        
        # Should be able to get embeddings
        embeddings = trainer.get_intermediate_embeddings()
        assert len(embeddings) == trainer.num_layers
    
    def test_gcn_vs_sage_temporal_handling(self):
        """Test temporal decay handling differences between GCN and SAGE."""
        reference_date = datetime(2024, 12, 19)
        graph = self.create_test_graph_with_dates(reference_date)
        
        # Test with GCN (uses edge weights natively)
        trainer_gcn = CricketGNNTrainer(
            graph=graph,
            temporal_decay_alpha=0.01,
            reference_date=reference_date,
            model_type="gcn"
        )
        
        # Test with SAGE (uses weights in loss)
        trainer_sage = CricketGNNTrainer(
            graph=graph,
            temporal_decay_alpha=0.01,
            reference_date=reference_date,
            model_type="sage"
        )
        
        # Both should have same edge weights
        assert torch.allclose(trainer_gcn.edge_weights, trainer_sage.edge_weights)
        
        # Both should train successfully
        trainer_gcn.train(epochs=3)
        trainer_sage.train(epochs=3)
        
        # Both should produce embeddings
        embeddings_gcn = trainer_gcn.get_intermediate_embeddings()
        embeddings_sage = trainer_sage.get_intermediate_embeddings()
        
        assert len(embeddings_gcn) == len(embeddings_sage)
    
    def test_exponential_decay_formula(self):
        """Test that decay formula is correctly applied."""
        reference_date = datetime(2024, 12, 19)
        alpha = 0.01
        
        # Create simple graph with known dates
        G = nx.DiGraph()
        G.add_node("A", type="batter")
        G.add_node("B", type="bowler")
        
        # Add edges with specific days ago
        test_cases = [
            (0, 1.0),      # Same day
            (1, np.exp(-0.01)),  # 1 day ago
            (7, np.exp(-0.07)),  # 1 week ago
            (30, np.exp(-0.30)), # 1 month ago
        ]
        
        for days_ago, expected_weight in test_cases:
            G.clear_edges()
            G.add_edge("A", "B", 
                edge_type="faced",
                match_date=reference_date - timedelta(days=days_ago),
                runs=10
            )
            
            trainer = CricketGNNTrainer(
                graph=G,
                temporal_decay_alpha=alpha,
                reference_date=reference_date
            )
            
            actual_weight = trainer.edge_weights.item()
            assert abs(actual_weight - expected_weight) < 0.0001, \
                f"Days ago: {days_ago}, Expected: {expected_weight}, Actual: {actual_weight}"
    
    def test_large_graph_performance(self):
        """Test temporal decay computation with a larger graph."""
        reference_date = datetime(2024, 12, 19)
        
        # Create larger graph
        G = nx.DiGraph()
        
        # Add many nodes
        for i in range(50):
            G.add_node(f"Player{i}", type="batter")
        
        # Add many edges with random dates
        np.random.seed(42)
        edge_count = 0
        for i in range(100):
            source = f"Player{i % 50}"
            target = f"Player{(i + 17) % 50}"  # Use different offset to avoid duplicates
            
            # Skip self-loops
            if source == target:
                continue
                
            days_ago = np.random.randint(0, 365)
            
            G.add_edge(source, target, 
                edge_type="faced",
                match_date=reference_date - timedelta(days=days_ago),
                runs=np.random.randint(0, 20)
            )
            edge_count += 1
        
        trainer = CricketGNNTrainer(
            graph=G,
            temporal_decay_alpha=0.01,
            reference_date=reference_date
        )
        
        # Should handle large graph efficiently
        actual_edge_count = len(list(G.edges()))
        assert len(trainer.edge_weights) == actual_edge_count
        
        # Should train successfully
        trainer.train(epochs=2)
        
        # Should compute stats
        stats = trainer.get_temporal_decay_stats()
        assert stats['num_edges'] == actual_edge_count
        assert 0 < stats['min_weight'] < stats['max_weight'] <= 1 
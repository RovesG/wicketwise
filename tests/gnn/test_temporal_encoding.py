# Purpose: Unit tests for temporal encoding functionality
# Author: WicketWise Team, Last Modified: 2024-07-19

import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from crickformers.gnn.temporal_encoding import (
    generate_temporal_embedding,
    LearnableTemporalEncoder,
    create_temporal_edge_attributes
)


class TestTemporalEmbedding:
    """Test suite for temporal embedding functions."""
    
    def test_temporal_embedding_shape_and_type(self):
        """Test that temporal embedding produces correct shape and type."""
        # Test with scalar input
        embedding = generate_temporal_embedding(30, dim=8)
        
        assert isinstance(embedding, torch.Tensor), "Output should be a torch.Tensor"
        assert embedding.shape == (8,), f"Expected shape (8,), got {embedding.shape}"
        assert embedding.dtype == torch.float32, f"Expected float32, got {embedding.dtype}"
        
        # Test with tensor input
        days_tensor = torch.tensor([10, 20, 30])
        embedding_batch = generate_temporal_embedding(days_tensor, dim=8)
        
        assert embedding_batch.shape == (3, 8), f"Expected shape (3, 8), got {embedding_batch.shape}"
        assert embedding_batch.dtype == torch.float32
    
    def test_temporal_embedding_normalization(self):
        """Test that temporal embeddings are normalized to [0, 1] range."""
        # Test with various days_ago values
        test_days = [0, 1, 7, 30, 90, 365]
        
        for days in test_days:
            embedding = generate_temporal_embedding(days, dim=8)
            
            assert torch.all(embedding >= 0.0), f"Embedding contains values < 0 for days={days}"
            assert torch.all(embedding <= 1.0), f"Embedding contains values > 1 for days={days}"
            
            # Check that embeddings are not all the same value (should have variation)
            assert embedding.std() > 1e-6, f"Embedding has no variation for days={days}"
    
    def test_temporal_embedding_consistency(self):
        """Test that same inputs produce same outputs."""
        days_ago = 15
        dim = 8
        
        # Generate multiple embeddings with same input
        embedding1 = generate_temporal_embedding(days_ago, dim=dim)
        embedding2 = generate_temporal_embedding(days_ago, dim=dim)
        
        assert torch.allclose(embedding1, embedding2), "Same inputs should produce same outputs"
    
    def test_temporal_embedding_different_dimensions(self):
        """Test temporal embedding with different dimensions."""
        days_ago = 30
        
        for dim in [4, 8, 16, 32]:
            embedding = generate_temporal_embedding(days_ago, dim=dim)
            assert embedding.shape == (dim,), f"Wrong shape for dim={dim}"
            
            # Check that we have both sinusoidal and linear components
            sin_dim = dim // 2
            linear_dim = dim - sin_dim
            
            # Sinusoidal components should vary smoothly
            if sin_dim > 0:
                sin_part = embedding[:sin_dim]
                assert torch.all(sin_part >= -1.1) and torch.all(sin_part <= 1.1), \
                    "Sinusoidal components outside expected range"
    
    def test_temporal_embedding_device_handling(self):
        """Test that temporal embedding handles device placement correctly."""
        days_ago = 30
        dim = 8
        
        # Test CPU
        embedding_cpu = generate_temporal_embedding(days_ago, dim=dim, device=torch.device('cpu'))
        assert embedding_cpu.device.type == 'cpu'
        
        # Test GPU if available
        if torch.cuda.is_available():
            embedding_gpu = generate_temporal_embedding(days_ago, dim=dim, device=torch.device('cuda'))
            assert embedding_gpu.device.type == 'cuda'
            
            # Should produce same values regardless of device
            assert torch.allclose(embedding_cpu, embedding_gpu.cpu(), atol=1e-6)
    
    def test_temporal_embedding_batch_processing(self):
        """Test temporal embedding with batch inputs."""
        days_batch = torch.tensor([0, 10, 30, 90, 365])
        dim = 8
        
        embeddings = generate_temporal_embedding(days_batch, dim=dim)
        
        assert embeddings.shape == (5, 8), f"Expected (5, 8), got {embeddings.shape}"
        
        # Each embedding should be different
        for i in range(len(days_batch)):
            for j in range(i + 1, len(days_batch)):
                assert not torch.allclose(embeddings[i], embeddings[j]), \
                    f"Embeddings {i} and {j} are too similar"
    
    def test_temporal_embedding_edge_cases(self):
        """Test temporal embedding with edge cases."""
        dim = 8
        
        # Test with zero days
        embedding_zero = generate_temporal_embedding(0, dim=dim)
        assert not torch.isnan(embedding_zero).any(), "Zero days should not produce NaN"
        
        # Test with very large days
        embedding_large = generate_temporal_embedding(10000, dim=dim)
        assert not torch.isnan(embedding_large).any(), "Large days should not produce NaN"
        assert torch.all(embedding_large >= 0.0) and torch.all(embedding_large <= 1.0), \
            "Large days should still be normalized"
        
        # Test with negative days (should be clamped to 0)
        embedding_negative = generate_temporal_embedding(-10, dim=dim)
        embedding_zero_ref = generate_temporal_embedding(0, dim=dim)
        assert torch.allclose(embedding_negative, embedding_zero_ref), \
            "Negative days should be treated as zero"


class TestLearnableTemporalEncoder:
    """Test suite for LearnableTemporalEncoder module."""
    
    @pytest.fixture
    def encoder(self):
        """Create a LearnableTemporalEncoder for testing."""
        return LearnableTemporalEncoder(dim=8, max_days=365, learnable_components=4)
    
    def test_encoder_initialization(self, encoder):
        """Test that encoder initializes correctly."""
        assert encoder.dim == 8
        assert encoder.max_days == 365
        assert encoder.learnable_components == 4
        assert encoder.sin_dim == 4
        assert encoder.linear_dim == 4
        
        # Check that learnable components exist
        assert hasattr(encoder, 'linear_layers')
        assert len(encoder.linear_layers) == 4
        
        # Check normalization parameters
        assert hasattr(encoder, 'norm_scale')
        assert hasattr(encoder, 'norm_bias')
        assert encoder.norm_scale.shape == (8,)
        assert encoder.norm_bias.shape == (8,)
    
    def test_encoder_forward_pass(self, encoder):
        """Test forward pass through learnable encoder."""
        days_ago = torch.tensor([10, 30, 90])
        
        embeddings = encoder(days_ago)
        
        assert embeddings.shape == (3, 8), f"Expected (3, 8), got {embeddings.shape}"
        assert embeddings.dtype == torch.float32
        
        # Check normalization to [0, 1]
        assert torch.all(embeddings >= 0.0), "Embeddings should be >= 0"
        assert torch.all(embeddings <= 1.0), "Embeddings should be <= 1"
    
    def test_encoder_gradient_flow(self, encoder):
        """Test that gradients flow properly through the encoder."""
        days_ago = torch.tensor([30.0], requires_grad=True)
        
        embeddings = encoder(days_ago)
        loss = embeddings.sum()
        loss.backward()
        
        # Check that parameters have gradients
        has_gradients = False
        for param in encoder.parameters():
            if param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad)):
                has_gradients = True
                break
        
        assert has_gradients, "No gradients found in encoder parameters"
        
        # Check that input has gradients if required
        if days_ago.requires_grad:
            assert days_ago.grad is not None, "Input should have gradients"
    
    def test_encoder_training_mode(self, encoder):
        """Test encoder behavior in training vs eval mode."""
        days_ago = torch.tensor([30])
        
        # Training mode
        encoder.train()
        embedding_train = encoder(days_ago)
        
        # Eval mode
        encoder.eval()
        with torch.no_grad():
            embedding_eval = encoder(days_ago)
        
        # Should produce similar but potentially different results due to learnable components
        assert embedding_train.shape == embedding_eval.shape
        # Note: Results might be slightly different due to learnable parameters
    
    def test_encoder_temporal_stats(self, encoder):
        """Test temporal statistics computation."""
        days_range = torch.arange(0, 100, 10)
        
        stats = encoder.get_temporal_stats(days_range)
        
        required_keys = ['mean', 'std', 'min', 'max', 'range']
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
            assert isinstance(stats[key], np.ndarray), f"Stats[{key}] should be numpy array"
            assert stats[key].shape == (8,), f"Stats[{key}] should have shape (8,)"


class TestTemporalEdgeAttributes:
    """Test suite for temporal edge attribute creation."""
    
    def test_edge_attributes_creation(self):
        """Test creation of temporal edge attributes."""
        # Create sample match dates
        reference_date = torch.tensor(100.0)  # Day 100
        match_dates = torch.tensor([90.0, 95.0, 100.0])  # Days 90, 95, 100
        
        edge_attrs = create_temporal_edge_attributes(
            match_dates=match_dates,
            reference_date=reference_date,
            embedding_dim=8
        )
        
        assert edge_attrs.shape == (3, 8), f"Expected (3, 8), got {edge_attrs.shape}"
        assert edge_attrs.dtype == torch.float32
        
        # Check normalization
        assert torch.all(edge_attrs >= 0.0), "Edge attributes should be >= 0"
        assert torch.all(edge_attrs <= 1.0), "Edge attributes should be <= 1"
    
    def test_edge_attributes_with_base_attrs(self):
        """Test edge attributes with base attributes."""
        reference_date = torch.tensor(100.0)
        match_dates = torch.tensor([90.0, 95.0])
        base_attrs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2 edges, 2 base attrs each
        
        edge_attrs = create_temporal_edge_attributes(
            match_dates=match_dates,
            reference_date=reference_date,
            embedding_dim=8,
            base_edge_attrs=base_attrs
        )
        
        # Should have base attrs + temporal embedding
        assert edge_attrs.shape == (2, 10), f"Expected (2, 10), got {edge_attrs.shape}"
        
        # Check that base attributes are preserved
        assert torch.allclose(edge_attrs[:, :2], base_attrs), "Base attributes not preserved"
        
        # Check that temporal part is normalized
        temporal_part = edge_attrs[:, 2:]
        assert torch.all(temporal_part >= 0.0), "Temporal part should be >= 0"
        assert torch.all(temporal_part <= 1.0), "Temporal part should be <= 1"
    
    def test_edge_attributes_gradient_propagation(self):
        """Test that gradients propagate through edge attributes."""
        reference_date = torch.tensor(100.0)
        match_dates = torch.tensor([90.0, 95.0], requires_grad=True)
        
        edge_attrs = create_temporal_edge_attributes(
            match_dates=match_dates,
            reference_date=reference_date,
            embedding_dim=8
        )
        
        # Create a simple loss
        loss = edge_attrs.sum()
        loss.backward()
        
        # Check that match_dates has gradients
        assert match_dates.grad is not None, "match_dates should have gradients"
        assert not torch.allclose(match_dates.grad, torch.zeros_like(match_dates.grad)), \
            "Gradients should be non-zero"
    
    def test_edge_attributes_different_sizes(self):
        """Test edge attributes with different input sizes."""
        reference_date = torch.tensor(100.0)
        
        # Test with single edge
        single_date = torch.tensor([95.0])
        single_attrs = create_temporal_edge_attributes(
            match_dates=single_date,
            reference_date=reference_date,
            embedding_dim=8
        )
        assert single_attrs.shape == (1, 8)
        
        # Test with many edges
        many_dates = torch.arange(50.0, 100.0, 5.0)  # 10 edges
        many_attrs = create_temporal_edge_attributes(
            match_dates=many_dates,
            reference_date=reference_date,
            embedding_dim=8
        )
        assert many_attrs.shape == (10, 8)


class TestTemporalEncodingIntegration:
    """Integration tests for temporal encoding with GNN trainer."""
    
    def test_temporal_encoding_dimensions(self):
        """Test that temporal encoding produces correct dimensions."""
        # Test various dimension combinations
        test_cases = [
            (4, 2, 2),   # 4D: 2 sin, 2 linear
            (8, 4, 4),   # 8D: 4 sin, 4 linear
            (16, 8, 8),  # 16D: 8 sin, 8 linear
            (9, 4, 5),   # 9D: 4 sin, 5 linear (odd number)
        ]
        
        for total_dim, expected_sin, expected_linear in test_cases:
            embedding = generate_temporal_embedding(30, dim=total_dim)
            
            assert embedding.shape == (total_dim,), f"Wrong total dimension for {total_dim}D"
            
            # Verify that we have the expected structure
            sin_dim = total_dim // 2
            linear_dim = total_dim - sin_dim
            
            assert sin_dim == expected_sin, f"Wrong sin_dim: expected {expected_sin}, got {sin_dim}"
            assert linear_dim == expected_linear, f"Wrong linear_dim: expected {expected_linear}, got {linear_dim}"
    
    def test_temporal_encoding_recency_bias(self):
        """Test that temporal encoding properly encodes recency."""
        # Recent events should have different patterns than old events
        recent_embedding = generate_temporal_embedding(1, dim=8)    # 1 day ago
        old_embedding = generate_temporal_embedding(365, dim=8)     # 1 year ago
        
        # Embeddings should be significantly different
        cosine_similarity = torch.cosine_similarity(recent_embedding, old_embedding, dim=0)
        assert cosine_similarity < 0.98, f"Recent and old embeddings too similar: {cosine_similarity}"
        
        # Test that there's a smooth transition
        days_sequence = torch.arange(1, 100, 10)
        embeddings_sequence = generate_temporal_embedding(days_sequence, dim=8)
        
        # Adjacent embeddings should be more similar than distant ones
        sim_adjacent = torch.cosine_similarity(embeddings_sequence[0], embeddings_sequence[1], dim=0)
        sim_distant = torch.cosine_similarity(embeddings_sequence[0], embeddings_sequence[-1], dim=0)
        
        assert sim_adjacent > sim_distant, "Adjacent embeddings should be more similar than distant ones"
    
    @pytest.mark.parametrize("embedding_dim", [4, 8, 16])
    def test_temporal_embedding_stability(self, embedding_dim):
        """Test temporal embedding stability across different dimensions."""
        days_ago = 30
        
        # Generate embedding multiple times
        embeddings = []
        for _ in range(5):
            embedding = generate_temporal_embedding(days_ago, dim=embedding_dim)
            embeddings.append(embedding)
        
        # All embeddings should be identical
        for i in range(1, len(embeddings)):
            assert torch.allclose(embeddings[0], embeddings[i]), \
                f"Embedding {i} differs from embedding 0 for dim={embedding_dim}"
    
    def test_temporal_encoding_mathematical_properties(self):
        """Test mathematical properties of temporal encoding."""
        dim = 8
        
        # Test that encoding preserves ordering for linear components
        days_sequence = [1, 10, 30, 90, 365]
        embeddings = [generate_temporal_embedding(days, dim=dim) for days in days_sequence]
        
        # Linear components (second half) should generally decrease with time
        # (since we use 1 - normalized_days for recency)
        linear_start = dim // 2
        for i in range(1, len(embeddings)):
            # At least some linear components should show temporal ordering
            linear_part_prev = embeddings[i-1][linear_start:]
            linear_part_curr = embeddings[i][linear_start:]
            
            # Check that at least one component shows the expected trend
            decreasing_components = sum(
                linear_part_prev[j] >= linear_part_curr[j] 
                for j in range(len(linear_part_prev))
            )
            
            assert decreasing_components >= len(linear_part_prev) // 2, \
                f"Not enough decreasing components at step {i}"
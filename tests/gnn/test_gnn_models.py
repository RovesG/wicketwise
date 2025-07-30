# Purpose: Unit tests for advanced GNN models
# Author: WicketWise Team, Last Modified: 2024-07-19

import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from crickformers.gnn.gnn_models import MultiHopGATv2


class TestMultiHopGATv2:
    """Test suite for MultiHopGATv2 model."""
    
    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph data for testing."""
        # Create a small graph with 10 nodes and 20 edges
        num_nodes = 10
        num_edges = 20
        
        # Node features: 9D as specified
        x = torch.randn(num_nodes, 9)
        
        # Random edge connectivity
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Optional edge attributes
        edge_attr = torch.randn(num_edges, 3)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    @pytest.fixture
    def model(self):
        """Create a MultiHopGATv2 model with default parameters."""
        return MultiHopGATv2(
            in_channels=9,
            hidden_channels=64,
            out_channels=128,
            num_layers=3,
            heads=4,
            dropout=0.1
        )
    
    def test_model_initialization(self):
        """Test that the model initializes correctly with specified parameters."""
        model = MultiHopGATv2(
            in_channels=9,
            hidden_channels=64,
            out_channels=128,
            num_layers=3,
            heads=4,
            dropout=0.1
        )
        
        # Check model attributes
        assert model.in_channels == 9
        assert model.hidden_channels == 64
        assert model.out_channels == 128
        assert model.num_layers == 3
        assert model.heads == 4
        assert model.dropout == 0.1
        
        # Check that we have the correct number of layers
        assert len(model.layers) == 3
        
        # Check layer types
        from torch_geometric.nn import GATv2Conv
        for layer in model.layers:
            assert isinstance(layer, GATv2Conv)
    
    def test_forward_pass_shape(self, model, sample_graph_data):
        """Test that forward pass produces correct output shape."""
        model.eval()
        
        with torch.no_grad():
            output = model(sample_graph_data.x, sample_graph_data.edge_index)
        
        # Check output shape
        expected_shape = (sample_graph_data.x.size(0), 128)  # [num_nodes, out_channels]
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Check that output is not all zeros or NaN
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.allclose(output, torch.zeros_like(output)), "Output is all zeros"
    
    def test_forward_pass_with_edge_attr(self, model, sample_graph_data):
        """Test forward pass with edge attributes (should work even if not used)."""
        model.eval()
        
        with torch.no_grad():
            output = model(
                sample_graph_data.x, 
                sample_graph_data.edge_index,
                sample_graph_data.edge_attr
            )
        
        # Should produce same shape regardless of edge attributes
        expected_shape = (sample_graph_data.x.size(0), 128)
        assert output.shape == expected_shape
    
    def test_output_embedding_norm(self, model, sample_graph_data):
        """Test that output embeddings have reasonable L2 norms."""
        model.eval()
        
        with torch.no_grad():
            output = model(sample_graph_data.x, sample_graph_data.edge_index)
        
        # Calculate L2 norms for each embedding
        norms = torch.norm(output, dim=1)
        
        # Check that norms are positive and reasonable
        assert torch.all(norms > 0), "Some embeddings have zero norm"
        assert torch.all(norms < 1000), "Some embeddings have unreasonably large norms"
        
        # Check mean norm is in reasonable range
        mean_norm = norms.mean().item()
        assert 0.1 < mean_norm < 100, f"Mean embedding norm {mean_norm} is outside reasonable range"
    
    def test_attention_head_usage(self, model):
        """Test that the model correctly uses 4 attention heads per layer."""
        # Check that each layer has the correct number of heads
        for i, layer in enumerate(model.layers):
            assert hasattr(layer, 'heads'), f"Layer {i} missing 'heads' attribute"
            assert layer.heads == 4, f"Layer {i} has {layer.heads} heads, expected 4"
            
            # Check that concat=False for mean pooling
            assert not layer.concat, f"Layer {i} should use mean pooling (concat=False)"
    
    def test_different_input_sizes(self, model):
        """Test model with different graph sizes."""
        # Test with small graph
        small_x = torch.randn(5, 9)
        small_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        
        output_small = model(small_x, small_edge_index)
        assert output_small.shape == (5, 128)
        
        # Test with larger graph
        large_x = torch.randn(50, 9)
        large_edge_index = torch.randint(0, 50, (2, 100))
        
        output_large = model(large_x, large_edge_index)
        assert output_large.shape == (50, 128)
    
    def test_model_training_mode(self, model, sample_graph_data):
        """Test that model behaves differently in training vs eval mode."""
        # Training mode
        model.train()
        output_train_1 = model(sample_graph_data.x, sample_graph_data.edge_index)
        output_train_2 = model(sample_graph_data.x, sample_graph_data.edge_index)
        
        # Due to dropout, outputs should be different in training mode
        assert not torch.allclose(output_train_1, output_train_2, atol=1e-6), \
            "Training outputs should differ due to dropout"
        
        # Eval mode
        model.eval()
        with torch.no_grad():
            output_eval_1 = model(sample_graph_data.x, sample_graph_data.edge_index)
            output_eval_2 = model(sample_graph_data.x, sample_graph_data.edge_index)
        
        # In eval mode, outputs should be identical
        assert torch.allclose(output_eval_1, output_eval_2), \
            "Eval outputs should be identical"
    
    def test_invalid_input_dimensions(self, model):
        """Test that model raises appropriate errors for invalid inputs."""
        # Wrong number of input features
        wrong_x = torch.randn(10, 5)  # Should be 9 features, not 5
        edge_index = torch.randint(0, 10, (2, 20))
        
        with pytest.raises(ValueError, match="Expected 9 input features, got 5"):
            model(wrong_x, edge_index)
    
    def test_single_layer_model(self):
        """Test model with only one layer."""
        single_layer_model = MultiHopGATv2(
            in_channels=9,
            hidden_channels=64,
            out_channels=128,
            num_layers=1,
            heads=4,
            dropout=0.1
        )
        
        # Should have only one layer
        assert len(single_layer_model.layers) == 1
        
        # Test forward pass
        x = torch.randn(10, 9)
        edge_index = torch.randint(0, 10, (2, 20))
        
        output = single_layer_model(x, edge_index)
        assert output.shape == (10, 128)
    
    def test_model_repr(self, model):
        """Test string representation of the model."""
        repr_str = repr(model)
        
        # Check that key parameters are in the string representation
        assert "MultiHopGATv2" in repr_str
        assert "in_channels=9" in repr_str
        assert "hidden_channels=64" in repr_str
        assert "out_channels=128" in repr_str
        assert "num_layers=3" in repr_str
        assert "heads=4" in repr_str
        assert "dropout=0.1" in repr_str
    
    def test_gradient_flow(self, model, sample_graph_data):
        """Test that gradients flow properly through the model."""
        model.train()
        
        # Forward pass
        output = model(sample_graph_data.x, sample_graph_data.edge_index)
        
        # Create a simple loss
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are non-zero for at least some parameters
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad)):
                has_gradients = True
                break
        
        assert has_gradients, "No gradients found in model parameters"
    
    def test_embedding_consistency(self, model, sample_graph_data):
        """Test that embeddings are consistent for the same input."""
        model.eval()
        
        with torch.no_grad():
            # Run multiple times with same input
            outputs = []
            for _ in range(3):
                output = model(sample_graph_data.x, sample_graph_data.edge_index)
                outputs.append(output)
            
            # All outputs should be identical in eval mode
            for i in range(1, len(outputs)):
                assert torch.allclose(outputs[0], outputs[i]), \
                    f"Output {i} differs from output 0"
    
    @pytest.mark.parametrize("heads", [1, 2, 4, 8])
    def test_different_attention_heads(self, heads):
        """Test model with different numbers of attention heads."""
        model = MultiHopGATv2(
            in_channels=9,
            hidden_channels=64,
            out_channels=128,
            num_layers=3,
            heads=heads,
            dropout=0.1
        )
        
        # Check that heads are set correctly
        for layer in model.layers:
            assert layer.heads == heads
        
        # Test forward pass
        x = torch.randn(10, 9)
        edge_index = torch.randint(0, 10, (2, 20))
        
        output = model(x, edge_index)
        assert output.shape == (10, 128)
    
    def test_model_device_handling(self, model, sample_graph_data):
        """Test that model handles device placement correctly."""
        # Test on CPU
        model_cpu = model.cpu()
        data_cpu = sample_graph_data.cpu()
        
        output_cpu = model_cpu(data_cpu.x, data_cpu.edge_index)
        assert output_cpu.device.type == 'cpu'
        
        # Test on GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            data_gpu = sample_graph_data.cuda()
            
            output_gpu = model_gpu(data_gpu.x, data_gpu.edge_index)
            assert output_gpu.device.type == 'cuda'
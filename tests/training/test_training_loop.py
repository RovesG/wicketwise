# Purpose: Tests for the training loop and associated components.
# Author: Shamus Rae, Last Modified: 2024-07-30

import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict

from crickformers.training.training_loop import _compute_loss, EarlyStopper, train_model

# --- Mocks and Fixtures ---

class MockModelWithLoss(nn.Module):
    """A mock model that has parameters and produces trainable outputs."""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 6) # Dummy layer
        self.win_prob_head = nn.Linear(10, 1)
        self.mispricing_head = nn.Linear(10, 1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Use the 'numeric' key from a mock batch
        x = inputs.get("numeric")
        if x is None:
             # Fallback for a case where numeric is not provided
             x = torch.randn(4, 10) # Assuming a default batch size for robustness

        return {
            "next_ball_outcome": self.layer(x),
            "win_probability": self.win_prob_head(x),
            "odds_mispricing": self.mispricing_head(x),
        }

@pytest.fixture
def mock_batch():
    """Provides a single mock batch of data."""
    batch_size = 4
    inputs = {"numeric": torch.randn(batch_size, 10)}
    targets = {
        "outcome": torch.randint(0, 6, (batch_size,)),
        "win_prob": torch.rand(batch_size, 1),
        "mispricing": torch.rand(batch_size, 1),
    }
    return {"inputs": inputs, "targets": targets}

# --- Test Functions ---

def test_compute_loss_aggregation():
    """Validates that the weighted sum of losses is calculated correctly."""
    loss_fns = {
        "outcome": nn.CrossEntropyLoss(),
        "win_prob": nn.BCEWithLogitsLoss(),
        "mispricing": nn.BCEWithLogitsLoss(),
    }
    loss_weights = {"outcome": 0.6, "win_prob": 0.2, "mispricing": 0.2}
    
    outputs = {
        "next_ball_outcome": torch.randn(4, 6),
        "win_probability": torch.randn(4, 1),
        "odds_mispricing": torch.randn(4, 1),
    }
    targets = {
        "outcome": torch.randint(0, 6, (4,)),
        "win_prob": torch.rand(4, 1),
        "mispricing": torch.rand(4, 1),
    }

    total_loss, individual_losses = _compute_loss(outputs, targets, loss_fns, loss_weights)
    
    # Manually compute expected loss
    l1 = loss_fns["outcome"](outputs["next_ball_outcome"], targets["outcome"])
    l2 = loss_fns["win_prob"](outputs["win_probability"], targets["win_prob"])
    l3 = loss_fns["mispricing"](outputs["odds_mispricing"], targets["mispricing"])
    expected_total_loss = l1 * 0.6 + l2 * 0.2 + l3 * 0.2
    
    assert torch.allclose(total_loss, expected_total_loss)
    assert "outcome" in individual_losses

def test_early_stopper_logic():
    """Tests the EarlyStopper logic for firing correctly."""
    stopper = EarlyStopper(patience=3) # Patience of 3 means it stops on the 3rd non-improving epoch

    assert not stopper(10.0) # Best loss is 10.0, counter is 0
    assert not stopper(9.0)  # Improves, best loss is 9.0, counter is 0
    
    assert not stopper(9.0)  # No improvement, counter becomes 1
    assert stopper.counter == 1
    
    assert not stopper(8.99) # Improves, best loss is 8.99, counter is 0
    
    assert not stopper(9.0)  # No improvement, counter becomes 1
    assert not stopper(8.995) # No improvement, counter becomes 2
    assert stopper(8.997)  # No improvement, counter becomes 3, so it stops

def test_train_model_runs_and_backward_step(mock_batch):
    """
    A minimal test to ensure train_model runs and performs a backward step.
    """
    model = MockModelWithLoss()
    assert all(p.grad is None for p in model.parameters())

    # Create a mock dataloader with a custom dataset
    class CustomDictDataset(TensorDataset):
        def __getitem__(self, idx):
            # Return data in the dict format our collate_fn expects
            tensors = super().__getitem__(idx)
            return {
                "inputs": {"numeric": tensors[0]},
                "targets": {
                    "outcome": tensors[1],
                    "win_prob": torch.rand(1),
                    "mispricing": torch.rand(1)
                }
            }
            
    dataset = CustomDictDataset(torch.randn(8, 10), torch.randint(0, 6, (8,)))
    loader = DataLoader(dataset, batch_size=4) # No collate_fn needed now
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_weights = {"outcome": 1.0, "win_prob": 0.1, "mispricing": 0.1}
    
    # Run for one epoch
    train_model(model, loader, loader, optimizer, None, loss_weights, 1, 1)

    assert any(p.grad is not None for p in model.parameters()) 
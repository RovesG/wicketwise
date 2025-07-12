# Purpose: Tests for the prediction head modules.
# Author: Shamus Rae, Last Modified: 2024-07-30

import torch
import pytest
from crickformers.model.prediction_heads import (
    NextBallOutcomeHead,
    WinProbabilityHead,
    OddsMispricingHead,
)

@pytest.fixture
def latent_vector():
    """Provides a sample latent vector for testing the heads."""
    batch_size = 8
    latent_dim = 64
    return torch.randn(batch_size, latent_dim)

def test_next_ball_outcome_head(latent_vector):
    """
    Tests the NextBallOutcomeHead for correct output shape and softmax behavior.
    """
    latent_dim = latent_vector.shape[1]
    num_outcomes = 6
    head = NextBallOutcomeHead(latent_dim, num_outcomes)
    head.eval()

    with torch.no_grad():
        logits = head(latent_vector)
        probabilities = torch.softmax(logits, dim=-1)

    assert logits.shape == (latent_vector.shape[0], num_outcomes)
    assert probabilities.shape == (latent_vector.shape[0], num_outcomes)
    # Check that probabilities for each item in the batch sum to 1
    assert torch.allclose(probabilities.sum(dim=-1), torch.tensor(1.0))

def test_win_probability_head(latent_vector):
    """
    Tests the WinProbabilityHead for correct output shape and sigmoid behavior.
    """
    latent_dim = latent_vector.shape[1]
    head = WinProbabilityHead(latent_dim)
    head.eval()

    with torch.no_grad():
        logits = head(latent_vector)
        probability = torch.sigmoid(logits)

    assert logits.shape == (latent_vector.shape[0], 1)
    assert probability.shape == (latent_vector.shape[0], 1)
    # Check that probabilities are all between 0 and 1
    assert torch.all(probability >= 0) and torch.all(probability <= 1)

def test_odds_mispricing_head(latent_vector):
    """
    Tests the OddsMispricingHead for correct output shape and sigmoid behavior.
    """
    latent_dim = latent_vector.shape[1]
    head = OddsMispricingHead(latent_dim)
    head.eval()

    with torch.no_grad():
        logits = head(latent_vector)
        probability = torch.sigmoid(logits)

    assert logits.shape == (latent_vector.shape[0], 1)
    assert probability.shape == (latent_vector.shape[0], 1)
    # Check that probabilities are all between 0 and 1
    assert torch.all(probability >= 0) and torch.all(probability <= 1)

@pytest.mark.parametrize("head_class", [NextBallOutcomeHead, WinProbabilityHead, OddsMispricingHead])
def test_dropout_behavior(head_class, latent_vector):
    """
    Tests that dropout is applied during training but not evaluation for all heads.
    """
    latent_dim = latent_vector.shape[1]
    # Use kwargs to handle the different constructor for NextBallOutcomeHead
    kwargs = {"num_outcomes": 6} if head_class == NextBallOutcomeHead else {}
    head = head_class(latent_dim, dropout_rate=0.5, **kwargs)

    # Train mode: outputs should differ
    head.train()
    with torch.no_grad():
        output1 = head(latent_vector)
        output2 = head(latent_vector)
    assert not torch.allclose(output1, output2)

    # Eval mode: outputs should be identical
    head.eval()
    with torch.no_grad():
        output1_eval = head(latent_vector)
        output2_eval = head(latent_vector)
    assert torch.allclose(output1_eval, output2_eval) 
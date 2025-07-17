# Purpose: Implements Monte Carlo dropout for uncertainty estimation in cricket predictions
# Author: Shamus Rae, Last Modified: 2024-12-19

"""
This module provides utilities for uncertainty quantification using Monte Carlo dropout.
It enables confidence estimation for any PyTorch model with dropout layers by performing
multiple forward passes with dropout enabled during inference.
"""

import torch
import torch.nn as nn
from typing import Tuple, Union, Optional
import numpy as np
from scipy import stats


def predict_with_uncertainty(
    model: nn.Module,
    inputs: Union[torch.Tensor, tuple, dict],
    n_samples: int = 20,
    confidence_level: float = 0.95,
    device: Optional[torch.device] = None
) -> Tuple[dict, dict, dict]:
    """
    Performs Monte Carlo dropout for uncertainty estimation.
    
    Args:
        model: PyTorch model with dropout layers
        inputs: Model inputs (tensor, tuple of tensors, or dict of tensors)
        n_samples: Number of forward passes to perform
        confidence_level: Confidence level for interval bounds (default: 0.95)
        device: Device to perform computations on
        
    Returns:
        Tuple containing:
        - mean_prediction: Dictionary of mean predictions for each output
        - std_prediction: Dictionary of standard deviations for each output
        - conf_intervals: Dictionary of confidence intervals for each output
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Move inputs to device if needed
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device)
    elif isinstance(inputs, tuple):
        inputs = tuple(inp.to(device) for inp in inputs)
    elif isinstance(inputs, dict):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Store original training mode
    original_mode = model.training
    
    # Enable dropout by setting model to train mode
    model.train()
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            if isinstance(inputs, torch.Tensor):
                pred = model(inputs)
            elif isinstance(inputs, tuple):
                pred = model(*inputs)
            elif isinstance(inputs, dict):
                pred = model(**inputs)
            else:
                raise ValueError(f"Unsupported input type: {type(inputs)}")
            
            predictions.append(pred)
    
    # Restore original training mode
    model.train(original_mode)
    
    # Handle both single tensor and dictionary outputs
    if isinstance(predictions[0], dict):
        # Multi-output model (like CrickformerModel)
        mean_predictions = {}
        std_predictions = {}
        conf_intervals = {}
        
        for key in predictions[0].keys():
            # Stack predictions for this output
            key_predictions = torch.stack([pred[key] for pred in predictions], dim=0)
            
            # Compute statistics
            mean_pred = key_predictions.mean(dim=0)
            std_pred = key_predictions.std(dim=0)
            
            # Compute confidence intervals
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha / 2)
            margin_of_error = z_score * std_pred
            
            mean_predictions[key] = mean_pred.to(device)
            std_predictions[key] = std_pred.to(device)
            conf_intervals[key] = (
                (mean_pred - margin_of_error).to(device),
                (mean_pred + margin_of_error).to(device)
            )
    else:
        # Single tensor output
        predictions = torch.stack(predictions, dim=0)
        
        # Compute mean and standard deviation
        mean_prediction = predictions.mean(dim=0)
        std_prediction = predictions.std(dim=0)
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)
        margin_of_error = z_score * std_prediction
        
        mean_predictions = {"output": mean_prediction.to(device)}
        std_predictions = {"output": std_prediction.to(device)}
        conf_intervals = {"output": (
            (mean_prediction - margin_of_error).to(device),
            (mean_prediction + margin_of_error).to(device)
        )}
    
    return mean_predictions, std_predictions, conf_intervals


def has_dropout_layers(model: nn.Module) -> bool:
    """
    Check if a model has dropout layers.
    
    Args:
        model: PyTorch model to check
        
    Returns:
        True if model contains dropout layers, False otherwise
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            return True
    return False


def get_dropout_modules(model: nn.Module) -> list:
    """
    Get all dropout modules in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of dropout modules
    """
    dropout_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            dropout_modules.append((name, module))
    return dropout_modules


def estimate_prediction_confidence(
    predictions: torch.Tensor,
    uncertainty: torch.Tensor,
    confidence_threshold: float = 0.1
) -> torch.Tensor:
    """
    Estimate confidence scores based on uncertainty.
    
    Args:
        predictions: Model predictions
        uncertainty: Uncertainty estimates (standard deviation)
        confidence_threshold: Threshold for high confidence (lower = more confident)
        
    Returns:
        Confidence scores (0 to 1, higher = more confident)
    """
    # Normalize uncertainty to [0, 1] range using sigmoid
    normalized_uncertainty = torch.sigmoid(uncertainty / confidence_threshold)
    
    # Convert to confidence (inverse of uncertainty)
    confidence = 1 - normalized_uncertainty
    
    return confidence


class ConfidenceTracker:
    """
    Tracks confidence metrics during inference.
    """
    
    def __init__(self):
        self.predictions = []
        self.uncertainties = []
        self.confidence_scores = []
        
    def add_prediction(self, prediction: torch.Tensor, uncertainty: torch.Tensor, confidence: torch.Tensor):
        """Add a prediction with its uncertainty and confidence."""
        self.predictions.append(prediction.cpu())
        self.uncertainties.append(uncertainty.cpu())
        self.confidence_scores.append(confidence.cpu())
    
    def get_statistics(self) -> dict:
        """Get aggregated statistics."""
        if not self.predictions:
            return {}
        
        uncertainties = torch.cat(self.uncertainties, dim=0)
        confidences = torch.cat(self.confidence_scores, dim=0)
        
        return {
            'mean_uncertainty': uncertainties.mean().item(),
            'std_uncertainty': uncertainties.std().item(),
            'mean_confidence': confidences.mean().item(),
            'std_confidence': confidences.std().item(),
            'min_confidence': confidences.min().item(),
            'max_confidence': confidences.max().item(),
            'num_predictions': len(self.predictions)
        }
    
    def clear(self):
        """Clear all stored predictions."""
        self.predictions.clear()
        self.uncertainties.clear()
        self.confidence_scores.clear()


def calculate_confidence_score(prediction: float, uncertainty: float) -> float:
    """
    Calculate a confidence score from prediction and uncertainty.
    
    Args:
        prediction: The mean prediction value
        uncertainty: The uncertainty (standard deviation) of the prediction
        
    Returns:
        A confidence score between 0 and 1, where 1 is highest confidence
    """
    # Simple confidence score based on inverse of relative uncertainty
    # Higher uncertainty -> lower confidence
    relative_uncertainty = uncertainty / (abs(prediction) + 1e-8)
    
    # Transform to 0-1 scale using sigmoid-like function
    confidence = 1.0 / (1.0 + relative_uncertainty)
    
    return max(0.0, min(1.0, confidence))  # Clamp to [0,1]
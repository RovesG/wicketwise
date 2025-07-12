# Purpose: Implements the training and validation loop for the Crickformer model.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains the primary `train_model` function which orchestrates
the model training process, including epoch iteration, loss computation,
backpropagation, validation, and early stopping.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Note: Batch stratification for class balance is best handled by a sampler
# (e.g., WeightedRandomSampler) passed to the DataLoader, not within the loop itself.

class EarlyStopper:
    """
    A simple early stopping handler.
    """
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, validation_loss: float) -> bool:
        """
        Returns True if training should stop, False otherwise.
        """
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            return False
        
        self.counter += 1
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.patience} epochs of no improvement.")
            return True
        return False

def _compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    loss_fns: Dict[str, nn.Module],
    loss_weights: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes the weighted total loss across all prediction heads.
    """
    losses = {
        "outcome": loss_fns["outcome"](outputs["next_ball_outcome"], targets["outcome"]),
        "win_prob": loss_fns["win_prob"](outputs["win_probability"], targets["win_prob"]),
        "mispricing": loss_fns["mispricing"](outputs["odds_mispricing"], targets["mispricing"]),
    }

    total_loss = sum(losses[key] * loss_weights[key] for key in losses)
    detached_losses = {key: loss.item() for key, loss in losses.items()}
    return total_loss, detached_losses


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any, # e.g., torch.optim.lr_scheduler._LRScheduler
    loss_weights: Dict[str, float],
    num_epochs: int,
    patience: int = 3,
):
    """
    Runs the full training and validation loop for the model.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        loss_weights: A dictionary weighting the loss for each prediction head.
        num_epochs: The total number of epochs to train for.
        patience: Number of epochs to wait for improvement before stopping early.
    """
    device = next(model.parameters()).device
    early_stopper = EarlyStopper(patience=patience)
    
    loss_fns = {
        "outcome": nn.CrossEntropyLoss(),
        "win_prob": nn.BCEWithLogitsLoss(),
        "mispricing": nn.BCEWithLogitsLoss(),
    }

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, _ = _compute_loss(outputs, targets, loss_fns, loss_weights)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Step ---
        model.eval()
        total_val_loss = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
                targets = {k: v.to(device) for k, v in batch["targets"].items()}
                
                outputs = model(inputs)
                loss, _ = _compute_loss(outputs, targets, loss_fns, loss_weights)
                total_val_loss += loss.item()

                # Collect preds for metrics (e.g., MAE, AUC)
                # This part would be more complex, calculating actual metrics.
                # For simplicity here, we focus on the validation loss.

        avg_val_loss = total_val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)
            
        if early_stopper(avg_val_loss):
            break
            
    logger.info("Training finished.") 
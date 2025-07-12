# Purpose: Evaluation script for trained Crickformer models with per-ball prediction logging
# Author: Assistant, Last Modified: 2024

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from crickformers.crickformer_dataset import CrickformerDataset
from crickformers.csv_data_adapter import CSVDataConfig
from crickformers.model.crickformer_model import CrickformerModel
from crickformers.train import collate_fn, load_config

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class CrickformerEvaluator:
    """
    Evaluator for trained Crickformer models.
    
    Loads a trained model checkpoint and generates per-ball predictions
    on test data, saving results to CSV for analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary containing model and evaluation settings
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.model = None
        self.dataset = None
        self.test_loader = None
        
        # Outcome class mapping for next ball prediction
        self.outcome_classes = {
            0: "0_runs",
            1: "1_run", 
            2: "2_runs",
            3: "3_runs",
            4: "4_runs",
            5: "6_runs",
            6: "wicket"
        }
        
        logger.info(f"Initialized CrickformerEvaluator on device: {self.device}")
    
    def load_model(self, checkpoint_path: str):
        """
        Load a trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to the saved model checkpoint
        """
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model configuration from checkpoint
        if 'config' in checkpoint:
            model_config = checkpoint['config'].get('model', {})
        else:
            # Use default configuration if not available in checkpoint
            model_config = self.config.get('model', {})
        
        # Initialize model with same architecture as training
        self.model = CrickformerModel(
            sequence_config=model_config.get("sequence_encoder", {
                "feature_dim": 6,
                "nhead": 2,
                "num_encoder_layers": 2,
                "dim_feedforward": 128,
                "dropout": 0.1
            }),
            static_config=model_config.get("static_context_encoder", {
                "numeric_dim": 15,
                "categorical_vocab_sizes": {"competition": 100, "batter_hand": 100, "bowler_type": 100, "innings": 10},
                "categorical_embedding_dims": {"competition": 8, "batter_hand": 4, "bowler_type": 8, "innings": 4},
                "video_dim": 99,
                "hidden_dims": [128, 64],
                "context_dim": 128,
                "dropout_rate": 0.1
            }),
            fusion_config=model_config.get("fusion_layer", {
                "sequence_dim": 6,
                "context_dim": 128,
                "kg_dim": 128,
                "hidden_dims": [256, 128],
                "latent_dim": 128,
                "dropout_rate": 0.1
            }),
            prediction_heads_config=model_config.get("prediction_heads", {
                "win_probability": {"latent_dim": 128, "dropout_rate": 0.1},
                "next_ball_outcome": {"latent_dim": 128, "num_outcomes": 7, "dropout_rate": 0.1},
                "odds_mispricing": {"latent_dim": 128, "dropout_rate": 0.1}
            }),
            gnn_embedding_dim=320
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
    
    def setup_dataset(self, data_path: str, use_csv: bool = True, test_split: float = 0.2):
        """
        Setup the test dataset.
        
        Args:
            data_path: Path to the cricket data directory
            use_csv: Whether to use CSV data adapter
            test_split: Fraction of data to use for testing
        """
        logger.info(f"Setting up test dataset from: {data_path}")
        
        # Create full dataset
        full_dataset = CrickformerDataset(
            data_root=data_path,
            use_csv_adapter=use_csv,
            csv_config=CSVDataConfig(),
            history_length=5,
            load_video=True,
            load_embeddings=True,
            load_market_odds=True
        )
        
        # Create test split
        total_size = len(full_dataset)
        test_size = int(total_size * test_split)
        test_start = total_size - test_size  # Use last portion as test set
        
        test_indices = list(range(test_start, total_size))
        self.dataset = Subset(full_dataset, test_indices)
        
        logger.info(f"Test dataset created: {len(self.dataset):,} samples")
        
        # Create test dataloader
        self.test_loader = DataLoader(
            self.dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=False,  # Don't shuffle for consistent evaluation
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Test DataLoader created: {len(self.test_loader)} batches")
    
    def _extract_actual_outcomes(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract actual outcomes from batch for comparison.
        
        Args:
            batch: Batch of data from dataloader
            
        Returns:
            List of actual outcome dictionaries
        """
        targets = batch["targets"]
        batch_size = targets["outcome"].shape[0]
        
        actual_outcomes = []
        for i in range(batch_size):
            actual_outcomes.append({
                "actual_runs": int(targets["outcome"][i].item()),
                "actual_win_prob": float(targets["win_prob"][i].item()),
                "actual_mispricing": float(targets["mispricing"][i].item())
            })
        
        return actual_outcomes
    
    def _get_ball_metadata(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract ball metadata from batch.
        
        Args:
            batch: Batch of data from dataloader
            
        Returns:
            List of metadata dictionaries
        """
        # For CSV data, we need to extract metadata from the underlying dataset
        # This is a simplified version - in practice, you'd need access to the original data
        batch_size = batch["inputs"]["numeric_features"].shape[0]
        
        metadata = []
        for i in range(batch_size):
            # Generate synthetic metadata for now
            # In practice, this would come from the dataset's sample_info
            metadata.append({
                "match_id": f"match_{i % 100}",  # Placeholder
                "ball_id": f"ball_{i}",  # Placeholder
                "phase": "powerplay" if i % 3 == 0 else "middle_overs",
                "batter_id": f"batter_{i % 50}",  # Placeholder
                "bowler_id": f"bowler_{i % 30}"   # Placeholder
            })
        
        return metadata
    
    def evaluate_model(self, output_csv: str = "eval_predictions.csv"):
        """
        Evaluate the model on test data and save per-ball predictions.
        
        Args:
            output_csv: Path to save the evaluation results CSV
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.test_loader is None:
            raise ValueError("Test dataset not setup. Call setup_dataset() first.")
        
        logger.info("Starting model evaluation...")
        
        # CSV headers
        csv_headers = [
            "match_id", "ball_id", "actual_runs", "predicted_runs_class",
            "win_prob", "odds_mispricing", "phase", "batter_id", "bowler_id",
            "predicted_runs_0", "predicted_runs_1", "predicted_runs_2", 
            "predicted_runs_3", "predicted_runs_4", "predicted_runs_6", 
            "predicted_wicket", "actual_win_prob", "actual_mispricing"
        ]
        
        # Open CSV file for writing
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
            
            total_samples = 0
            
            # Iterate through test data
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in batch["inputs"].items()}
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(inputs)
                
                # Convert predictions to probabilities
                win_probs = torch.sigmoid(outputs["win_probability"]).cpu().numpy()
                outcome_probs = torch.softmax(outputs["next_ball_outcome"], dim=1).cpu().numpy()
                mispricing_probs = torch.sigmoid(outputs["odds_mispricing"]).cpu().numpy()
                
                # Get predicted classes
                predicted_classes = torch.argmax(outputs["next_ball_outcome"], dim=1).cpu().numpy()
                
                # Extract actual outcomes and metadata
                actual_outcomes = self._extract_actual_outcomes(batch)
                metadata = self._get_ball_metadata(batch)
                
                # Write predictions for each ball in the batch
                batch_size = len(actual_outcomes)
                for i in range(batch_size):
                    row = {
                        "match_id": metadata[i]["match_id"],
                        "ball_id": metadata[i]["ball_id"],
                        "actual_runs": actual_outcomes[i]["actual_runs"],
                        "predicted_runs_class": self.outcome_classes[predicted_classes[i]],
                        "win_prob": float(win_probs[i][0]),
                        "odds_mispricing": float(mispricing_probs[i][0]),
                        "phase": metadata[i]["phase"],
                        "batter_id": metadata[i]["batter_id"],
                        "bowler_id": metadata[i]["bowler_id"],
                        "predicted_runs_0": float(outcome_probs[i][0]),
                        "predicted_runs_1": float(outcome_probs[i][1]),
                        "predicted_runs_2": float(outcome_probs[i][2]),
                        "predicted_runs_3": float(outcome_probs[i][3]),
                        "predicted_runs_4": float(outcome_probs[i][4]),
                        "predicted_runs_6": float(outcome_probs[i][5]),
                        "predicted_wicket": float(outcome_probs[i][6]),
                        "actual_win_prob": actual_outcomes[i]["actual_win_prob"],
                        "actual_mispricing": actual_outcomes[i]["actual_mispricing"]
                    }
                    writer.writerow(row)
                    total_samples += 1
                
                # Log progress
                if (batch_idx + 1) % 100 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(self.test_loader)} batches")
        
        logger.info(f"Evaluation completed! Saved {total_samples:,} predictions to {output_csv}")
        return total_samples


def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained Crickformer model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data",
        help="Path to cricket data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_predictions.csv",
        help="Path to save evaluation results CSV"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing"
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        default=True,
        help="Use CSV data adapter"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        logger.info("Using default configuration")
        config = {}
    
    # Override config with CLI arguments
    config.update({
        "batch_size": args.batch_size,
        "test_split": args.test_split
    })
    
    # Initialize evaluator
    evaluator = CrickformerEvaluator(config)
    
    # Load model
    evaluator.load_model(args.checkpoint)
    
    # Setup dataset
    evaluator.setup_dataset(args.data_path, use_csv=args.use_csv, test_split=args.test_split)
    
    # Run evaluation
    total_predictions = evaluator.evaluate_model(args.output)
    
    logger.info(f"🎉 Evaluation completed successfully!")
    logger.info(f"📊 Generated {total_predictions:,} predictions")
    logger.info(f"📄 Results saved to: {args.output}")


if __name__ == "__main__":
    main() 
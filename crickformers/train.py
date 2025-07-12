# Purpose: Command-line interface for training the Crickformer model.
# Author: Shamus Rae, Last Modified: 2024-07-30

import argparse
import json
import logging
from typing import Any, Dict

import torch
import yaml
from crickformers.model.crickformer_model import CrickformerModel
from crickformers.training.training_loop import train_model

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a JSON or YAML configuration file."""
    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)
    elif config_path.endswith((".yaml", ".yml")):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Unsupported config file format. Use JSON or YAML.")


def main():
    """Main function to parse arguments and launch the training process."""
    parser = argparse.ArgumentParser(description="Train the Crickformer model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the training config file."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Optional path to a model to resume training from."
    )
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Assemble the model from components based on the config
    logger.info("Assembling Crickformer model...")
    model = CrickformerModel(
        sequence_config=config["model"]["sequence_encoder"],
        static_config=config["model"]["static_context_encoder"],
        fusion_config=config["model"]["fusion_layer"],
        prediction_heads_config=config["model"]["prediction_heads"],
    )

    if args.resume:
        logger.info(f"Resuming training from {args.resume}")
        model.load_state_dict(torch.load(args.resume))

    # For now, using placeholder data loaders
    # future: replace with actual data loaders
    train_loader = torch.utils.data.DataLoader([torch.randn(10)], batch_size=1)
    val_loader = torch.utils.data.DataLoader([torch.randn(10)], batch_size=1)

    logger.info("Starting training loop...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        **config["training_params"],
    )

    logger.info(f"Training complete. Saving model to {args.save_path}")
    torch.save(trained_model.state_dict(), args.save_path)
    logger.info("Model saved successfully.")


if __name__ == "__main__":
    main() 
# Purpose: Mock configuration loader for WicketWise UI dropdowns and options
# Author: WicketWise Team, Last Modified: 2024-01-15

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import json

class ConfigLoader:
    """Mock configuration loader for development UI"""
    
    def __init__(self, config_dir: str = "config", checkpoints_dir: str = "checkpoints"):
        self.config_dir = Path(config_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        
        # Create directories if they don't exist
        self.config_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
    
    def get_available_configs(self) -> List[str]:
        """Get list of available configuration files"""
        # Check for actual config files
        config_files = []
        
        # Look for YAML config files
        if self.config_dir.exists():
            yaml_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
            config_files.extend([f.name for f in yaml_files])
        
        # Add default mock configs if no real ones exist
        if not config_files:
            config_files = [
                "train_config.yaml",
                "default_config.yaml", 
                "test_config.yaml",
                "production_config.yaml"
            ]
        
        return sorted(config_files)
    
    def get_model_checkpoints(self) -> List[str]:
        """Get list of available model checkpoint files"""
        checkpoint_files = []
        
        # Look for actual checkpoint files
        if self.checkpoints_dir.exists():
            pt_files = list(self.checkpoints_dir.glob("*.pt")) + list(self.checkpoints_dir.glob("*.pth"))
            checkpoint_files.extend([f.name for f in pt_files])
        
        # Also check current directory for checkpoints
        current_dir_checkpoints = glob.glob("*.pt") + glob.glob("*.pth")
        checkpoint_files.extend(current_dir_checkpoints)
        
        # Add default mock checkpoints if no real ones exist
        if not checkpoint_files:
            checkpoint_files = [
                "model_checkpoint_latest.pt",
                "model_checkpoint_best.pt",
                "crickformer_v1.pt",
                "crickformer_trained.pt"
            ]
        
        return sorted(list(set(checkpoint_files)))  # Remove duplicates
    
    def get_sample_data_files(self) -> List[str]:
        """Get list of available sample data files"""
        sample_files = []
        
        # Look for actual sample files
        samples_dir = Path("samples")
        if samples_dir.exists():
            csv_files = list(samples_dir.glob("*.csv"))
            sample_files.extend([str(f) for f in csv_files])
        
        # Add default mock sample files
        if not sample_files:
            sample_files = [
                "samples/test_match.csv",
                "samples/evaluation_data.csv",
                "samples/live_match_data.csv",
                "samples/historical_matches.csv"
            ]
        
        return sorted(sample_files)
    
    def get_training_options(self) -> Dict[str, Any]:
        """Get training configuration options"""
        return {
            "batch_sizes": [16, 32, 64, 128],
            "learning_rates": [0.001, 0.0001, 0.00001],
            "epochs": [10, 25, 50, 100],
            "optimizers": ["adam", "sgd", "adamw"],
            "schedulers": ["step", "cosine", "plateau"]
        }
    
    def get_evaluation_options(self) -> Dict[str, Any]:
        """Get evaluation configuration options"""
        return {
            "metrics": ["accuracy", "f1", "precision", "recall", "auc"],
            "test_splits": [0.1, 0.2, 0.3],
            "batch_sizes": [32, 64, 128],
            "output_formats": ["csv", "json", "xlsx"]
        }
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a specific configuration file"""
        config_path = self.config_dir / config_name
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    if config_path.suffix.lower() in ['.yaml', '.yml']:
                        return yaml.safe_load(f)
                    elif config_path.suffix.lower() == '.json':
                        return json.load(f)
            except Exception as e:
                print(f"Error loading config {config_name}: {e}")
        
        # Return mock config if file doesn't exist
        return self.get_mock_config(config_name)
    
    def get_mock_config(self, config_name: str) -> Dict[str, Any]:
        """Generate a mock configuration"""
        base_config = {
            "model": {
                "name": "CrickformerModel",
                "embedding_dim": 256,
                "num_heads": 8,
                "num_layers": 6,
                "dropout": 0.1
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 50,
                "optimizer": "adam",
                "scheduler": "step"
            },
            "data": {
                "train_split": 0.7,
                "val_split": 0.2,
                "test_split": 0.1,
                "max_sequence_length": 300
            },
            "paths": {
                "data_dir": "data/",
                "output_dir": "outputs/",
                "checkpoint_dir": "checkpoints/"
            }
        }
        
        # Modify based on config name
        if "test" in config_name.lower():
            base_config["training"]["epochs"] = 5
            base_config["training"]["batch_size"] = 16
        elif "production" in config_name.lower():
            base_config["training"]["epochs"] = 100
            base_config["training"]["batch_size"] = 64
        
        return base_config
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> bool:
        """Save a configuration to file"""
        try:
            config_path = self.config_dir / config_name
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config, f, default_flow_style=False)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config {config_name}: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for the UI"""
        return {
            "python_version": "3.9+",
            "required_packages": [
                "torch>=1.9.0",
                "transformers>=4.0.0",
                "pandas>=1.3.0",
                "numpy>=1.21.0",
                "streamlit>=1.0.0",
                "pyyaml>=5.4.0"
            ],
            "gpu_available": "Check PyTorch CUDA availability",
            "memory_recommendation": "8GB+ RAM recommended"
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a configuration and return validation results"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required sections
        required_sections = ["model", "training", "data"]
        for section in required_sections:
            if section not in config:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Missing required section: {section}")
        
        # Check model parameters
        if "model" in config:
            model_config = config["model"]
            if "embedding_dim" in model_config and model_config["embedding_dim"] < 64:
                validation_results["warnings"].append("Embedding dimension is quite small")
        
        # Check training parameters
        if "training" in config:
            training_config = config["training"]
            if "batch_size" in training_config and training_config["batch_size"] > 128:
                validation_results["warnings"].append("Large batch size may require significant memory")
        
        return validation_results 
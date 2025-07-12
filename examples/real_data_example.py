#!/usr/bin/env python3
# Purpose: Example of using CrickformerDataset with real CSV cricket data
# Author: Assistant, Last Modified: 2024

"""
Example script demonstrating how to use the CrickformerDataset with real CSV cricket data.

This script shows:
1. Loading the dataset with CSV adapter
2. Exploring the data structure
3. Creating data loaders for training
4. Accessing individual samples and their components
"""

import sys
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from crickformers.crickformer_dataset import CrickformerDataset
from crickformers.csv_data_adapter import CSVDataConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main example function"""
    
    # Path to your real cricket data
    data_path = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data"
    
    logger.info("🏏 Cricket Data Loading Example")
    logger.info("=" * 50)
    
    # 1. Create dataset with CSV adapter
    logger.info("1. Loading dataset...")
    dataset = CrickformerDataset(
        data_root=data_path,
        use_csv_adapter=True,
        csv_config=CSVDataConfig(),
        history_length=5,
        load_video=True,
        load_embeddings=True,
        load_market_odds=True
    )
    
    logger.info(f"✅ Dataset loaded: {len(dataset):,} samples")
    
    # 2. Explore dataset structure
    logger.info("\n2. Dataset overview:")
    match_ids = dataset.get_match_ids()
    logger.info(f"   • Total matches: {len(match_ids)}")
    logger.info(f"   • Total balls: {len(dataset):,}")
    logger.info(f"   • Avg balls per match: {len(dataset) / len(match_ids):.1f}")
    
    # Show some example matches
    logger.info(f"   • Example matches:")
    for i, match in enumerate(match_ids[:5]):
        logger.info(f"     {i+1}. {match}")
    
    # 3. Examine a sample
    logger.info("\n3. Sample data structure:")
    sample = dataset[0]
    sample_info = dataset.get_sample_info(0)
    
    logger.info(f"   • Sample info: {sample_info}")
    logger.info(f"   • Tensor components:")
    for key, tensor in sample.items():
        logger.info(f"     - {key}: {tensor.shape} ({tensor.dtype})")
    
    # 4. Show actual values from a sample
    logger.info("\n4. Sample values:")
    logger.info(f"   • Numeric features (first 5): {sample['numeric_ball_features'][:5].tolist()}")
    logger.info(f"   • Categorical features: {sample['categorical_ball_features'].tolist()}")
    logger.info(f"   • Video mask: {sample['video_mask'].item()}")
    logger.info(f"   • Market odds mask: {sample['market_odds_mask'].item()}")
    
    # 5. Create data loaders
    logger.info("\n5. Creating data loaders...")
    
    # Split dataset into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"   • Training samples: {len(train_dataset):,}")
    logger.info(f"   • Validation samples: {len(val_dataset):,}")
    logger.info(f"   • Training batches: {len(train_loader):,}")
    logger.info(f"   • Validation batches: {len(val_loader):,}")
    
    # 6. Test batch loading
    logger.info("\n6. Testing batch loading...")
    batch = next(iter(train_loader))
    
    logger.info(f"   • Batch size: {len(batch['numeric_ball_features'])}")
    logger.info(f"   • Batch tensor shapes:")
    for key, tensor in batch.items():
        logger.info(f"     - {key}: {tensor.shape}")
    
    # 7. Filter by specific matches
    logger.info("\n7. Filtering by matches...")
    specific_matches = match_ids[:3]  # First 3 matches
    filtered_dataset = dataset.filter_by_match(specific_matches)
    
    logger.info(f"   • Filtered to {len(specific_matches)} matches")
    logger.info(f"   • Filtered dataset size: {len(filtered_dataset):,} samples")
    
    # 8. Performance metrics
    logger.info("\n8. Performance info:")
    logger.info(f"   • Data loading from: {data_path}")
    logger.info(f"   • CSV files processed: nvplay_data_v3.csv, decimal_data_v3.csv")
    logger.info(f"   • Memory usage: ~{len(dataset) * 8 * 500 / 1024 / 1024:.1f} MB (estimated)")
    
    logger.info("\n🎉 Example completed successfully!")
    logger.info("\nNext steps:")
    logger.info("  • Use train_loader and val_loader for model training")
    logger.info("  • Customize CSVDataConfig for different preprocessing")
    logger.info("  • Filter dataset by specific matches for targeted analysis")
    logger.info("  • Implement custom transforms for data augmentation")

def quick_stats():
    """Quick function to show dataset statistics"""
    data_path = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data"
    
    dataset = CrickformerDataset(
        data_root=data_path,
        use_csv_adapter=True,
        load_video=False,  # Faster loading
        load_embeddings=False,
        load_market_odds=False
    )
    
    print(f"Dataset Statistics:")
    print(f"  Total samples: {len(dataset):,}")
    print(f"  Total matches: {len(dataset.get_match_ids())}")
    
    # Sample some data points
    for i in range(min(5, len(dataset))):
        info = dataset.get_sample_info(i)
        print(f"  Sample {i}: {info['match_id']}, Ball {info['ball_number']}")

if __name__ == "__main__":
    # Run full example
    main()
    
    # Uncomment for quick stats only:
    # quick_stats() 
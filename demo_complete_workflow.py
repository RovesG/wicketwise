# Purpose: Demonstrates complete WicketWise training workflow with match-level splitting
# Author: Assistant, Last Modified: 2024

import logging
import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from match_aligner import MatchAligner
from wicketwise.match_splitter import split_matches
from crickformers.crickformer_dataset import CrickformerDataset
from crickformers.csv_data_adapter import CSVDataConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate the complete WicketWise workflow."""
    
    # Configuration
    data_path = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data"
    output_dir = Path("./workflow_output")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("🏏 WicketWise Complete Workflow Demo")
    logger.info("=" * 50)
    
    # Step 1: Match Alignment
    logger.info("\n1️⃣ STEP 1: Match Alignment")
    logger.info("Aligning matches between NVPlay and decimal data...")
    
    try:
        aligner = MatchAligner(
            nvplay_path=Path(data_path) / "nvplay_data_v3.csv",
            decimal_path=Path(data_path) / "decimal_data_v3.csv"
        )
        
        aligned_matches = aligner.align_matches()
        aligned_file = output_dir / "aligned_matches.csv"
        aligned_matches.to_csv(aligned_file, index=False)
        
        logger.info(f"✅ Aligned {len(aligned_matches)} matches")
        logger.info(f"   Saved to: {aligned_file}")
        
    except Exception as e:
        logger.error(f"❌ Match alignment failed: {e}")
        logger.info("Skipping to next step with mock data...")
        
        # Create mock aligned matches for demo
        import pandas as pd
        aligned_matches = pd.DataFrame({
            'match_id': [f'match_{i}' for i in range(100)]
        })
        aligned_file = output_dir / "aligned_matches.csv"
        aligned_matches.to_csv(aligned_file, index=False)
        logger.info(f"   Created mock aligned matches: {aligned_file}")
    
    # Step 2: Match Splitting
    logger.info("\n2️⃣ STEP 2: Match-Level Splitting")
    logger.info("Splitting matches into train/validation/test sets...")
    
    try:
        train_matches, val_matches, test_matches = split_matches(
            input_csv=aligned_file,
            output_dir=output_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=42,
            match_id_column="match_id"
        )
        
        logger.info(f"✅ Split completed:")
        logger.info(f"   Train: {len(train_matches)} matches")
        logger.info(f"   Validation: {len(val_matches)} matches")
        logger.info(f"   Test: {len(test_matches)} matches")
        logger.info(f"   Files saved to: {output_dir}/")
        
    except Exception as e:
        logger.error(f"❌ Match splitting failed: {e}")
        return
    
    # Step 3: Dataset Loading with Filtering
    logger.info("\n3️⃣ STEP 3: Dataset Loading with Match Filtering")
    logger.info("Loading datasets with match-level filtering...")
    
    try:
        # Load training dataset
        train_dataset = CrickformerDataset(
            data_root=data_path,
            use_csv_adapter=True,
            csv_config=CSVDataConfig(),
            history_length=5,
            load_video=True,
            load_embeddings=True,
            load_market_odds=True,
            match_id_list_path=str(output_dir / "train_matches.csv")
        )
        
        # Load validation dataset
        val_dataset = CrickformerDataset(
            data_root=data_path,
            use_csv_adapter=True,
            csv_config=CSVDataConfig(),
            history_length=5,
            load_video=True,
            load_embeddings=True,
            load_market_odds=True,
            match_id_list_path=str(output_dir / "val_matches.csv")
        )
        
        # Load test dataset
        test_dataset = CrickformerDataset(
            data_root=data_path,
            use_csv_adapter=True,
            csv_config=CSVDataConfig(),
            history_length=5,
            load_video=True,
            load_embeddings=True,
            load_market_odds=True,
            match_id_list_path=str(output_dir / "test_matches.csv")
        )
        
        logger.info(f"✅ Datasets loaded successfully:")
        logger.info(f"   Training: {len(train_dataset):,} samples from {len(train_dataset.get_match_ids())} matches")
        logger.info(f"   Validation: {len(val_dataset):,} samples from {len(val_dataset.get_match_ids())} matches")
        logger.info(f"   Test: {len(test_dataset):,} samples from {len(test_dataset.get_match_ids())} matches")
        
        # Verify no overlap between datasets
        train_matches_set = set(train_dataset.get_match_ids())
        val_matches_set = set(val_dataset.get_match_ids())
        test_matches_set = set(test_dataset.get_match_ids())
        
        train_val_overlap = train_matches_set & val_matches_set
        train_test_overlap = train_matches_set & test_matches_set
        val_test_overlap = val_matches_set & test_matches_set
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            logger.warning(f"⚠️  Found overlapping matches between datasets!")
            logger.warning(f"   Train-Val overlap: {len(train_val_overlap)}")
            logger.warning(f"   Train-Test overlap: {len(train_test_overlap)}")
            logger.warning(f"   Val-Test overlap: {len(val_test_overlap)}")
        else:
            logger.info("✅ No match overlap detected between datasets")
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        logger.info("This is expected if the actual data files are not available")
        return
    
    # Step 4: Training Command
    logger.info("\n4️⃣ STEP 4: Training Command")
    logger.info("Ready to train! Use this command:")
    
    train_command = f"""
PYTHONPATH=. python3 crickformers/train.py \\
    --data-path "{data_path}" \\
    --train-matches "{output_dir}/train_matches.csv" \\
    --val-matches "{output_dir}/val_matches.csv" \\
    --epochs 10 \\
    --batch-size 32 \\
    --save-path "models/crickformer_match_level.pt"
"""
    
    logger.info(train_command)
    
    # Step 5: Summary
    logger.info("\n📋 WORKFLOW SUMMARY")
    logger.info("=" * 50)
    logger.info("✅ Complete workflow demonstrated:")
    logger.info("   1. Match alignment between data sources")
    logger.info("   2. Match-level splitting (train/val/test)")
    logger.info("   3. Dataset loading with match filtering")
    logger.info("   4. Training command generation")
    logger.info("")
    logger.info("🎯 Key Benefits:")
    logger.info("   • No data leakage between train/val/test")
    logger.info("   • Proper match-level evaluation")
    logger.info("   • Reproducible splits with fixed seed")
    logger.info("   • Scalable to any dataset size")
    logger.info("")
    logger.info("📁 Output files:")
    logger.info(f"   • {output_dir}/aligned_matches.csv")
    logger.info(f"   • {output_dir}/train_matches.csv")
    logger.info(f"   • {output_dir}/val_matches.csv")
    logger.info(f"   • {output_dir}/test_matches.csv")


if __name__ == "__main__":
    main() 
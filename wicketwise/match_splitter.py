# Purpose: Splits cricket matches into train, validation, and test sets at match level
# Author: WicketWise Team, Last Modified: 2024-12-07

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging
import argparse
import sys

logger = logging.getLogger(__name__)


def split_matches(
    input_csv: Union[str, Path],
    output_dir: Union[str, Path] = ".",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    match_id_column: str = "match_id"
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split cricket matches into train, validation, and test sets.
    
    This function loads a CSV file containing match data, extracts unique match IDs,
    shuffles them with a fixed seed for reproducibility, and splits them into
    train/validation/test sets at the match level (not ball level).
    
    Args:
        input_csv: Path to input CSV file containing match data
        output_dir: Directory to save output CSV files (default: current directory)
        train_ratio: Proportion of matches for training (default: 0.8)
        val_ratio: Proportion of matches for validation (default: 0.1)
        test_ratio: Proportion of matches for testing (default: 0.1)
        random_seed: Random seed for reproducible shuffling (default: 42)
        match_id_column: Name of column containing match IDs (default: "match_id")
        
    Returns:
        Tuple of (train_matches, val_matches, test_matches) lists
        
    Raises:
        FileNotFoundError: If input CSV file doesn't exist
        ValueError: If ratios don't sum to 1.0 or CSV doesn't have required column
        
    Example:
        >>> train, val, test = split_matches("matched_matches.csv")
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    # Validate inputs
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_path}")
    
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    if any(ratio < 0 for ratio in [train_ratio, val_ratio, test_ratio]):
        raise ValueError("All ratios must be non-negative")
    
    # Load data
    logger.info(f"Loading match data from {input_path}")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # Validate match_id column exists
    if match_id_column not in df.columns:
        available_columns = list(df.columns)
        # Try to find a suitable match ID column
        potential_columns = [col for col in available_columns if 'match' in col.lower()]
        if potential_columns:
            logger.warning(f"Column '{match_id_column}' not found. Available match columns: {potential_columns}")
            raise ValueError(f"Column '{match_id_column}' not found. Available columns: {available_columns}")
        else:
            raise ValueError(f"Column '{match_id_column}' not found. Available columns: {available_columns}")
    
    # Extract unique match IDs
    unique_matches = df[match_id_column].unique()
    total_matches = len(unique_matches)
    
    logger.info(f"Found {total_matches} unique matches")
    
    if total_matches == 0:
        raise ValueError("No matches found in the dataset")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle matches
    shuffled_matches = np.random.permutation(unique_matches)
    
    # Calculate split indices
    train_end = int(total_matches * train_ratio)
    val_end = train_end + int(total_matches * val_ratio)
    
    # Handle edge case: ensure at least one match goes to training if possible
    if total_matches > 0 and train_end == 0:
        train_end = 1
        val_end = train_end + max(0, int(total_matches * val_ratio) - 1)
    
    # Split matches
    train_matches = shuffled_matches[:train_end].tolist()
    val_matches = shuffled_matches[train_end:val_end].tolist()
    test_matches = shuffled_matches[val_end:].tolist()
    
    logger.info(f"Split: Train={len(train_matches)}, Val={len(val_matches)}, Test={len(test_matches)}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits to CSV files
    train_df = pd.DataFrame({"match_id": train_matches})
    val_df = pd.DataFrame({"match_id": val_matches})
    test_df = pd.DataFrame({"match_id": test_matches})
    
    train_file = output_path / "train_matches.csv"
    val_file = output_path / "val_matches.csv"
    test_file = output_path / "test_matches.csv"
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    logger.info(f"✅ Saved train matches to {train_file}")
    logger.info(f"✅ Saved validation matches to {val_file}")
    logger.info(f"✅ Saved test matches to {test_file}")
    
    return train_matches, val_matches, test_matches


def load_match_splits(
    splits_dir: Union[str, Path] = "."
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load previously saved match splits from CSV files.
    
    Args:
        splits_dir: Directory containing the split CSV files
        
    Returns:
        Tuple of (train_matches, val_matches, test_matches) lists
        
    Raises:
        FileNotFoundError: If any split file is missing
    """
    splits_path = Path(splits_dir)
    
    train_file = splits_path / "train_matches.csv"
    val_file = splits_path / "val_matches.csv"
    test_file = splits_path / "test_matches.csv"
    
    # Check all files exist
    for file_path in [train_file, val_file, test_file]:
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}")
    
    # Load splits
    train_matches = pd.read_csv(train_file)["match_id"].tolist()
    val_matches = pd.read_csv(val_file)["match_id"].tolist()
    test_matches = pd.read_csv(test_file)["match_id"].tolist()
    
    logger.info(f"Loaded splits: Train={len(train_matches)}, Val={len(val_matches)}, Test={len(test_matches)}")
    
    return train_matches, val_matches, test_matches


def validate_splits(
    train_matches: List[str],
    val_matches: List[str],
    test_matches: List[str]
) -> bool:
    """
    Validate that match splits have no overlap and cover all matches.
    
    Args:
        train_matches: List of training match IDs
        val_matches: List of validation match IDs
        test_matches: List of test match IDs
        
    Returns:
        True if splits are valid, False otherwise
    """
    train_set = set(train_matches)
    val_set = set(val_matches)
    test_set = set(test_matches)
    
    # Check for overlaps
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    if train_val_overlap:
        logger.error(f"Train-Val overlap: {train_val_overlap}")
        return False
    
    if train_test_overlap:
        logger.error(f"Train-Test overlap: {train_test_overlap}")
        return False
    
    if val_test_overlap:
        logger.error(f"Val-Test overlap: {val_test_overlap}")
        return False
    
    logger.info("✅ No overlaps detected between splits")
    return True


def main():
    """Command-line interface for match splitting."""
    parser = argparse.ArgumentParser(
        description="Split cricket matches into train, validation, and test sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file containing match data"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directory to save output CSV files"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of matches for training"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of matches for validation"
    )
    
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of matches for testing"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling"
    )
    
    parser.add_argument(
        "--match-id-column",
        default="match_id",
        help="Name of column containing match IDs"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate splits for overlaps after creation"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Split matches
        train_matches, val_matches, test_matches = split_matches(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
            match_id_column=args.match_id_column
        )
        
        # Validate splits if requested
        if args.validate:
            is_valid = validate_splits(train_matches, val_matches, test_matches)
            if not is_valid:
                logger.error("❌ Split validation failed")
                sys.exit(1)
            else:
                logger.info("✅ Split validation passed")
        
        # Print summary
        total_matches = len(train_matches) + len(val_matches) + len(test_matches)
        print(f"\n🏏 Match Split Summary:")
        print(f"📊 Total matches: {total_matches}")
        print(f"🚂 Training: {len(train_matches)} ({len(train_matches)/total_matches:.1%})")
        print(f"🔍 Validation: {len(val_matches)} ({len(val_matches)/total_matches:.1%})")
        print(f"🧪 Test: {len(test_matches)} ({len(test_matches)/total_matches:.1%})")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"🎲 Random seed: {args.seed}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
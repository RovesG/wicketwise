# Purpose: Tests for match_splitter.py module
# Author: WicketWise Team, Last Modified: 2024-12-07

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add the parent directory to the path so we can import match_splitter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wicketwise.match_splitter import (
    split_matches,
    load_match_splits,
    validate_splits,
    main
)


class TestMatchSplitter:
    """Test suite for match_splitter.py functionality."""
    
    @pytest.fixture
    def sample_match_data(self):
        """Create sample match data for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create sample data with 10 unique matches
        data = {
            'match_id': ['match_001', 'match_002', 'match_003', 'match_004', 'match_005',
                        'match_006', 'match_007', 'match_008', 'match_009', 'match_010'] * 5,  # 50 balls
            'ball_id': [f"{i//5 + 1}.{i%5 + 1}" for i in range(50)],
            'actual_runs': np.random.randint(0, 7, 50),
            'predicted_runs_class': ['0_runs'] * 50,
            'win_prob': np.random.rand(50),
            'odds_mispricing': np.random.rand(50),
            'phase': ['powerplay'] * 30 + ['middle_overs'] * 20,
            'batter_id': [f'batter_{i%5 + 1}' for i in range(50)],
            'bowler_id': [f'bowler_{i%3 + 1}' for i in range(50)]
        }
        
        df = pd.DataFrame(data)
        csv_path = Path(temp_dir) / "sample_matches.csv"
        df.to_csv(csv_path, index=False)
        
        return {
            'temp_dir': temp_dir,
            'csv_path': str(csv_path),
            'expected_matches': 10
        }
    
    @pytest.fixture
    def large_match_data(self):
        """Create larger sample data for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create data with 100 unique matches
        match_ids = [f'match_{i:03d}' for i in range(1, 101)]
        data = {
            'match_id': match_ids * 2,  # 200 balls total
            'ball_id': [f"{i//2 + 1}.{i%2 + 1}" for i in range(200)],
            'actual_runs': np.random.randint(0, 7, 200),
            'predicted_runs_class': ['0_runs'] * 200,
            'win_prob': np.random.rand(200),
            'odds_mispricing': np.random.rand(200),
            'phase': ['powerplay'] * 200,
            'batter_id': [f'batter_{i%10 + 1}' for i in range(200)],
            'bowler_id': [f'bowler_{i%5 + 1}' for i in range(200)]
        }
        
        df = pd.DataFrame(data)
        csv_path = Path(temp_dir) / "large_matches.csv"
        df.to_csv(csv_path, index=False)
        
        return {
            'temp_dir': temp_dir,
            'csv_path': str(csv_path),
            'expected_matches': 100
        }
    
    def test_basic_split_functionality(self, sample_match_data):
        """Test basic split functionality with default parameters."""
        train, val, test = split_matches(
            input_csv=sample_match_data['csv_path'],
            output_dir=sample_match_data['temp_dir']
        )
        
        # Check lengths
        assert len(train) == 8  # 80% of 10 matches
        assert len(val) == 1    # 10% of 10 matches
        assert len(test) == 1   # 10% of 10 matches
        
        # Check no overlaps
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0
        
        # Check all matches are covered
        all_matches = set(train + val + test)
        assert len(all_matches) == 10
    
    def test_reproducibility_with_same_seed(self, sample_match_data):
        """Test that same seed produces identical splits."""
        # First split
        train1, val1, test1 = split_matches(
            input_csv=sample_match_data['csv_path'],
            output_dir=sample_match_data['temp_dir'],
            random_seed=42
        )
        
        # Second split with same seed
        train2, val2, test2 = split_matches(
            input_csv=sample_match_data['csv_path'],
            output_dir=sample_match_data['temp_dir'],
            random_seed=42
        )
        
        # Should be identical
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2
    
    def test_different_seeds_produce_different_splits(self, sample_match_data):
        """Test that different seeds produce different splits."""
        # First split
        train1, val1, test1 = split_matches(
            input_csv=sample_match_data['csv_path'],
            output_dir=sample_match_data['temp_dir'],
            random_seed=42
        )
        
        # Second split with different seed
        train2, val2, test2 = split_matches(
            input_csv=sample_match_data['csv_path'],
            output_dir=sample_match_data['temp_dir'],
            random_seed=123
        )
        
        # Should be different (with high probability)
        assert train1 != train2 or val1 != val2 or test1 != test2
    
    def test_custom_ratios(self, sample_match_data):
        """Test custom train/val/test ratios."""
        train, val, test = split_matches(
            input_csv=sample_match_data['csv_path'],
            output_dir=sample_match_data['temp_dir'],
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # Check lengths (6/2/2 for 10 matches)
        assert len(train) == 6
        assert len(val) == 2
        assert len(test) == 2
        
        # Check total
        assert len(train) + len(val) + len(test) == 10
    
    def test_large_dataset_split(self, large_match_data):
        """Test splitting with a larger dataset."""
        train, val, test = split_matches(
            input_csv=large_match_data['csv_path'],
            output_dir=large_match_data['temp_dir']
        )
        
        # Check approximate lengths (80/10/10 for 100 matches)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
        
        # Check no overlaps
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0
    
    def test_output_files_created(self, sample_match_data):
        """Test that output CSV files are created correctly."""
        output_dir = sample_match_data['temp_dir']
        
        split_matches(
            input_csv=sample_match_data['csv_path'],
            output_dir=output_dir
        )
        
        # Check files exist
        train_file = Path(output_dir) / "train_matches.csv"
        val_file = Path(output_dir) / "val_matches.csv"
        test_file = Path(output_dir) / "test_matches.csv"
        
        assert train_file.exists()
        assert val_file.exists()
        assert test_file.exists()
        
        # Check file contents
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        
        assert 'match_id' in train_df.columns
        assert 'match_id' in val_df.columns
        assert 'match_id' in test_df.columns
        
        assert len(train_df) == 8
        assert len(val_df) == 1
        assert len(test_df) == 1
    
    def test_load_match_splits(self, sample_match_data):
        """Test loading previously saved match splits."""
        output_dir = sample_match_data['temp_dir']
        
        # First create splits
        original_train, original_val, original_test = split_matches(
            input_csv=sample_match_data['csv_path'],
            output_dir=output_dir
        )
        
        # Then load them
        loaded_train, loaded_val, loaded_test = load_match_splits(output_dir)
        
        # Should be identical
        assert original_train == loaded_train
        assert original_val == loaded_val
        assert original_test == loaded_test
    
    def test_validate_splits_valid_case(self, sample_match_data):
        """Test validation of valid splits."""
        train, val, test = split_matches(
            input_csv=sample_match_data['csv_path'],
            output_dir=sample_match_data['temp_dir']
        )
        
        # Should be valid
        assert validate_splits(train, val, test) is True
    
    def test_validate_splits_overlap_case(self):
        """Test validation detects overlaps."""
        train = ['match_001', 'match_002', 'match_003']
        val = ['match_003', 'match_004']  # Overlap with train
        test = ['match_005', 'match_006']
        
        # Should detect overlap
        assert validate_splits(train, val, test) is False
    
    def test_invalid_input_file(self):
        """Test handling of non-existent input file."""
        with pytest.raises(FileNotFoundError):
            split_matches("nonexistent_file.csv")
    
    def test_invalid_ratios(self, sample_match_data):
        """Test handling of invalid ratio combinations."""
        # Ratios don't sum to 1.0
        with pytest.raises(ValueError):
            split_matches(
                input_csv=sample_match_data['csv_path'],
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3  # Sum = 1.1
            )
        
        # Negative ratio
        with pytest.raises(ValueError):
            split_matches(
                input_csv=sample_match_data['csv_path'],
                train_ratio=0.9,
                val_ratio=-0.1,
                test_ratio=0.2
            )
    
    def test_missing_match_id_column(self, sample_match_data):
        """Test handling of missing match_id column."""
        # Create CSV without match_id column
        temp_dir = tempfile.mkdtemp()
        data = {
            'game_id': ['game_001', 'game_002', 'game_003'],
            'ball_id': ['1.1', '1.2', '1.3'],
            'runs': [0, 1, 4]
        }
        df = pd.DataFrame(data)
        csv_path = Path(temp_dir) / "no_match_id.csv"
        df.to_csv(csv_path, index=False)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Column 'match_id' not found"):
            split_matches(str(csv_path))
    
    def test_custom_match_id_column(self, sample_match_data):
        """Test using custom match ID column name."""
        # Create CSV with different column name
        temp_dir = tempfile.mkdtemp()
        df = pd.read_csv(sample_match_data['csv_path'])
        df = df.rename(columns={'match_id': 'game_id'})
        csv_path = Path(temp_dir) / "custom_column.csv"
        df.to_csv(csv_path, index=False)
        
        # Should work with custom column name
        train, val, test = split_matches(
            input_csv=str(csv_path),
            output_dir=temp_dir,
            match_id_column='game_id'
        )
        
        assert len(train) + len(val) + len(test) == 10
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        temp_dir = tempfile.mkdtemp()
        
        # Create empty CSV
        df = pd.DataFrame(columns=['match_id', 'ball_id', 'runs'])
        csv_path = Path(temp_dir) / "empty.csv"
        df.to_csv(csv_path, index=False)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="No matches found"):
            split_matches(str(csv_path))
    
    def test_single_match_dataset(self):
        """Test handling of dataset with single match."""
        temp_dir = tempfile.mkdtemp()
        
        # Create single match data
        data = {
            'match_id': ['match_001'] * 5,
            'ball_id': ['1.1', '1.2', '1.3', '1.4', '1.5'],
            'runs': [0, 1, 4, 0, 2]
        }
        df = pd.DataFrame(data)
        csv_path = Path(temp_dir) / "single_match.csv"
        df.to_csv(csv_path, index=False)
        
        # Should work but put single match in train
        train, val, test = split_matches(
            input_csv=str(csv_path),
            output_dir=temp_dir
        )
        
        assert len(train) == 1
        assert len(val) == 0
        assert len(test) == 0
        assert train[0] == 'match_001'
    
    def test_load_nonexistent_splits(self):
        """Test loading splits when files don't exist."""
        temp_dir = tempfile.mkdtemp()
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_match_splits(temp_dir)
    
    def test_different_match_id_formats(self):
        """Test with different match ID formats."""
        temp_dir = tempfile.mkdtemp()
        
        # Create data with various match ID formats
        data = {
            'match_id': ['BBL_2024_001', 'IPL_2024_002', 'PSL_2024_003', 'CPL_2024_004', 'T20WC_2024_005'],
            'ball_id': ['1.1', '1.2', '1.3', '1.4', '1.5'],
            'runs': [0, 1, 4, 0, 2]
        }
        df = pd.DataFrame(data)
        csv_path = Path(temp_dir) / "different_formats.csv"
        df.to_csv(csv_path, index=False)
        
        # Should work with different formats
        train, val, test = split_matches(
            input_csv=str(csv_path),
            output_dir=temp_dir
        )
        
        total_matches = len(train) + len(val) + len(test)
        assert total_matches == 5
        
        # Check all match IDs are preserved
        all_matches = set(train + val + test)
        expected_matches = set(data['match_id'])
        assert all_matches == expected_matches


class TestCommandLineInterface:
    """Test the command-line interface."""
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for CLI testing."""
        temp_dir = tempfile.mkdtemp()
        
        data = {
            'match_id': ['match_001', 'match_002', 'match_003', 'match_004', 'match_005'] * 2,
            'ball_id': [f"{i//2 + 1}.{i%2 + 1}" for i in range(10)],
            'runs': [0, 1, 4, 0, 2, 1, 0, 6, 0, 1]
        }
        df = pd.DataFrame(data)
        csv_path = Path(temp_dir) / "cli_test.csv"
        df.to_csv(csv_path, index=False)
        
        return {
            'temp_dir': temp_dir,
            'csv_path': str(csv_path)
        }
    
    def test_cli_basic_usage(self, sample_csv_file, capsys):
        """Test basic CLI usage."""
        # Mock sys.argv
        import sys
        original_argv = sys.argv
        
        try:
            sys.argv = [
                'match_splitter.py',
                sample_csv_file['csv_path'],
                '--output-dir', sample_csv_file['temp_dir'],
                '--verbose'
            ]
            
            # Should not raise exception
            main()
            
            # Check output files were created
            output_dir = Path(sample_csv_file['temp_dir'])
            assert (output_dir / "train_matches.csv").exists()
            assert (output_dir / "val_matches.csv").exists()
            assert (output_dir / "test_matches.csv").exists()
            
        finally:
            sys.argv = original_argv
    
    def test_cli_with_custom_ratios(self, sample_csv_file):
        """Test CLI with custom ratios."""
        import sys
        original_argv = sys.argv
        
        try:
            sys.argv = [
                'match_splitter.py',
                sample_csv_file['csv_path'],
                '--output-dir', sample_csv_file['temp_dir'],
                '--train-ratio', '0.6',
                '--val-ratio', '0.2',
                '--test-ratio', '0.2',
                '--seed', '123'
            ]
            
            # Should not raise exception
            main()
            
            # Check splits have expected sizes
            output_dir = Path(sample_csv_file['temp_dir'])
            train_df = pd.read_csv(output_dir / "train_matches.csv")
            val_df = pd.read_csv(output_dir / "val_matches.csv")
            test_df = pd.read_csv(output_dir / "test_matches.csv")
            
            assert len(train_df) == 3  # 60% of 5
            assert len(val_df) == 1    # 20% of 5
            assert len(test_df) == 1   # 20% of 5
            
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    pytest.main([__file__]) 
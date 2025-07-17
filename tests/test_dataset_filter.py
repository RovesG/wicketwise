# Purpose: Tests for CrickformerDataset match filtering functionality
# Author: WicketWise Team, Last Modified: 2024-12-07

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import crickformers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crickformers.crickformer_dataset import CrickformerDataset
from crickformers.csv_data_adapter import CSVDataAdapter, CSVDataConfig


class TestDatasetFiltering:
    """Test suite for CrickformerDataset match filtering functionality."""
    
    @pytest.fixture
    def mock_csv_data_dir(self):
        """Create mock CSV data directory with multiple matches."""
        temp_dir = tempfile.mkdtemp()
        
        # Create mock NVPlay data with 5 matches
        nvplay_data = {
            'Competition': ['Test League'] * 25,
            'Match': ['match_001'] * 5 + ['match_002'] * 5 + ['match_003'] * 5 + ['match_004'] * 5 + ['match_005'] * 5,
            'Date': ['2024-01-01'] * 25,
            'Innings': [1] * 25,
            'Over': [1, 1, 1, 2, 2] * 5,
            'Ball': [1, 2, 3, 1, 2] * 5,
            'Innings Ball': [1, 2, 3, 4, 5] * 5,
            'Batter': [f'Batter_{i%3 + 1}' for i in range(25)],
            'Batter ID': [f'BAT_{i%3 + 1:03d}' for i in range(25)],
            'Bowler': [f'Bowler_{i%2 + 1}' for i in range(25)],
            'Bowler ID': [f'BOW_{i%2 + 1:03d}' for i in range(25)],
            'Runs': [0, 1, 4, 0, 2] * 5,
            'Extra Runs': [0] * 25,
            'Wicket': ['No Wicket'] * 25,
            'Team Runs': [0, 1, 5, 5, 7] * 5,
            'Team Wickets': [0] * 25,
            'Batting Team': ['Team_A'] * 25,
            'Bowling Team': ['Team_B'] * 25,
            'Batting Hand': ['RHB'] * 25,
            'Bowler Type': ['RM'] * 25,
            'FieldX': [100.0] * 25,
            'FieldY': [150.0] * 25,
            'PitchX': [0.0] * 25,
            'PitchY': [0.0] * 25,
            'Power Play': [1.0] * 25,
            'Run Rate After': [6.0] * 25,
            'Req Run Rate After': [8.0] * 25,
            'Venue': ['Test Ground'] * 25
        }
        
        # Create mock decimal data
        decimal_data = {
            'date': ['2024-01-01'] * 25,
            'competition': ['Test League'] * 25,
            'home': ['Team_A'] * 25,
            'away': ['Team_B'] * 25,
            'innings': [1] * 25,
            'ball': [1, 2, 3, 4, 5] * 5,
            'win_prob': [0.5] * 25
        }
        
        # Save as CSV files
        nvplay_df = pd.DataFrame(nvplay_data)
        decimal_df = pd.DataFrame(decimal_data)
        
        nvplay_path = Path(temp_dir) / 'nvplay_data_v3.csv'
        decimal_path = Path(temp_dir) / 'decimal_data_v3.csv'
        
        nvplay_df.to_csv(nvplay_path, index=False)
        decimal_df.to_csv(decimal_path, index=False)
        
        return {
            'temp_dir': temp_dir,
            'nvplay_path': str(nvplay_path),
            'decimal_path': str(decimal_path),
            'all_matches': ['match_001', 'match_002', 'match_003', 'match_004', 'match_005']
        }
    
    @pytest.fixture
    def mock_directory_data(self):
        """Create mock directory structure with multiple matches."""
        temp_dir = tempfile.mkdtemp()
        
        # Create 4 match directories
        match_ids = ['match_001', 'match_002', 'match_003', 'match_004']
        
        for match_id in match_ids:
            match_dir = Path(temp_dir) / match_id
            match_dir.mkdir()
            
            # Create current_ball_features directory
            features_dir = match_dir / 'current_ball_features'
            features_dir.mkdir()
            
            # Create 3 ball files per match
            for ball_num in range(1, 4):
                ball_id = f"ball_{ball_num:03d}"
                ball_file = features_dir / f"{ball_id}.json"
                
                # Create mock ball features
                ball_features = {
                    "match_id": match_id,
                    "competition_name": "Test League",
                    "venue": "Test Ground",
                    "venue_city": "Test City",
                    "venue_country": "Test Country",
                    "innings": 1,
                    "over": 1.0,
                    "ball_in_over": ball_num,
                    "innings_ball": ball_num,
                    "batter_name": f"Batter_{ball_num}",
                    "batter_id": f"BAT_{ball_num:03d}",
                    "bowler_name": "Bowler_1",
                    "bowler_id": "BOW_001",
                    "runs_scored": ball_num - 1,
                    "extras": 0,
                    "is_wicket": False,
                    "team_score": ball_num,
                    "team_wickets": 0,
                    "batting_team": "Team_A",
                    "bowling_team": "Team_B",
                    "batter_hand": "RHB",
                    "bowler_type": "RM",
                    "field_x": 100.0,
                    "field_y": 150.0,
                    "pitch_x": 0.0,
                    "pitch_y": 0.0,
                    "powerplay": 1.0,
                    "run_rate": 6.0,
                    "req_run_rate": 8.0
                }
                
                with open(ball_file, 'w') as f:
                    json.dump(ball_features, f)
        
        return {
            'temp_dir': temp_dir,
            'all_matches': match_ids
        }
    
    def test_no_filtering_csv_adapter(self, mock_csv_data_dir):
        """Test dataset loading without filtering (CSV adapter)."""
        # Create dataset without filtering
        dataset = CrickformerDataset(
            data_root=mock_csv_data_dir['temp_dir'],
            use_csv_adapter=True,
            csv_config=CSVDataConfig(),
            load_video=False,
            load_embeddings=False,
            load_market_odds=False
        )
        
        # Should load all matches
        assert len(dataset) == 25  # 5 balls per match * 5 matches
        assert len(dataset.get_match_ids()) == 5
        assert set(dataset.get_match_ids()) == set(mock_csv_data_dir['all_matches'])
    
    def test_no_filtering_directory_structure(self, mock_directory_data):
        """Test dataset loading without filtering (directory structure)."""
        # Create dataset without filtering
        dataset = CrickformerDataset(
            data_root=mock_directory_data['temp_dir'],
            use_csv_adapter=False,
            load_video=False,
            load_embeddings=False,
            load_market_odds=False
        )
        
        # Should load all matches
        assert len(dataset) == 12  # 3 balls per match * 4 matches
        assert len(dataset.get_match_ids()) == 4
        assert set(dataset.get_match_ids()) == set(mock_directory_data['all_matches'])
    
    def test_filtering_with_csv_adapter(self, mock_csv_data_dir):
        """Test dataset filtering with CSV adapter."""
        temp_dir = mock_csv_data_dir['temp_dir']
        
        # Create filter file with 2 matches
        filter_matches = ['match_001', 'match_003']
        filter_df = pd.DataFrame({'match_id': filter_matches})
        filter_path = Path(temp_dir) / 'filter_matches.csv'
        filter_df.to_csv(filter_path, index=False)
        
        # Create dataset with filtering
        dataset = CrickformerDataset(
            data_root=temp_dir,
            use_csv_adapter=True,
            csv_config=CSVDataConfig(),
            match_id_list_path=str(filter_path),
            load_video=False,
            load_embeddings=False,
            load_market_odds=False
        )
        
        # Should only load filtered matches
        assert len(dataset) == 10  # 5 balls per match * 2 matches
        assert len(dataset.get_match_ids()) == 2
        assert set(dataset.get_match_ids()) == set(filter_matches)
        
        # Check that samples are from correct matches
        for i in range(len(dataset)):
            sample_info = dataset.get_sample_info(i)
            assert sample_info['match_id'] in filter_matches
    
    def test_filtering_with_directory_structure(self, mock_directory_data):
        """Test dataset filtering with directory structure."""
        temp_dir = mock_directory_data['temp_dir']
        
        # Create filter file with 2 matches
        filter_matches = ['match_002', 'match_004']
        filter_df = pd.DataFrame({'match_id': filter_matches})
        filter_path = Path(temp_dir) / 'filter_matches.csv'
        filter_df.to_csv(filter_path, index=False)
        
        # Create dataset with filtering
        dataset = CrickformerDataset(
            data_root=temp_dir,
            use_csv_adapter=False,
            match_id_list_path=str(filter_path),
            load_video=False,
            load_embeddings=False,
            load_market_odds=False
        )
        
        # Should only load filtered matches
        assert len(dataset) == 6  # 3 balls per match * 2 matches
        assert len(dataset.get_match_ids()) == 2
        assert set(dataset.get_match_ids()) == set(filter_matches)
        
        # Check that samples are from correct matches
        for i in range(len(dataset)):
            sample_info = dataset.get_sample_info(i)
            assert sample_info['match_id'] in filter_matches
    
    def test_filtering_nonexistent_matches(self, mock_csv_data_dir):
        """Test filtering with matches that don't exist in the dataset."""
        temp_dir = mock_csv_data_dir['temp_dir']
        
        # Create filter file with mix of existing and non-existing matches
        filter_matches = ['match_001', 'match_999', 'match_002']  # match_999 doesn't exist
        filter_df = pd.DataFrame({'match_id': filter_matches})
        filter_path = Path(temp_dir) / 'filter_matches.csv'
        filter_df.to_csv(filter_path, index=False)
        
        # Create dataset with filtering
        dataset = CrickformerDataset(
            data_root=temp_dir,
            use_csv_adapter=True,
            csv_config=CSVDataConfig(),
            match_id_list_path=str(filter_path),
            load_video=False,
            load_embeddings=False,
            load_market_odds=False
        )
        
        # Should only load existing matches from filter
        assert len(dataset) == 10  # 5 balls per match * 2 existing matches
        assert len(dataset.get_match_ids()) == 2
        assert set(dataset.get_match_ids()) == {'match_001', 'match_002'}
    
    def test_filtering_empty_result(self, mock_csv_data_dir):
        """Test filtering that results in empty dataset."""
        temp_dir = mock_csv_data_dir['temp_dir']
        
        # Create filter file with non-existing matches
        filter_matches = ['match_999', 'match_888']
        filter_df = pd.DataFrame({'match_id': filter_matches})
        filter_path = Path(temp_dir) / 'filter_matches.csv'
        filter_df.to_csv(filter_path, index=False)
        
        # Create dataset with filtering
        dataset = CrickformerDataset(
            data_root=temp_dir,
            use_csv_adapter=True,
            csv_config=CSVDataConfig(),
            match_id_list_path=str(filter_path),
            load_video=False,
            load_embeddings=False,
            load_market_odds=False
        )
        
        # Should result in empty dataset
        assert len(dataset) == 0
        assert len(dataset.get_match_ids()) == 0
    
    def test_filter_file_not_found(self, mock_csv_data_dir):
        """Test error handling when filter file doesn't exist."""
        temp_dir = mock_csv_data_dir['temp_dir']
        nonexistent_path = Path(temp_dir) / 'nonexistent_filter.csv'
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Match filter file not found"):
            CrickformerDataset(
                data_root=temp_dir,
                use_csv_adapter=True,
                csv_config=CSVDataConfig(),
                match_id_list_path=str(nonexistent_path),
                load_video=False,
                load_embeddings=False,
                load_market_odds=False
            )
    
    def test_filter_file_wrong_format(self, mock_csv_data_dir):
        """Test error handling when filter file has wrong format."""
        temp_dir = mock_csv_data_dir['temp_dir']
        
        # Create filter file with wrong column name
        filter_df = pd.DataFrame({'game_id': ['match_001', 'match_002']})  # Wrong column name
        filter_path = Path(temp_dir) / 'wrong_format.csv'
        filter_df.to_csv(filter_path, index=False)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="CSV file must contain 'match_id' column"):
            CrickformerDataset(
                data_root=temp_dir,
                use_csv_adapter=True,
                csv_config=CSVDataConfig(),
                match_id_list_path=str(filter_path),
                load_video=False,
                load_embeddings=False,
                load_market_odds=False
            )
    
    def test_filter_file_with_duplicates(self, mock_csv_data_dir):
        """Test filtering with duplicate match IDs in filter file."""
        temp_dir = mock_csv_data_dir['temp_dir']
        
        # Create filter file with duplicates
        filter_matches = ['match_001', 'match_002', 'match_001', 'match_002']  # Duplicates
        filter_df = pd.DataFrame({'match_id': filter_matches})
        filter_path = Path(temp_dir) / 'filter_with_duplicates.csv'
        filter_df.to_csv(filter_path, index=False)
        
        # Create dataset with filtering
        dataset = CrickformerDataset(
            data_root=temp_dir,
            use_csv_adapter=True,
            csv_config=CSVDataConfig(),
            match_id_list_path=str(filter_path),
            load_video=False,
            load_embeddings=False,
            load_market_odds=False
        )
        
        # Should handle duplicates correctly
        assert len(dataset) == 10  # 5 balls per match * 2 unique matches
        assert len(dataset.get_match_ids()) == 2
        assert set(dataset.get_match_ids()) == {'match_001', 'match_002'}
    
    def test_filter_file_with_na_values(self, mock_csv_data_dir):
        """Test filtering with NaN values in filter file."""
        temp_dir = mock_csv_data_dir['temp_dir']
        
        # Create filter file with NaN values
        filter_df = pd.DataFrame({
            'match_id': ['match_001', np.nan, 'match_002', None, 'match_003']
        })
        filter_path = Path(temp_dir) / 'filter_with_na.csv'
        filter_df.to_csv(filter_path, index=False)
        
        # Create dataset with filtering
        dataset = CrickformerDataset(
            data_root=temp_dir,
            use_csv_adapter=True,
            csv_config=CSVDataConfig(),
            match_id_list_path=str(filter_path),
            load_video=False,
            load_embeddings=False,
            load_market_odds=False
        )
        
        # Should ignore NaN values
        assert len(dataset) == 15  # 5 balls per match * 3 valid matches
        assert len(dataset.get_match_ids()) == 3
        assert set(dataset.get_match_ids()) == {'match_001', 'match_002', 'match_003'}
    
    def test_filtering_logging(self, mock_csv_data_dir, caplog):
        """Test that filtering produces appropriate log messages."""
        temp_dir = mock_csv_data_dir['temp_dir']
        
        # Create filter file
        filter_matches = ['match_001', 'match_003']
        filter_df = pd.DataFrame({'match_id': filter_matches})
        filter_path = Path(temp_dir) / 'filter_matches.csv'
        filter_df.to_csv(filter_path, index=False)
        
        # Create dataset with filtering
        with caplog.at_level('INFO'):
            dataset = CrickformerDataset(
                data_root=temp_dir,
                use_csv_adapter=True,
                csv_config=CSVDataConfig(),
                match_id_list_path=str(filter_path),
                load_video=False,
                load_embeddings=False,
                load_market_odds=False
            )
        
        # Check log messages
        log_messages = [record.message for record in caplog.records]
        
        # Should log filter file loading
        assert any("Loaded 2 match IDs from filter file" in msg for msg in log_messages)
        
        # Should log filtering results
        assert any("Applied match filtering: 25 → 10 samples" in msg for msg in log_messages)
        assert any("Filtered to 2 matches from 2 requested matches" in msg for msg in log_messages)
    
    def test_no_filtering_logging(self, mock_csv_data_dir, caplog):
        """Test logging when no filtering is applied."""
        temp_dir = mock_csv_data_dir['temp_dir']
        
        # Create dataset without filtering
        with caplog.at_level('INFO'):
            dataset = CrickformerDataset(
                data_root=temp_dir,
                use_csv_adapter=True,
                csv_config=CSVDataConfig(),
                load_video=False,
                load_embeddings=False,
                load_market_odds=False
            )
        
        # Check log messages
        log_messages = [record.message for record in caplog.records]
        
        # Should log no filtering applied
        assert any("no filtering applied" in msg for msg in log_messages)
        assert any("Loaded 25 samples from 5 matches" in msg for msg in log_messages)
    
    def test_filtering_integration_with_existing_methods(self, mock_csv_data_dir):
        """Test that filtering works correctly with existing dataset methods."""
        temp_dir = mock_csv_data_dir['temp_dir']
        
        # Create filter file
        filter_matches = ['match_001', 'match_002']
        filter_df = pd.DataFrame({'match_id': filter_matches})
        filter_path = Path(temp_dir) / 'filter_matches.csv'
        filter_df.to_csv(filter_path, index=False)
        
        # Create dataset with filtering
        dataset = CrickformerDataset(
            data_root=temp_dir,
            use_csv_adapter=True,
            csv_config=CSVDataConfig(),
            match_id_list_path=str(filter_path),
            load_video=False,
            load_embeddings=False,
            load_market_odds=False
        )
        
        # Test get_match_ids() method
        match_ids = dataset.get_match_ids()
        assert len(match_ids) == 2
        assert set(match_ids) == set(filter_matches)
        
        # Test get_sample_info() method
        for i in range(len(dataset)):
            sample_info = dataset.get_sample_info(i)
            assert sample_info['match_id'] in filter_matches
        
        # Test filter_by_match() method (should work with already filtered dataset)
        further_filtered = dataset.filter_by_match(['match_001'])
        assert len(further_filtered) == 5  # Only match_001 balls
        assert len(further_filtered.get_match_ids()) == 1
        assert further_filtered.get_match_ids()[0] == 'match_001'
    
    def test_filtering_with_manifest_file(self, mock_csv_data_dir):
        """Test that filtering works when manifest file is provided."""
        temp_dir = mock_csv_data_dir['temp_dir']
        
        # Create a manifest file
        manifest_data = {
            'samples': [
                {'ball_id': 'ball_001', 'match_id': 'match_001', 'index': 0},
                {'ball_id': 'ball_002', 'match_id': 'match_001', 'index': 1},
                {'ball_id': 'ball_003', 'match_id': 'match_002', 'index': 5},
                {'ball_id': 'ball_004', 'match_id': 'match_003', 'index': 10},
            ]
        }
        manifest_path = Path(temp_dir) / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f)
        
        # Create filter file
        filter_matches = ['match_001', 'match_003']
        filter_df = pd.DataFrame({'match_id': filter_matches})
        filter_path = Path(temp_dir) / 'filter_matches.csv'
        filter_df.to_csv(filter_path, index=False)
        
        # Create dataset with manifest and filtering
        dataset = CrickformerDataset(
            data_root=temp_dir,
            use_csv_adapter=True,
            csv_config=CSVDataConfig(),
            manifest_file='manifest.json',
            match_id_list_path=str(filter_path),
            load_video=False,
            load_embeddings=False,
            load_market_odds=False
        )
        
        # Should filter manifest samples
        assert len(dataset) == 3  # 2 from match_001 + 1 from match_003
        assert len(dataset.get_match_ids()) == 2
        assert set(dataset.get_match_ids()) == set(filter_matches)


if __name__ == "__main__":
    pytest.main([__file__]) 
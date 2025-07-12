# Purpose: Tests for match_aligner.py module
# Author: Assistant, Last Modified: 2024

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add the parent directory to the path so we can import match_aligner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from match_aligner import MatchAligner, align_matches


class TestMatchAligner:
    """Test suite for MatchAligner class."""
    
    @pytest.fixture
    def mock_csv_data(self):
        """Create mock CSV data for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Mock NVPlay data with 3 matches
        # Match 1: Perfect overlap with decimal Match A
        # Match 2: Partial overlap with decimal Match B  
        # Match 3: No overlap (unique to nvplay)
        nvplay_data = {
            'Competition': ['Test League'] * 15,
            'Match': ['Match_1'] * 5 + ['Match_2'] * 5 + ['Match_3'] * 5,
            'Date': ['2024-01-01'] * 15,
            'Innings': [1] * 15,
            'Over': [1, 1, 1, 2, 2] * 3,
            'Ball': [1, 2, 3, 1, 2] * 3,
            'Innings Ball': [1, 2, 3, 4, 5] * 3,
            'Batter': ['Player_A', 'Player_A', 'Player_B', 'Player_B', 'Player_C'] * 3,
            'Bowler': ['Bowler_X', 'Bowler_X', 'Bowler_Y', 'Bowler_Y', 'Bowler_Z'] * 3,
            'Runs': [1, 0, 4, 2, 0] + [1, 0, 4, 1, 0] + [2, 1, 6, 0, 3],  # Match 2 has slight differences
            'Extra Runs': [0] * 15,
            'Wicket': ['No Wicket'] * 15,
            'Team Runs': [1, 1, 5, 7, 7] * 3,
            'Team Wickets': [0] * 15,
            'Batting Team': ['Team_A'] * 15,
            'Bowling Team': ['Team_B'] * 15,
            'Batting Hand': ['RHB'] * 15,
            'Bowler Type': ['RM'] * 15,
            'FieldX': [100.0] * 15,
            'FieldY': [150.0] * 15,
            'PitchX': [0.0] * 15,
            'PitchY': [0.0] * 15,
            'Power Play': [1.0] * 15,
            'Run Rate After': [6.0] * 15,
            'Req Run Rate After': [8.0] * 15,
            'Venue': ['Test Ground'] * 15
        }
        
        # Mock decimal data with 3 matches
        # Match A: Perfect overlap with nvplay Match 1 (same players, same runs)
        # Match B: Partial overlap with nvplay Match 2 (same players, different runs)
        # Match C: No overlap (unique to decimal - different players)
        decimal_data = {
            'date': ['2024-01-01'] * 5 + ['2024-01-02'] * 5 + ['2024-01-03'] * 5,
            'competition': ['Test League'] * 15,
            'home': ['Team_A'] * 5 + ['Team_C'] * 5 + ['Team_E'] * 5,
            'away': ['Team_B'] * 5 + ['Team_D'] * 5 + ['Team_F'] * 5,
            'innings': [1] * 15,
            'ball': [1, 2, 3, 4, 5] * 3,
            'batter': ['Player_A', 'Player_A', 'Player_B', 'Player_B', 'Player_C'] + ['Player_A', 'Player_A', 'Player_B', 'Player_B', 'Player_C'] + ['Player_X', 'Player_X', 'Player_Y', 'Player_Y', 'Player_Z'],
            'bowler': ['Bowler_X', 'Bowler_X', 'Bowler_Y', 'Bowler_Y', 'Bowler_Z'] + ['Bowler_X', 'Bowler_X', 'Bowler_Y', 'Bowler_Y', 'Bowler_Z'] + ['Bowler_P', 'Bowler_P', 'Bowler_Q', 'Bowler_Q', 'Bowler_R'],
            'runs': [1, 0, 4, 2, 0] + [1, 0, 4, 1, 0] + [0, 2, 1, 3, 4],  # First match identical, second partial, third different
            'win_prob': [0.5] * 15
        }
        
        # Save as CSV files
        nvplay_df = pd.DataFrame(nvplay_data)
        decimal_df = pd.DataFrame(decimal_data)
        
        nvplay_path = Path(temp_dir) / 'nvplay_test.csv'
        decimal_path = Path(temp_dir) / 'decimal_test.csv'
        
        nvplay_df.to_csv(nvplay_path, index=False)
        decimal_df.to_csv(decimal_path, index=False)
        
        return {
            'temp_dir': temp_dir,
            'nvplay_path': str(nvplay_path),
            'decimal_path': str(decimal_path)
        }
    
    def test_match_aligner_initialization(self, mock_csv_data):
        """Test MatchAligner initialization and data loading."""
        aligner = MatchAligner(
            mock_csv_data['nvplay_path'],
            mock_csv_data['decimal_path'],
            fingerprint_length=10
        )
        
        # Check data loaded correctly
        assert aligner.nvplay_df is not None
        assert aligner.decimal_df is not None
        assert len(aligner.nvplay_df) == 15
        assert len(aligner.decimal_df) == 15
        
        # Check fingerprints extracted
        assert len(aligner.nvplay_fingerprints) == 3  # 3 matches
        assert len(aligner.decimal_fingerprints) == 3  # 3 matches
    
    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            MatchAligner("nonexistent_nvplay.csv", "nonexistent_decimal.csv")
    
    def test_fingerprint_extraction(self, mock_csv_data):
        """Test fingerprint extraction from both data sources."""
        aligner = MatchAligner(
            mock_csv_data['nvplay_path'],
            mock_csv_data['decimal_path'],
            fingerprint_length=5
        )
        
        # Check nvplay fingerprints
        nvplay_fingerprints = aligner.nvplay_fingerprints
        assert 'Match_1' in nvplay_fingerprints
        assert 'Match_2' in nvplay_fingerprints
        assert 'Match_3' in nvplay_fingerprints
        
        # Check fingerprint structure for Match_1
        match1_fingerprint = nvplay_fingerprints['Match_1']
        assert len(match1_fingerprint) == 5  # 5 balls
        
        # Check tuple structure: (over, ball, batter, bowler, runs)
        first_ball = match1_fingerprint[0]
        assert len(first_ball) == 5
        assert first_ball == (1.0, 1, 'Player_A', 'Bowler_X', 1)
        
        # Check decimal fingerprints
        decimal_fingerprints = aligner.decimal_fingerprints
        assert len(decimal_fingerprints) == 3
        
        # Check that decimal match IDs are created correctly
        decimal_match_ids = list(decimal_fingerprints.keys())
        expected_match_ids = [
            'Test League_Team_A_vs_Team_B_2024-01-01',
            'Test League_Team_C_vs_Team_D_2024-01-02',
            'Test League_Team_E_vs_Team_F_2024-01-03'
        ]
        for expected_id in expected_match_ids:
            assert expected_id in decimal_match_ids
    
    def test_similarity_calculation(self, mock_csv_data):
        """Test similarity calculation between fingerprints."""
        aligner = MatchAligner(
            mock_csv_data['nvplay_path'],
            mock_csv_data['decimal_path'],
            fingerprint_length=5
        )
        
        # Test identical fingerprints
        fingerprint1 = [(1.0, 1, 'Player_A', 'Bowler_X', 1), (1.0, 2, 'Player_A', 'Bowler_X', 0)]
        fingerprint2 = [(1.0, 1, 'Player_A', 'Bowler_X', 1), (1.0, 2, 'Player_A', 'Bowler_X', 0)]
        
        similarity = aligner._calculate_similarity(fingerprint1, fingerprint2)
        assert similarity == 1.0
        
        # Test partially matching fingerprints
        fingerprint3 = [(1.0, 1, 'Player_A', 'Bowler_X', 1), (1.0, 2, 'Player_B', 'Bowler_Y', 2)]
        similarity = aligner._calculate_similarity(fingerprint1, fingerprint3)
        assert similarity == 0.5  # 1 out of 2 matches
        
        # Test completely different fingerprints
        fingerprint4 = [(1.0, 1, 'Player_C', 'Bowler_Z', 3), (1.0, 2, 'Player_D', 'Bowler_W', 4)]
        similarity = aligner._calculate_similarity(fingerprint1, fingerprint4)
        assert similarity == 0.0
        
        # Test empty fingerprints
        similarity = aligner._calculate_similarity([], fingerprint1)
        assert similarity == 0.0
    
    def test_find_matches_perfect_overlap(self, mock_csv_data):
        """Test finding matches with perfect overlap."""
        aligner = MatchAligner(
            mock_csv_data['nvplay_path'],
            mock_csv_data['decimal_path'],
            fingerprint_length=5
        )
        
        # Find matches with high threshold
        matches = aligner.find_matches(similarity_threshold=0.9)
        
        # Should find at least one perfect match (Match_1 with decimal Match A)
        assert len(matches) >= 1
        
        # Check that we found the expected perfect match
        perfect_matches = [m for m in matches if m['similarity_score'] == 1.0]
        assert len(perfect_matches) >= 1
        
        # Verify match structure
        match = matches[0]
        assert 'nvplay_match_id' in match
        assert 'decimal_match_id' in match
        assert 'similarity_score' in match
        assert 0.0 <= match['similarity_score'] <= 1.0
    
    def test_find_matches_partial_overlap(self, mock_csv_data):
        """Test finding matches with partial overlap."""
        aligner = MatchAligner(
            mock_csv_data['nvplay_path'],
            mock_csv_data['decimal_path'],
            fingerprint_length=5
        )
        
        # Find matches with lower threshold to catch partial matches
        matches = aligner.find_matches(similarity_threshold=0.6)
        
        # Should find more matches with lower threshold
        assert len(matches) >= 1
        
        # Check that we have matches with various similarity scores
        similarities = [m['similarity_score'] for m in matches]
        assert min(similarities) >= 0.6
        assert max(similarities) <= 1.0
    
    def test_find_matches_no_overlap(self, mock_csv_data):
        """Test finding matches when there's no overlap."""
        aligner = MatchAligner(
            mock_csv_data['nvplay_path'],
            mock_csv_data['decimal_path'],
            fingerprint_length=5
        )
        
        # Find matches with very high threshold
        matches = aligner.find_matches(similarity_threshold=1.0)
        
        # Should only find perfect matches
        for match in matches:
            assert match['similarity_score'] == 1.0
    
    def test_save_matches(self, mock_csv_data):
        """Test saving matches to CSV file."""
        aligner = MatchAligner(
            mock_csv_data['nvplay_path'],
            mock_csv_data['decimal_path'],
            fingerprint_length=5
        )
        
        # Find matches
        matches = aligner.find_matches(similarity_threshold=0.8)
        
        # Save to temporary file
        output_path = Path(mock_csv_data['temp_dir']) / 'test_matches.csv'
        aligner.save_matches(matches, str(output_path))
        
        # Verify file was created
        assert output_path.exists()
        
        # Load and verify content
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == len(matches)
        assert 'nvplay_match_id' in saved_df.columns
        assert 'decimal_match_id' in saved_df.columns
        assert 'similarity_score' in saved_df.columns
        
        # Verify data integrity
        if len(matches) > 0:
            assert saved_df['similarity_score'].min() >= 0.0
            assert saved_df['similarity_score'].max() <= 1.0
    
    def test_save_matches_empty_list(self, mock_csv_data):
        """Test saving empty matches list."""
        aligner = MatchAligner(
            mock_csv_data['nvplay_path'],
            mock_csv_data['decimal_path'],
            fingerprint_length=5
        )
        
        # Save empty matches list
        output_path = Path(mock_csv_data['temp_dir']) / 'empty_matches.csv'
        aligner.save_matches([], str(output_path))
        
        # File should not be created for empty matches
        assert not output_path.exists()
    
    def test_fingerprint_length_parameter(self, mock_csv_data):
        """Test that fingerprint_length parameter works correctly."""
        # Test with different fingerprint lengths
        aligner_short = MatchAligner(
            mock_csv_data['nvplay_path'],
            mock_csv_data['decimal_path'],
            fingerprint_length=3
        )
        
        aligner_long = MatchAligner(
            mock_csv_data['nvplay_path'],
            mock_csv_data['decimal_path'],
            fingerprint_length=10
        )
        
        # Check that fingerprint lengths are respected
        for match_id, fingerprint in aligner_short.nvplay_fingerprints.items():
            assert len(fingerprint) <= 3
        
        for match_id, fingerprint in aligner_long.nvplay_fingerprints.items():
            assert len(fingerprint) <= 5  # Limited by available data (5 balls per match)


class TestAlignMatchesFunction:
    """Test suite for the align_matches function."""
    
    @pytest.fixture
    def mock_csv_data(self):
        """Create mock CSV data for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Simple mock data with one clear match
        nvplay_data = {
            'Competition': ['Test League'] * 3,
            'Match': ['Perfect_Match'] * 3,
            'Date': ['2024-01-01'] * 3,
            'Innings': [1] * 3,
            'Over': [1, 1, 1],
            'Ball': [1, 2, 3],
            'Innings Ball': [1, 2, 3],
            'Batter': ['Player_A', 'Player_A', 'Player_B'],
            'Bowler': ['Bowler_X', 'Bowler_X', 'Bowler_Y'],
            'Runs': [1, 0, 4],
            'Extra Runs': [0] * 3,
            'Wicket': ['No Wicket'] * 3,
            'Team Runs': [1, 1, 5],
            'Team Wickets': [0] * 3,
            'Batting Team': ['Team_A'] * 3,
            'Bowling Team': ['Team_B'] * 3,
            'Batting Hand': ['RHB'] * 3,
            'Bowler Type': ['RM'] * 3,
            'FieldX': [100.0] * 3,
            'FieldY': [150.0] * 3,
            'PitchX': [0.0] * 3,
            'PitchY': [0.0] * 3,
            'Power Play': [1.0] * 3,
            'Run Rate After': [6.0] * 3,
            'Req Run Rate After': [8.0] * 3,
            'Venue': ['Test Ground'] * 3
        }
        
        decimal_data = {
            'date': ['2024-01-01'] * 3,
            'competition': ['Test League'] * 3,
            'home': ['Team_A'] * 3,
            'away': ['Team_B'] * 3,
            'innings': [1] * 3,
            'ball': [1, 2, 3],
            'batter': ['Player_A', 'Player_A', 'Player_B'],
            'bowler': ['Bowler_X', 'Bowler_X', 'Bowler_Y'],
            'runs': [1, 0, 4],
            'win_prob': [0.5] * 3
        }
        
        # Save as CSV files
        nvplay_df = pd.DataFrame(nvplay_data)
        decimal_df = pd.DataFrame(decimal_data)
        
        nvplay_path = Path(temp_dir) / 'nvplay_simple.csv'
        decimal_path = Path(temp_dir) / 'decimal_simple.csv'
        
        nvplay_df.to_csv(nvplay_path, index=False)
        decimal_df.to_csv(decimal_path, index=False)
        
        return {
            'temp_dir': temp_dir,
            'nvplay_path': str(nvplay_path),
            'decimal_path': str(decimal_path)
        }
    
    def test_align_matches_function(self, mock_csv_data):
        """Test the main align_matches function."""
        output_path = Path(mock_csv_data['temp_dir']) / 'function_test_matches.csv'
        
        matches = align_matches(
            mock_csv_data['nvplay_path'],
            mock_csv_data['decimal_path'],
            str(output_path),
            fingerprint_length=10,
            similarity_threshold=0.9
        )
        
        # Should find at least one match
        assert len(matches) >= 1
        
        # Output file should be created
        assert output_path.exists()
        
        # Verify the perfect match was found
        perfect_matches = [m for m in matches if m['similarity_score'] == 1.0]
        assert len(perfect_matches) >= 1
    
    def test_align_matches_with_different_parameters(self, mock_csv_data):
        """Test align_matches with different parameters."""
        output_path = Path(mock_csv_data['temp_dir']) / 'param_test_matches.csv'
        
        # Test with different fingerprint length and threshold
        matches = align_matches(
            mock_csv_data['nvplay_path'],
            mock_csv_data['decimal_path'],
            str(output_path),
            fingerprint_length=5,
            similarity_threshold=0.8
        )
        
        # Should still find matches
        assert isinstance(matches, list)
        assert output_path.exists()
        
        # Load and verify CSV structure
        saved_df = pd.read_csv(output_path)
        expected_columns = ['nvplay_match_id', 'decimal_match_id', 'similarity_score']
        for col in expected_columns:
            assert col in saved_df.columns


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_csv_files(self):
        """Test handling of empty CSV files."""
        temp_dir = tempfile.mkdtemp()
        
        # Create empty CSV files
        nvplay_path = Path(temp_dir) / 'empty_nvplay.csv'
        decimal_path = Path(temp_dir) / 'empty_decimal.csv'
        
        # Create empty DataFrames with correct columns
        empty_nvplay = pd.DataFrame(columns=['Match', 'Innings', 'Innings Ball', 'Over', 'Ball', 'Batter', 'Bowler', 'Runs'])
        empty_decimal = pd.DataFrame(columns=['competition', 'home', 'away', 'date', 'innings', 'ball', 'batter', 'bowler', 'runs'])
        
        empty_nvplay.to_csv(nvplay_path, index=False)
        empty_decimal.to_csv(decimal_path, index=False)
        
        # Should handle empty files gracefully
        aligner = MatchAligner(str(nvplay_path), str(decimal_path))
        matches = aligner.find_matches()
        
        assert len(matches) == 0
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns in decimal data."""
        temp_dir = tempfile.mkdtemp()
        
        # Create nvplay data with required columns
        nvplay_data = {
            'Competition': ['Test League'],
            'Match': ['Test_Match'],
            'Date': ['2024-01-01'],
            'Innings': [1],
            'Over': [1],
            'Ball': [1],
            'Innings Ball': [1],
            'Batter': ['Player_A'],
            'Bowler': ['Bowler_X'],
            'Runs': [1],
            'Extra Runs': [0],
            'Wicket': ['No Wicket'],
            'Team Runs': [1],
            'Team Wickets': [0],
            'Batting Team': ['Team_A'],
            'Bowling Team': ['Team_B'],
            'Batting Hand': ['RHB'],
            'Bowler Type': ['RM'],
            'FieldX': [100.0],
            'FieldY': [150.0],
            'PitchX': [0.0],
            'PitchY': [0.0],
            'Power Play': [1.0],
            'Run Rate After': [6.0],
            'Req Run Rate After': [8.0],
            'Venue': ['Test Ground']
        }
        
        # Create decimal data missing batter/bowler columns
        decimal_data = {
            'date': ['2024-01-01'],
            'competition': ['Test League'],
            'home': ['Team_A'],
            'away': ['Team_B'],
            'innings': [1],
            'ball': [1],
            'win_prob': [0.5]
            # Missing 'batter', 'bowler', 'runs' columns
        }
        
        nvplay_df = pd.DataFrame(nvplay_data)
        decimal_df = pd.DataFrame(decimal_data)
        
        nvplay_path = Path(temp_dir) / 'nvplay_complete.csv'
        decimal_path = Path(temp_dir) / 'decimal_incomplete.csv'
        
        nvplay_df.to_csv(nvplay_path, index=False)
        decimal_df.to_csv(decimal_path, index=False)
        
        # Should handle missing columns gracefully
        aligner = MatchAligner(str(nvplay_path), str(decimal_path))
        matches = aligner.find_matches()
        
        # Should not crash, but may not find matches due to missing data
        assert isinstance(matches, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
# Purpose: Unit tests for video_sync.py - ball metadata to video file matching
# Author: Shamus Rae, Last Modified: July 17, 2025

import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, mock_open
from video_sync import (
    find_video_for_ball,
    normalize_name,
    create_search_patterns,
    format_date_for_matching,
    match_pattern_in_filename,
    get_video_files_for_match,
    validate_ball_metadata,
    VIDEO_EXTENSIONS
)

class TestVideoSync:
    """Test class for video synchronization functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Create test video files with relevant metadata in names
        self.test_video_files = [
            'viratkohli_over15_ball3_20240315.mp4',  # Perfect match
            'rohitsharma_over12_ball1_20240315.avi',  # Different player
            'viratkohli_innings_highlights_20240315.mov',  # Batter + date only
            'match_20240315_full_coverage.mkv',  # Date only
            'jaspritbumrah_bowling_over15.mp4',  # Bowler + over
            'random_video_file.webm',  # No relevant metadata
            'Virat_Kohli_batting_2024-03-15.m4v',  # Alternative formats
        ]
        
        # Create the test video files
        for video_file in self.test_video_files:
            file_path = os.path.join(self.test_dir, video_file)
            with open(file_path, 'w') as f:
                f.write("fake video content")
    
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        shutil.rmtree(self.test_dir)
    
    def test_find_video_for_ball_perfect_match(self):
        """Test finding video with perfect metadata match."""
        ball_metadata = {
            'batter': 'Virat Kohli',
            'bowler': 'Jasprit Bumrah',
            'over': 15,
            'ball': 3,
            'match_date': '2024-03-15'
        }
        
        result = find_video_for_ball(ball_metadata, self.test_dir)
        
        assert result is not None
        assert 'viratkohli_over15_ball3_20240315.mp4' in result
        assert os.path.exists(result)
    
    def test_find_video_for_ball_batter_over_match(self):
        """Test finding video with batter and over/ball match (no date)."""
        ball_metadata = {
            'batter': 'Virat Kohli',
            'over': 15,
            'ball': 3
        }
        
        result = find_video_for_ball(ball_metadata, self.test_dir)
        
        assert result is not None
        assert 'viratkohli_over15_ball3_20240315.mp4' in result
    
    def test_find_video_for_ball_batter_date_match(self):
        """Test finding video with batter and date match only."""
        ball_metadata = {
            'batter': 'Virat Kohli',
            'over': 99,  # Non-existent over
            'ball': 99,  # Non-existent ball
            'match_date': '2024-03-15'
        }
        
        result = find_video_for_ball(ball_metadata, self.test_dir)
        
        assert result is not None
        # Should match either the perfect match file or the highlights file
        assert any(filename in result for filename in [
            'viratkohli_over15_ball3_20240315.mp4',
            'viratkohli_innings_highlights_20240315.mov',
            'Virat_Kohli_batting_2024-03-15.m4v'
        ])
    
    def test_find_video_for_ball_batter_only_match(self):
        """Test finding video with batter name only."""
        ball_metadata = {
            'batter': 'Virat Kohli'
        }
        
        result = find_video_for_ball(ball_metadata, self.test_dir)
        
        assert result is not None
        assert any(filename in result for filename in [
            'viratkohli_over15_ball3_20240315.mp4',
            'viratkohli_innings_highlights_20240315.mov',
            'Virat_Kohli_batting_2024-03-15.m4v'
        ])
    
    def test_find_video_for_ball_no_match(self):
        """Test when no video file matches the ball metadata."""
        ball_metadata = {
            'batter': 'MS Dhoni',  # Player not in test files
            'over': 20,
            'ball': 6,
            'match_date': '2023-01-01'
        }
        
        result = find_video_for_ball(ball_metadata, self.test_dir)
        
        assert result is None
    
    def test_find_video_for_ball_directory_not_exists(self):
        """Test behavior when video directory doesn't exist."""
        ball_metadata = {
            'batter': 'Virat Kohli',
            'over': 15,
            'ball': 3
        }
        
        result = find_video_for_ball(ball_metadata, '/nonexistent/directory')
        
        assert result is None
    
    def test_find_video_for_ball_empty_directory(self):
        """Test behavior when video directory is empty."""
        empty_dir = tempfile.mkdtemp()
        
        try:
            ball_metadata = {
                'batter': 'Virat Kohli',
                'over': 15,
                'ball': 3
            }
            
            result = find_video_for_ball(ball_metadata, empty_dir)
            
            assert result is None
        finally:
            shutil.rmtree(empty_dir)
    
    def test_find_video_for_ball_bowler_match(self):
        """Test finding video using bowler information."""
        ball_metadata = {
            'batter': 'Unknown Player',  # Not in files
            'bowler': 'Jasprit Bumrah',
            'over': 15,
            'ball': 1
        }
        
        result = find_video_for_ball(ball_metadata, self.test_dir)
        
        assert result is not None
        assert 'jaspritbumrah_bowling_over15.mp4' in result
    
    def test_normalize_name(self):
        """Test name normalization function."""
        test_cases = [
            ('Virat Kohli', 'viratkohli'),
            ('M.S. Dhoni', 'msdhoni'),
            ('Rohit Sharma', 'rohitsharma'),
            ('K.L. Rahul', 'klrahul'),
            ('', ''),
            ('   ', ''),
            ('Test123', 'test123'),
            ('Name-With-Hyphens', 'namewithhyphens')
        ]
        
        for input_name, expected in test_cases:
            result = normalize_name(input_name)
            assert result == expected, f"Failed for input: {input_name}"
    
    def test_create_search_patterns(self):
        """Test search pattern creation with different priority levels."""
        patterns = create_search_patterns(
            batter='viratkohli',
            bowler='jaspritbumrah',
            over=15,
            ball=3,
            match_date='2024-03-15'
        )
        
        # Should have multiple priority levels
        assert len(patterns) > 0
        
        # Priority 1 should include batter + over/ball + date
        assert 1 in patterns
        assert any('viratkohli' in pattern and '153' in pattern for pattern in patterns[1])
        
        # Priority 2 should include batter + over/ball
        assert 2 in patterns
        assert any('viratkohli' in pattern and '153' in pattern for pattern in patterns[2])
        
        # Priority 4 should include batter only
        assert 4 in patterns
        assert 'viratkohli' in patterns[4]
    
    def test_format_date_for_matching(self):
        """Test date formatting for filename matching."""
        test_cases = [
            ('2024-03-15', '20240315'),
            ('15/03/2024', '20240315'),
            ('03/15/2024', '20240315'),
            ('20240315', '20240315'),
            ('15-03-2024', '20240315'),
            ('03-15-2024', '20240315'),
            ('', ''),
            ('invalid-date', 'invaliddate')
        ]
        
        for input_date, expected_contains in test_cases:
            result = format_date_for_matching(input_date)
            if expected_contains:
                assert expected_contains in result, f"Failed for input: {input_date}"
            else:
                assert result == '', f"Failed for input: {input_date}"
    
    def test_match_pattern_in_filename(self):
        """Test pattern matching in filenames."""
        test_cases = [
            ('viratkohli', 'viratkohli_over15_ball3_20240315.mp4', True),
            ('rohitsharma', 'viratkohli_over15_ball3_20240315.mp4', False),
            ('153', 'viratkohli_over15_ball3_20240315.mp4', True),
            ('20240315', 'viratkohli_over15_ball3_20240315.mp4', True),
            ('viratkohli.*153', 'viratkohli_over15_ball3_20240315.mp4', True),
            ('rohitsharma.*153', 'viratkohli_over15_ball3_20240315.mp4', False),
            ('20240315|2024-03-15', 'Virat_Kohli_batting_2024-03-15.m4v', True),
        ]
        
        for pattern, filename, expected in test_cases:
            result = match_pattern_in_filename(pattern, filename)
            assert result == expected, f"Failed for pattern: {pattern}, filename: {filename}"
    
    def test_get_video_files_for_match(self):
        """Test getting all video files for a specific match."""
        result = get_video_files_for_match(self.test_dir, '20240315')
        
        assert len(result) >= 4  # Should find multiple files with this date
        assert all(os.path.exists(file) for file in result)
        assert all(any(file.endswith(ext) for ext in VIDEO_EXTENSIONS) for file in result)
    
    def test_get_video_files_for_match_no_directory(self):
        """Test getting video files when directory doesn't exist."""
        result = get_video_files_for_match('/nonexistent/directory', '20240315')
        
        assert result == []
    
    def test_validate_ball_metadata(self):
        """Test ball metadata validation."""
        # Valid metadata
        valid_metadata = {
            'batter': 'Virat Kohli',
            'over': 15,
            'ball': 3
        }
        assert validate_ball_metadata(valid_metadata) is True
        
        # Missing batter
        invalid_metadata = {
            'over': 15,
            'ball': 3
        }
        assert validate_ball_metadata(invalid_metadata) is False
        
        # Empty batter
        invalid_metadata2 = {
            'batter': '',
            'over': 15,
            'ball': 3
        }
        assert validate_ball_metadata(invalid_metadata2) is False
    
    def test_video_extensions_recognition(self):
        """Test that all video file extensions are recognized."""
        # Create files with different extensions
        temp_dir = tempfile.mkdtemp()
        
        try:
            test_files = [
                'test.mp4',
                'test.avi',
                'test.mov',
                'test.mkv',
                'test.wmv',
                'test.flv',
                'test.webm',
                'test.m4v',
                'test.txt',  # Not a video file
                'test.jpg'   # Not a video file
            ]
            
            for file in test_files:
                file_path = os.path.join(temp_dir, file)
                with open(file_path, 'w') as f:
                    f.write("test content")
            
            ball_metadata = {
                'batter': 'test'
            }
            
            result = find_video_for_ball(ball_metadata, temp_dir)
            
            # Should find one of the video files, not the txt or jpg files
            assert result is not None
            assert any(result.endswith(ext) for ext in VIDEO_EXTENSIONS)
            assert not result.endswith('.txt')
            assert not result.endswith('.jpg')
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_case_insensitive_matching(self):
        """Test that matching works regardless of case."""
        # Create file with mixed case
        mixed_case_file = 'VIRAT_KOHLI_Over15_Ball3.MP4'
        file_path = os.path.join(self.test_dir, mixed_case_file)
        with open(file_path, 'w') as f:
            f.write("test content")
        
        ball_metadata = {
            'batter': 'virat kohli',  # lowercase
            'over': 15,
            'ball': 3
        }
        
        result = find_video_for_ball(ball_metadata, self.test_dir)
        
        assert result is not None
        assert mixed_case_file in result
    
    def test_priority_matching_order(self):
        """Test that higher priority matches are returned first."""
        # Remove existing files to control the test better
        for file in self.test_video_files:
            os.remove(os.path.join(self.test_dir, file))
        
        # Create files with different priority levels
        test_files = [
            'viratkohli_generic.mp4',  # Priority 4: batter only
            'viratkohli_over15_ball3_20240315.mp4',  # Priority 1: batter + over/ball + date
        ]
        
        for file in test_files:
            file_path = os.path.join(self.test_dir, file)
            with open(file_path, 'w') as f:
                f.write("test content")
        
        ball_metadata = {
            'batter': 'Virat Kohli',
            'over': 15,
            'ball': 3,
            'match_date': '2024-03-15'
        }
        
        result = find_video_for_ball(ball_metadata, self.test_dir)
        
        # Should match the higher priority file (with over/ball/date)
        assert result is not None
        assert 'viratkohli_over15_ball3_20240315.mp4' in result
    
    def test_alternative_date_formats(self):
        """Test matching with various date formats."""
        # Create file with different date format
        date_file = 'viratkohli_batting_2024-03-15.mp4'
        file_path = os.path.join(self.test_dir, date_file)
        with open(file_path, 'w') as f:
            f.write("test content")
        
        ball_metadata = {
            'batter': 'Virat Kohli',
            'match_date': '15/03/2024'  # Different format
        }
        
        result = find_video_for_ball(ball_metadata, self.test_dir)
        
        assert result is not None
        assert date_file in result or 'viratkohli_over15_ball3_20240315.mp4' in result
    
    def test_partial_name_matching(self):
        """Test that partial player names work."""
        ball_metadata = {
            'batter': 'Virat'  # Partial name
        }
        
        result = find_video_for_ball(ball_metadata, self.test_dir)
        
        assert result is not None
        assert any(filename in result for filename in [
            'viratkohli_over15_ball3_20240315.mp4',
            'viratkohli_innings_highlights_20240315.mov',
            'Virat_Kohli_batting_2024-03-15.m4v'
        ])

# Integration tests
class TestVideoSyncIntegration:
    """Integration tests for video sync functionality."""
    
    def test_realistic_video_file_names(self):
        """Test with realistic cricket video file names."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create realistic video file names
            realistic_files = [
                'IND_vs_AUS_T20_2024-03-15_Virat_Kohli_over12_ball4_SIX.mp4',
                'IPL_2024_CSK_vs_MI_MS_Dhoni_finishing_over19_ball6.avi',
                'World_Cup_2024_Rohit_Sharma_century_highlights.mov',
                'BBL_2024_01_15_David_Warner_explosive_batting.mkv',
                'test_match_day1_session2_generic.mp4'
            ]
            
            for file in realistic_files:
                file_path = os.path.join(temp_dir, file)
                with open(file_path, 'w') as f:
                    f.write("test content")
            
            # Test specific ball
            ball_metadata = {
                'batter': 'Virat Kohli',
                'over': 12,
                'ball': 4,
                'match_date': '2024-03-15'
            }
            
            result = find_video_for_ball(ball_metadata, temp_dir)
            
            assert result is not None
            assert 'IND_vs_AUS_T20_2024-03-15_Virat_Kohli_over12_ball4_SIX.mp4' in result
            
            # Test player-only search
            ball_metadata2 = {
                'batter': 'MS Dhoni'
            }
            
            result2 = find_video_for_ball(ball_metadata2, temp_dir)
            
            assert result2 is not None
            assert 'IPL_2024_CSK_vs_MI_MS_Dhoni_finishing_over19_ball6.avi' in result2
        
        finally:
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    pytest.main([__file__]) 
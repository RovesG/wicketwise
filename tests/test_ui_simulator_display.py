# Purpose: Unit tests for enhanced simulator display functionality in ui_launcher.py
# Author: Phi1618 Cricket AI Team, Last Modified: 2024

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
from simulator_engine import MatchSimulator
from video_sync import find_video_for_ball

class TestUISimulatorDisplay:
    """Test class for UI simulator display functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock match data
        self.mock_match_data = pd.DataFrame({
            'match_id': ['M001'] * 12 + ['M002'] * 8,
            'ball_id': list(range(1, 21)),
            'over': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4],
            'ball': [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2],
            'batter': ['SA(Kumar) Yadav'] * 12 + ['Rohit Sharma'] * 8,
            'bowler': ['Jasprit Bumrah'] * 6 + ['Mohammed Shami'] * 6 + ['Yuzvendra Chahal'] * 8,
            'runs': [1, 0, 4, 0, 2, 1, 0, 0, 6, 1, 0, 0, 1, 1, 0, 4, 0, 2, 1, 0],
            'extras': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'wicket': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'dismissal': [None] * 11 + ['bowled'] + [None] * 8,
            'win_probability': [0.45, 0.46, 0.52, 0.52, 0.55, 0.56, 0.56, 0.56, 0.65, 0.67, 0.67, 0.62, 0.63, 0.64, 0.64, 0.72, 0.72, 0.75, 0.76, 0.76]
        })
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.mock_match_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
        # Mock video directory and files
        self.mock_video_dir = '/mock/video/directory'
        self.mock_video_files = [
            'Match_1_1_001_01.mp4',
            'Match_1_1_001_02.mp4',
            'Match_1_1_001_03.mp4',
            'Match_1_1_001_04.mp4',
            'Match_1_1_002_01.mp4',
            'Match_1_1_002_02.mp4',
        ]
    
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_match_simulator_initialization(self):
        """Test MatchSimulator initialization with match data."""
        simulator = MatchSimulator(self.temp_file.name)
        
        assert simulator.csv_path == self.temp_file.name
        assert simulator.match_id is None
        assert simulator._cached is False
        
        # Test with specific match ID
        simulator_with_match = MatchSimulator(self.temp_file.name, 'M001')
        assert simulator_with_match.match_id == 'M001'
    
    def test_match_simulator_load_data(self):
        """Test loading match data from CSV file."""
        simulator = MatchSimulator(self.temp_file.name)
        
        # Get match ball count
        ball_count = simulator.get_match_ball_count()
        assert ball_count == 20
        
        # Get match summary
        summary = simulator.get_match_summary()
        assert summary['total_balls'] == 20
        assert summary['total_overs'] == 20/6
        assert 'total_runs' in summary
        assert 'total_wickets' in summary
    
    def test_match_simulator_filter_by_match_id(self):
        """Test filtering match data by match ID."""
        # Test with match M001
        simulator_m001 = MatchSimulator(self.temp_file.name, 'M001')
        ball_count_m001 = simulator_m001.get_match_ball_count()
        assert ball_count_m001 == 12
        
        # Test with match M002
        simulator_m002 = MatchSimulator(self.temp_file.name, 'M002')
        ball_count_m002 = simulator_m002.get_match_ball_count()
        assert ball_count_m002 == 8
    
    def test_match_simulator_get_ball(self):
        """Test getting specific ball information."""
        simulator = MatchSimulator(self.temp_file.name, 'M001')
        
        # Test first ball
        ball_0 = simulator.get_ball(0)
        assert ball_0['ball_index'] == 0
        assert ball_0['ball_number'] == 1
        assert ball_0['batter'] == 'SA(Kumar) Yadav'
        assert ball_0['bowler'] == 'Jasprit Bumrah'
        assert ball_0['over'] == 1
        assert ball_0['ball'] == 1
        assert ball_0['runs'] == 1
        
        # Test boundary ball
        ball_2 = simulator.get_ball(2)
        assert ball_2['runs'] == 4
        
        # Test six ball
        ball_8 = simulator.get_ball(8)
        assert ball_8['runs'] == 6
        
        # Test wicket ball
        ball_11 = simulator.get_ball(11)
        assert ball_11['wicket'] == 1
        assert ball_11['dismissal'] == 'bowled'
    
    def test_match_simulator_get_ball_out_of_range(self):
        """Test getting ball with invalid index."""
        simulator = MatchSimulator(self.temp_file.name, 'M001')
        
        # Test negative index
        with pytest.raises(IndexError):
            simulator.get_ball(-1)
        
        # Test index too high
        with pytest.raises(IndexError):
            simulator.get_ball(20)
    
    def test_match_simulator_match_state(self):
        """Test match state information in ball data."""
        simulator = MatchSimulator(self.temp_file.name, 'M001')
        
        # Test first ball match state
        ball_0 = simulator.get_ball(0)
        match_state = ball_0['match_state']
        
        assert match_state['balls_bowled'] == 1
        assert match_state['total_balls'] == 12
        assert match_state['progress_percentage'] == 100/12
        
        # Test middle ball match state
        ball_5 = simulator.get_ball(5)
        match_state_5 = ball_5['match_state']
        
        assert match_state_5['balls_bowled'] == 6
        assert match_state_5['progress_percentage'] == 50.0
    
    def test_match_simulator_get_available_match_ids(self):
        """Test getting available match IDs from data."""
        # Read raw data to get match IDs
        raw_data = pd.read_csv(self.temp_file.name)
        available_matches = sorted(raw_data['match_id'].unique())
        
        assert len(available_matches) == 2
        assert 'M001' in available_matches
        assert 'M002' in available_matches
    
    def test_match_simulator_win_probability(self):
        """Test win probability display in ball data."""
        simulator = MatchSimulator(self.temp_file.name, 'M001')
        
        # Test first ball win probability
        ball_0 = simulator.get_ball(0)
        assert 'win_probability' in ball_0['raw_data']
        assert ball_0['raw_data']['win_probability'] == 0.45
        
        # Test ball with higher win probability
        ball_8 = simulator.get_ball(8)
        assert ball_8['raw_data']['win_probability'] == 0.65
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('video_sync.find_video_for_ball')
    def test_video_sync_integration(self, mock_find_video, mock_listdir, mock_exists):
        """Test video sync integration with simulator."""
        # Mock video directory and files
        mock_exists.return_value = True
        mock_listdir.return_value = self.mock_video_files
        
        # Mock video file found
        mock_video_path = '/mock/video/directory/Match_1_1_001_01.mp4'
        mock_find_video.return_value = mock_video_path
        
        # Test ball metadata for video search
        simulator = MatchSimulator(self.temp_file.name, 'M001')
        ball_0 = simulator.get_ball(0)
        
        ball_metadata = {
            'batter': ball_0.get('batter', ''),
            'bowler': ball_0.get('bowler', ''),
            'over': ball_0.get('over'),
            'ball': ball_0.get('ball'),
            'match_date': None
        }
        
        # Find video for ball
        video_path = find_video_for_ball(ball_metadata, self.mock_video_dir)
        
        # Verify video was found
        assert video_path == mock_video_path
        mock_find_video.assert_called_once_with(ball_metadata, self.mock_video_dir)
    
    @patch('os.path.exists')
    @patch('video_sync.find_video_for_ball')
    def test_video_sync_no_video_found(self, mock_find_video, mock_exists):
        """Test video sync when no video is found."""
        # Mock video directory exists but no video found
        mock_exists.return_value = True
        mock_find_video.return_value = None
        
        # Test ball metadata for video search
        simulator = MatchSimulator(self.temp_file.name, 'M001')
        ball_0 = simulator.get_ball(0)
        
        ball_metadata = {
            'batter': ball_0.get('batter', ''),
            'bowler': ball_0.get('bowler', ''),
            'over': ball_0.get('over'),
            'ball': ball_0.get('ball'),
            'match_date': None
        }
        
        # Find video for ball
        video_path = find_video_for_ball(ball_metadata, self.mock_video_dir)
        
        # Verify no video was found
        assert video_path is None
    
    @patch('os.path.exists')
    def test_video_sync_directory_not_found(self, mock_exists):
        """Test video sync when video directory doesn't exist."""
        # Mock video directory doesn't exist
        mock_exists.return_value = False
        
        # Test ball metadata for video search
        simulator = MatchSimulator(self.temp_file.name, 'M001')
        ball_0 = simulator.get_ball(0)
        
        ball_metadata = {
            'batter': ball_0.get('batter', ''),
            'bowler': ball_0.get('bowler', ''),
            'over': ball_0.get('over'),
            'ball': ball_0.get('ball'),
            'match_date': None
        }
        
        # Find video for ball
        video_path = find_video_for_ball(ball_metadata, self.mock_video_dir)
        
        # Verify no video was found
        assert video_path is None
    
    def test_simulator_display_elements_populate(self):
        """Test that display elements populate correctly with match data."""
        simulator = MatchSimulator(self.temp_file.name, 'M001')
        
        # Get ball information
        ball_0 = simulator.get_ball(0)
        
        # Verify all required display elements have data
        assert ball_0['ball_number'] == 1
        assert ball_0['batter'] == 'SA(Kumar) Yadav'
        assert ball_0['bowler'] == 'Jasprit Bumrah'
        assert ball_0['over'] == 1
        assert ball_0['ball'] == 1
        assert ball_0['runs'] == 1
        assert ball_0['extras'] == 0
        assert ball_0['dismissal'] is None
        
        # Verify match state elements
        match_state = ball_0['match_state']
        assert 'balls_bowled' in match_state
        assert 'total_balls' in match_state
        assert 'progress_percentage' in match_state
    
    def test_simulator_display_dismissal_info(self):
        """Test dismissal information display."""
        simulator = MatchSimulator(self.temp_file.name, 'M001')
        
        # Get ball with dismissal
        ball_11 = simulator.get_ball(11)
        
        # Verify dismissal information
        assert ball_11['wicket'] == 1
        assert ball_11['dismissal'] == 'bowled'
        
        # Get ball without dismissal
        ball_0 = simulator.get_ball(0)
        assert ball_0['wicket'] == 0
        assert ball_0['dismissal'] is None
    
    def test_simulator_display_run_rate_calculation(self):
        """Test run rate calculation in match state."""
        simulator = MatchSimulator(self.temp_file.name, 'M001')
        
        # Get ball after some runs
        ball_5 = simulator.get_ball(5)
        match_state = ball_5['match_state']
        
        # Should have run rate calculated
        assert 'run_rate' in match_state
        assert match_state['run_rate'] is not None
        assert match_state['run_rate'] > 0
    
    def test_simulator_display_boundaries_and_sixes(self):
        """Test boundary and six detection in match summary."""
        simulator = MatchSimulator(self.temp_file.name, 'M001')
        
        # Get match summary
        summary = simulator.get_match_summary()
        
        # Should have boundary and six counts
        assert 'boundaries' in summary
        assert 'sixes' in summary
        assert summary['boundaries'] >= summary['sixes']
    
    def test_simulator_display_player_lists(self):
        """Test player lists in match summary."""
        simulator = MatchSimulator(self.temp_file.name, 'M001')
        
        # Get match summary
        summary = simulator.get_match_summary()
        
        # Should have player lists
        assert 'batters' in summary
        assert 'bowlers' in summary
        assert 'SA(Kumar) Yadav' in summary['batters']
        assert 'Jasprit Bumrah' in summary['bowlers']
    
    def test_simulator_display_raw_data(self):
        """Test raw data availability for debugging."""
        simulator = MatchSimulator(self.temp_file.name, 'M001')
        
        # Get ball information
        ball_0 = simulator.get_ball(0)
        
        # Should have raw data
        assert 'raw_data' in ball_0
        assert isinstance(ball_0['raw_data'], dict)
        assert 'match_id' in ball_0['raw_data']
        assert 'batter' in ball_0['raw_data']
        assert 'bowler' in ball_0['raw_data']
    
    def test_simulator_csv_without_match_id(self):
        """Test simulator with CSV file without match_id column."""
        # Create data without match_id
        data_no_match_id = self.mock_match_data.drop('match_id', axis=1)
        
        # Create temporary file
        temp_file_no_match = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        data_no_match_id.to_csv(temp_file_no_match.name, index=False)
        temp_file_no_match.close()
        
        try:
            simulator = MatchSimulator(temp_file_no_match.name)
            ball_count = simulator.get_match_ball_count()
            assert ball_count == 20
            
            # Should still work for getting ball info
            ball_0 = simulator.get_ball(0)
            assert ball_0['batter'] == 'SA(Kumar) Yadav'
            
        finally:
            os.unlink(temp_file_no_match.name)
    
    def test_simulator_error_handling(self):
        """Test error handling in simulator."""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            simulator = MatchSimulator('/non/existent/file.csv')
            simulator.get_match_ball_count()
        
        # Test with invalid match_id
        with pytest.raises(ValueError):
            simulator = MatchSimulator(self.temp_file.name, 'INVALID_MATCH')
            simulator.get_match_ball_count()

class TestVideoSyncIntegration:
    """Integration tests for video sync functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_video_dir = '/mock/video/directory'
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_video_path_resolution(self, mock_getsize, mock_listdir, mock_exists):
        """Test video path resolution with various file patterns."""
        # Mock video directory exists
        mock_exists.return_value = True
        mock_getsize.return_value = 1024000  # 1MB
        
        # Mock video files with different patterns
        mock_video_files = [
            'Match_1_1_007_02.mp4',
            'Match_1_1_008_01.mp4',
            'Match_1_1_009_04.mp4',
            'sakumaryadav_over7_ball2.mp4',
            'SA_Kumar_Yadav_batting_highlights.mp4'
        ]
        mock_listdir.return_value = mock_video_files
        
        # Test exact match pattern
        ball_metadata = {
            'batter': 'SA(Kumar) Yadav',
            'bowler': 'Jasprit Bumrah',
            'over': 7,
            'ball': 2,
            'match_date': None
        }
        
        video_path = find_video_for_ball(ball_metadata, self.mock_video_dir)
        
        # Should find a video file
        assert video_path is not None
        assert os.path.basename(video_path) in mock_video_files
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_video_fallback_scenarios(self, mock_listdir, mock_exists):
        """Test video fallback scenarios when exact matches aren't found."""
        # Mock video directory exists
        mock_exists.return_value = True
        
        # Mock video files with only partial matches
        mock_video_files = [
            'sakumaryadav_generic_highlights.mp4',
            'other_player_over7_ball2.mp4',
            'random_video.mp4'
        ]
        mock_listdir.return_value = mock_video_files
        
        # Test with specific ball that might not have exact match
        ball_metadata = {
            'batter': 'SA(Kumar) Yadav',
            'bowler': 'Jasprit Bumrah',
            'over': 99,  # Non-existent over
            'ball': 99,  # Non-existent ball
            'match_date': None
        }
        
        video_path = find_video_for_ball(ball_metadata, self.mock_video_dir)
        
        # Should find generic highlights video or return None
        if video_path:
            assert 'sakumaryadav' in os.path.basename(video_path).lower()

if __name__ == '__main__':
    pytest.main([__file__]) 
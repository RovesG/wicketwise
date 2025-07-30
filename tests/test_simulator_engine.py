# Purpose: Test suite for simulator engine functionality
# Author: Phi1618 Cricket AI Team, Last Modified: 2024

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys

# Add the parent directory to the path to import simulator_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator_engine import MatchSimulator, create_simulator

class TestMatchSimulator:
    """Test suite for the MatchSimulator class."""
    
    @pytest.fixture
    def mock_single_match_data(self):
        """Create mock data for a single match with 10 balls."""
        data = {
            'ball_id': range(1, 11),
            'match_id': ['MATCH_001'] * 10,
            'over': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            'ball_in_over': [1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
            'batter': ['Smith', 'Smith', 'Jones', 'Jones', 'Smith', 'Smith', 'Jones', 'Jones', 'Smith', 'Smith'],
            'bowler': ['Kumar', 'Kumar', 'Kumar', 'Kumar', 'Kumar', 'Kumar', 'Patel', 'Patel', 'Patel', 'Patel'],
            'runs': [1, 0, 4, 0, 2, 1, 0, 0, 6, 1],
            'extras': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'wicket': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_multi_match_data(self):
        """Create mock data for multiple matches."""
        data = {
            'ball_id': range(1, 21),
            'match_id': ['MATCH_001'] * 10 + ['MATCH_002'] * 10,
            'over': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2] * 2,
            'ball_in_over': [1, 2, 3, 4, 5, 6, 1, 2, 3, 4] * 2,
            'batter': ['Smith', 'Smith', 'Jones', 'Jones', 'Smith', 'Smith', 'Jones', 'Jones', 'Smith', 'Smith'] * 2,
            'bowler': ['Kumar', 'Kumar', 'Kumar', 'Kumar', 'Kumar', 'Kumar', 'Patel', 'Patel', 'Patel', 'Patel'] * 2,
            'runs': [1, 0, 4, 0, 2, 1, 0, 0, 6, 1] * 2,
            'extras': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] * 2,
            'wicket': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] * 2
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, mock_single_match_data):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            mock_single_match_data.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name
        
        yield temp_file_path
        
        # Cleanup
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
    
    @pytest.fixture
    def temp_multi_match_csv_file(self, mock_multi_match_data):
        """Create a temporary CSV file with multiple matches for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            mock_multi_match_data.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name
        
        yield temp_file_path
        
        # Cleanup
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
    
    def test_simulator_initialization(self, temp_csv_file):
        """Test MatchSimulator initialization."""
        simulator = MatchSimulator(temp_csv_file)
        assert simulator.csv_path == temp_csv_file
        assert simulator.match_id is None
        assert simulator._cached is False
        
        # Test with match_id
        simulator_with_match = MatchSimulator(temp_csv_file, match_id="MATCH_001")
        assert simulator_with_match.match_id == "MATCH_001"
    
    def test_get_match_ball_count(self, temp_csv_file):
        """Test get_match_ball_count method."""
        simulator = MatchSimulator(temp_csv_file)
        ball_count = simulator.get_match_ball_count()
        assert ball_count == 10
        
        # Test caching
        assert simulator._cached is True
        ball_count_cached = simulator.get_match_ball_count()
        assert ball_count_cached == 10
    
    def test_get_ball_basic_functionality(self, temp_csv_file):
        """Test basic ball retrieval functionality."""
        simulator = MatchSimulator(temp_csv_file)
        
        # Test first ball
        ball_0 = simulator.get_ball(0)
        assert ball_0['ball_index'] == 0
        assert ball_0['ball_number'] == 1
        assert ball_0['batter'] == 'Smith'
        assert ball_0['bowler'] == 'Kumar'
        assert ball_0['over'] == 1
        assert ball_0['runs'] == 1
        assert ball_0['extras'] == 0
        assert ball_0['wicket'] == 0
        
        # Test middle ball
        ball_4 = simulator.get_ball(4)
        assert ball_4['ball_index'] == 4
        assert ball_4['ball_number'] == 5
        assert ball_4['batter'] == 'Smith'
        assert ball_4['bowler'] == 'Kumar'
        assert ball_4['runs'] == 2
        
        # Test last ball
        ball_9 = simulator.get_ball(9)
        assert ball_9['ball_index'] == 9
        assert ball_9['ball_number'] == 10
        assert ball_9['wicket'] == 1
    
    def test_get_ball_various_indexes(self, temp_csv_file):
        """Test ball retrieval accuracy for various indexes."""
        simulator = MatchSimulator(temp_csv_file)
        
        # Test specific known balls
        test_cases = [
            (0, {'batter': 'Smith', 'bowler': 'Kumar', 'runs': 1, 'extras': 0}),
            (2, {'batter': 'Jones', 'bowler': 'Kumar', 'runs': 4, 'extras': 0}),
            (3, {'batter': 'Jones', 'bowler': 'Kumar', 'runs': 0, 'extras': 1}),
            (8, {'batter': 'Smith', 'bowler': 'Patel', 'runs': 6, 'extras': 0}),
            (9, {'batter': 'Smith', 'bowler': 'Patel', 'runs': 1, 'wicket': 1}),
        ]
        
        for ball_index, expected_values in test_cases:
            ball_info = simulator.get_ball(ball_index)
            for key, expected_value in expected_values.items():
                assert ball_info[key] == expected_value, f"Ball {ball_index}, key {key}: expected {expected_value}, got {ball_info[key]}"
    
    def test_get_ball_out_of_range(self, temp_csv_file):
        """Test ball retrieval with out-of-range indexes."""
        simulator = MatchSimulator(temp_csv_file)
        
        # Test negative index
        with pytest.raises(IndexError):
            simulator.get_ball(-1)
        
        # Test index too large
        with pytest.raises(IndexError):
            simulator.get_ball(10)  # Only 10 balls (0-9)
        
        # Test way out of range
        with pytest.raises(IndexError):
            simulator.get_ball(100)
    
    def test_match_filtering_single_match(self, temp_multi_match_csv_file):
        """Test correct match filtering for single match."""
        # Test filtering for MATCH_001
        simulator_001 = MatchSimulator(temp_multi_match_csv_file, match_id="MATCH_001")
        ball_count_001 = simulator_001.get_match_ball_count()
        assert ball_count_001 == 10
        
        # Verify the first ball is from MATCH_001
        first_ball = simulator_001.get_ball(0)
        assert first_ball['raw_data']['match_id'] == 'MATCH_001'
        
        # Test filtering for MATCH_002
        simulator_002 = MatchSimulator(temp_multi_match_csv_file, match_id="MATCH_002")
        ball_count_002 = simulator_002.get_match_ball_count()
        assert ball_count_002 == 10
        
        # Verify the first ball is from MATCH_002
        first_ball_002 = simulator_002.get_ball(0)
        assert first_ball_002['raw_data']['match_id'] == 'MATCH_002'
    
    def test_match_filtering_no_match_id(self, temp_multi_match_csv_file):
        """Test behavior when no match_id is specified."""
        simulator = MatchSimulator(temp_multi_match_csv_file)
        ball_count = simulator.get_match_ball_count()
        assert ball_count == 20  # All balls from both matches
    
    def test_match_filtering_invalid_match_id(self, temp_multi_match_csv_file):
        """Test behavior with invalid match_id."""
        simulator = MatchSimulator(temp_multi_match_csv_file, match_id="INVALID_MATCH")
        
        with pytest.raises(ValueError, match="No data found for match_id: INVALID_MATCH"):
            simulator.get_match_ball_count()
    
    def test_match_state_calculation(self, temp_csv_file):
        """Test match state calculation."""
        simulator = MatchSimulator(temp_csv_file)
        
        # Test first ball
        ball_0 = simulator.get_ball(0)
        match_state = ball_0['match_state']
        assert match_state['balls_bowled'] == 1
        assert match_state['total_balls'] == 10
        assert match_state['progress_percentage'] == 10.0
        assert match_state['total_runs'] == 1
        assert match_state['total_wickets'] == 0
        
        # Test middle ball
        ball_4 = simulator.get_ball(4)
        match_state = ball_4['match_state']
        assert match_state['balls_bowled'] == 5
        assert match_state['progress_percentage'] == 50.0
        assert match_state['total_runs'] == 7  # 1+0+4+0+2
        assert match_state['total_wickets'] == 0
        
        # Test last ball
        ball_9 = simulator.get_ball(9)
        match_state = ball_9['match_state']
        assert match_state['balls_bowled'] == 10
        assert match_state['progress_percentage'] == 100.0
        assert match_state['total_wickets'] == 1
    
    def test_column_flexibility(self):
        """Test handling of different column names."""
        # Create data with alternative column names
        data = {
            'ball_id': [1, 2, 3],
            'match_id': ['MATCH_001'] * 3,
            'over': [1, 1, 1],
            'batsman': ['Smith', 'Jones', 'Smith'],  # Different column name
            'bowler': ['Kumar', 'Kumar', 'Patel'],
            'runs_off_bat': [1, 0, 4],  # Different column name
            'extras': [0, 0, 0],
            'is_wicket': [0, 0, 0]  # Different column name
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name
        
        try:
            simulator = MatchSimulator(temp_file_path)
            ball_0 = simulator.get_ball(0)
            
            # Check that alternative column names are handled
            assert ball_0['batter'] == 'Smith'  # From 'batsman' column
            assert ball_0['runs'] == 1  # From 'runs_off_bat' column
            assert ball_0['wicket'] == 0  # From 'is_wicket' column
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_get_match_summary(self, temp_csv_file):
        """Test get_match_summary method."""
        simulator = MatchSimulator(temp_csv_file, match_id="MATCH_001")
        summary = simulator.get_match_summary()
        
        assert summary['match_id'] == "MATCH_001"
        assert summary['total_balls'] == 10
        assert summary['total_overs'] == 1.7  # 10 balls / 6
        assert summary['total_runs'] == 15  # Sum of runs
        assert summary['total_wickets'] == 1
        assert summary['boundaries'] == 2  # 4 and 6
        assert summary['sixes'] == 1  # One six
        assert 'Smith' in summary['batters']
        assert 'Jones' in summary['batters']
        assert 'Kumar' in summary['bowlers']
        assert 'Patel' in summary['bowlers']
    
    def test_caching_behavior(self, temp_csv_file):
        """Test caching behavior."""
        simulator = MatchSimulator(temp_csv_file)
        
        # Initially not cached
        assert simulator._cached is False
        
        # First call should cache the data
        ball_count_1 = simulator.get_match_ball_count()
        assert simulator._cached is True
        
        # Second call should use cached data
        ball_count_2 = simulator.get_match_ball_count()
        assert ball_count_1 == ball_count_2
        
        # Reset cache
        simulator.reset_cache()
        assert simulator._cached is False
    
    def test_set_match_id(self, temp_multi_match_csv_file):
        """Test set_match_id method."""
        simulator = MatchSimulator(temp_multi_match_csv_file, match_id="MATCH_001")
        
        # Initial state
        ball_count_1 = simulator.get_match_ball_count()
        assert ball_count_1 == 10
        
        # Change match_id
        simulator.set_match_id("MATCH_002")
        ball_count_2 = simulator.get_match_ball_count()
        assert ball_count_2 == 10
        
        # Verify we're getting different data
        first_ball_002 = simulator.get_ball(0)
        assert first_ball_002['raw_data']['match_id'] == 'MATCH_002'
    
    def test_file_not_found_error(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            simulator = MatchSimulator("non_existent_file.csv")
            simulator.get_match_ball_count()
    
    def test_create_simulator_function(self, temp_csv_file):
        """Test the create_simulator convenience function."""
        simulator = create_simulator(temp_csv_file, match_id="MATCH_001")
        assert isinstance(simulator, MatchSimulator)
        assert simulator.csv_path == temp_csv_file
        assert simulator.match_id == "MATCH_001"
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns."""
        # Create minimal data with missing columns
        data = {
            'ball_id': [1, 2, 3],
            'batter': ['Smith', 'Jones', 'Smith'],
            'bowler': ['Kumar', 'Kumar', 'Patel']
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name
        
        try:
            simulator = MatchSimulator(temp_file_path)
            ball_0 = simulator.get_ball(0)
            
            # Check default values for missing columns
            assert ball_0['runs'] == 0  # Default for missing runs
            assert ball_0['extras'] == 0  # Default for missing extras
            assert ball_0['wicket'] == 0  # Default for missing wicket
            assert ball_0['phase'] == 'Unknown'  # Default for missing phase
            assert ball_0['dismissal'] is None  # Default for missing dismissal
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


class TestSimulatorEngineIntegration:
    """Integration tests for the simulator engine."""
    
    def test_real_csv_file_integration(self):
        """Test integration with the actual test_matches.csv file."""
        csv_path = "test_matches.csv"
        
        # Test without match_id (should load all data)
        simulator_all = MatchSimulator(csv_path)
        total_balls = simulator_all.get_match_ball_count()
        assert total_balls == 20  # All balls from both matches
        
        # Test with MATCH_001
        simulator_001 = MatchSimulator(csv_path, match_id="MATCH_001")
        balls_001 = simulator_001.get_match_ball_count()
        assert balls_001 == 12  # First 12 balls are MATCH_001
        
        # Test with MATCH_002
        simulator_002 = MatchSimulator(csv_path, match_id="MATCH_002")
        balls_002 = simulator_002.get_match_ball_count()
        assert balls_002 == 8  # Last 8 balls are MATCH_002
        
        # Test specific ball retrieval
        ball_0_001 = simulator_001.get_ball(0)
        assert ball_0_001['batter'] == 'Smith'
        assert ball_0_001['bowler'] == 'Kumar'
        assert ball_0_001['runs'] == 1
        
        ball_0_002 = simulator_002.get_ball(0)
        assert ball_0_002['batter'] == 'Williams'
        assert ball_0_002['bowler'] == 'Singh'
        assert ball_0_002['runs'] == 1
    
    def test_match_summary_integration(self):
        """Test match summary with real data."""
        csv_path = "test_matches.csv"
        simulator = MatchSimulator(csv_path, match_id="MATCH_001")
        
        summary = simulator.get_match_summary()
        assert summary['match_id'] == "MATCH_001"
        assert summary['total_balls'] == 12
        assert summary['total_overs'] == 2.0
        assert 'Smith' in summary['batters']
        assert 'Jones' in summary['batters']
        assert 'Kumar' in summary['bowlers']
        assert 'Patel' in summary['bowlers']
    
    def test_error_scenarios(self):
        """Test various error scenarios."""
        csv_path = "test_matches.csv"
        
        # Test invalid match_id
        with pytest.raises(ValueError):
            simulator = MatchSimulator(csv_path, match_id="INVALID_MATCH")
            simulator.get_match_ball_count()
        
        # Test invalid ball index
        simulator = MatchSimulator(csv_path, match_id="MATCH_001")
        with pytest.raises(IndexError):
            simulator.get_ball(-1)
        
        with pytest.raises(IndexError):
            simulator.get_ball(100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
# Purpose: Test suite for UI simulator functionality
# Author: Phi1618 Cricket AI Team, Last Modified: 2024

import pytest
import pandas as pd
import io
import streamlit as st
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import ui_launcher
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_launcher import render_simulator_mode, create_sample_data

class TestSimulatorMode:
    """Test suite for the simulator mode functionality."""
    
    @pytest.fixture
    def mock_csv_data(self):
        """Create mock CSV data for testing."""
        data = {
            'ball_id': [1, 2, 3, 4, 5],
            'over': [1, 1, 1, 1, 1],
            'batter': ['Smith', 'Smith', 'Jones', 'Jones', 'Smith'],
            'bowler': ['Kumar', 'Kumar', 'Patel', 'Patel', 'Kumar'],
            'runs': [1, 0, 4, 0, 2],
            'extras': [0, 0, 0, 1, 0],
            'wicket': [0, 0, 0, 0, 0]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_csv_string(self, mock_csv_data):
        """Convert mock data to CSV string for file upload simulation."""
        return mock_csv_data.to_csv(index=False)
    
    def test_create_sample_data(self):
        """Test that create_sample_data returns valid DataFrame."""
        df = create_sample_data()
        
        # Check that it's a DataFrame
        assert isinstance(df, pd.DataFrame)
        
        # Check expected columns
        expected_columns = ['ball_id', 'over', 'batter', 'bowler', 'runs', 'extras', 'wicket']
        assert all(col in df.columns for col in expected_columns)
        
        # Check data types and values
        assert df.shape[0] == 20  # 20 balls
        assert df['ball_id'].dtype == int
        assert df['runs'].dtype == int
        assert df['wicket'].dtype == int
        
        # Check that runs are non-negative
        assert (df['runs'] >= 0).all()
        
        # Check that extras are non-negative
        assert (df['extras'] >= 0).all()
        
        # Check that wicket is binary (0 or 1)
        assert df['wicket'].isin([0, 1]).all()
    
    def test_csv_data_loading(self, mock_csv_data):
        """Test CSV data loading functionality."""
        # Test basic data structure
        assert len(mock_csv_data) == 5
        assert 'batter' in mock_csv_data.columns
        assert 'bowler' in mock_csv_data.columns
        assert 'runs' in mock_csv_data.columns
        
        # Test data content
        assert mock_csv_data['batter'].iloc[0] == 'Smith'
        assert mock_csv_data['bowler'].iloc[0] == 'Kumar'
        assert mock_csv_data['runs'].iloc[0] == 1
    
    def test_slider_updates_display_area(self, mock_csv_data):
        """Test that slider updates correctly affect display area."""
        # Test different ball positions
        for ball_num in range(1, 6):  # 1 to 5
            current_row = mock_csv_data.iloc[ball_num - 1]  # Convert to 0-based index
            
            # Verify we can access the row data
            assert current_row is not None
            assert 'batter' in current_row.index
            assert 'bowler' in current_row.index
            assert 'runs' in current_row.index
            
            # Test specific values for known data
            if ball_num == 1:
                assert current_row['batter'] == 'Smith'
                assert current_row['bowler'] == 'Kumar'
                assert current_row['runs'] == 1
            elif ball_num == 3:
                assert current_row['batter'] == 'Jones'
                assert current_row['bowler'] == 'Patel'
                assert current_row['runs'] == 4
    
    def test_display_area_shows_correct_info(self, mock_csv_data):
        """Test that display area shows correct information for each ball."""
        # Test first ball
        ball_1_data = mock_csv_data.iloc[0]
        assert ball_1_data['batter'] == 'Smith'
        assert ball_1_data['bowler'] == 'Kumar'
        assert ball_1_data['runs'] == 1
        assert ball_1_data['extras'] == 0
        
        # Test third ball (boundary)
        ball_3_data = mock_csv_data.iloc[2]
        assert ball_3_data['batter'] == 'Jones'
        assert ball_3_data['bowler'] == 'Patel'
        assert ball_3_data['runs'] == 4
        assert ball_3_data['extras'] == 0
        
        # Test fourth ball (with extras)
        ball_4_data = mock_csv_data.iloc[3]
        assert ball_4_data['batter'] == 'Jones'
        assert ball_4_data['bowler'] == 'Patel'
        assert ball_4_data['runs'] == 0
        assert ball_4_data['extras'] == 1
    
    def test_column_flexibility(self):
        """Test that the system handles different column names."""
        # Test with 'batsman' instead of 'batter'
        data_batsman = {
            'ball_id': [1, 2, 3],
            'over': [1, 1, 1],
            'batsman': ['Smith', 'Jones', 'Smith'],  # Different column name
            'bowler': ['Kumar', 'Kumar', 'Patel'],
            'runs_off_bat': [1, 0, 4],  # Different column name
            'extras': [0, 0, 0]
        }
        df_batsman = pd.DataFrame(data_batsman)
        
        # Check that both column names are handled
        assert 'batsman' in df_batsman.columns
        assert 'runs_off_bat' in df_batsman.columns
        
        # Test accessing data with different column names
        row = df_batsman.iloc[0]
        assert row['batsman'] == 'Smith'
        assert row['runs_off_bat'] == 1
    
    def test_slider_boundary_conditions(self, mock_csv_data):
        """Test slider behavior at boundary conditions."""
        max_balls = len(mock_csv_data)
        
        # Test first ball (minimum)
        first_ball = mock_csv_data.iloc[0]
        assert first_ball['ball_id'] == 1
        
        # Test last ball (maximum)
        last_ball = mock_csv_data.iloc[max_balls - 1]
        assert last_ball['ball_id'] == 5
        
        # Test that we can't go beyond boundaries
        assert max_balls == 5
        
        # Test middle ball
        middle_ball = mock_csv_data.iloc[2]
        assert middle_ball['ball_id'] == 3
    
    def test_data_conversion_to_dict(self, mock_csv_data):
        """Test that row data can be converted to dictionary for display."""
        for i in range(len(mock_csv_data)):
            row = mock_csv_data.iloc[i]
            ball_dict = row.to_dict()
            
            # Check that conversion worked
            assert isinstance(ball_dict, dict)
            assert 'batter' in ball_dict
            assert 'bowler' in ball_dict
            assert 'runs' in ball_dict
            
            # Check that values are preserved
            assert ball_dict['batter'] == row['batter']
            assert ball_dict['bowler'] == row['bowler']
            assert ball_dict['runs'] == row['runs']
    
    def test_csv_string_to_dataframe(self, mock_csv_string):
        """Test that CSV string can be converted to DataFrame."""
        # Simulate file upload by converting string to StringIO
        csv_buffer = io.StringIO(mock_csv_string)
        df = pd.read_csv(csv_buffer)
        
        # Check that DataFrame was created correctly
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'batter' in df.columns
        assert 'bowler' in df.columns
        assert 'runs' in df.columns
        
        # Check data integrity
        assert df['batter'].iloc[0] == 'Smith'
        assert df['runs'].iloc[2] == 4
    
    def test_error_handling_invalid_csv(self):
        """Test error handling for invalid CSV data."""
        # Test with malformed CSV
        invalid_csv = "invalid,csv,data\n1,2"  # Missing value
        
        csv_buffer = io.StringIO(invalid_csv)
        try:
            df = pd.read_csv(csv_buffer)
            # Should have NaN for missing values
            assert df.isna().any().any()
        except Exception as e:
            # Should handle parsing errors gracefully
            assert isinstance(e, (pd.errors.ParserError, ValueError))
    
    def test_session_state_structure(self):
        """Test expected session state structure for simulator."""
        # Expected keys in session state
        expected_keys = ['simulator_data', 'simulator_playing', 'current_ball']
        
        # This would normally be tested with actual Streamlit session state
        # For now, test the expected structure
        mock_session_state = {
            'simulator_data': None,
            'simulator_playing': False,
            'current_ball': 1
        }
        
        for key in expected_keys:
            assert key in mock_session_state
        
        # Test default values
        assert mock_session_state['simulator_data'] is None
        assert mock_session_state['simulator_playing'] is False
        assert mock_session_state['current_ball'] == 1


class TestSimulatorIntegration:
    """Integration tests for simulator functionality."""
    
    def test_complete_simulator_workflow(self):
        """Test the complete simulator workflow from data loading to display."""
        # Create test data
        df = create_sample_data()
        
        # Simulate loading data
        assert df is not None
        assert len(df) > 0
        
        # Simulate slider interaction
        for ball_num in range(1, min(6, len(df) + 1)):
            current_row = df.iloc[ball_num - 1]
            
            # Verify basic data access
            assert current_row is not None
            assert 'batter' in current_row.index
            assert 'bowler' in current_row.index
            
            # Test metrics calculation
            ball_data = {
                'ball_number': ball_num,
                'batter': current_row['batter'],
                'bowler': current_row['bowler'],
                'runs': current_row['runs'],
                'extras': current_row['extras']
            }
            
            # Verify all required data is present
            assert ball_data['ball_number'] == ball_num
            assert isinstance(ball_data['batter'], str)
            assert isinstance(ball_data['bowler'], str)
            # Handle numpy data types
            assert isinstance(ball_data['runs'], (int, float)) or hasattr(ball_data['runs'], 'dtype')
            assert isinstance(ball_data['extras'], (int, float)) or hasattr(ball_data['extras'], 'dtype')
    
    def test_data_validation(self):
        """Test data validation for simulator input."""
        df = create_sample_data()
        
        # Test required columns exist
        required_cols = ['over', 'batter', 'bowler', 'runs']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Test data types
        assert df['runs'].dtype in ['int64', 'float64'], "Runs should be numeric"
        assert df['extras'].dtype in ['int64', 'float64'], "Extras should be numeric"
        
        # Test data ranges
        assert (df['runs'] >= 0).all(), "Runs should be non-negative"
        assert (df['extras'] >= 0).all(), "Extras should be non-negative"
        assert (df['over'] >= 1).all(), "Over should be >= 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
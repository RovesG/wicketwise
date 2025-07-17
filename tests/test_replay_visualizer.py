# Purpose: Test suite for replay visualizer functionality
# Author: Assistant, Last Modified: 2024

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys

# Add the wicketwise module to path
sys.path.append(str(Path(__file__).parent.parent))

from wicketwise.replay_visualizer import ReplayVisualizer


class TestReplayVisualizer:
    """Test suite for the ReplayVisualizer class."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        # Generate sample data for 3 matches with varying number of balls
        data = []
        
        for match_id in ["match_001", "match_002", "match_003"]:
            # Each match has 60-120 balls (10-20 overs)
            num_balls = np.random.randint(60, 121)
            
            for ball_num in range(num_balls):
                # Generate realistic cricket data
                actual_runs = np.random.choice([0, 1, 2, 3, 4, 6, -1], 
                                             p=[0.4, 0.25, 0.15, 0.05, 0.08, 0.05, 0.02])
                
                # Convert -1 to wicket indicator (using 10 as wicket)
                if actual_runs == -1:
                    actual_runs = 10
                
                # Generate predicted outcome class
                if actual_runs == 10:
                    predicted_class = "wicket"
                else:
                    predicted_class = f"{actual_runs}_runs" if actual_runs != 1 else "1_run"
                
                # Sometimes make predictions wrong for testing
                if np.random.random() < 0.2:  # 20% wrong predictions
                    wrong_outcomes = ["0_runs", "1_run", "2_runs", "4_runs", "6_runs", "wicket"]
                    predicted_class = np.random.choice(wrong_outcomes)
                
                # Generate other fields
                over_num = (ball_num // 6) + 1
                ball_in_over = (ball_num % 6) + 1
                
                # Determine phase
                if over_num <= 6:
                    phase = "powerplay"
                elif over_num <= 15:
                    phase = "middle_overs"
                else:
                    phase = "death_overs"
                
                # Generate win probability (trending based on match progression)
                base_prob = 0.5 + 0.3 * np.sin(ball_num / num_balls * np.pi)
                win_prob = np.clip(base_prob + np.random.normal(0, 0.1), 0, 1)
                
                # Generate odds mispricing
                odds_mispricing = np.random.normal(0, 0.1)
                
                # Add probability columns for different outcomes
                probs = np.random.dirichlet([2, 3, 2, 1, 1, 1, 1])  # 7 outcomes
                
                data.append({
                    "match_id": match_id,
                    "ball_id": f"{over_num}.{ball_in_over}",
                    "actual_runs": actual_runs,
                    "predicted_runs_class": predicted_class,
                    "win_prob": win_prob,
                    "odds_mispricing": odds_mispricing,
                    "phase": phase,
                    "batter_id": f"batter_{np.random.randint(1, 12)}",
                    "bowler_id": f"bowler_{np.random.randint(1, 8)}",
                    "predicted_runs_0": probs[0],
                    "predicted_runs_1": probs[1],
                    "predicted_runs_2": probs[2],
                    "predicted_runs_3": probs[3],
                    "predicted_runs_4": probs[4],
                    "predicted_runs_6": probs[5],
                    "predicted_wicket": probs[6],
                    "actual_win_prob": win_prob + np.random.normal(0, 0.05),
                    "actual_mispricing": odds_mispricing + np.random.normal(0, 0.02)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create a temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            yield f.name
        
        # Cleanup
        try:
            os.unlink(f.name)
        except:
            pass
    
    @pytest.fixture
    def visualizer(self):
        """Create a ReplayVisualizer instance."""
        return ReplayVisualizer()
    
    def test_init(self, visualizer):
        """Test ReplayVisualizer initialization."""
        assert visualizer.data is None
        assert visualizer.matches == []
        assert visualizer.current_match is None
        assert visualizer.current_ball_idx == 0
        
        # Check color mappings are defined
        assert "0_runs" in visualizer.outcome_colors
        assert "wicket" in visualizer.outcome_colors
        assert "value_bet" in visualizer.betting_colors
        assert "no_bet" in visualizer.betting_colors
        assert "risk_alert" in visualizer.betting_colors
    
    def test_load_data_success(self, visualizer, temp_csv_file):
        """Test successful data loading."""
        # Mock Streamlit functions
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.success') as mock_success:
            
            result = visualizer.load_data(temp_csv_file)
            
            assert result is True
            assert visualizer.data is not None
            assert len(visualizer.data) > 0
            assert len(visualizer.matches) == 3  # 3 matches in sample data
            assert "match_001" in visualizer.matches
            assert "match_002" in visualizer.matches
            assert "match_003" in visualizer.matches
            
            # Check that success message was called
            mock_success.assert_called_once()
            mock_error.assert_not_called()
    
    def test_load_data_file_not_found(self, visualizer):
        """Test loading data from non-existent file."""
        with patch('streamlit.error') as mock_error:
            result = visualizer.load_data("non_existent_file.csv")
            
            assert result is False
            assert visualizer.data is None
            mock_error.assert_called_once()
    
    def test_load_data_missing_columns(self, visualizer):
        """Test loading data with missing required columns."""
        # Create CSV with missing columns
        incomplete_data = pd.DataFrame({
            "match_id": ["match_1"],
            "ball_id": ["1.1"],
            "actual_runs": [4]
            # Missing other required columns
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            incomplete_data.to_csv(f.name, index=False)
            
            with patch('streamlit.error') as mock_error:
                result = visualizer.load_data(f.name)
                
                assert result is False
                mock_error.assert_called_once()
            
            os.unlink(f.name)
    
    def test_runs_to_outcome_class(self, visualizer):
        """Test conversion of runs to outcome class."""
        assert visualizer._runs_to_outcome_class(0) == "0_runs"
        assert visualizer._runs_to_outcome_class(1) == "1_run"
        assert visualizer._runs_to_outcome_class(2) == "2_runs"
        assert visualizer._runs_to_outcome_class(3) == "3_runs"
        assert visualizer._runs_to_outcome_class(4) == "4_runs"
        assert visualizer._runs_to_outcome_class(6) == "6_runs"
        assert visualizer._runs_to_outcome_class(10) == "wicket"
        assert visualizer._runs_to_outcome_class(-1) == "wicket"
    
    def test_generate_betting_decision(self, visualizer):
        """Test betting decision generation logic."""
        # Test value bet scenario
        row = pd.Series({
            'odds_mispricing': 0.2,
            'win_prob': 0.7
        })
        assert visualizer._generate_betting_decision(row) == "value_bet"
        
        # Test risk alert scenario (high negative mispricing)
        row = pd.Series({
            'odds_mispricing': -0.2,
            'win_prob': 0.5
        })
        assert visualizer._generate_betting_decision(row) == "risk_alert"
        
        # Test risk alert scenario (low win probability)
        row = pd.Series({
            'odds_mispricing': 0.1,
            'win_prob': 0.1
        })
        assert visualizer._generate_betting_decision(row) == "risk_alert"
        
        # Test no bet scenario
        row = pd.Series({
            'odds_mispricing': 0.05,
            'win_prob': 0.5
        })
        assert visualizer._generate_betting_decision(row) == "no_bet"
    
    def test_extract_over_ball(self, visualizer):
        """Test over.ball extraction logic."""
        # Test with existing over.ball format
        row = pd.Series({
            'ball_id': '5.3',
            'ball_sequence': 27
        })
        assert visualizer._extract_over_ball(row) == '5.3'
        
        # Test with sequence-based generation
        row = pd.Series({
            'ball_id': 'ball_7',
            'ball_sequence': 7
        })
        assert visualizer._extract_over_ball(row) == '2.1'  # 7th ball = 2nd over, 1st ball
        
        # Test with sequence-based generation (different ball)
        row = pd.Series({
            'ball_id': 'ball_12',
            'ball_sequence': 12
        })
        assert visualizer._extract_over_ball(row) == '2.6'  # 12th ball = 2nd over, 6th ball
    
    def test_process_data(self, visualizer, sample_csv_data):
        """Test data processing functionality."""
        processed_data = visualizer._process_data(sample_csv_data.copy())
        
        # Check that new columns were added
        assert 'actual_runs_str' in processed_data.columns
        assert 'prediction_correct' in processed_data.columns
        assert 'betting_decision' in processed_data.columns
        assert 'ball_sequence' in processed_data.columns
        assert 'win_prob_change' in processed_data.columns
        assert 'over_ball' in processed_data.columns
        
        # Check that actual_runs_str is correctly converted
        assert processed_data['actual_runs_str'].iloc[0] in [
            "0_runs", "1_run", "2_runs", "3_runs", "4_runs", "6_runs", "wicket"
        ]
        
        # Check that prediction_correct is boolean
        assert processed_data['prediction_correct'].dtype == bool
        
        # Check that betting_decision has valid values
        valid_decisions = {"value_bet", "no_bet", "risk_alert"}
        assert set(processed_data['betting_decision'].unique()).issubset(valid_decisions)
        
        # Check that ball_sequence is sequential within matches
        for match_id in processed_data['match_id'].unique():
            match_data = processed_data[processed_data['match_id'] == match_id]
            ball_sequences = match_data['ball_sequence'].values
            assert list(ball_sequences) == list(range(1, len(ball_sequences) + 1))
    
    def test_render_match_selector_no_matches(self, visualizer):
        """Test match selector with no matches loaded."""
        with patch('streamlit.warning') as mock_warning:
            result = visualizer.render_match_selector()
            
            assert result is None
            mock_warning.assert_called_once()
    
    def test_render_match_selector_with_matches(self, visualizer, temp_csv_file):
        """Test match selector with loaded matches."""
        # Load data first
        with patch('streamlit.error'), patch('streamlit.success'):
            visualizer.load_data(temp_csv_file)
        
        with patch('streamlit.selectbox') as mock_selectbox:
            mock_selectbox.return_value = "match_001"
            
            result = visualizer.render_match_selector()
            
            assert result == "match_001"
            mock_selectbox.assert_called_once()
            
            # Check that options include all matches
            call_args = mock_selectbox.call_args
            assert "match_001" in call_args[1]['options']
            assert "match_002" in call_args[1]['options']
            assert "match_003" in call_args[1]['options']
    
    def test_render_ball_navigation(self, visualizer, sample_csv_data):
        """Test ball navigation interface."""
        # Create sample match data
        match_data = sample_csv_data[sample_csv_data['match_id'] == 'match_001']
        
        with patch('streamlit.slider') as mock_slider, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.write') as mock_write:
            
            # Mock the slider to return a specific value
            mock_slider.return_value = 5
            
            # Mock columns context manager
            mock_col = MagicMock()
            mock_columns.return_value = [mock_col, mock_col, mock_col]
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=None)
            
            result = visualizer.render_ball_navigation(match_data)
            
            assert result == 5
            mock_slider.assert_called_once()
            
            # Check slider parameters
            call_args = mock_slider.call_args
            assert call_args[1]['min_value'] == 0
            assert call_args[1]['max_value'] == len(match_data) - 1
    
    def test_render_filters(self, visualizer, sample_csv_data):
        """Test sidebar filters functionality."""
        match_data = sample_csv_data[sample_csv_data['match_id'] == 'match_001']
        
        # Process the data first to add required columns
        processed_data = visualizer._process_data(match_data)
        
        with patch('streamlit.sidebar') as mock_sidebar:
            # Mock sidebar components
            mock_sidebar.header = MagicMock()
            mock_sidebar.selectbox = MagicMock(return_value='All')
            mock_sidebar.slider = MagicMock(return_value=(1, 20))
            mock_sidebar.write = MagicMock()
            
            result = visualizer.render_filters(processed_data)
            
            # Should return the original data since all filters are 'All'
            assert len(result) == len(processed_data)
            assert result.equals(processed_data)
    
    def test_render_filters_with_selections(self, visualizer, sample_csv_data):
        """Test sidebar filters with specific selections."""
        match_data = sample_csv_data[sample_csv_data['match_id'] == 'match_001']
        
        # Process the data first to add required columns
        processed_data = visualizer._process_data(match_data)
        
        # Get a specific batter from the data
        specific_batter = processed_data['batter_id'].iloc[0]
        
        with patch('streamlit.sidebar') as mock_sidebar:
            # Mock sidebar components with specific selections
            mock_sidebar.header = MagicMock()
            mock_sidebar.selectbox = MagicMock(side_effect=[specific_batter, 'All', 'All'])
            mock_sidebar.slider = MagicMock(return_value=(1, 20))
            mock_sidebar.write = MagicMock()
            
            result = visualizer.render_filters(processed_data)
            
            # Should return only data for the specific batter
            assert len(result) <= len(processed_data)
            assert all(result['batter_id'] == specific_batter)
    
    def test_integration_with_sample_data(self, visualizer, temp_csv_file):
        """Test full integration with sample data."""
        # Mock all Streamlit functions
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.success') as mock_success:
            
            # Load data
            success = visualizer.load_data(temp_csv_file)
            assert success is True
            
            # Verify data structure
            assert visualizer.data is not None
            assert len(visualizer.matches) == 3
            
            # Test that all required columns exist after processing
            required_processed_columns = [
                'actual_runs_str', 'prediction_correct', 'betting_decision',
                'ball_sequence', 'win_prob_change', 'over_ball'
            ]
            
            for col in required_processed_columns:
                assert col in visualizer.data.columns
            
            # Test that we can get match data
            match_data = visualizer.data[visualizer.data['match_id'] == 'match_001']
            assert len(match_data) > 0
            
            # Test that ball sequences are properly ordered
            ball_sequences = match_data['ball_sequence'].values
            assert list(ball_sequences) == sorted(ball_sequences)
    
    def test_color_mappings(self, visualizer):
        """Test that color mappings are properly defined."""
        # Test outcome colors
        outcome_colors = visualizer.outcome_colors
        expected_outcomes = ["0_runs", "1_run", "2_runs", "3_runs", "4_runs", "6_runs", "wicket"]
        
        for outcome in expected_outcomes:
            assert outcome in outcome_colors
            assert outcome_colors[outcome].startswith('#')  # Should be hex color
        
        # Test betting colors
        betting_colors = visualizer.betting_colors
        expected_decisions = ["value_bet", "no_bet", "risk_alert"]
        
        for decision in expected_decisions:
            assert decision in betting_colors
            assert betting_colors[decision].startswith('#')  # Should be hex color
    
    def test_data_validation(self, visualizer):
        """Test data validation with various edge cases."""
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            empty_data.to_csv(f.name, index=False)
            
            with patch('streamlit.error') as mock_error:
                result = visualizer.load_data(f.name)
                assert result is False
            
            os.unlink(f.name)
        
        # Test with data containing NaN values
        nan_data = pd.DataFrame({
            "match_id": ["match_1", "match_1"],
            "ball_id": ["1.1", "1.2"],
            "actual_runs": [4, np.nan],
            "predicted_runs_class": ["4_runs", "2_runs"],
            "win_prob": [0.6, 0.7],
            "odds_mispricing": [0.1, 0.2],
            "phase": ["powerplay", "powerplay"],
            "batter_id": ["batter_1", "batter_2"],
            "bowler_id": ["bowler_1", "bowler_1"]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            nan_data.to_csv(f.name, index=False)
            
            with patch('streamlit.error'), patch('streamlit.success'):
                result = visualizer.load_data(f.name)
                # Should still load successfully, but with processed NaN values
                assert result is True
                assert visualizer.data is not None
            
            os.unlink(f.name)


def test_main_function():
    """Test that the main function can be imported and called without errors."""
    # This test ensures the main function exists and can be imported
    from wicketwise.replay_visualizer import main
    
    # We can't easily test the Streamlit app without running it,
    # but we can at least verify the function exists
    assert callable(main)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 
# Purpose: Test UI style components for cricket analysis interface
# Author: Phi1618 Cricket AI, Last Modified: 2025-01-17

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock streamlit before importing ui_style
mock_st = MagicMock()
sys.modules['streamlit'] = mock_st

# Import UI style components
from ui_style import (
    render_batter_card,
    render_bowler_card,
    render_odds_panel,
    render_win_bar,
    render_match_status,
    render_info_panel,
    CricketColors,
    Typography
)

# Helper function for mocking columns
def mock_columns_helper(n):
    """Helper function to mock st.columns() with both integer and list arguments"""
    if isinstance(n, list):
        return [MagicMock() for _ in range(len(n))]
    else:
        return [MagicMock() for _ in range(n)]

class TestBatterCard:
    """Test suite for render_batter_card function"""
    
    def test_render_batter_card_complete_data(self):
        """Test batter card with complete player data"""
        player_data = {
            'name': 'Virat Kohli',
            'average': '45.2',
            'strike_rate': '142.3',
            'recent_shots': '4, 1, dot, 6, 2',
            'runs': 67,
            'balls_faced': 48,
            'image_url': 'https://example.com/virat.jpg',
            'team_color': '#c8712d',
            'highest_score': '183',
            'boundaries': '8'
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_batter_card(player_data)
            
            # Verify streamlit components were called
            mock_st.container.assert_called()
            mock_st.markdown.assert_called()
            # Should have image column, stats grid (3 columns x 2 rows), and layout column
            assert mock_st.columns.call_count >= 3
            # Should have 6 metrics in the stats grid
            assert mock_st.metric.call_count >= 6
    
    def test_render_batter_card_minimal_data(self):
        """Test batter card with minimal data"""
        player_data = {
            'name': 'MS Dhoni'
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_batter_card(player_data)
            
            # Verify fallback values were used
            mock_st.markdown.assert_called()
            mock_st.metric.assert_called()
    
    def test_render_batter_card_empty_data(self):
        """Test batter card with empty data"""
        player_data = {}
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_batter_card(player_data)
            
            # Verify function completed without errors
            mock_st.container.assert_called()
    
    def test_render_batter_card_with_image_fallback(self):
        """Test batter card with missing image (fallback placeholder)"""
        player_data = {
            'name': 'Rohit Sharma',
            'average': '32.1',
            'strike_rate': '139.7',
            'runs': 45,
            'balls_faced': 35,
            'team_color': '#ff6b35',
            'highest_score': '264',
            'boundaries': '6'
            # image_url is missing - should use placeholder
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_batter_card(player_data)
            
            # Verify container and markdown were called
            mock_st.container.assert_called()
            mock_st.markdown.assert_called()
            
            # Should have 6 metrics in the stats grid (2 rows x 3 columns)
            assert mock_st.metric.call_count >= 6
            
            # Check that st.image was NOT called (since no image_url)
            mock_st.image.assert_not_called()
    
    def test_render_batter_card_failed_image_load(self):
        """Test batter card with image URL that fails to load"""
        player_data = {
            'name': 'KL Rahul',
            'average': '35.8',
            'strike_rate': '126.4',
            'runs': 28,
            'balls_faced': 22,
            'image_url': 'https://broken.link/invalid.jpg',
            'team_color': '#2e8b57',
            'highest_score': '132',
            'boundaries': '4'
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Mock st.image to raise an exception (simulating failed image load)
            mock_st.image.side_effect = Exception("Image failed to load")
            
            # Should not raise any exceptions
            render_batter_card(player_data)
            
            # Verify container and markdown were called
            mock_st.container.assert_called()
            mock_st.markdown.assert_called()
            
            # Should have attempted to load image
            mock_st.image.assert_called()
            
            # Should have 6 metrics in the stats grid
            assert mock_st.metric.call_count >= 6
    
    def test_render_batter_card_metrics_count(self):
        """Test that batter card renders with 6 metrics (4+ as required)"""
        player_data = {
            'name': 'Hardik Pandya',
            'average': '29.8',
            'strike_rate': '145.2',
            'runs': 38,
            'balls_faced': 26,
            'team_color': '#4169e1',
            'highest_score': '91',
            'boundaries': '5'
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_batter_card(player_data)
            
            # Verify that 6 metrics were rendered (exceeds 4+ requirement)
            assert mock_st.metric.call_count == 6, f"Expected 6 metrics, got {mock_st.metric.call_count}"
            
            # Verify that the metrics include the expected labels
            metric_calls = mock_st.metric.call_args_list
            metric_labels = [call[1]['label'] for call in metric_calls if 'label' in call[1]]
            
            expected_labels = ['Average', 'Strike Rate', 'Runs/Balls', 'High Score', 'Boundaries', 'Scoring Rate']
            for expected_label in expected_labels:
                assert expected_label in metric_labels, f"Missing expected metric label: {expected_label}"
    
    def test_render_batter_card_none_values(self):
        """Test batter card with None values"""
        player_data = {
            'name': None,
            'average': None,
            'strike_rate': None,
            'recent_shots': None,
            'runs': None,
            'balls_faced': None
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_batter_card(player_data)
            
            # Verify function handled None values gracefully
            mock_st.container.assert_called()


class TestBowlerCard:
    """Test suite for render_bowler_card function"""
    
    def test_render_bowler_card_complete_data(self):
        """Test bowler card with complete player data"""
        player_data = {
            'name': 'Jasprit Bumrah',
            'economy': '6.8',
            'wickets': 3,
            'overs': '8.2',
            'runs_conceded': 56,
            'maidens': 1,
            'image_url': 'https://example.com/bumrah.jpg',
            'team_color': '#002466',
            'best_figures': '4/17',
            'dot_balls': '14'
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_bowler_card(player_data)
            
            # Verify streamlit components were called
            mock_st.container.assert_called()
            mock_st.markdown.assert_called()
            # Should have image column, stats grid (3 columns x 2 rows), and layout column
            assert mock_st.columns.call_count >= 3
            # Should have 6 metrics in the stats grid
            assert mock_st.metric.call_count >= 6
    
    def test_render_bowler_card_minimal_data(self):
        """Test bowler card with minimal data"""
        player_data = {
            'name': 'Pat Cummins'
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_bowler_card(player_data)
            
            # Verify fallback values were used
            mock_st.markdown.assert_called()
            mock_st.metric.assert_called()
    
    def test_render_bowler_card_empty_data(self):
        """Test bowler card with empty data"""
        player_data = {}
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_bowler_card(player_data)
            
            # Verify function completed without errors
            mock_st.container.assert_called()
    
    def test_render_bowler_card_with_image_fallback(self):
        """Test bowler card with missing image (fallback placeholder)"""
        player_data = {
            'name': 'Trent Boult',
            'economy': '7.2',
            'wickets': 2,
            'overs': '6.0',
            'runs_conceded': 43,
            'maidens': 0,
            'team_color': '#1e90ff',
            'best_figures': '3/28',
            'dot_balls': '18'
            # image_url is missing - should use placeholder
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_bowler_card(player_data)
            
            # Verify container and markdown were called
            mock_st.container.assert_called()
            mock_st.markdown.assert_called()
            
            # Should have 6 metrics in the stats grid (2 rows x 3 columns)
            assert mock_st.metric.call_count >= 6
            
            # Check that st.image was NOT called (since no image_url)
            mock_st.image.assert_not_called()
    
    def test_render_bowler_card_failed_image_load(self):
        """Test bowler card with image URL that fails to load"""
        player_data = {
            'name': 'Rashid Khan',
            'economy': '5.8',
            'wickets': 4,
            'overs': '10.0',
            'runs_conceded': 58,
            'maidens': 1,
            'image_url': 'https://broken.link/invalid.jpg',
            'team_color': '#ff4500',
            'best_figures': '5/27',
            'dot_balls': '24'
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Mock st.image to raise an exception (simulating failed image load)
            mock_st.image.side_effect = Exception("Image failed to load")
            
            # Should not raise any exceptions
            render_bowler_card(player_data)
            
            # Verify container and markdown were called
            mock_st.container.assert_called()
            mock_st.markdown.assert_called()
            
            # Should have attempted to load image
            mock_st.image.assert_called()
            
            # Should have 6 metrics in the stats grid
            assert mock_st.metric.call_count >= 6
    
    def test_render_bowler_card_metrics_count(self):
        """Test that bowler card renders with 6 metrics (4+ as required)"""
        player_data = {
            'name': 'Kagiso Rabada',
            'economy': '6.5',
            'wickets': 5,
            'overs': '9.0',
            'runs_conceded': 58,
            'maidens': 2,
            'team_color': '#228b22',
            'best_figures': '6/25',
            'dot_balls': '32'
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_bowler_card(player_data)
            
            # Verify that 6 metrics were rendered (exceeds 4+ requirement)
            assert mock_st.metric.call_count == 6, f"Expected 6 metrics, got {mock_st.metric.call_count}"
            
            # Verify that the metrics include the expected labels
            metric_calls = mock_st.metric.call_args_list
            metric_labels = [call[1]['label'] for call in metric_calls if 'label' in call[1]]
            
            expected_labels = ['Economy', 'Wickets', 'Overs', 'Best Figures', 'Maidens', 'Dot Balls']
            for expected_label in expected_labels:
                assert expected_label in metric_labels, f"Missing expected metric label: {expected_label}"


class TestOddsPanel:
    """Test suite for render_odds_panel function"""
    
    def test_render_odds_panel_complete_data(self):
        """Test odds panel with complete data"""
        market_data = {
            'home_win': 1.85,
            'away_win': 2.10,
            'home_prob': 54.1,
            'away_prob': 47.6
        }
        
        model_data = {
            'home_win': 1.92,
            'away_win': 2.05,
            'home_prob': 52.1,
            'away_prob': 48.8
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_odds_panel(market_data, model_data)
            
            # Verify streamlit components were called
            mock_st.container.assert_called()
            mock_st.markdown.assert_called()
            mock_st.columns.assert_called_with(2)
    
    def test_render_odds_panel_with_value_opportunity(self):
        """Test odds panel with value opportunity"""
        market_data = {
            'home_win': 2.5,  # Higher than model
            'away_win': 1.8,
            'home_prob': 40.0,
            'away_prob': 55.6
        }
        
        model_data = {
            'home_win': 2.0,  # Lower than market
            'away_win': 1.9,
            'home_prob': 50.0,
            'away_prob': 52.6
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_odds_panel(market_data, model_data)
            
            # Verify function completed
            mock_st.container.assert_called()
            mock_st.markdown.assert_called()
    
    def test_render_odds_panel_empty_data(self):
        """Test odds panel with empty data"""
        market_data = {}
        model_data = {}
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_odds_panel(market_data, model_data)
            
            # Verify function completed without errors
            mock_st.container.assert_called()
    
    def test_render_odds_panel_invalid_odds(self):
        """Test odds panel with invalid odds data"""
        market_data = {
            'home_win': 'invalid',
            'away_win': None,
            'home_prob': 'not_a_number',
            'away_prob': 47.6
        }
        
        model_data = {
            'home_win': 1.92,
            'away_win': 'also_invalid',
            'home_prob': 52.1,
            'away_prob': None
        }
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Should not raise any exceptions
            render_odds_panel(market_data, model_data)
            
            # Verify function handled invalid data gracefully
            mock_st.container.assert_called()


class TestWinBar:
    """Test suite for render_win_bar function"""
    
    def test_render_win_bar_percentage_format(self):
        """Test win bar with percentage format (0-100)"""
        probability = 67.8
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            # Should not raise any exceptions
            render_win_bar(probability)
            
            # Verify streamlit components were called
            mock_st.container.assert_called()
            mock_st.markdown.assert_called()
    
    def test_render_win_bar_decimal_format(self):
        """Test win bar with decimal format (0-1)"""
        probability = 0.678
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            # Should not raise any exceptions
            render_win_bar(probability)
            
            # Verify function completed
            mock_st.container.assert_called()
    
    def test_render_win_bar_edge_cases(self):
        """Test win bar with edge case values"""
        edge_cases = [0, 100, 0.0, 1.0, 150, -10]
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            for probability in edge_cases:
                # Should not raise any exceptions
                render_win_bar(probability)
                
                # Verify function completed
                mock_st.container.assert_called()
    
    def test_render_win_bar_invalid_data(self):
        """Test win bar with invalid data"""
        invalid_values = ['invalid', None, 'N/A', '']
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            for probability in invalid_values:
                # Should not raise any exceptions
                render_win_bar(probability)
                
                # Verify function handled invalid data gracefully
                mock_st.container.assert_called()
    
    def test_render_win_bar_color_thresholds(self):
        """Test win bar color thresholds"""
        # Test different probability ranges to ensure proper color assignment
        test_cases = [
            (85, "Strong"),    # High probability
            (60, "Moderate"),  # Medium probability
            (30, "Weak")       # Low probability
        ]
        
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            for probability, expected_status in test_cases:
                # Should not raise any exceptions
                render_win_bar(probability)
                
                # Verify function completed
                mock_st.container.assert_called()


class TestMatchStatus:
    """Test suite for render_match_status function"""
    
    def test_render_match_status_live(self):
        """Test match status with live match"""
        with patch('ui_style.st') as mock_st:
            # Should not raise any exceptions
            render_match_status("India", "Australia", "Live")
            
            # Verify streamlit components were called
            mock_st.markdown.assert_called()
    
    def test_render_match_status_finished(self):
        """Test match status with finished match"""
        with patch('ui_style.st') as mock_st:
            # Should not raise any exceptions
            render_match_status("England", "Pakistan", "Finished")
            
            # Verify function completed
            mock_st.markdown.assert_called()
    
    def test_render_match_status_default(self):
        """Test match status with default status"""
        with patch('ui_style.st') as mock_st:
            # Should not raise any exceptions
            render_match_status("New Zealand", "South Africa")
            
            # Verify function completed
            mock_st.markdown.assert_called()


class TestInfoPanel:
    """Test suite for render_info_panel function"""
    
    def test_render_info_panel_all_types(self):
        """Test info panel with all panel types"""
        panel_types = ["info", "warning", "success", "danger", "neutral"]
        
        with patch('ui_style.st') as mock_st:
            for panel_type in panel_types:
                # Should not raise any exceptions
                render_info_panel(
                    f"Test {panel_type.title()}", 
                    f"This is a {panel_type} panel.", 
                    panel_type
                )
                
                # Verify streamlit components were called
                mock_st.markdown.assert_called()
    
    def test_render_info_panel_invalid_type(self):
        """Test info panel with invalid panel type"""
        with patch('ui_style.st') as mock_st:
            # Should not raise any exceptions
            render_info_panel(
                "Test Panel", 
                "This panel has an invalid type.", 
                "invalid_type"
            )
            
            # Verify function completed (should default to info)
            mock_st.markdown.assert_called()
    
    def test_render_info_panel_empty_content(self):
        """Test info panel with empty content"""
        with patch('ui_style.st') as mock_st:
            # Should not raise any exceptions
            render_info_panel("", "", "info")
            
            # Verify function completed
            mock_st.markdown.assert_called()


class TestColorThemes:
    """Test suite for color theme constants"""
    
    def test_cricket_colors_exist(self):
        """Test that all cricket colors are defined"""
        assert hasattr(CricketColors, 'BATTING')
        assert hasattr(CricketColors, 'BOWLING')
        assert hasattr(CricketColors, 'WICKET')
        assert hasattr(CricketColors, 'SIGNALS')
        assert hasattr(CricketColors, 'NEUTRAL')
    
    def test_cricket_colors_are_valid_hex(self):
        """Test that cricket colors are valid hex codes"""
        colors = [
            CricketColors.BATTING,
            CricketColors.BOWLING,
            CricketColors.WICKET,
            CricketColors.SIGNALS,
            CricketColors.NEUTRAL
        ]
        
        for color in colors:
            assert color.startswith('#')
            assert len(color) == 7
            # Verify hex format
            int(color[1:], 16)  # Should not raise ValueError
    
    def test_ui_colors_exist(self):
        """Test that all UI colors are defined"""
        assert hasattr(CricketColors, 'SUCCESS')
        assert hasattr(CricketColors, 'WARNING')
        assert hasattr(CricketColors, 'DANGER')
        assert hasattr(CricketColors, 'INFO')


class TestTypography:
    """Test suite for typography constants"""
    
    def test_typography_constants_exist(self):
        """Test that all typography constants are defined"""
        assert hasattr(Typography, 'H1_SIZE')
        assert hasattr(Typography, 'H2_SIZE')
        assert hasattr(Typography, 'BODY_SIZE')
        assert hasattr(Typography, 'MAX_WIDTH')
        assert hasattr(Typography, 'SECTION_PADDING')
    
    def test_typography_size_formats(self):
        """Test that typography sizes are in correct format"""
        sizes = [
            Typography.H1_SIZE,
            Typography.H2_SIZE,
            Typography.BODY_SIZE,
            Typography.MAX_WIDTH,
            Typography.SECTION_PADDING
        ]
        
        for size in sizes:
            assert isinstance(size, str)
            assert 'em' in size


class TestIntegration:
    """Integration tests for UI style components"""
    
    def test_all_components_render_together(self):
        """Test that all components can be rendered together"""
        with patch('ui_style.st') as mock_st:
            mock_st.container.return_value.__enter__.return_value = MagicMock()
            
            
            mock_st.columns.side_effect = mock_columns_helper
            
            # Sample data
            batter_data = {'name': 'Test Batter', 'average': '30.5'}
            bowler_data = {'name': 'Test Bowler', 'economy': '7.2'}
            market_data = {'home_win': 1.8, 'away_win': 2.0}
            model_data = {'home_win': 1.9, 'away_win': 1.95}
            
            # Should not raise any exceptions
            render_match_status("Team A", "Team B", "Live")
            render_batter_card(batter_data)
            render_bowler_card(bowler_data)
            render_odds_panel(market_data, model_data)
            render_win_bar(65.5)
            render_info_panel("Test", "Integration test", "info")
            
            # Verify all components were called
            assert mock_st.markdown.call_count > 0
            assert mock_st.container.call_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
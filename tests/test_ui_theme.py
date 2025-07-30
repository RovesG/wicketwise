# Purpose: Test ui_theme.py global style system and custom Streamlit theming
# Author: Phi1618 Cricket AI, Last Modified: 2025-01-17

import pytest
from unittest.mock import patch, MagicMock
from ui_theme import (
    ThemeColors, 
    ThemeTypography, 
    set_streamlit_theme, 
    get_card_style, 
    style_win_bar, 
    get_win_bar_style,
    get_odds_panel_style,
    get_match_status_style
)

class TestThemeColors:
    """Test ThemeColors class constants"""
    
    def test_cricket_colors_exist(self):
        """Test that cricket-specific colors are defined"""
        assert ThemeColors.BATTING == "#c8712d"
        assert ThemeColors.BOWLING == "#002466"
        assert ThemeColors.WICKET == "#660003"
        assert ThemeColors.SIGNALS == "#819f3d"
        assert ThemeColors.NEUTRAL == "#404041"
    
    def test_ui_colors_exist(self):
        """Test that UI colors are defined"""
        assert ThemeColors.PRIMARY_BUTTON == "#d38c55"
        assert ThemeColors.PRIMARY_BUTTON_HOVER == "#c8712d"
        assert ThemeColors.BACKGROUND == "#ffffff"
        assert ThemeColors.CARD_BACKGROUND == "#f8f9fa"
        assert ThemeColors.BORDER == "#e9ecef"
    
    def test_win_probability_colors_exist(self):
        """Test that win probability colors are defined"""
        assert ThemeColors.WIN_STRONG == "#28a745"
        assert ThemeColors.WIN_MODERATE == "#ffc107"
        assert ThemeColors.WIN_WEAK == "#dc3545"
    
    def test_color_format_valid(self):
        """Test that all colors are valid hex format"""
        import re
        hex_pattern = r'^#[0-9a-fA-F]{6}$'
        
        # Test all color attributes
        for attr_name in dir(ThemeColors):
            if not attr_name.startswith('_'):
                color_value = getattr(ThemeColors, attr_name)
                assert re.match(hex_pattern, color_value), f"Invalid color format for {attr_name}: {color_value}"

class TestThemeTypography:
    """Test ThemeTypography class constants"""
    
    def test_font_family_exists(self):
        """Test that font family is defined"""
        assert "system-ui" in ThemeTypography.FONT_FAMILY
        assert "Roboto" in ThemeTypography.FONT_FAMILY
        assert "Arial" in ThemeTypography.FONT_FAMILY
    
    def test_font_sizes_exist(self):
        """Test that font sizes are defined"""
        assert ThemeTypography.H1_SIZE == "1.75em"
        assert ThemeTypography.H2_SIZE == "1.5em"
        assert ThemeTypography.BODY_SIZE == "1em"
    
    def test_layout_constants_exist(self):
        """Test that layout constants are defined"""
        assert ThemeTypography.MAX_WIDTH == "65em"
        assert ThemeTypography.SECTION_PADDING == "2em"
        assert ThemeTypography.BUTTON_PADDING == "0.65em 0.75em"

class TestSetStreamlitTheme:
    """Test set_streamlit_theme function"""
    
    @patch('ui_theme.st.markdown')
    def test_set_streamlit_theme_called(self, mock_markdown):
        """Test that set_streamlit_theme calls st.markdown"""
        set_streamlit_theme()
        
        mock_markdown.assert_called_once()
        args, kwargs = mock_markdown.call_args
        assert kwargs.get('unsafe_allow_html') is True
    
    @patch('ui_theme.st.markdown')
    def test_theme_css_contains_expected_styles(self, mock_markdown):
        """Test that generated CSS contains expected style elements"""
        set_streamlit_theme()
        
        args, kwargs = mock_markdown.call_args
        css_content = args[0]
        
        # Check for key CSS elements
        assert "font-family" in css_content
        assert ThemeTypography.FONT_FAMILY in css_content
        assert ThemeColors.PRIMARY_BUTTON in css_content
        assert ThemeColors.PRIMARY_BUTTON_HOVER in css_content
        assert "button" in css_content
        assert "h1" in css_content
        assert "h2" in css_content
    
    @patch('ui_theme.st.markdown')
    def test_theme_css_includes_accessibility(self, mock_markdown):
        """Test that CSS includes accessibility features"""
        set_streamlit_theme()
        
        args, kwargs = mock_markdown.call_args
        css_content = args[0]
        
        # Check for accessibility features
        assert "focus-visible" in css_content
        assert "outline" in css_content
        assert "prefers-contrast" in css_content
    
    @patch('ui_theme.st.markdown')
    def test_theme_css_includes_animations(self, mock_markdown):
        """Test that CSS includes smooth transitions"""
        set_streamlit_theme()
        
        args, kwargs = mock_markdown.call_args
        css_content = args[0]
        
        assert "transition" in css_content
        assert "transform" in css_content
        assert "box-shadow" in css_content

class TestGetCardStyle:
    """Test get_card_style function"""
    
    def test_get_card_style_with_valid_color(self):
        """Test get_card_style with valid hex color"""
        color = "#c8712d"
        css = get_card_style(color)
        
        # Check that CSS is returned as string
        assert isinstance(css, str)
        assert "<style>" in css
        assert "</style>" in css
        
        # Check that color is included in CSS
        assert color in css
    
    def test_get_card_style_contains_expected_classes(self):
        """Test that CSS contains expected class names"""
        css = get_card_style("#c8712d")
        
        expected_classes = [
            "cricket-card",
            "cricket-card-header",
            "cricket-card-content",
            "cricket-card-stat",
            "cricket-card-grid",
            "cricket-card-metric"
        ]
        
        for class_name in expected_classes:
            assert class_name in css
    
    def test_get_card_style_includes_theme_colors(self):
        """Test that CSS includes theme colors"""
        css = get_card_style("#c8712d")
        
        assert ThemeColors.CARD_BACKGROUND in css
        assert ThemeColors.BORDER in css
        assert ThemeColors.NEUTRAL in css
        assert ThemeColors.BACKGROUND in css
    
    def test_get_card_style_includes_typography(self):
        """Test that CSS includes typography settings"""
        css = get_card_style("#c8712d")
        
        assert ThemeTypography.H2_SIZE in css
    
    def test_get_card_style_includes_responsive_design(self):
        """Test that CSS includes responsive design features"""
        css = get_card_style("#c8712d")
        
        assert "grid-template-columns" in css
        assert "auto-fit" in css
        assert "minmax" in css
    
    def test_get_card_style_includes_hover_effects(self):
        """Test that CSS includes hover effects"""
        css = get_card_style("#c8712d")
        
        assert ":hover" in css
        assert "transform" in css
        assert "box-shadow" in css

class TestStyleWinBar:
    """Test style_win_bar function"""
    
    def test_style_win_bar_strong_probability(self):
        """Test style_win_bar with strong probability (>= 0.7)"""
        color, width = style_win_bar(0.8)
        
        assert color == ThemeColors.WIN_STRONG
        assert width == "80.0%"
    
    def test_style_win_bar_moderate_probability(self):
        """Test style_win_bar with moderate probability (0.4-0.69)"""
        color, width = style_win_bar(0.5)
        
        assert color == ThemeColors.WIN_MODERATE
        assert width == "50.0%"
    
    def test_style_win_bar_weak_probability(self):
        """Test style_win_bar with weak probability (< 0.4)"""
        color, width = style_win_bar(0.2)
        
        assert color == ThemeColors.WIN_WEAK
        assert width == "20.0%"
    
    def test_style_win_bar_edge_cases(self):
        """Test style_win_bar with edge case probabilities"""
        # Test 0.0
        color, width = style_win_bar(0.0)
        assert color == ThemeColors.WIN_WEAK
        assert width == "0.0%"
        
        # Test 1.0
        color, width = style_win_bar(1.0)
        assert color == ThemeColors.WIN_STRONG
        assert width == "100.0%"
        
        # Test exactly 0.7
        color, width = style_win_bar(0.7)
        assert color == ThemeColors.WIN_STRONG
        assert width == "70.0%"
        
        # Test exactly 0.4
        color, width = style_win_bar(0.4)
        assert color == ThemeColors.WIN_MODERATE
        assert width == "40.0%"
    
    def test_style_win_bar_invalid_values(self):
        """Test style_win_bar with invalid probability values"""
        # Test negative value
        color, width = style_win_bar(-0.5)
        assert color == ThemeColors.WIN_WEAK
        assert width == "0.0%"
        
        # Test value > 1.0
        color, width = style_win_bar(1.5)
        assert color == ThemeColors.WIN_STRONG
        assert width == "100.0%"
    
    def test_style_win_bar_returns_tuple(self):
        """Test that style_win_bar returns a tuple"""
        result = style_win_bar(0.6)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        color, width = result
        assert isinstance(color, str)
        assert isinstance(width, str)
        assert color.startswith("#")
        assert width.endswith("%")
    
    def test_style_win_bar_precision(self):
        """Test that style_win_bar returns precise width values"""
        color, width = style_win_bar(0.123)
        assert width == "12.3%"
        
        color, width = style_win_bar(0.999)
        assert width == "99.9%"

class TestGetWinBarStyle:
    """Test get_win_bar_style function"""
    
    def test_get_win_bar_style_returns_css(self):
        """Test that get_win_bar_style returns CSS string"""
        css = get_win_bar_style(0.7)
        
        assert isinstance(css, str)
        assert "<style>" in css
        assert "</style>" in css
    
    def test_get_win_bar_style_includes_expected_classes(self):
        """Test that CSS contains expected class names"""
        css = get_win_bar_style(0.7)
        
        expected_classes = [
            "win-bar-container",
            "win-bar-fill",
            "win-bar-text",
            "win-bar-label"
        ]
        
        for class_name in expected_classes:
            assert class_name in css
    
    def test_get_win_bar_style_includes_probability_values(self):
        """Test that CSS includes probability-based values"""
        css = get_win_bar_style(0.8)
        
        # Should include width percentage
        assert "80.0%" in css
        # Should include strong color
        assert ThemeColors.WIN_STRONG in css
    
    def test_get_win_bar_style_includes_animations(self):
        """Test that CSS includes smooth transitions"""
        css = get_win_bar_style(0.6)
        
        assert "transition" in css
        assert "ease-in-out" in css

class TestGetOddsPanelStyle:
    """Test get_odds_panel_style function"""
    
    def test_get_odds_panel_style_returns_css(self):
        """Test that get_odds_panel_style returns CSS string"""
        css = get_odds_panel_style()
        
        assert isinstance(css, str)
        assert "<style>" in css
        assert "</style>" in css
    
    def test_get_odds_panel_style_includes_expected_classes(self):
        """Test that CSS contains expected class names"""
        css = get_odds_panel_style()
        
        expected_classes = [
            "odds-panel",
            "odds-header",
            "odds-comparison",
            "odds-item",
            "odds-value-opportunity",
            "odds-value-caution",
            "odds-value-danger"
        ]
        
        for class_name in expected_classes:
            assert class_name in css
    
    def test_get_odds_panel_style_includes_theme_colors(self):
        """Test that CSS includes theme colors"""
        css = get_odds_panel_style()
        
        assert ThemeColors.CARD_BACKGROUND in css
        assert ThemeColors.BORDER in css
        assert ThemeColors.SIGNALS in css
        assert ThemeColors.WIN_STRONG in css
        assert ThemeColors.WIN_MODERATE in css
        assert ThemeColors.WIN_WEAK in css
    
    def test_get_odds_panel_style_includes_grid_layout(self):
        """Test that CSS includes grid layout"""
        css = get_odds_panel_style()
        
        assert "grid-template-columns" in css
        assert "1fr 1fr" in css

class TestGetMatchStatusStyle:
    """Test get_match_status_style function"""
    
    def test_get_match_status_style_live(self):
        """Test match status style for live matches"""
        css = get_match_status_style("live")
        
        assert ThemeColors.WIN_STRONG in css
        assert "animation: pulse" in css
    
    def test_get_match_status_style_completed(self):
        """Test match status style for completed matches"""
        css = get_match_status_style("completed")
        
        assert ThemeColors.NEUTRAL in css
    
    def test_get_match_status_style_upcoming(self):
        """Test match status style for upcoming matches"""
        css = get_match_status_style("upcoming")
        
        assert ThemeColors.WIN_MODERATE in css
    
    def test_get_match_status_style_interrupted(self):
        """Test match status style for interrupted matches"""
        css = get_match_status_style("interrupted")
        
        assert ThemeColors.WIN_WEAK in css
    
    def test_get_match_status_style_case_insensitive(self):
        """Test that status matching is case insensitive"""
        css_lower = get_match_status_style("live")
        css_upper = get_match_status_style("LIVE")
        css_mixed = get_match_status_style("Live")
        
        # All should contain the same color
        assert ThemeColors.WIN_STRONG in css_lower
        assert ThemeColors.WIN_STRONG in css_upper
        assert ThemeColors.WIN_STRONG in css_mixed
    
    def test_get_match_status_style_unknown_status(self):
        """Test match status style for unknown status"""
        css = get_match_status_style("unknown")
        
        assert ThemeColors.NEUTRAL in css
    
    def test_get_match_status_style_includes_expected_classes(self):
        """Test that CSS contains expected class names"""
        css = get_match_status_style("live")
        
        expected_classes = [
            "match-status",
            "match-status-live"
        ]
        
        for class_name in expected_classes:
            assert class_name in css
    
    def test_get_match_status_style_includes_animation(self):
        """Test that CSS includes keyframe animation"""
        css = get_match_status_style("live")
        
        assert "@keyframes pulse" in css
        assert "opacity" in css

class TestIntegration:
    """Integration tests for ui_theme module"""
    
    def test_all_functions_return_strings(self):
        """Test that all style functions return strings"""
        # Test functions that return CSS strings
        assert isinstance(get_card_style("#c8712d"), str)
        assert isinstance(get_win_bar_style(0.7), str)
        assert isinstance(get_odds_panel_style(), str)
        assert isinstance(get_match_status_style("live"), str)
    
    def test_style_win_bar_integration(self):
        """Test style_win_bar integration with other functions"""
        color, width = style_win_bar(0.8)
        
        # Should be compatible with win bar style
        css = get_win_bar_style(0.8)
        assert color in css
        assert width in css
    
    def test_color_consistency(self):
        """Test that colors are consistent across all functions"""
        # Test that theme colors are used consistently
        card_css = get_card_style(ThemeColors.BATTING)
        win_bar_css = get_win_bar_style(0.8)
        odds_css = get_odds_panel_style()
        status_css = get_match_status_style("live")
        
        # All should use consistent border color
        assert ThemeColors.BORDER in card_css
        assert ThemeColors.BORDER in win_bar_css
        assert ThemeColors.BORDER in odds_css
    
    def test_typography_consistency(self):
        """Test that typography is consistent across functions"""
        card_css = get_card_style("#c8712d")
        
        # Should use consistent typography
        assert ThemeTypography.H2_SIZE in card_css
    
    @patch('ui_theme.st.markdown')
    def test_theme_integration_with_streamlit(self, mock_markdown):
        """Test that theme integrates properly with Streamlit"""
        set_streamlit_theme()
        
        # Should call st.markdown with unsafe_allow_html=True
        mock_markdown.assert_called_once()
        args, kwargs = mock_markdown.call_args
        assert kwargs.get('unsafe_allow_html') is True
        
        # CSS should be properly formatted
        css_content = args[0]
        assert css_content.startswith("\n    <style>")
        assert css_content.endswith("\n    </style>\n    ")

if __name__ == "__main__":
    pytest.main([__file__]) 
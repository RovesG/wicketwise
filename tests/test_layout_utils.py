# Purpose: Test suite for responsive layout utilities
# Author: Claude, Last Modified: 2025-01-17

import pytest
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
from layout_utils import (
    LayoutMode, 
    BreakpointConfig, 
    detect_layout_mode,
    inject_responsive_css,
    render_wide_desktop_layout,
    render_medium_tablet_layout,
    render_mobile_layout,
    render_responsive_dashboard,
    create_component_wrapper,
    get_layout_recommendations
)


class TestLayoutMode:
    """Test LayoutMode enum"""
    
    def test_layout_mode_values(self):
        """Test layout mode enum values"""
        assert LayoutMode.WIDE_DESKTOP.value == "wide_desktop"
        assert LayoutMode.MEDIUM_TABLET.value == "medium_tablet"
        assert LayoutMode.MOBILE.value == "mobile"


class TestBreakpointConfig:
    """Test BreakpointConfig class"""
    
    def test_breakpoint_values(self):
        """Test breakpoint configuration values"""
        assert BreakpointConfig.MOBILE_MAX_WIDTH == 768
        assert BreakpointConfig.TABLET_MAX_WIDTH == 1024
        assert BreakpointConfig.DESKTOP_MIN_WIDTH == 1025
    
    def test_css_breakpoints_generation(self):
        """Test CSS breakpoints generation"""
        css = BreakpointConfig.get_css_breakpoints()
        
        # Check that all breakpoints are included
        assert "max-width: 768px" in css
        assert "min-width: 769px" in css
        assert "max-width: 1024px" in css
        assert "min-width: 1025px" in css
        
        # Check that layout classes are defined
        assert ".mobile-layout" in css
        assert ".tablet-layout" in css
        assert ".desktop-layout" in css
        assert ".responsive-container" in css
        
        # Check responsive utilities
        assert ".collapsible-chat" in css
        assert ".layout-section" in css
        assert ".side-by-side" in css


class TestLayoutDetection:
    """Test layout mode detection logic"""
    
    def test_detect_layout_mode_mobile(self):
        """Test mobile layout detection"""
        assert detect_layout_mode(320) == LayoutMode.MOBILE
        assert detect_layout_mode(768) == LayoutMode.MOBILE
        assert detect_layout_mode(600) == LayoutMode.MOBILE
    
    def test_detect_layout_mode_tablet(self):
        """Test tablet layout detection"""
        assert detect_layout_mode(769) == LayoutMode.MEDIUM_TABLET
        assert detect_layout_mode(1024) == LayoutMode.MEDIUM_TABLET
        assert detect_layout_mode(900) == LayoutMode.MEDIUM_TABLET
    
    def test_detect_layout_mode_desktop(self):
        """Test desktop layout detection"""
        assert detect_layout_mode(1025) == LayoutMode.WIDE_DESKTOP
        assert detect_layout_mode(1920) == LayoutMode.WIDE_DESKTOP
        assert detect_layout_mode(2560) == LayoutMode.WIDE_DESKTOP
    
    def test_detect_layout_mode_none(self):
        """Test layout detection with no width provided"""
        # Should default to tablet
        assert detect_layout_mode(None) == LayoutMode.MEDIUM_TABLET
        assert detect_layout_mode() == LayoutMode.MEDIUM_TABLET


class TestCSSInjection:
    """Test CSS injection functionality"""
    
    @patch('streamlit.markdown')
    def test_inject_responsive_css(self, mock_markdown):
        """Test CSS injection into Streamlit"""
        inject_responsive_css()
        
        # Check that st.markdown was called
        mock_markdown.assert_called_once()
        
        # Check that the call includes CSS style tag
        call_args = mock_markdown.call_args
        assert '<style>' in call_args[0][0]
        assert '</style>' in call_args[0][0]
        assert call_args[1]['unsafe_allow_html'] is True


class TestLayoutRendering:
    """Test layout rendering functions"""
    
    def setup_method(self):
        """Setup mock components for testing"""
        self.mock_video = Mock()
        self.mock_player_cards = [Mock(), Mock()]
        self.mock_win_probability = Mock()
        self.mock_chat = Mock()
        self.mock_additional = [Mock()]
    
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    def test_render_wide_desktop_layout(self, mock_subheader, mock_columns, mock_markdown):
        """Test wide desktop layout rendering"""
        # Mock columns to return mock column objects with context manager support
        mock_col1, mock_col2 = Mock(), Mock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        
        # Mock columns to return list of context manager objects
        mock_columns.return_value = [mock_col1, mock_col2]
        
        render_wide_desktop_layout(
            self.mock_video, self.mock_player_cards, 
            self.mock_win_probability, self.mock_chat,
            self.mock_additional
        )
        
        # Check that components were called
        self.mock_video.assert_called_once()
        self.mock_win_probability.assert_called_once()
        self.mock_chat.assert_called_once()
        for card in self.mock_player_cards:
            card.assert_called_once()
        for component in self.mock_additional:
            component.assert_called_once()
        
        # Check that markdown was called for layout classes
        markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
        assert any('desktop-layout' in call for call in markdown_calls)
        assert any('layout-section' in call for call in markdown_calls)
    
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    @patch('streamlit.expander')
    def test_render_medium_tablet_layout(self, mock_expander, mock_subheader, mock_columns, mock_markdown):
        """Test medium/tablet layout rendering"""
        # Mock expander context manager
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        # Mock columns to return mock column objects with context manager support
        mock_col1, mock_col2 = Mock(), Mock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        
        # Mock columns to return list of context manager objects
        mock_columns.return_value = [mock_col1, mock_col2]
        
        render_medium_tablet_layout(
            self.mock_video, self.mock_player_cards, 
            self.mock_win_probability, self.mock_chat,
            self.mock_additional
        )
        
        # Check that components were called
        self.mock_video.assert_called_once()
        self.mock_win_probability.assert_called_once()
        self.mock_chat.assert_called_once()
        for card in self.mock_player_cards:
            card.assert_called_once()
        
        # Check that expanders were created
        assert mock_expander.call_count >= 1
        
        # Check tablet layout class
        markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
        assert any('tablet-layout' in call for call in markdown_calls)
    
    @patch('streamlit.markdown')
    @patch('streamlit.subheader')
    @patch('streamlit.expander')
    def test_render_mobile_layout(self, mock_expander, mock_subheader, mock_markdown):
        """Test mobile layout rendering"""
        # Mock expander context manager
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        render_mobile_layout(
            self.mock_video, self.mock_player_cards, 
            self.mock_win_probability, self.mock_chat,
            self.mock_additional
        )
        
        # Check that components were called
        self.mock_video.assert_called_once()
        self.mock_win_probability.assert_called_once()
        self.mock_chat.assert_called_once()
        for card in self.mock_player_cards:
            card.assert_called_once()
        
        # Check that expanders were created (for collapsible chat)
        assert mock_expander.call_count >= 1
        
        # Check mobile layout class
        markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
        assert any('mobile-layout' in call for call in markdown_calls)


class TestResponsiveDashboard:
    """Test the main responsive dashboard function"""
    
    def setup_method(self):
        """Setup mock components for testing"""
        self.mock_video = Mock()
        self.mock_player_cards = [Mock(), Mock()]
        self.mock_win_probability = Mock()
        self.mock_chat = Mock()
        self.mock_additional = [Mock()]
    
    @patch('layout_utils.render_wide_desktop_layout')
    @patch('layout_utils.inject_responsive_css')
    @patch('streamlit.markdown')
    @patch('streamlit.checkbox')
    def test_render_responsive_dashboard_desktop(self, mock_checkbox, mock_markdown, mock_css, mock_desktop_layout):
        """Test responsive dashboard renders desktop layout"""
        mock_checkbox.return_value = False
        
        render_responsive_dashboard(
            self.mock_video, self.mock_player_cards, 
            self.mock_win_probability, self.mock_chat,
            layout_mode=LayoutMode.WIDE_DESKTOP
        )
        
        # Check that CSS was injected
        mock_css.assert_called_once()
        
        # Check that desktop layout was called
        mock_desktop_layout.assert_called_once_with(
            self.mock_video, self.mock_player_cards, 
            self.mock_win_probability, self.mock_chat, None
        )
    
    @patch('layout_utils.render_medium_tablet_layout')
    @patch('layout_utils.inject_responsive_css')
    @patch('streamlit.markdown')
    @patch('streamlit.checkbox')
    def test_render_responsive_dashboard_tablet(self, mock_checkbox, mock_markdown, mock_css, mock_tablet_layout):
        """Test responsive dashboard renders tablet layout"""
        mock_checkbox.return_value = False
        
        render_responsive_dashboard(
            self.mock_video, self.mock_player_cards, 
            self.mock_win_probability, self.mock_chat,
            layout_mode=LayoutMode.MEDIUM_TABLET
        )
        
        # Check that CSS was injected
        mock_css.assert_called_once()
        
        # Check that tablet layout was called
        mock_tablet_layout.assert_called_once_with(
            self.mock_video, self.mock_player_cards, 
            self.mock_win_probability, self.mock_chat, None
        )
    
    @patch('layout_utils.render_mobile_layout')
    @patch('layout_utils.inject_responsive_css')
    @patch('streamlit.markdown')
    @patch('streamlit.checkbox')
    def test_render_responsive_dashboard_mobile(self, mock_checkbox, mock_markdown, mock_css, mock_mobile_layout):
        """Test responsive dashboard renders mobile layout"""
        mock_checkbox.return_value = False
        
        render_responsive_dashboard(
            self.mock_video, self.mock_player_cards, 
            self.mock_win_probability, self.mock_chat,
            layout_mode=LayoutMode.MOBILE
        )
        
        # Check that CSS was injected
        mock_css.assert_called_once()
        
        # Check that mobile layout was called
        mock_mobile_layout.assert_called_once_with(
            self.mock_video, self.mock_player_cards, 
            self.mock_win_probability, self.mock_chat, None
        )
    
    @patch('layout_utils.detect_layout_mode')
    @patch('layout_utils.render_medium_tablet_layout')
    @patch('layout_utils.inject_responsive_css')
    @patch('streamlit.markdown')
    @patch('streamlit.checkbox')
    def test_render_responsive_dashboard_auto_detect(self, mock_checkbox, mock_markdown, mock_css, mock_tablet_layout, mock_detect):
        """Test responsive dashboard auto-detects layout mode"""
        mock_checkbox.return_value = False
        mock_detect.return_value = LayoutMode.MEDIUM_TABLET
        
        render_responsive_dashboard(
            self.mock_video, self.mock_player_cards, 
            self.mock_win_probability, self.mock_chat,
            container_width=900
        )
        
        # Check that detection was called
        mock_detect.assert_called_once_with(900)
        
        # Check that tablet layout was called
        mock_tablet_layout.assert_called_once()
    
    @patch('layout_utils.render_medium_tablet_layout')
    @patch('layout_utils.inject_responsive_css')
    @patch('streamlit.markdown')
    @patch('streamlit.checkbox')
    @patch('streamlit.info')
    def test_render_responsive_dashboard_debug_info(self, mock_info, mock_checkbox, mock_markdown, mock_css, mock_tablet_layout):
        """Test responsive dashboard shows debug info when enabled"""
        mock_checkbox.return_value = True
        
        render_responsive_dashboard(
            self.mock_video, self.mock_player_cards, 
            self.mock_win_probability, self.mock_chat,
            layout_mode=LayoutMode.MEDIUM_TABLET,
            container_width=900
        )
        
        # Check that debug info was shown
        assert mock_info.call_count >= 1
        info_calls = [call[0][0] for call in mock_info.call_args_list]
        assert any('medium_tablet' in call for call in info_calls)
        assert any('900' in call for call in info_calls)


class TestComponentWrapper:
    """Test component wrapper functionality"""
    
    def test_create_component_wrapper(self):
        """Test creating component wrapper with arguments"""
        mock_component = Mock()
        test_arg = "test_value"
        test_kwarg = "test_kwarg"
        
        wrapper = create_component_wrapper(mock_component, arg=test_arg, kwarg=test_kwarg)
        
        # Call the wrapper
        wrapper()
        
        # Check that the original component was called with the right arguments
        mock_component.assert_called_once_with(arg=test_arg, kwarg=test_kwarg)
    
    def test_create_component_wrapper_no_args(self):
        """Test creating component wrapper without arguments"""
        mock_component = Mock()
        
        wrapper = create_component_wrapper(mock_component)
        
        # Call the wrapper
        wrapper()
        
        # Check that the original component was called without arguments
        mock_component.assert_called_once_with()


class TestLayoutRecommendations:
    """Test layout recommendations functionality"""
    
    def test_get_layout_recommendations_desktop(self):
        """Test getting recommendations for desktop layout"""
        recommendations = get_layout_recommendations(LayoutMode.WIDE_DESKTOP)
        
        assert recommendations["max_columns"] == 4
        assert recommendations["card_width"] == "auto"
        assert recommendations["sidebar_visible"] is True
        assert recommendations["video_aspect_ratio"] == "16:9"
        assert recommendations["chat_height"] == "400px"
    
    def test_get_layout_recommendations_tablet(self):
        """Test getting recommendations for tablet layout"""
        recommendations = get_layout_recommendations(LayoutMode.MEDIUM_TABLET)
        
        assert recommendations["max_columns"] == 2
        assert recommendations["card_width"] == "300px"
        assert recommendations["sidebar_visible"] is True
        assert recommendations["video_aspect_ratio"] == "16:9"
        assert recommendations["chat_height"] == "300px"
    
    def test_get_layout_recommendations_mobile(self):
        """Test getting recommendations for mobile layout"""
        recommendations = get_layout_recommendations(LayoutMode.MOBILE)
        
        assert recommendations["max_columns"] == 1
        assert recommendations["card_width"] == "100%"
        assert recommendations["sidebar_visible"] is False
        assert recommendations["video_aspect_ratio"] == "16:9"
        assert recommendations["chat_height"] == "250px"
    
    def test_get_layout_recommendations_invalid(self):
        """Test getting recommendations for invalid layout mode"""
        # Create a mock invalid enum value
        invalid_mode = Mock()
        invalid_mode.value = "invalid"
        
        recommendations = get_layout_recommendations(invalid_mode)
        
        # Should return mobile recommendations as fallback
        assert recommendations["max_columns"] == 1
        assert recommendations["card_width"] == "100%"
        assert recommendations["sidebar_visible"] is False


class TestLayoutIntegration:
    """Test layout integration scenarios"""
    
    def test_layout_mode_progression(self):
        """Test that layout modes change correctly with screen size"""
        # Test progression from mobile to desktop
        widths_and_modes = [
            (320, LayoutMode.MOBILE),
            (768, LayoutMode.MOBILE),
            (769, LayoutMode.MEDIUM_TABLET),
            (1024, LayoutMode.MEDIUM_TABLET),
            (1025, LayoutMode.WIDE_DESKTOP),
            (1920, LayoutMode.WIDE_DESKTOP)
        ]
        
        for width, expected_mode in widths_and_modes:
            assert detect_layout_mode(width) == expected_mode
    
    def test_breakpoint_boundaries(self):
        """Test layout detection at exact breakpoint boundaries"""
        # Test exact boundaries
        assert detect_layout_mode(768) == LayoutMode.MOBILE
        assert detect_layout_mode(769) == LayoutMode.MEDIUM_TABLET
        assert detect_layout_mode(1024) == LayoutMode.MEDIUM_TABLET
        assert detect_layout_mode(1025) == LayoutMode.WIDE_DESKTOP
    
    @patch('layout_utils.render_wide_desktop_layout')
    @patch('layout_utils.render_medium_tablet_layout')
    @patch('layout_utils.render_mobile_layout')
    @patch('layout_utils.inject_responsive_css')
    @patch('streamlit.markdown')
    @patch('streamlit.checkbox')
    def test_layout_routing(self, mock_checkbox, mock_markdown, mock_css, mock_mobile, mock_tablet, mock_desktop):
        """Test that correct layout function is called for each mode"""
        mock_checkbox.return_value = False
        
        # Setup mock components
        mock_video = Mock()
        mock_cards = [Mock()]
        mock_win_prob = Mock()
        mock_chat = Mock()
        
        # Test desktop routing
        render_responsive_dashboard(
            mock_video, mock_cards, mock_win_prob, mock_chat,
            layout_mode=LayoutMode.WIDE_DESKTOP
        )
        mock_desktop.assert_called_once()
        
        # Test tablet routing
        render_responsive_dashboard(
            mock_video, mock_cards, mock_win_prob, mock_chat,
            layout_mode=LayoutMode.MEDIUM_TABLET
        )
        mock_tablet.assert_called_once()
        
        # Test mobile routing
        render_responsive_dashboard(
            mock_video, mock_cards, mock_win_prob, mock_chat,
            layout_mode=LayoutMode.MOBILE
        )
        mock_mobile.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__]) 
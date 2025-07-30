# Purpose: Test file for UI launcher structure and functionality
# Author: WicketWise Team, Last Modified: 2024-12-19

"""
Test suite for ui_launcher.py

Tests the basic structure and functionality of the Streamlit UI launcher:
- Verifies that the file can be imported without errors
- Checks that the expected UI components are present
- Validates the tab structure and content
- Confirms the version footer is displayed
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_ui_launcher_imports():
    """Test that the UI launcher file can be imported without errors."""
    try:
        import ui_launcher
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import ui_launcher: {e}")

def test_ui_launcher_structure():
    """Test that the UI launcher has the expected structure."""
    # Mock all streamlit functions
    with patch('streamlit.set_page_config') as mock_config, \
         patch('streamlit.title') as mock_title, \
         patch('streamlit.tabs') as mock_tabs, \
         patch('streamlit.header') as mock_header, \
         patch('streamlit.write') as mock_write, \
         patch('streamlit.markdown') as mock_markdown:
        
        # Mock tabs return value (simulate context managers)
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tab3 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2, mock_tab3]
        
        # Mock the context managers
        mock_tab1.__enter__ = MagicMock(return_value=mock_tab1)
        mock_tab1.__exit__ = MagicMock(return_value=None)
        mock_tab2.__enter__ = MagicMock(return_value=mock_tab2)
        mock_tab2.__exit__ = MagicMock(return_value=None)
        mock_tab3.__enter__ = MagicMock(return_value=mock_tab3)
        mock_tab3.__exit__ = MagicMock(return_value=None)
        
        # Clear any existing import of ui_launcher
        import sys
        if 'ui_launcher' in sys.modules:
            del sys.modules['ui_launcher']
        
        # Import and run the app
        import ui_launcher
        
        # Verify page config was called with correct parameters
        mock_config.assert_called_once_with(
            page_title="WicketWise Cricket AI",
            page_icon="🏏",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Verify title was called
        mock_title.assert_called_once_with("🏏 WicketWise Cricket AI")
        
        # Verify tabs were created with correct names
        mock_tabs.assert_called_once_with(["Live Match Dashboard", "Simulator Mode", "Admin Panel"])
        
        # Verify headers were called for each tab
        expected_headers = ["Live Match Dashboard", "Simulator Mode", "Admin Panel"]
        header_calls = [call.args[0] for call in mock_header.call_args_list]
        assert header_calls == expected_headers
        
        # Verify footer markdown was called
        markdown_calls = [call.args[0] for call in mock_markdown.call_args_list]
        assert "---" in markdown_calls
        assert "**WicketWise version 0.1**" in markdown_calls

def test_ui_launcher_placeholder_text():
    """Test that placeholder text is present for each tab."""
    with patch('streamlit.set_page_config'), \
         patch('streamlit.title'), \
         patch('streamlit.tabs') as mock_tabs, \
         patch('streamlit.header'), \
         patch('streamlit.write') as mock_write, \
         patch('streamlit.markdown'):
        
        # Mock tabs return value with context managers
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tab3 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2, mock_tab3]
        
        # Mock the context managers
        mock_tab1.__enter__ = MagicMock(return_value=mock_tab1)
        mock_tab1.__exit__ = MagicMock(return_value=None)
        mock_tab2.__enter__ = MagicMock(return_value=mock_tab2)
        mock_tab2.__exit__ = MagicMock(return_value=None)
        mock_tab3.__enter__ = MagicMock(return_value=mock_tab3)
        mock_tab3.__exit__ = MagicMock(return_value=None)
        
        # Clear any existing import of ui_launcher
        import sys
        if 'ui_launcher' in sys.modules:
            del sys.modules['ui_launcher']
        
        # Import and run the app
        import ui_launcher
        
        # Verify placeholder text was written for each tab
        expected_placeholders = [
            "Real-time cricket match analysis and predictions will appear here.",
            "Cricket match simulation and scenario testing will appear here.",
            "System administration and configuration options will appear here."
        ]
        write_calls = [call.args[0] for call in mock_write.call_args_list]
        assert write_calls == expected_placeholders

def test_ui_launcher_tab_structure():
    """Test that each tab has the correct structure."""
    with patch('streamlit.set_page_config'), \
         patch('streamlit.title'), \
         patch('streamlit.tabs') as mock_tabs, \
         patch('streamlit.header') as mock_header, \
         patch('streamlit.write') as mock_write, \
         patch('streamlit.markdown'):
        
        # Mock tabs return value with context managers
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tab3 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2, mock_tab3]
        
        # Mock the context managers
        mock_tab1.__enter__ = MagicMock(return_value=mock_tab1)
        mock_tab1.__exit__ = MagicMock(return_value=None)
        mock_tab2.__enter__ = MagicMock(return_value=mock_tab2)
        mock_tab2.__exit__ = MagicMock(return_value=None)
        mock_tab3.__enter__ = MagicMock(return_value=mock_tab3)
        mock_tab3.__exit__ = MagicMock(return_value=None)
        
        # Clear any existing import of ui_launcher
        import sys
        if 'ui_launcher' in sys.modules:
            del sys.modules['ui_launcher']
        
        # Import and run the app
        import ui_launcher
        
        # Verify each tab context manager was used
        mock_tab1.__enter__.assert_called_once()
        mock_tab1.__exit__.assert_called_once()
        mock_tab2.__enter__.assert_called_once()
        mock_tab2.__exit__.assert_called_once()
        mock_tab3.__enter__.assert_called_once()
        mock_tab3.__exit__.assert_called_once()
        
        # Verify the correct number of headers and write calls
        assert mock_header.call_count == 3
        assert mock_write.call_count == 3

def test_ui_launcher_version_footer():
    """Test that the version footer is correctly displayed."""
    with patch('streamlit.set_page_config'), \
         patch('streamlit.title'), \
         patch('streamlit.tabs') as mock_tabs, \
         patch('streamlit.header'), \
         patch('streamlit.write'), \
         patch('streamlit.markdown') as mock_markdown:
        
        # Mock tabs return value with context managers
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tab3 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2, mock_tab3]
        
        # Mock the context managers
        mock_tab1.__enter__ = MagicMock(return_value=mock_tab1)
        mock_tab1.__exit__ = MagicMock(return_value=None)
        mock_tab2.__enter__ = MagicMock(return_value=mock_tab2)
        mock_tab2.__exit__ = MagicMock(return_value=None)
        mock_tab3.__enter__ = MagicMock(return_value=mock_tab3)
        mock_tab3.__exit__ = MagicMock(return_value=None)
        
        # Clear any existing import of ui_launcher
        import sys
        if 'ui_launcher' in sys.modules:
            del sys.modules['ui_launcher']
        
        # Import and run the app
        import ui_launcher
        
        # Verify footer elements are present
        markdown_calls = [call.args[0] for call in mock_markdown.call_args_list]
        
        # Check that both separator and version are present
        assert "---" in markdown_calls
        assert "**WicketWise version 0.1**" in markdown_calls
        
        # Check that they are called in the correct order (separator first, then version)
        separator_index = markdown_calls.index("---")
        version_index = markdown_calls.index("**WicketWise version 0.1**")
        assert separator_index < version_index

def test_ui_launcher_tab_names():
    """Test that the tab names are exactly as specified."""
    with patch('streamlit.set_page_config'), \
         patch('streamlit.title'), \
         patch('streamlit.tabs') as mock_tabs, \
         patch('streamlit.header'), \
         patch('streamlit.write'), \
         patch('streamlit.markdown'):
        
        # Mock tabs return value with context managers
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tab3 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2, mock_tab3]
        
        # Mock the context managers
        mock_tab1.__enter__ = MagicMock(return_value=mock_tab1)
        mock_tab1.__exit__ = MagicMock(return_value=None)
        mock_tab2.__enter__ = MagicMock(return_value=mock_tab2)
        mock_tab2.__exit__ = MagicMock(return_value=None)
        mock_tab3.__enter__ = MagicMock(return_value=mock_tab3)
        mock_tab3.__exit__ = MagicMock(return_value=None)
        
        # Clear any existing import of ui_launcher
        import sys
        if 'ui_launcher' in sys.modules:
            del sys.modules['ui_launcher']
        
        # Import and run the app
        import ui_launcher
        
        # Verify exact tab names
        expected_tab_names = ["Live Match Dashboard", "Simulator Mode", "Admin Panel"]
        mock_tabs.assert_called_once_with(expected_tab_names)

if __name__ == "__main__":
    pytest.main([__file__]) 
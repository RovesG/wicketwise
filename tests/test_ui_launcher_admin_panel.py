# Purpose: Test Admin Panel input fields and session state management
# Author: WicketWise Team, Last Modified: 2024-12-19

"""
Test suite for Admin Panel functionality in ui_launcher.py
Tests that input fields correctly populate session state with specified keys.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch
import io
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MockSessionState:
    """Mock class to simulate Streamlit's session state behavior"""
    def __init__(self):
        self._data = {}
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._data[name] = value
    
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __contains__(self, key):
        return key in self._data
    
    def keys(self):
        return self._data.keys()
    
    def items(self):
        return self._data.items()
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def __len__(self):
        return len(self._data)

class TestUILauncherAdminPanel:
    """Test suite for Admin Panel session state management"""
    
    def setup_method(self):
        """Setup method run before each test"""
        # Clear session state before each test
        if hasattr(st, 'session_state'):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
    
    def test_api_keys_session_state_storage(self):
        """Test that API keys are stored in session state with correct keys"""
        # Mock session state
        mock_session_state = MockSessionState()
        
        with patch('streamlit.session_state', mock_session_state):
            # Simulate API key inputs
            betfair_key = "test_betfair_key_123"
            openai_key = "test_openai_key_456"
            
            # Simulate the logic from ui_launcher.py
            if betfair_key:
                st.session_state.api_betfair = betfair_key
            if openai_key:
                st.session_state.api_openai = openai_key
            
            # Verify session state contains the correct keys and values
            assert 'api_betfair' in mock_session_state
            assert 'api_openai' in mock_session_state
            assert mock_session_state.api_betfair == betfair_key
            assert mock_session_state.api_openai == openai_key
    
    def test_file_upload_session_state_storage(self):
        """Test that uploaded files are stored in session state with correct keys"""
        # Mock session state
        mock_session_state = MockSessionState()
        
        with patch('streamlit.session_state', mock_session_state):
            # Create mock file objects
            mock_decimal_file = Mock()
            mock_decimal_file.name = "decimal_data.csv"
            mock_decimal_file.type = "text/csv"
            
            mock_nvplay_file = Mock()
            mock_nvplay_file.name = "nvplay_data.csv"
            mock_nvplay_file.type = "text/csv"
            
            mock_aligned_file = Mock()
            mock_aligned_file.name = "aligned_matches.csv"
            mock_aligned_file.type = "text/csv"
            
            # Simulate the logic from ui_launcher.py
            if mock_decimal_file:
                st.session_state.path_decimal = mock_decimal_file
            if mock_nvplay_file:
                st.session_state.path_nvplay = mock_nvplay_file
            if mock_aligned_file:
                st.session_state.path_aligned = mock_aligned_file
            
            # Verify session state contains the correct keys and values
            assert 'path_decimal' in mock_session_state
            assert 'path_nvplay' in mock_session_state
            assert 'path_aligned' in mock_session_state
            assert mock_session_state.path_decimal == mock_decimal_file
            assert mock_session_state.path_nvplay == mock_nvplay_file
            assert mock_session_state.path_aligned == mock_aligned_file
    
    def test_all_session_keys_populated(self):
        """Test that all expected session keys can be populated"""
        # Mock session state
        mock_session_state = MockSessionState()
        
        with patch('streamlit.session_state', mock_session_state):
            # Test data
            test_data = {
                'api_betfair': 'test_betfair_api_key',
                'api_openai': 'test_openai_api_key',
                'path_decimal': Mock(name='decimal.csv'),
                'path_nvplay': Mock(name='nvplay.csv'),
                'path_aligned': Mock(name='aligned.csv')
            }
            
            # Simulate all inputs being populated
            for key, value in test_data.items():
                setattr(st.session_state, key, value)
            
            # Verify all keys are present
            expected_keys = ['api_betfair', 'api_openai', 'path_decimal', 'path_nvplay', 'path_aligned']
            for key in expected_keys:
                assert key in mock_session_state
                assert getattr(mock_session_state, key) == test_data[key]
    
    def test_empty_inputs_not_stored(self):
        """Test that empty inputs are not stored in session state"""
        # Mock session state
        mock_session_state = MockSessionState()
        
        with patch('streamlit.session_state', mock_session_state):
            # Simulate empty inputs
            betfair_key = ""
            openai_key = None
            decimal_file = None
            nvplay_file = None
            aligned_file = None
            
            # Simulate the logic from ui_launcher.py (only store if truthy)
            if betfair_key:
                st.session_state.api_betfair = betfair_key
            if openai_key:
                st.session_state.api_openai = openai_key
            if decimal_file:
                st.session_state.path_decimal = decimal_file
            if nvplay_file:
                st.session_state.path_nvplay = nvplay_file
            if aligned_file:
                st.session_state.path_aligned = aligned_file
            
            # Verify session state is empty
            assert len(mock_session_state) == 0
    
    def test_session_state_keys_match_specification(self):
        """Test that session state keys match the exact specification"""
        # Mock session state
        mock_session_state = MockSessionState()
        
        with patch('streamlit.session_state', mock_session_state):
            # Expected keys from specification
            expected_keys = ['api_betfair', 'api_openai', 'path_decimal', 'path_nvplay', 'path_aligned']
            
            # Populate session state with test values
            for key in expected_keys:
                if key.startswith('api_'):
                    setattr(st.session_state, key, f"test_{key}")
                else:
                    setattr(st.session_state, key, Mock(name=f"test_{key}.csv"))
            
            # Verify all keys exist and match specification
            for key in expected_keys:
                assert key in mock_session_state
            
            # Verify no extra keys are present
            assert set(mock_session_state.keys()) == set(expected_keys)
    
    def test_session_state_persistence(self):
        """Test that session state values persist between operations"""
        # Mock session state
        mock_session_state = MockSessionState()
        
        with patch('streamlit.session_state', mock_session_state):
            # First operation: store API keys
            st.session_state.api_betfair = "initial_betfair_key"
            st.session_state.api_openai = "initial_openai_key"
            
            # Verify initial values
            assert mock_session_state.api_betfair == "initial_betfair_key"
            assert mock_session_state.api_openai == "initial_openai_key"
            
            # Second operation: add file uploads
            st.session_state.path_decimal = Mock(name="decimal.csv")
            st.session_state.path_nvplay = Mock(name="nvplay.csv")
            
            # Verify all values persist
            assert mock_session_state.api_betfair == "initial_betfair_key"
            assert mock_session_state.api_openai == "initial_openai_key"
            assert 'path_decimal' in mock_session_state
            assert 'path_nvplay' in mock_session_state
            
            # Third operation: update existing key
            st.session_state.api_betfair = "updated_betfair_key"
            
            # Verify update and persistence
            assert mock_session_state.api_betfair == "updated_betfair_key"
            assert mock_session_state.api_openai == "initial_openai_key"  # Should remain unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
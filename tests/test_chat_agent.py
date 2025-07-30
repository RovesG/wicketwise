# Purpose: Test chat agent functionality with mocked OpenAI responses
# Author: Phi1618 Cricket AI Team, Last Modified: 2024

import pytest
from unittest.mock import patch, MagicMock, Mock
import json
import sys
import os
from typing import Dict, Any

# Add the parent directory to the path so we can import chat_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_agent import ChatAgent, ask_llm_with_tools, get_chat_agent, reset_chat_agent


class TestChatAgentInitialization:
    """Test chat agent initialization and configuration."""
    
    def setup_method(self):
        """Reset chat agent before each test."""
        reset_chat_agent()
    
    @patch('chat_agent.OpenAI')
    def test_initialization_with_api_key(self, mock_openai):
        """Test chat agent initialization with API key."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        agent = ChatAgent(api_key="test-key", model="gpt-4o")
        
        assert agent.api_key == "test-key"
        assert agent.model == "gpt-4o"
        assert agent.client == mock_client
        mock_openai.assert_called_once_with(api_key="test-key")
    
    @patch('chat_agent.OpenAI')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'})
    def test_initialization_with_env_key(self, mock_openai):
        """Test chat agent initialization with environment API key."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        agent = ChatAgent()
        
        assert agent.api_key == "env-key"
        assert agent.model == "gpt-4o"  # Default model
        mock_openai.assert_called_once_with(api_key="env-key")
    
    @patch('chat_agent.OPENAI_AVAILABLE', False)
    def test_initialization_without_openai_library(self):
        """Test chat agent initialization when OpenAI library is not available."""
        agent = ChatAgent()
        
        assert agent.client is None
        assert agent.api_key is None
    
    @patch('chat_agent.OpenAI')
    def test_initialization_without_api_key(self, mock_openai):
        """Test chat agent initialization without API key."""
        # Clear environment variable
        with patch.dict(os.environ, {}, clear=True):
            agent = ChatAgent()
            
            assert agent.api_key is None
            assert agent.client is None
            mock_openai.assert_not_called()


class TestChatAgentFunctionCalling:
    """Test chat agent function calling with mocked OpenAI responses."""
    
    def setup_method(self):
        """Reset chat agent and set up common mocks."""
        reset_chat_agent()
        self.mock_client = MagicMock()
        
        # Mock OpenAI completion response with function call
        self.mock_function_call_response = MagicMock()
        self.mock_function_call_response.choices = [MagicMock()]
        self.mock_function_call_response.choices[0].message.tool_calls = [
            MagicMock(
                id="call_123",
                function=MagicMock(
                    name="get_form_vector",
                    arguments='{"player_name": "Virat Kohli"}'
                )
            )
        ]
        
        # Mock final response after function call
        self.mock_final_response = MagicMock()
        self.mock_final_response.choices = [MagicMock()]
        self.mock_final_response.choices[0].message.content = "Based on the form analysis, Virat Kohli is in excellent form with a score of 0.75."
    
    @patch('chat_agent.OpenAI')
    def test_function_calling_with_get_form_vector(self, mock_openai):
        """Test function calling with get_form_vector tool."""
        mock_openai.return_value = self.mock_client
        
        # Configure mock to return function call first, then final response
        self.mock_client.chat.completions.create.side_effect = [
            self.mock_function_call_response,
            self.mock_final_response
        ]
        
        agent = ChatAgent(api_key="test-key")
        result = agent.ask_llm_with_tools("What is Virat Kohli's form?")
        
        # Verify the result
        assert result["success"] == True
        assert result["tool_used"] == "get_form_vector"
        assert result["tool_result"] is not None
        assert "Virat Kohli" in result["answer"]
        assert len(result["raw_logs"]) >= 2
        assert result["error"] is None
        
        # Verify OpenAI was called twice (initial + final)
        assert self.mock_client.chat.completions.create.call_count == 2
    
    @patch('chat_agent.OpenAI')
    def test_function_calling_with_predict_ball_outcome(self, mock_openai):
        """Test function calling with predict_ball_outcome tool."""
        mock_openai.return_value = self.mock_client
        
        # Mock function call for prediction
        self.mock_function_call_response.choices[0].message.tool_calls = [
            MagicMock(
                id="call_456",
                function=MagicMock(
                    name="predict_ball_outcome",
                    arguments='{"scenario": "Next ball prediction"}'
                )
            )
        ]
        
        self.mock_final_response.choices[0].message.content = "Based on the prediction model, there's a 68% chance of a favorable outcome."
        
        self.mock_client.chat.completions.create.side_effect = [
            self.mock_function_call_response,
            self.mock_final_response
        ]
        
        agent = ChatAgent(api_key="test-key")
        result = agent.ask_llm_with_tools("Predict the outcome of the next ball")
        
        # Verify the result
        assert result["success"] == True
        assert result["tool_used"] == "predict_ball_outcome"
        assert result["tool_result"] is not None
        assert "prediction" in result["answer"].lower()
        assert len(result["raw_logs"]) >= 2


class TestChatAgentFallbackScenarios:
    """Test chat agent fallback scenarios when no function is called."""
    
    def setup_method(self):
        """Reset chat agent and set up common mocks."""
        reset_chat_agent()
        self.mock_client = MagicMock()
        
        # Mock OpenAI response without function calls
        self.mock_direct_response = MagicMock()
        self.mock_direct_response.choices = [MagicMock()]
        self.mock_direct_response.choices[0].message.tool_calls = None
        self.mock_direct_response.choices[0].message.content = "Cricket is a complex sport with many strategic elements."
    
    @patch('chat_agent.OpenAI')
    def test_direct_response_without_function_call(self, mock_openai):
        """Test direct response when no function is called."""
        mock_openai.return_value = self.mock_client
        self.mock_client.chat.completions.create.return_value = self.mock_direct_response
        
        agent = ChatAgent(api_key="test-key")
        result = agent.ask_llm_with_tools("Tell me about cricket rules")
        
        # Verify the result
        assert result["success"] == True
        assert result["tool_used"] is None
        assert result["tool_result"] is None
        assert "cricket" in result["answer"].lower()
        assert len(result["raw_logs"]) >= 1
        
        # Verify OpenAI was called only once
        assert self.mock_client.chat.completions.create.call_count == 1
    
    @patch('chat_agent.OpenAI')
    def test_api_error_handling(self, mock_openai):
        """Test error handling when OpenAI API fails."""
        mock_openai.return_value = self.mock_client
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        agent = ChatAgent(api_key="test-key")
        result = agent.ask_llm_with_tools("What is Virat Kohli's form?")
        
        # Verify fallback response
        assert result["success"] == True  # Should succeed with fallback
        assert result["tool_used"] == "fallback_handler"
        assert "Fallback Mode" in result["answer"]
        assert result["error"] == "API Error"
    
    @patch('chat_agent.OPENAI_AVAILABLE', False)
    def test_openai_not_available(self):
        """Test behavior when OpenAI library is not available."""
        agent = ChatAgent()
        result = agent.ask_llm_with_tools("What is Virat Kohli's form?")
        
        # Verify fallback response
        assert result["success"] == True
        assert result["tool_used"] == "fallback_handler"
        assert "Fallback Mode" in result["answer"]
        assert len(result["raw_logs"]) >= 1


class TestChatAgentConvenienceFunctions:
    """Test convenience functions and global agent management."""
    
    def setup_method(self):
        """Reset chat agent before each test."""
        reset_chat_agent()
    
    @patch('chat_agent.OpenAI')
    def test_get_chat_agent_singleton(self, mock_openai):
        """Test that get_chat_agent returns singleton instance."""
        mock_openai.return_value = MagicMock()
        
        agent1 = get_chat_agent(api_key="test-key")
        agent2 = get_chat_agent(api_key="different-key")  # Should return same instance
        
        assert agent1 is agent2
        assert agent1.api_key == "test-key"  # Should keep original key
        
        # Only one OpenAI client should be created
        assert mock_openai.call_count == 1
    
    @patch('chat_agent.OpenAI')
    def test_ask_llm_with_tools_convenience_function(self, mock_openai):
        """Test the convenience function ask_llm_with_tools."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "Test response"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        result = ask_llm_with_tools("Test query")
        
        assert result["success"] == True
        assert result["answer"] == "Test response"
        assert mock_client.chat.completions.create.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

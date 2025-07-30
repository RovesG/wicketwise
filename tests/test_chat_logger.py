# Purpose: Test chat logger functionality for cricket analysis conversations
# Author: Phi1618 Cricket AI, Last Modified: 2025-01-17

import pytest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the chat logger module
from chat_logger import (
    log_chat_interaction,
    read_chat_logs,
    get_session_stats,
    clear_chat_logs,
    get_log_file_info,
    reset_session,
    ensure_log_directory,
    generate_session_id,
    LOG_FILE_PATH,
    LOG_DIRECTORY
)

class TestChatLogger:
    """Test suite for chat logger functionality"""
    
    def setup_method(self):
        """Set up test environment before each test"""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.original_log_path = LOG_FILE_PATH
        self.original_log_dir = LOG_DIRECTORY
        
        # Patch the log paths to use temporary directory
        self.log_file_path = os.path.join(self.temp_dir, "chat_history.jsonl")
        self.log_directory = self.temp_dir
        
        # Apply patches
        patch_log_file = patch('chat_logger.LOG_FILE_PATH', self.log_file_path)
        patch_log_dir = patch('chat_logger.LOG_DIRECTORY', self.log_directory)
        
        self.log_file_patcher = patch_log_file.start()
        self.log_dir_patcher = patch_log_dir.start()
        
        # Reset session for clean testing
        if hasattr(log_chat_interaction, '_session_id'):
            delattr(log_chat_interaction, '_session_id')
    
    def teardown_method(self):
        """Clean up after each test"""
        # Stop patches
        patch.stopall()
        
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Reset session
        if hasattr(log_chat_interaction, '_session_id'):
            delattr(log_chat_interaction, '_session_id')
    
    def test_log_chat_interaction_basic(self):
        """Test basic chat interaction logging"""
        # Test data
        user_message = "What is Virat Kohli's form?"
        bot_response = "Virat Kohli is in excellent form with a current form score of 0.85."
        tool_used = "get_form_vector"
        context = {
            "match_id": "IND_vs_AUS_2024_T20_001",
            "ball_number": 42,
            "total_balls": 120
        }
        
        # Log the interaction
        log_chat_interaction(user_message, bot_response, tool_used, context)
        
        # Verify log file was created
        assert os.path.exists(self.log_file_path)
        
        # Read and verify log content
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            log_line = f.readline().strip()
            log_entry = json.loads(log_line)
        
        # Verify required fields
        assert log_entry["user_message"] == user_message
        assert log_entry["assistant_response"] == bot_response
        assert log_entry["tool_used"] == tool_used
        assert log_entry["context"]["match_id"] == context["match_id"]
        assert log_entry["context"]["ball_number"] == context["ball_number"]
        assert log_entry["context"]["total_balls"] == context["total_balls"]
        
        # Verify metadata
        assert log_entry["metadata"]["response_length"] == len(bot_response)
        assert log_entry["metadata"]["query_length"] == len(user_message)
        assert log_entry["metadata"]["has_tool"] is True
        assert log_entry["metadata"]["has_context"] is True
        
        # Verify timestamp format
        datetime.fromisoformat(log_entry["timestamp"])  # Should not raise exception
        
        # Verify session ID exists
        assert "session_id" in log_entry
        assert len(log_entry["session_id"]) == 8
    
    def test_log_chat_interaction_without_tool(self):
        """Test chat interaction logging without tool usage"""
        user_message = "Hello, how are you?"
        bot_response = "Hello! I'm your cricket analysis assistant."
        
        # Log without tool
        log_chat_interaction(user_message, bot_response, None, None)
        
        # Verify log file was created
        assert os.path.exists(self.log_file_path)
        
        # Read and verify log content
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            log_line = f.readline().strip()
            log_entry = json.loads(log_line)
        
        # Verify fields
        assert log_entry["user_message"] == user_message
        assert log_entry["assistant_response"] == bot_response
        assert log_entry["tool_used"] is None
        assert log_entry["context"]["match_id"] is None
        assert log_entry["metadata"]["has_tool"] is False
        assert log_entry["metadata"]["has_context"] is False
    
    def test_write_two_mock_logs(self):
        """Test writing two mock logs and validating JSON structure"""
        # First mock log
        log_chat_interaction(
            user="What is the current match situation?",
            bot="The match is at a crucial stage with 45 runs needed from 30 balls.",
            tool_used="match_status",
            context={
                "match_id": "IND_vs_AUS_2024_T20_001",
                "ball_number": 90,
                "total_balls": 120,
                "current_data": {
                    "batter": "MS Dhoni",
                    "bowler": "Mitchell Starc",
                    "over": 15,
                    "runs": 2
                },
                "match_state": {
                    "playing": True,
                    "progress": 75.0
                }
            }
        )
        
        # Second mock log
        log_chat_interaction(
            user="Who has the batting advantage?",
            bot="Based on current form and matchup analysis, the batting team has a slight advantage.",
            tool_used="query_kg_relationship",
            context={
                "match_id": "IND_vs_AUS_2024_T20_001",
                "ball_number": 91,
                "total_balls": 120
            }
        )
        
        # Verify log file exists
        assert os.path.exists(self.log_file_path)
        
        # Read and validate both logs
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        # Validate first log
        log1 = json.loads(lines[0].strip())
        assert log1["user_message"] == "What is the current match situation?"
        assert log1["tool_used"] == "match_status"
        assert log1["context"]["match_id"] == "IND_vs_AUS_2024_T20_001"
        assert log1["context"]["ball_number"] == 90
        assert log1["context"]["current_data"]["batter"] == "MS Dhoni"
        
        # Validate second log
        log2 = json.loads(lines[1].strip())
        assert log2["user_message"] == "Who has the batting advantage?"
        assert log2["tool_used"] == "query_kg_relationship"
        assert log2["context"]["match_id"] == "IND_vs_AUS_2024_T20_001"
        assert log2["context"]["ball_number"] == 91
        
        # Verify both logs have same session ID
        assert log1["session_id"] == log2["session_id"]
    
    def test_required_fields_presence(self):
        """Test that all required fields are present in log entries"""
        # Log a comprehensive interaction
        log_chat_interaction(
            user="Predict the next ball outcome",
            bot="Based on current conditions, there's a 35% chance of a boundary.",
            tool_used="predict_ball_outcome",
            context={
                "match_id": "ENG_vs_PAK_2024_T20_005",
                "ball_number": 67,
                "total_balls": 120,
                "current_data": {
                    "batter": "Babar Azam",
                    "bowler": "Jofra Archer",
                    "over": 11,
                    "runs": 0
                },
                "match_state": {
                    "playing": True,
                    "progress": 55.8
                },
                "additional_field": "extra_context"
            }
        )
        
        # Read the log
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.readline().strip())
        
        # Verify all required fields are present
        required_fields = [
            "timestamp", "session_id", "user_message", "assistant_response",
            "tool_used", "context", "metadata"
        ]
        
        for field in required_fields:
            assert field in log_entry, f"Required field '{field}' is missing"
        
        # Verify context sub-fields
        context_fields = [
            "match_id", "ball_number", "total_balls", "current_data",
            "match_state", "additional_context"
        ]
        
        for field in context_fields:
            assert field in log_entry["context"], f"Context field '{field}' is missing"
        
        # Verify metadata sub-fields
        metadata_fields = [
            "response_length", "query_length", "has_tool", "has_context"
        ]
        
        for field in metadata_fields:
            assert field in log_entry["metadata"], f"Metadata field '{field}' is missing"
        
        # Verify additional context was captured
        assert "additional_field" in log_entry["context"]["additional_context"]
        assert log_entry["context"]["additional_context"]["additional_field"] == "extra_context"
    
    def test_json_validity(self):
        """Test that log file contains valid JSON"""
        # Log multiple interactions
        interactions = [
            ("Query 1", "Response 1", "tool_1", {"match_id": "match_1"}),
            ("Query 2", "Response 2", "tool_2", {"match_id": "match_2"}),
            ("Query 3", "Response 3", None, None),
        ]
        
        for user, bot, tool, context in interactions:
            log_chat_interaction(user, bot, tool, context)
        
        # Verify each line is valid JSON
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Invalid JSON on line {line_num}: {e}")
    
    def test_read_chat_logs(self):
        """Test reading chat logs functionality"""
        # Log some interactions
        log_chat_interaction("Test 1", "Response 1", "tool_1", {"match_id": "test_match"})
        log_chat_interaction("Test 2", "Response 2", "tool_2", {"match_id": "test_match"})
        log_chat_interaction("Test 3", "Response 3", None, None)
        
        # Read all logs
        logs = read_chat_logs()
        assert len(logs) == 3
        
        # Verify most recent first
        assert logs[0]["user_message"] == "Test 3"
        assert logs[1]["user_message"] == "Test 2"
        assert logs[2]["user_message"] == "Test 1"
        
        # Test with limit
        limited_logs = read_chat_logs(limit=2)
        assert len(limited_logs) == 2
        assert limited_logs[0]["user_message"] == "Test 3"
        assert limited_logs[1]["user_message"] == "Test 2"
    
    def test_session_management(self):
        """Test session ID management"""
        # Log first interaction
        log_chat_interaction("Query 1", "Response 1", "tool_1", {"match_id": "test"})
        
        # Log second interaction (should have same session)
        log_chat_interaction("Query 2", "Response 2", "tool_2", {"match_id": "test"})
        
        # Read logs and verify same session ID
        logs = read_chat_logs()
        assert logs[0]["session_id"] == logs[1]["session_id"]
        
        # Reset session
        new_session_id = reset_session()
        
        # Log third interaction (should have different session)
        log_chat_interaction("Query 3", "Response 3", "tool_3", {"match_id": "test"})
        
        # Read logs and verify different session ID
        logs = read_chat_logs()
        assert logs[0]["session_id"] != logs[1]["session_id"]
        assert logs[0]["session_id"] == new_session_id
    
    def test_get_log_file_info(self):
        """Test log file information retrieval"""
        # Test when file doesn't exist
        info = get_log_file_info()
        assert info["exists"] is False
        assert info["size_bytes"] == 0
        assert info["line_count"] == 0
        
        # Log some interactions
        log_chat_interaction("Test", "Response", "tool", {"match_id": "test"})
        
        # Test when file exists
        info = get_log_file_info()
        assert info["exists"] is True
        assert info["size_bytes"] > 0
        assert info["line_count"] == 1
        assert info["path"] == self.log_file_path
    
    def test_clear_chat_logs(self):
        """Test clearing chat logs"""
        # Log some interactions
        log_chat_interaction("Test", "Response", "tool", {"match_id": "test"})
        assert os.path.exists(self.log_file_path)
        
        # Clear logs
        success = clear_chat_logs()
        assert success is True
        assert not os.path.exists(self.log_file_path)
    
    def test_get_session_stats(self):
        """Test session statistics calculation"""
        # Log interactions in same session
        log_chat_interaction("Query 1", "Response 1", "tool_1", {"match_id": "test"})
        log_chat_interaction("Query 2", "Response 2", "tool_2", {"match_id": "test"})
        log_chat_interaction("Query 3", "Response 3", None, {"match_id": "test"})
        
        # Get session stats
        logs = read_chat_logs()
        session_id = logs[0]["session_id"]
        stats = get_session_stats(session_id)
        
        # Verify stats
        assert stats["session_id"] == session_id
        assert stats["total_interactions"] == 3
        assert len(stats["tools_used"]) == 2  # tool_1, tool_2
        assert len(stats["unique_tools"]) == 2
        assert stats["tool_usage_rate"] == 2/3
        assert stats["avg_response_length"] > 0
        assert stats["avg_query_length"] > 0
    
    def test_ensure_log_directory(self):
        """Test log directory creation"""
        # Remove the directory
        if os.path.exists(self.log_directory):
            shutil.rmtree(self.log_directory)
        
        # Ensure directory
        ensure_log_directory()
        
        # Verify directory exists
        assert os.path.exists(self.log_directory)
        assert os.path.isdir(self.log_directory)
    
    def test_generate_session_id(self):
        """Test session ID generation"""
        session_id = generate_session_id()
        
        # Verify format
        assert isinstance(session_id, str)
        assert len(session_id) == 8
        
        # Verify uniqueness
        session_id2 = generate_session_id()
        assert session_id != session_id2
    
    def test_error_handling(self):
        """Test error handling in logging"""
        # Test with read-only directory (if possible)
        # This is a basic test - in production, we'd use more sophisticated mocking
        
        # Test with empty strings
        log_chat_interaction("", "", None, None)
        
        # Should not crash, log should be created
        assert os.path.exists(self.log_file_path)
        
        # Read and verify
        logs = read_chat_logs()
        assert len(logs) == 1
        assert logs[0]["user_message"] == ""
        assert logs[0]["assistant_response"] == ""
        assert logs[0]["metadata"]["query_length"] == 0
        assert logs[0]["metadata"]["response_length"] == 0
    
    def test_unicode_handling(self):
        """Test handling of unicode characters"""
        user_message = "What about 🏏 cricket analysis? 中文 test"
        bot_response = "Cricket analysis with emojis: 🎯 📊 ⚡"
        
        log_chat_interaction(user_message, bot_response, "unicode_tool", {"match_id": "unicode_test"})
        
        # Read and verify
        logs = read_chat_logs()
        assert logs[0]["user_message"] == user_message
        assert logs[0]["assistant_response"] == bot_response
        assert logs[0]["context"]["match_id"] == "unicode_test"


class TestChatLoggerIntegration:
    """Integration tests for chat logger with various scenarios"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file_path = os.path.join(self.temp_dir, "chat_history.jsonl")
        
        # Patch the log paths
        patch_log_file = patch('chat_logger.LOG_FILE_PATH', self.log_file_path)
        patch_log_dir = patch('chat_logger.LOG_DIRECTORY', self.temp_dir)
        
        self.log_file_patcher = patch_log_file.start()
        self.log_dir_patcher = patch_log_dir.start()
        
        # Reset session
        if hasattr(log_chat_interaction, '_session_id'):
            delattr(log_chat_interaction, '_session_id')
    
    def teardown_method(self):
        """Clean up after tests"""
        patch.stopall()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if hasattr(log_chat_interaction, '_session_id'):
            delattr(log_chat_interaction, '_session_id')
    
    def test_cricket_analysis_conversation(self):
        """Test a realistic cricket analysis conversation"""
        # Simulate a cricket analysis conversation
        conversation = [
            ("What is Virat Kohli's recent form?", "Analyzing form...", "get_form_vector"),
            ("How does he perform against fast bowlers?", "Checking matchups...", "query_kg_relationship"),
            ("What's the probability of a boundary next ball?", "Calculating probability...", "predict_ball_outcome"),
            ("Thanks for the analysis!", "You're welcome!", None)
        ]
        
        context = {
            "match_id": "IND_vs_AUS_2024_T20_001",
            "ball_number": 45,
            "total_balls": 120,
            "current_data": {
                "batter": "Virat Kohli",
                "bowler": "Pat Cummins",
                "over": 8,
                "runs": 1
            },
            "match_state": {
                "playing": True,
                "progress": 37.5
            }
        }
        
        # Log the conversation
        for user, bot, tool in conversation:
            log_chat_interaction(user, bot, tool, context)
            # Update context for next interaction
            context["ball_number"] += 1
        
        # Verify conversation was logged correctly
        logs = read_chat_logs()
        assert len(logs) == 4
        
        # Verify conversation order (most recent first)
        assert logs[0]["user_message"] == "Thanks for the analysis!"
        assert logs[1]["user_message"] == "What's the probability of a boundary next ball?"
        assert logs[2]["user_message"] == "How does he perform against fast bowlers?"
        assert logs[3]["user_message"] == "What is Virat Kohli's recent form?"
        
        # Verify tool usage
        assert logs[0]["tool_used"] is None
        assert logs[1]["tool_used"] == "predict_ball_outcome"
        assert logs[2]["tool_used"] == "query_kg_relationship"
        assert logs[3]["tool_used"] == "get_form_vector"
        
        # Verify context progression
        assert logs[0]["context"]["ball_number"] == 48
        assert logs[1]["context"]["ball_number"] == 47
        assert logs[2]["context"]["ball_number"] == 46
        assert logs[3]["context"]["ball_number"] == 45
        
        # Verify same session ID
        session_ids = [log["session_id"] for log in logs]
        assert len(set(session_ids)) == 1  # All should be the same
    
    def test_multiple_sessions(self):
        """Test logging across multiple sessions"""
        # First session
        log_chat_interaction("Session 1 Query", "Session 1 Response", "tool_1", {"match_id": "match_1"})
        
        # Reset session
        reset_session()
        
        # Second session
        log_chat_interaction("Session 2 Query", "Session 2 Response", "tool_2", {"match_id": "match_2"})
        
        # Read logs
        logs = read_chat_logs()
        assert len(logs) == 2
        
        # Verify different session IDs
        assert logs[0]["session_id"] != logs[1]["session_id"]
        
        # Test session stats
        session_1_stats = get_session_stats(logs[1]["session_id"])
        session_2_stats = get_session_stats(logs[0]["session_id"])
        
        assert session_1_stats["total_interactions"] == 1
        assert session_2_stats["total_interactions"] == 1
        assert session_1_stats["session_id"] != session_2_stats["session_id"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
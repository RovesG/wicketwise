# Purpose: Chat interaction logger for cricket analysis conversations
# Author: Phi1618 Cricket AI, Last Modified: 2025-01-17

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union
import uuid

# Constants
LOG_FILE_PATH = "logs/chat_history.jsonl"
LOG_DIRECTORY = "logs"

def ensure_log_directory() -> None:
    """
    Ensure the logs directory exists.
    Creates the directory if it doesn't exist.
    """
    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)

def generate_session_id() -> str:
    """
    Generate a unique session ID for tracking conversation sessions.
    
    Returns:
        str: Unique session identifier
    """
    return str(uuid.uuid4())[:8]

def log_chat_interaction(
    user: str,
    bot: str,
    tool_used: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a chat interaction to the structured JSON log file.
    
    Args:
        user: User message/query
        bot: Assistant response
        tool_used: Name of the tool used (if any)
        context: Match context and metadata
    """
    
    # Ensure logs directory exists
    ensure_log_directory()
    
    # Generate timestamp
    timestamp = datetime.now().isoformat()
    
    # Extract context information
    if context is None:
        context = {}
    
    # Build log entry
    log_entry = {
        "timestamp": timestamp,
        "session_id": getattr(log_chat_interaction, '_session_id', generate_session_id()),
        "user_message": user,
        "assistant_response": bot,
        "tool_used": tool_used,
        "context": {
            "match_id": context.get("match_id"),
            "ball_number": context.get("ball_number"),
            "total_balls": context.get("total_balls"),
            "current_data": context.get("current_data"),
            "match_state": context.get("match_state"),
            "additional_context": {k: v for k, v in context.items() 
                                 if k not in ["match_id", "ball_number", "total_balls", "current_data", "match_state"]}
        },
        "metadata": {
            "response_length": len(bot) if bot else 0,
            "query_length": len(user) if user else 0,
            "has_tool": tool_used is not None,
            "has_context": bool(context and any(context.values()))
        }
    }
    
    # Store session ID for subsequent calls
    if not hasattr(log_chat_interaction, '_session_id'):
        log_chat_interaction._session_id = log_entry["session_id"]
    
    # Append to log file
    try:
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        # Silent failure - don't break the chat if logging fails
        print(f"Warning: Failed to write to chat log: {e}")

def read_chat_logs(limit: Optional[int] = None) -> list[Dict[str, Any]]:
    """
    Read chat logs from the log file.
    
    Args:
        limit: Maximum number of logs to return (most recent first)
        
    Returns:
        List of log entries
    """
    
    if not os.path.exists(LOG_FILE_PATH):
        return []
    
    logs = []
    try:
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        log_entry = json.loads(line)
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue
        
        # Return most recent first
        logs.reverse()
        
        if limit:
            logs = logs[:limit]
            
        return logs
        
    except Exception as e:
        print(f"Warning: Failed to read chat logs: {e}")
        return []

def get_session_stats(session_id: str) -> Dict[str, Any]:
    """
    Get statistics for a specific chat session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dictionary with session statistics
    """
    
    logs = read_chat_logs()
    session_logs = [log for log in logs if log.get("session_id") == session_id]
    
    if not session_logs:
        return {}
    
    total_interactions = len(session_logs)
    tools_used = [log.get("tool_used") for log in session_logs if log.get("tool_used")]
    unique_tools = list(set(tools_used))
    
    # Calculate timing
    timestamps = [datetime.fromisoformat(log["timestamp"]) for log in session_logs]
    session_start = min(timestamps)
    session_end = max(timestamps)
    session_duration = (session_end - session_start).total_seconds()
    
    return {
        "session_id": session_id,
        "total_interactions": total_interactions,
        "tools_used": tools_used,
        "unique_tools": unique_tools,
        "session_start": session_start.isoformat(),
        "session_end": session_end.isoformat(),
        "session_duration_seconds": session_duration,
        "avg_response_length": sum(log["metadata"]["response_length"] for log in session_logs) / total_interactions,
        "avg_query_length": sum(log["metadata"]["query_length"] for log in session_logs) / total_interactions,
        "tool_usage_rate": len(tools_used) / total_interactions if total_interactions > 0 else 0
    }

def clear_chat_logs() -> bool:
    """
    Clear all chat logs.
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        if os.path.exists(LOG_FILE_PATH):
            os.remove(LOG_FILE_PATH)
        return True
    except Exception as e:
        print(f"Warning: Failed to clear chat logs: {e}")
        return False

def get_log_file_info() -> Dict[str, Any]:
    """
    Get information about the log file.
    
    Returns:
        Dictionary with log file information
    """
    
    if not os.path.exists(LOG_FILE_PATH):
        return {
            "exists": False,
            "path": LOG_FILE_PATH,
            "size_bytes": 0,
            "line_count": 0
        }
    
    try:
        file_size = os.path.getsize(LOG_FILE_PATH)
        
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f if line.strip())
        
        return {
            "exists": True,
            "path": LOG_FILE_PATH,
            "size_bytes": file_size,
            "line_count": line_count,
            "size_mb": round(file_size / (1024 * 1024), 2)
        }
        
    except Exception as e:
        print(f"Warning: Failed to get log file info: {e}")
        return {
            "exists": True,
            "path": LOG_FILE_PATH,
            "size_bytes": 0,
            "line_count": 0,
            "error": str(e)
        }

def reset_session() -> str:
    """
    Reset the current session and generate a new session ID.
    
    Returns:
        str: New session identifier
    """
    
    new_session_id = generate_session_id()
    log_chat_interaction._session_id = new_session_id
    return new_session_id

# Example usage for testing
if __name__ == "__main__":
    # Test the logging functionality
    print("=== Chat Logger Demo ===\n")
    
    # Test basic logging
    print("1. Testing basic logging...")
    log_chat_interaction(
        user="What is Virat Kohli's form?",
        bot="Virat Kohli is in excellent form with a current form score of 0.85.",
        tool_used="get_form_vector",
        context={
            "match_id": "IND_vs_AUS_2024_T20_001",
            "ball_number": 42,
            "total_balls": 120,
            "current_data": {
                "batter": "Virat Kohli",
                "bowler": "Pat Cummins",
                "over": 7,
                "runs": 4
            },
            "match_state": {
                "playing": True,
                "progress": 35.0
            }
        }
    )
    
    # Test logging without tool
    print("2. Testing logging without tool...")
    log_chat_interaction(
        user="Hello, how are you?",
        bot="Hello! I'm your cricket analysis assistant. How can I help you today?",
        tool_used=None,
        context={}
    )
    
    # Test logging with minimal context
    print("3. Testing logging with minimal context...")
    log_chat_interaction(
        user="What's the current score?",
        bot="The current score is 156/3 after 18 overs.",
        tool_used="match_status",
        context={"match_id": "IND_vs_AUS_2024_T20_001"}
    )
    
    # Display log file info
    print("\n4. Log file information:")
    log_info = get_log_file_info()
    print(f"   Exists: {log_info['exists']}")
    print(f"   Path: {log_info['path']}")
    print(f"   Size: {log_info['size_bytes']} bytes")
    print(f"   Lines: {log_info['line_count']}")
    
    # Display recent logs
    print("\n5. Recent logs:")
    recent_logs = read_chat_logs(limit=2)
    for i, log in enumerate(recent_logs, 1):
        print(f"   Log {i}: {log['user_message'][:50]}...")
        print(f"           Tool: {log['tool_used']}")
        print(f"           Time: {log['timestamp']}")
    
    print("\n=== Demo Complete ===") 
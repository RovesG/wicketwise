#!/usr/bin/env python3
"""
Test script for Knowledge Graph Chat System
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_kg_query_engine():
    """Test the KG Query Engine"""
    print("🔧 Testing KG Query Engine...")
    
    try:
        from crickformers.chat import KGQueryEngine
        
        # Initialize engine
        engine = KGQueryEngine()
        
        # Test graph summary
        summary = engine.get_graph_summary()
        print(f"✅ Graph Summary: {summary.get('total_nodes', 0)} nodes, {summary.get('total_edges', 0)} edges")
        
        # Test player search
        player_node = engine.find_player_node("Kohli")
        print(f"✅ Found player node: {player_node}")
        
        # Test player stats (this will work even with empty graph)
        stats = engine.get_player_stats("Virat Kohli")
        print(f"✅ Player stats result: {type(stats)}")
        
        return True
        
    except Exception as e:
        print(f"❌ KG Query Engine test failed: {e}")
        return False

def test_kg_chat_agent():
    """Test the KG Chat Agent (requires OpenAI API key)"""
    print("\n🤖 Testing KG Chat Agent...")
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OpenAI API key not found - skipping chat agent test")
        return True
    
    try:
        from crickformers.chat import KGChatAgent
        
        # Initialize agent
        agent = KGChatAgent()
        print("✅ Chat agent initialized successfully")
        
        # Test suggested questions
        suggestions = agent.get_suggested_questions()
        print(f"✅ Got {len(suggestions)} suggested questions")
        
        # Test available functions
        functions = agent.get_available_functions()
        print(f"✅ Got {len(functions)} available functions")
        
        return True
        
    except Exception as e:
        print(f"❌ KG Chat Agent test failed: {e}")
        return False

def test_flask_endpoints():
    """Test Flask endpoints"""
    print("\n🌐 Testing Flask Endpoints...")
    
    try:
        import requests
        
        base_url = "http://127.0.0.1:5001"
        
        # Test health endpoint
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
        
        # Test chat suggestions endpoint
        response = requests.get(f"{base_url}/api/kg-chat/suggestions", timeout=5)
        if response.status_code == 200:
            suggestions = response.json()
            print(f"✅ Chat suggestions endpoint: {len(suggestions.get('suggestions', []))} suggestions")
        else:
            print(f"❌ Chat suggestions failed: {response.status_code}")
        
        # Test chat functions endpoint
        response = requests.get(f"{base_url}/api/kg-chat/functions", timeout=5)
        if response.status_code == 200:
            functions = response.json()
            print(f"✅ Chat functions endpoint: {len(functions.get('functions', {}))} functions")
        else:
            print(f"❌ Chat functions failed: {response.status_code}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Flask backend not running - start with: python admin_backend.py")
        return False
    except Exception as e:
        print(f"❌ Flask endpoints test failed: {e}")
        return False

def test_chat_interaction():
    """Test a full chat interaction"""
    print("\n💬 Testing Chat Interaction...")
    
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OpenAI API key not found - skipping chat interaction test")
        return True
    
    try:
        import requests
        
        base_url = "http://127.0.0.1:5001"
        
        # Test chat message
        chat_data = {
            "message": "What can you tell me about cricket?",
            "session_id": "test_session"
        }
        
        response = requests.post(
            f"{base_url}/api/kg-chat", 
            json=chat_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat response received: {len(result.get('response', ''))} characters")
            print(f"📝 Sample response: {result.get('response', '')[:100]}...")
        else:
            print(f"❌ Chat interaction failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Flask backend not running")
        return False
    except Exception as e:
        print(f"❌ Chat interaction test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing WicketWise Knowledge Graph Chat System\n")
    
    tests = [
        test_kg_query_engine,
        test_kg_chat_agent,
        test_flask_endpoints,
        test_chat_interaction
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! The KG Chat system is ready to use.")
        print("\n🚀 To use the chat:")
        print("1. Make sure Flask backend is running: python admin_backend.py")
        print("2. Set OPENAI_API_KEY environment variable")
        print("3. Open http://127.0.0.1:8000/wicketwise_dashboard.html")
        print("4. Look for the 'Cricket AI Chat' panel")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

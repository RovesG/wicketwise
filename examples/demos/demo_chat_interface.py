#!/usr/bin/env python3
# Purpose: Demonstration script for chat interface functionality in ui_launcher.py
# Author: Phi1618 Cricket AI, Last Modified: 2025-01-17

import sys
import pandas as pd
sys.path.insert(0, '.')

from chat_tools import handle_chat_query

def extract_tool_used(response):
    """Extract tool information from chat response."""
    if "**Form Analysis Tool Used**" in response:
        return "get_form_vector()"
    elif "**Knowledge Graph Tool Used**" in response:
        return "query_kg_player_relationship()"
    elif "**Prediction Tool Used**" in response:
        return "predict_with_uncertainty()"
    elif "**General Query Handler**" in response:
        return "fallback_handler()"
    else:
        return "unknown_tool()"

def main():
    """Demonstrate chat interface functionality"""
    
    print("=== Cricket AI Chat Interface Demonstration ===\n")
    
    # Test queries for different routing scenarios
    test_queries = [
        ("What is Virat Kohli's form?", "form"),
        ("What is the matchup between Virat Kohli and Pat Cummins?", "matchup"),
        ("What is the prediction for next match?", "prediction"),
        ("Hello there", "fallback")
    ]
    
    for i, (query, expected_type) in enumerate(test_queries, 1):
        print(f"--- Test Query {i}: {expected_type.upper()} ---")
        print(f"Query: {query}")
        print()
        
        # Get response from chat tools
        response = handle_chat_query(query)
        
        # Extract tool used information
        tool_used = extract_tool_used(response)
        
        # Display response
        print(response)
        print(f"\n🔧 Tool used: {tool_used}")
        print("\n" + "="*80 + "\n")
    
    print("=== Integration Test Summary ===\n")
    
    # Test all three tool types
    integration_tests = [
        ("What is Virat Kohli's form?", "get_form_vector()"),
        ("What is the matchup between Virat Kohli and Pat Cummins?", "query_kg_player_relationship()"),
        ("What is the prediction for next match?", "predict_with_uncertainty()"),
        ("Random question", "fallback_handler()")
    ]
    
    results = []
    for query, expected_tool in integration_tests:
        response = handle_chat_query(query)
        tool_used = extract_tool_used(response)
        match = tool_used == expected_tool
        results.append(match)
        
        print(f"Query: {query}")
        print(f"Expected: {expected_tool}")
        print(f"Actual: {tool_used}")
        print(f"Result: {'✅ PASS' if match else '❌ FAIL'}")
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print(f"=== Test Results Summary ===")
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Overall Status: {'✅ ALL TESTS PASSED' if passed == total else '❌ SOME TESTS FAILED'}")
    
    print("\n=== Chat Interface Demo Complete ===")
    print("The chat interface is ready for integration with the Streamlit UI!")

if __name__ == "__main__":
    main()

# Purpose: Knowledge Graph Chat Agent with OpenAI LLM Integration
# Author: WicketWise Team, Last Modified: 2025-08-16

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI
import os

from .kg_query_engine import KGQueryEngine
from .function_tools import get_function_tools, get_function_descriptions

logger = logging.getLogger(__name__)


class KGChatAgent:
    """
    Knowledge Graph Chat Agent
    
    Integrates OpenAI LLM with cricket knowledge graph to provide
    intelligent responses to natural language queries about cricket data.
    
    Features:
    - OpenAI function calling for safe KG queries
    - Context-aware responses with current match info
    - Markdown formatting for rich responses
    - Chat history management
    """
    
    def __init__(self, graph_path: str = "models/cricket_knowledge_graph.pkl"):
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=api_key)
        
        # Initialize KG Query Engine (try unified first, fallback to legacy)
        try:
            from .unified_kg_query_engine import UnifiedKGQueryEngine
            unified_path = graph_path.replace('cricket_knowledge_graph.pkl', 'unified_cricket_kg.pkl')
            if Path(unified_path).exists():
                self.kg_engine = UnifiedKGQueryEngine(unified_path)
                logger.info("Using Unified Knowledge Graph Query Engine")
            else:
                from .kg_query_engine import KGQueryEngine
                self.kg_engine = KGQueryEngine(graph_path)
                logger.info("Using Legacy Knowledge Graph Query Engine")
        except ImportError:
            from .kg_query_engine import KGQueryEngine
            self.kg_engine = KGQueryEngine(graph_path)
            logger.info("Using Legacy Knowledge Graph Query Engine")
        
        # Get available function tools
        self.function_tools = get_function_tools()
        self.function_descriptions = get_function_descriptions()
        
        # System prompt for cricket expertise
        self.system_prompt = self._build_system_prompt()
        
        logger.info("KG Chat Agent initialized successfully")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with cricket knowledge and function descriptions"""
        
        # Get KG summary for context
        kg_summary = self.kg_engine.get_graph_summary()
        
        prompt = f"""You are WicketWise, an expert cricket analytics AI assistant with access to a comprehensive cricket knowledge graph.

KNOWLEDGE GRAPH OVERVIEW:
- Total Players: {kg_summary.get('node_types', {}).get('player', 0):,}
- Total Venues: {kg_summary.get('node_types', {}).get('venue', 0):,}
- Total Teams: {kg_summary.get('node_types', {}).get('team', 0):,}
- Total Matches: {kg_summary.get('node_types', {}).get('match', 0):,}
- Database Size: {kg_summary.get('total_nodes', 0):,} nodes, {kg_summary.get('total_edges', 0):,} edges

AVAILABLE FUNCTIONS:
You have access to these cricket analysis functions:
"""
        
        for func_name, description in self.function_descriptions.items():
            prompt += f"- {func_name}: {description}\n"
        
        prompt += """
RESPONSE GUIDELINES:
1. **Be Cricket Expert**: Use cricket terminology naturally and provide insightful analysis
2. **Use Functions Strategically**: Call functions to get specific data, then provide expert analysis
3. **Format Responses**: Use markdown for tables, lists, and emphasis
4. **Trust Knowledge Graph Data**: NEVER dismiss KG data as "incomplete" or "problematic" - always present the actual data returned
5. **Handle Data Types**: If you get bowling stats for a famous batsman, present them as valid bowling statistics for that player
6. **Be Conversational**: Respond naturally while being informative and helpful
7. **Provide Context**: Explain what the numbers mean in cricket terms
8. **Compare & Contrast**: When showing stats, provide perspective and comparisons
9. **Data Interpretation**: Look for "data_type" field to understand if stats are batting or bowling data

EXAMPLE RESPONSE PATTERNS:
- For player queries: Get stats, then analyze performance patterns, strengths, key venues
- For comparisons: Create markdown tables, highlight key differences, provide insights
- For venue queries: Discuss pitch characteristics, historical trends, team performances
- For team H2H: Show win-loss records, discuss key matchups, historical context

CURRENT CONTEXT:
- You have access to comprehensive cricket data across all formats
- Data includes career statistics, venue performance, team records, and match history
- Always prioritize accuracy and provide data-driven insights

Remember: You're not just showing data, you're providing expert cricket analysis and insights!"""

        return prompt
    
    def _execute_function_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function call on the KG Query Engine"""
        try:
            # Map function names to KG engine methods
            function_map = {
                'get_player_stats': self.kg_engine.get_player_stats,
                'compare_players': self.kg_engine.compare_players,
                'get_venue_history': getattr(self.kg_engine, 'get_venue_history', None),
                'get_head_to_head': getattr(self.kg_engine, 'get_head_to_head', None),
                'find_similar_players': getattr(self.kg_engine, 'find_similar_players', None),
                'get_graph_summary': self.kg_engine.get_graph_summary,
                'explain_data_limitations': self.kg_engine.explain_data_limitations,
                # New unified functions
                'get_complete_player_profile': getattr(self.kg_engine, 'get_complete_player_profile', None),
                'get_situational_analysis': getattr(self.kg_engine, 'get_situational_analysis', None),
                'compare_players_advanced': getattr(self.kg_engine, 'compare_players_advanced', None),
                'find_best_performers': getattr(self.kg_engine, 'find_best_performers', None)
            }
            
            if function_name not in function_map:
                return {"error": f"Unknown function: {function_name}"}
            
            # Execute the function
            func = function_map[function_name]
            
            # Check if function exists (for backwards compatibility)
            if func is None:
                return {"error": f"Function {function_name} not available in current KG engine"}
            
            # Enhanced logging
            logger.info(f"🎯 KG FUNCTION CALL: {function_name} with args: {arguments}")
            
            # Handle functions with no arguments
            if function_name in ['get_graph_summary']:
                result = func()
            else:
                result = func(**arguments)
            
            # Log result summary
            if isinstance(result, dict) and 'error' not in result:
                logger.info(f"📊 KG SUCCESS: {function_name} returned data successfully")
            elif isinstance(result, dict) and 'error' in result:
                logger.warning(f"⚠️ KG WARNING: {function_name} returned error: {result['error']}")
            else:
                logger.info(f"📊 KG RESULT: {function_name} returned: {str(result)[:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")
            return {"error": f"Function execution failed: {str(e)}"}
    
    def _format_function_result(self, result: Dict[str, Any], function_name: str) -> str:
        """Format function results for LLM consumption"""
        if "error" in result:
            return f"Error: {result['error']}"
        
        # Convert result to a formatted string
        try:
            # For complex results, provide structured summary
            if function_name == "compare_players" and "comparison" in result:
                formatted = f"Player Comparison Results ({result.get('metric', 'unknown')} metric):\n"
                for i, player_data in enumerate(result["comparison"], 1):
                    formatted += f"{i}. {player_data.get('player', 'Unknown')}: "
                    formatted += f"SR: {player_data.get('strike_rate', 0):.1f}, "
                    formatted += f"Avg: {player_data.get('average', 0):.1f}, "
                    formatted += f"Runs: {player_data.get('total_runs', 0)}, "
                    formatted += f"Matches: {player_data.get('matches', 0)}\n"
                return formatted
            
            elif function_name == "get_player_stats" and "career_stats" in result:
                stats = result["career_stats"]
                formatted = f"Player: {result.get('player', 'Unknown')}\n"
                formatted += f"Career Stats: {stats.get('total_runs', 0)} runs, "
                formatted += f"{stats.get('strike_rate', 0):.1f} SR, "
                formatted += f"{stats.get('average', 0):.1f} avg, "
                formatted += f"{stats.get('matches', 0)} matches\n"
                formatted += f"Venues: {', '.join(result.get('venues_played', [])[:5])}\n"
                formatted += f"Teams: {', '.join(result.get('teams_played_for', []))}\n"
                return formatted
            
            elif function_name == "get_venue_history" and "venue_stats" in result:
                stats = result["venue_stats"]
                formatted = f"Venue: {result.get('venue', 'Unknown')}\n"
                formatted += f"Total Matches: {stats.get('total_matches', 0)}, "
                formatted += f"Teams: {stats.get('teams_played', 0)}, "
                formatted += f"Avg Score: {stats.get('avg_score', 0):.1f}\n"
                if "characteristics" in result:
                    char = result["characteristics"]
                    formatted += f"Pitch Type: {char.get('pitch_type', 'unknown')}, "
                    formatted += f"Chase Success: {char.get('chase_success_rate', 0):.1f}%\n"
                return formatted
            
            elif function_name == "get_head_to_head":
                formatted = f"Head-to-Head: {result.get('team1', 'Team1')} vs {result.get('team2', 'Team2')}\n"
                formatted += f"Matches: {result.get('matches_played', 0)}, "
                formatted += f"{result.get('team1', 'Team1')}: {result.get('team1_wins', 0)} wins, "
                formatted += f"{result.get('team2', 'Team2')}: {result.get('team2_wins', 0)} wins\n"
                win_pct = result.get('win_percentage', {})
                formatted += f"Win %: {win_pct.get('team1', 0):.1f}% vs {win_pct.get('team2', 0):.1f}%\n"
                return formatted
            
            else:
                # Fallback to JSON representation
                return json.dumps(result, indent=2)
                
        except Exception as e:
            logger.warning(f"Error formatting result: {e}")
            return json.dumps(result, indent=2)
    
    def chat(self, user_message: str, chat_history: Optional[List[Dict[str, str]]] = None,
             current_match_context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[Dict[str, str]]]:
        """
        Process a chat message and return response with updated history
        
        Args:
            user_message: User's question/message
            chat_history: Previous chat messages
            current_match_context: Optional current match information
            
        Returns:
            Tuple of (response_message, updated_chat_history)
        """
        try:
            # Initialize chat history if not provided
            if chat_history is None:
                chat_history = []
            
            # Build messages for OpenAI
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add current match context if available
            if current_match_context:
                context_msg = f"CURRENT MATCH CONTEXT: {json.dumps(current_match_context, indent=2)}"
                messages.append({"role": "system", "content": context_msg})
            
            # Add chat history
            for msg in chat_history[-10:]:  # Keep last 10 messages for context
                messages.append(msg)
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Call OpenAI with function calling
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use gpt-4o for mixed modality and speed
                messages=messages,
                tools=self.function_tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=2000
            )
            
            # Process the response
            assistant_message = response.choices[0].message
            
            # Handle function calls
            if assistant_message.tool_calls:
                # Execute function calls
                function_results = []
                kg_functions_used = []
                
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Track KG functions used for debug display
                    kg_functions_used.append(f"{function_name}({', '.join(f'{k}={v}' for k, v in function_args.items())})")
                    
                    # Execute the function
                    result = self._execute_function_call(function_name, function_args)
                    formatted_result = self._format_function_result(result, function_name)
                    
                    function_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": formatted_result
                    })
                
                # Add function call and results to messages
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls
                })
                
                for result in function_results:
                    messages.append(result)
                
                # Get final response from OpenAI
                final_response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                final_message = final_response.choices[0].message.content
                
                # Add KG debug info to response
                if kg_functions_used:
                    kg_debug = f"\n\n---\n**🔍 Knowledge Graph Insights Used:**\n" + "\n".join(f"• `{func}`" for func in kg_functions_used)
                    final_message = final_message + kg_debug
                
            else:
                # No function calls, use the direct response
                final_message = assistant_message.content
                
                # Add debug info for AI-only responses
                final_message = final_message + "\n\n---\n**🤖 Response Source:** AI General Knowledge (no KG functions called)"
            
            # Update chat history
            updated_history = chat_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": final_message}
            ]
            
            # Keep history manageable (last 20 messages)
            if len(updated_history) > 20:
                updated_history = updated_history[-20:]
            
            logger.info(f"Chat response generated for: {user_message[:50]}...")
            return final_message, updated_history
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            error_message = "I apologize, but I encountered an error processing your request. Please try rephrasing your question or ask about something else."
            
            # Still update history with error
            updated_history = chat_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": error_message}
            ]
            
            return error_message, updated_history
    
    def get_suggested_questions(self) -> List[str]:
        """Get a list of suggested questions users can ask"""
        return [
            "How does Virat Kohli perform at the MCG?",
            "Compare MS Dhoni and Jos Buttler's T20 stats",
            "What's the head-to-head record between India and Australia?",
            "Tell me about Eden Gardens' pitch characteristics",
            "Find players similar to AB de Villiers in strike rate",
            "Which venues has Rohit Sharma performed best at?",
            "Compare England vs New Zealand in T20 matches",
            "Who are the top performers at Lord's?",
            "Show me Jasprit Bumrah's bowling statistics",
            "What's the chase success rate at Wankhede Stadium?"
        ]
    
    def get_available_functions(self) -> Dict[str, str]:
        """Get list of available functions for the UI"""
        return self.function_descriptions

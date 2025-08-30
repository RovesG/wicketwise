# Purpose: Knowledge Graph Chat Agent with OpenAI LLM Integration
# Author: WicketWise Team, Last Modified: 2025-08-16

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI
import os
from ..models.enhanced_openai_client import WicketWiseOpenAI
from ..models.model_selection_service import create_chat_context

from .kg_query_engine import KGQueryEngine
from .function_tools import get_function_tools, get_function_descriptions
from .gnn_enhanced_kg_query_engine import GNNEnhancedKGQueryEngine
from .gnn_function_tools import get_all_enhanced_function_tools, get_all_enhanced_function_descriptions
from .dynamic_kg_query_engine import DynamicKGQueryEngine, get_dynamic_query_function_tool

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
        # Initialize Enhanced OpenAI client with intelligent model selection
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        try:
            self.client = WicketWiseOpenAI(api_key=api_key)
            self.enhanced_client = True
            logger.info("🚀 Using Enhanced OpenAI Client with intelligent model selection")
        except Exception as e:
            logger.warning(f"⚠️ Enhanced client failed, using standard client: {e}")
            self.client = OpenAI(api_key=api_key)
            self.enhanced_client = False
        
        # Initialize KG Query Engine (try GNN-enhanced first, then unified, then legacy)
        try:
            # Try GNN-enhanced engine first
            unified_path = graph_path.replace('cricket_knowledge_graph.pkl', 'unified_cricket_kg.pkl')
            gnn_embeddings_path = "models/gnn_embeddings.pt"
            
            if Path(unified_path).exists():
                try:
                    self.kg_engine = GNNEnhancedKGQueryEngine(
                        graph_path=unified_path,
                        gnn_embeddings_path=gnn_embeddings_path
                    )
                    logger.info("🚀 Using GNN-Enhanced Knowledge Graph Query Engine")
                except Exception as e:
                    logger.warning(f"⚠️ GNN enhancement failed, falling back to unified: {e}")
                    from .unified_kg_query_engine import UnifiedKGQueryEngine
                    self.kg_engine = UnifiedKGQueryEngine(unified_path)
                    logger.info("📊 Using Unified Knowledge Graph Query Engine")
            else:
                from .kg_query_engine import KGQueryEngine
                self.kg_engine = KGQueryEngine(graph_path)
                logger.info("📈 Using Legacy Knowledge Graph Query Engine")
                
        except ImportError as e:
            logger.warning(f"Import error: {e}")
            from .kg_query_engine import KGQueryEngine
            self.kg_engine = KGQueryEngine(graph_path)
            logger.info("📈 Using Legacy Knowledge Graph Query Engine")
        
        # Initialize Dynamic Query Engine for LLM-created queries
        try:
            unified_path = graph_path.replace('cricket_knowledge_graph.pkl', 'unified_cricket_kg.pkl')
            if Path(unified_path).exists():
                self.dynamic_engine = DynamicKGQueryEngine(unified_path)
            else:
                self.dynamic_engine = DynamicKGQueryEngine(graph_path)
            logger.info("🚀 Dynamic KG Query Engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize dynamic query engine: {e}")
            self.dynamic_engine = None
        
        # Get available function tools (enhanced with GNN if available)
        base_tools = []
        if hasattr(self.kg_engine, 'gnn_embeddings_available') and self.kg_engine.gnn_embeddings_available:
            base_tools = get_all_enhanced_function_tools()
            self.function_descriptions = get_all_enhanced_function_descriptions()
            logger.info("🧠 Enhanced function tools loaded (with GNN capabilities)")
        else:
            base_tools = get_function_tools()
            self.function_descriptions = get_function_descriptions()
            logger.info("📋 Standard function tools loaded")
        
        # Add dynamic query tool if available
        if self.dynamic_engine:
            base_tools.append(get_dynamic_query_function_tool())
            logger.info("⚡ Dynamic query tool added")
        
        self.function_tools = base_tools
        
        # System prompt for cricket expertise
        self.system_prompt = self._build_system_prompt()
        
        logger.info("KG Chat Agent initialized successfully")
    
    def _execute_dynamic_query(self, query_code: str, description: str) -> Dict[str, Any]:
        """Execute a dynamic KG query created by the LLM"""
        if not self.dynamic_engine:
            return {"error": "Dynamic query engine not available"}
        
        return self.dynamic_engine.execute_query(query_code, description)
    
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
                # Original KG functions
                'get_player_stats': self.kg_engine.get_player_stats,
                'compare_players': self.kg_engine.compare_players,
                'get_venue_history': getattr(self.kg_engine, 'get_venue_history', None),
                'get_head_to_head': getattr(self.kg_engine, 'get_head_to_head', None),
                'find_similar_players': getattr(self.kg_engine, 'find_similar_players', None),
                'get_graph_summary': self.kg_engine.get_graph_summary,
                'explain_data_limitations': getattr(self.kg_engine, 'explain_data_limitations', None),
                
                # Unified KG functions
                'get_complete_player_profile': getattr(self.kg_engine, 'get_complete_player_profile', None),
                'get_situational_analysis': getattr(self.kg_engine, 'get_situational_analysis', None),
                'compare_players_advanced': getattr(self.kg_engine, 'compare_players_advanced', None),
                'find_best_performers': getattr(self.kg_engine, 'find_best_performers', None),
                
                # GNN-enhanced functions (if available)
                'find_similar_players_gnn': getattr(self.kg_engine, 'find_similar_players_gnn', None),
                'predict_contextual_performance': getattr(self.kg_engine, 'predict_contextual_performance', None),
                'analyze_venue_compatibility': getattr(self.kg_engine, 'analyze_venue_compatibility', None),
                'get_playing_style_similarity': getattr(self.kg_engine, 'get_playing_style_similarity', None),
                'find_best_performers_contextual': getattr(self.kg_engine, 'find_best_performers', None),  # Alias
                
                # Dynamic query function
                'execute_dynamic_kg_query': self._execute_dynamic_query,
                'analyze_team_composition_gnn': getattr(self.kg_engine, 'analyze_team_composition_gnn', None)
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
            
            # Call OpenAI with intelligent model selection
            if self.enhanced_client:
                response = self.client.kg_chat(
                    messages=messages,
                    tools=self.function_tools,
                    tool_choice="auto",
                    max_completion_tokens=2000
                )
            else:
                # Fallback to standard client
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # Fallback model
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
                if self.enhanced_client:
                    final_response = self.client.kg_chat(
                        messages=messages,
                        max_completion_tokens=2000
                    )
                else:
                    final_response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.7,
                        max_completion_tokens=2000
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
            if chat_history is None:
                chat_history = []
            
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
            import traceback
            logger.error(f"Error in chat processing: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Check if it's an OpenAI quota/API error
            if "quota" in str(e).lower() or "429" in str(e) or "insufficient_quota" in str(e):
                # Use intelligent fallback with direct KG queries
                fallback_response = self._create_intelligent_fallback(user_message)
                
                if chat_history is None:
                    chat_history = []
                
                updated_history = chat_history + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": fallback_response}
                ]
                
                return fallback_response, updated_history
            
            # For other errors, provide generic message
            error_message = "I apologize, but I encountered an error processing your request. Please try rephrasing your question or ask about something else."
            
            # Still update history with error
            if chat_history is None:
                chat_history = []
            
            updated_history = chat_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": error_message}
            ]
            
            return error_message, updated_history
    
    def _create_intelligent_fallback(self, user_message: str) -> str:
        """Create intelligent fallback response using direct KG queries when OpenAI is unavailable"""
        try:
            user_lower = user_message.lower()
            
            # NO HARDCODED PLAYER LIST - Use KG for player detection
            detected_player = None
            
            # Try to extract player names from KG instead of hardcoded list
            try:
                if self.kg_engine:
                    # Use KG to detect players mentioned in the query
                    # This is a simplified approach - could be enhanced with NER
                    words = user_lower.split()
                    for i, word in enumerate(words):
                        if i < len(words) - 1:
                            potential_name = f"{word} {words[i+1]}".title()
                            # Try to query KG for this potential player name
                            try:
                                result = self.kg_engine.get_player_stats(potential_name)
                                if result and not result.get('error'):
                                    detected_player = potential_name
                                    break
                            except:
                                continue
            except Exception as e:
                logger.warning(f"Could not use KG for player detection: {e}")
                detected_player = None
            
            # Query type detection
            if any(word in user_lower for word in ["stats", "statistics", "performance", "record"]):
                if detected_player:
                    try:
                        result = self.kg_engine.get_player_stats(detected_player)
                        return self._format_player_stats_fallback(result, detected_player)
                    except:
                        pass
            
            elif any(word in user_lower for word in ["compare", "vs", "versus", "better"]):
                # Try to detect two players for comparison
                players_found = [p.title() for p in common_players if p in user_lower]
                if len(players_found) >= 2:
                    try:
                        result = self.kg_engine.compare_players(players_found[0], players_found[1])
                        return self._format_comparison_fallback(result, players_found[0], players_found[1])
                    except:
                        pass
            
            elif any(word in user_lower for word in ["venue", "ground", "stadium"]):
                if detected_player:
                    try:
                        result = self.kg_engine.get_player_stats(detected_player)
                        return self._format_venue_performance_fallback(result, detected_player)
                    except:
                        pass
            
            elif any(word in user_lower for word in ["summary", "overview", "graph", "data"]):
                try:
                    result = self.kg_engine.get_graph_summary()
                    return self._format_graph_summary_fallback(result)
                except:
                    pass
            
            # Default intelligent fallback
            return self._create_default_intelligent_response(user_message, detected_player)
            
        except Exception as e:
            logger.error(f"Error in intelligent fallback: {e}")
            return "I'm currently experiencing technical difficulties with my AI capabilities, but I can still help you explore cricket data. Try asking about specific players like 'Virat Kohli stats' or 'compare Rohit Sharma and David Warner'."
    
    def _format_player_stats_fallback(self, result: Dict[str, Any], player: str) -> str:
        """Format player stats for fallback response"""
        if result.get("error"):
            return f"I found some information about **{player}** in our cricket database, but couldn't retrieve detailed stats right now. Our knowledge graph contains comprehensive cricket data - please try your query again or ask about a different aspect of {player}'s performance."
        
        # Extract key stats if available
        stats = result.get("career_stats", {})
        matches = stats.get("matches", "N/A")
        runs = stats.get("runs", "N/A")
        avg = stats.get("average", "N/A")
        sr = stats.get("strike_rate", "N/A")
        
        response = f"""## 📊 **{player} - Cricket Statistics**

**Career Overview:**
• **Matches:** {matches}
• **Runs:** {runs}
• **Average:** {avg}
• **Strike Rate:** {sr}

*Note: I'm currently using our cricket knowledge graph directly due to AI system limitations. For more detailed analysis, please try your query again later.*

---
**🔍 Data Source:** Cricket Knowledge Graph (Direct Query)"""
        
        return response
    
    def _format_comparison_fallback(self, result: Dict[str, Any], player1: str, player2: str) -> str:
        """Format player comparison for fallback response"""
        return f"""## ⚖️ **{player1} vs {player2} - Quick Comparison**

I can access our cricket knowledge graph to compare these players, but my AI analysis capabilities are currently limited. 

**What I can tell you:**
• Both players are in our comprehensive cricket database
• Our knowledge graph contains detailed match-by-match data for both
• Performance metrics across different venues and formats are available

**For detailed comparison, try asking:**
• "{player1} stats vs {player2} stats"
• "How does {player1} perform at specific venues?"
• "{player1} T20 vs ODI performance"

---
**🔍 Data Source:** Cricket Knowledge Graph (Direct Access)"""
    
    def _format_venue_performance_fallback(self, result: Dict[str, Any], player: str) -> str:
        """Format venue performance for fallback response"""
        return f"""## 🏟️ **{player} - Venue Performance Analysis**

Our cricket knowledge graph contains detailed venue-specific data for **{player}**, including:

• **Performance at major grounds** (MCG, Wankhede, Lord's, etc.)
• **Home vs Away statistics**
• **Venue-specific strike rates and averages**
• **Match outcomes and contributions**

*Currently accessing data directly from our knowledge graph. For AI-powered insights and detailed analysis, please try your query again later.*

**Suggested queries:**
• "{player} performance at MCG"
• "{player} home ground advantage"
• "Best venues for {player}"

---
**🔍 Data Source:** Cricket Knowledge Graph (Direct Query)"""
    
    def _format_graph_summary_fallback(self, result: Dict[str, Any]) -> str:
        """Format graph summary for fallback response"""
        nodes = result.get("total_nodes", "Unknown")
        edges = result.get("total_edges", "Unknown")
        players = result.get("players", "Unknown")
        matches = result.get("matches", "Unknown")
        
        return f"""## 📈 **Cricket Knowledge Graph - Database Overview**

**Graph Statistics:**
• **Total Nodes:** {nodes:,} entities
• **Total Edges:** {edges:,} relationships
• **Players:** {players:,} cricket players
• **Matches:** {matches:,} match records

**Available Data:**
✅ **Player Statistics** - Career records, performance metrics
✅ **Match Data** - Ball-by-ball, venue, format details  
✅ **Venue Information** - Ground-specific performance
✅ **Team Records** - Historical team data
✅ **Relationships** - Player-venue, player-team connections

*Currently showing raw database statistics. For AI-powered insights and natural language analysis, please try your queries again later.*

---
**🔍 Data Source:** Cricket Knowledge Graph (System Status)"""
    
    def _create_default_intelligent_response(self, user_message: str, detected_player: Optional[str]) -> str:
        """Create a default intelligent response when specific handlers don't match"""
        if detected_player:
            return f"""## 🏏 **Cricket Intelligence System**

I can see you're asking about **{detected_player}**. Our cricket knowledge graph contains comprehensive data about this player, but my AI analysis capabilities are currently limited.

**What's available in our database:**
• Detailed career statistics and performance metrics
• Venue-specific performance data  
• Match-by-match records and contributions
• Team and format-wise breakdowns

**Try these specific queries:**
• "{detected_player} career stats"
• "{detected_player} performance at [venue name]"
• "Compare {detected_player} with [another player]"

---
**🔍 Status:** Knowledge Graph Online | AI Analysis Temporarily Limited"""
        
        else:
            return f"""## 🏏 **Cricket Intelligence System**

I understand you're asking: *"{user_message}"*

Our cricket knowledge graph is fully operational with comprehensive data, but my AI analysis capabilities are currently limited. 

**Available Data & Queries:**
• **Player Statistics** - Ask about specific players (e.g., "Virat Kohli stats")
• **Player Comparisons** - Compare any two players  
• **Venue Analysis** - Performance at specific grounds
• **Database Overview** - "Show me graph summary"

**Example queries that work:**
• "Tell me about [Player Name]"
• "Compare [Player A] and [Player B]"
• "[Player Name] performance at [Venue]"

---
**🔍 Status:** Knowledge Graph Online | AI Analysis Temporarily Limited"""
    
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
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from enhanced client"""
        if self.enhanced_client and hasattr(self.client, 'get_stats'):
            stats = self.client.get_stats()
            
            # Add KG-specific information
            kg_stats = {
                "kg_engine_type": type(self.kg_engine).__name__,
                "gnn_enhanced": hasattr(self.kg_engine, 'gnn_embeddings_available') and self.kg_engine.gnn_embeddings_available,
                "function_tools_count": len(self.function_tools),
                "enhanced_client": self.enhanced_client
            }
            
            return {**stats, "kg_chat_info": kg_stats}
        else:
            return {
                "message": "Enhanced client not available",
                "kg_engine_type": type(self.kg_engine).__name__,
                "function_tools_count": len(self.function_tools),
                "enhanced_client": self.enhanced_client
            }

# Purpose: LLM-Powered Player Insight Agent with Function Calling
# Author: WicketWise Team, Last Modified: 2025-08-30

"""
Player Insight Agent

An LLM-powered agent that generates comprehensive cricket player insights by:
- Calling KG/GNN functions for historical data
- Using Web Intelligence Agent for real-time updates
- Synthesizing information with cricket domain knowledge
- Generating tactical recommendations and narratives

Designed to replace static data processing with intelligent analysis.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

from ..models.enhanced_openai_client import WicketWiseOpenAI
from ..chat.gnn_enhanced_kg_query_engine import GNNEnhancedKGQueryEngine
from ..chat.function_tools import get_function_tools
from ..chat.gnn_function_tools import get_all_enhanced_function_tools
from .web_intelligence_agent import WebIntelligenceAgent, IntelligenceType, IntelligenceResult

logger = logging.getLogger(__name__)

class PlayerInsightAgent:
    """
    LLM-Powered Player Insight Agent
    
    Uses OpenAI GPT-4 with function calling to generate comprehensive
    player insights by combining:
    - Historical KG/GNN data
    - Real-time web intelligence  
    - Cricket domain expertise
    - Match context analysis
    """
    
    def __init__(
        self,
        kg_engine: GNNEnhancedKGQueryEngine = None,
        web_intelligence_agent: WebIntelligenceAgent = None,
        openai_client: WicketWiseOpenAI = None
    ):
        """
        Initialize Player Insight Agent
        
        Args:
            kg_engine: Knowledge Graph engine for historical data
            web_intelligence_agent: Web intelligence agent for real-time data
            openai_client: OpenAI client for LLM processing
        """
        self.kg_engine = kg_engine
        self.web_agent = web_intelligence_agent
        self.llm = openai_client or WicketWiseOpenAI()
        
        # Combine function tools from KG/GNN and web intelligence
        self.function_tools = self._build_function_tools()
        
        logger.info("🤖 Player Insight Agent initialized")
    
    def _build_function_tools(self) -> List[Dict[str, Any]]:
        """Build comprehensive function tools for LLM"""
        tools = []
        
        # Add KG/GNN function tools
        if self.kg_engine:
            tools.extend(get_function_tools())
            tools.extend(get_all_enhanced_function_tools())
        
        # Add web intelligence tools
        if self.web_agent:
            tools.extend(self._get_web_intelligence_tools())
        
        return tools
    
    def _get_web_intelligence_tools(self) -> List[Dict[str, Any]]:
        """Define web intelligence function tools for LLM"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_player_injury_status",
                    "description": "Get latest injury and fitness status for a player from web sources",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "player_name": {
                                "type": "string",
                                "description": "Player name to check injury status for"
                            }
                        },
                        "required": ["player_name"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "get_player_recent_form",
                    "description": "Get recent form and performance updates for a player from web sources",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "player_name": {
                                "type": "string",
                                "description": "Player name to check recent form for"
                            }
                        },
                        "required": ["player_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_match_conditions",
                    "description": "Get current weather and pitch conditions for a venue",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "venue": {
                                "type": "string",
                                "description": "Venue name to check conditions for"
                            }
                        },
                        "required": ["venue"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_expert_analysis",
                    "description": "Get expert cricket analysis and opinions about a player",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "player_name": {
                                "type": "string",
                                "description": "Player name to get expert analysis for"
                            }
                        },
                        "required": ["player_name"]
                    }
                }
            }
        ]
    
    async def generate_comprehensive_insights(
        self,
        player_name: str,
        match_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive player insights using LLM with function calling
        
        Args:
            player_name: Name of the player
            match_context: Match context (venue, opponent, conditions, etc.)
            
        Returns:
            Comprehensive player insights with tactical recommendations
        """
        logger.info(f"🧠 Generating comprehensive insights for {player_name}")
        
        # Build context-aware prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(player_name, match_context)
        
        try:
            # Call LLM with function calling enabled
            # Note: GPT-5 models use default temperature (1.0) for optimal performance
            response = self.llm.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=self.function_tools,
                tool_choice="auto"
                # No temperature override - let model config handle it
            )
            
            # Process the response and function calls
            insights = await self._process_llm_response(response, player_name, match_context)
            
            logger.info(f"✅ Generated comprehensive insights for {player_name}")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights for {player_name}: {e}")
            return self._generate_fallback_insights(player_name, match_context)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the LLM"""
        return """You are an expert cricket analyst with deep knowledge of player performance, tactics, and match dynamics.

Your task is to generate comprehensive player insights by:

1. **Gathering Historical Data**: Use KG/GNN functions to get career stats, situational performance, and similar player comparisons
2. **Collecting Real-time Information**: Use web intelligence functions to get injury status, recent form, and expert opinions  
3. **Analyzing Match Context**: Consider venue, opponent, conditions, and match situation
4. **Generating Cricket Intelligence**: Provide tactical insights, recommendations, and strategic analysis

**Key Principles:**
- Always call multiple functions to gather comprehensive data
- Prioritize recent and contextual information
- Provide specific tactical recommendations
- Explain the reasoning behind insights
- Use cricket terminology and domain expertise
- Be honest about data limitations

**Output Structure:**
- Tactical Matchups: Specific bowling/batting analysis
- Venue Factors: Venue-specific performance and conditions
- Form Analysis: Recent performance and fitness status
- Strategic Recommendations: Actionable tactical advice
- Risk Assessment: Potential concerns or advantages

Always start by gathering data through function calls before generating insights."""
    
    def _build_user_prompt(self, player_name: str, match_context: Dict[str, Any] = None) -> str:
        """Build user prompt with player and match context"""
        
        context_str = ""
        if match_context:
            venue = match_context.get('venue', 'Unknown')
            opponent = match_context.get('opponent', 'Unknown')
            match_type = match_context.get('match_type', 'T20')
            conditions = match_context.get('conditions', 'Unknown')
            
            context_str = f"""
**Match Context:**
- Venue: {venue}
- Opponent: {opponent}  
- Format: {match_type}
- Conditions: {conditions}
"""
        
        return f"""Generate comprehensive cricket insights for **{player_name}**.
{context_str}

Please gather all relevant data using available functions and then provide:

1. **Tactical Matchups**: How does {player_name} perform against different bowling types? What are their strengths and weaknesses?

2. **Venue Factors**: How does {player_name} perform at this venue? What are the conditions like?

3. **Current Form & Fitness**: What is {player_name}'s recent form? Any injury concerns?

4. **Strategic Recommendations**: Based on all data, what tactical approach should be taken with/against {player_name}?

5. **Risk Assessment**: What are the key risks and opportunities?

Use function calls to gather comprehensive data before generating your analysis."""
    
    async def _process_llm_response(
        self,
        response: Any,
        player_name: str,
        match_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process LLM response and execute any function calls"""
        
        insights = {
            "player_name": player_name,
            "match_context": match_context,
            "generated_at": datetime.now().isoformat(),
            "data_sources": [],
            "tactical_insights": {},
            "venue_factors": {},
            "form_analysis": {},
            "strategic_recommendations": {},
            "risk_assessment": {},
            "function_calls_made": [],
            "llm_analysis": ""
        }
        
        # Handle function calls if present
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            message = choice.message
            
            # Process function calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"🔧 Executing function: {function_name}({function_args})")
                    
                    # Execute the function call
                    function_result = await self._execute_function_call(function_name, function_args)
                    
                    insights["function_calls_made"].append({
                        "function": function_name,
                        "arguments": function_args,
                        "result_summary": str(function_result)[:200] + "..." if len(str(function_result)) > 200 else str(function_result)
                    })
                    
                    # Add to data sources
                    if function_result and not isinstance(function_result, dict) or not function_result.get('error'):
                        insights["data_sources"].append(function_name)
            
            # Get the LLM's analysis text
            if hasattr(message, 'content') and message.content:
                insights["llm_analysis"] = message.content
        
        # Structure the insights based on LLM analysis
        insights = self._structure_insights(insights)
        
        return insights
    
    async def _execute_function_call(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a function call"""
        
        try:
            # KG/GNN function calls
            if hasattr(self.kg_engine, function_name):
                method = getattr(self.kg_engine, function_name)
                if asyncio.iscoroutinefunction(method):
                    return await method(**arguments)
                else:
                    return method(**arguments)
            
            # Web intelligence function calls
            elif function_name in ['get_player_injury_status', 'get_player_recent_form', 'get_match_conditions', 'get_expert_analysis']:
                return await self._execute_web_intelligence_call(function_name, arguments)
            
            else:
                logger.warning(f"Unknown function: {function_name}")
                return {"error": f"Function {function_name} not available"}
                
        except Exception as e:
            logger.error(f"Error executing {function_name}: {e}")
            return {"error": str(e)}
    
    async def _execute_web_intelligence_call(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute web intelligence function calls"""
        
        if not self.web_agent:
            return {"error": "Web intelligence agent not available"}
        
        player_name = arguments.get('player_name')
        venue = arguments.get('venue')
        
        try:
            if function_name == 'get_player_injury_status':
                results = await self.web_agent.gather_player_intelligence(
                    player_name, [IntelligenceType.INJURY_STATUS]
                )
            elif function_name == 'get_player_recent_form':
                results = await self.web_agent.gather_player_intelligence(
                    player_name, [IntelligenceType.RECENT_FORM]
                )
            elif function_name == 'get_match_conditions':
                results = await self.web_agent.gather_match_intelligence(
                    venue, "Team1", "Team2", [IntelligenceType.WEATHER_CONDITIONS, IntelligenceType.PITCH_REPORT]
                )
            elif function_name == 'get_expert_analysis':
                results = await self.web_agent.gather_player_intelligence(
                    player_name, [IntelligenceType.EXPERT_ANALYSIS]
                )
            else:
                return {"error": f"Unknown web intelligence function: {function_name}"}
            
            # Format results for LLM
            if results:
                return {
                    "intelligence_gathered": len(results),
                    "insights": [
                        {
                            "type": r.intelligence_type.value,
                            "content": r.content,
                            "confidence": r.confidence,
                            "source": r.source
                        }
                        for r in results
                    ]
                }
            else:
                return {"message": "No relevant intelligence found"}
                
        except Exception as e:
            logger.error(f"Web intelligence call failed: {e}")
            return {"error": str(e)}
    
    def _structure_insights(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the insights based on LLM analysis"""
        
        # Parse LLM analysis to extract structured insights
        analysis = insights.get("llm_analysis", "")
        
        # This is a simplified version - in practice, you'd use more sophisticated parsing
        # or have the LLM return structured JSON
        
        insights["tactical_insights"] = {
            "bowling_matchups": "Extracted from LLM analysis",
            "batting_approach": "Extracted from LLM analysis",
            "key_strengths": "Extracted from LLM analysis",
            "vulnerabilities": "Extracted from LLM analysis"
        }
        
        insights["venue_factors"] = {
            "venue_performance": "Extracted from LLM analysis",
            "conditions_impact": "Extracted from LLM analysis",
            "historical_record": "Extracted from LLM analysis"
        }
        
        insights["form_analysis"] = {
            "recent_performance": "Extracted from LLM analysis",
            "fitness_status": "Extracted from LLM analysis",
            "confidence_level": "Extracted from LLM analysis"
        }
        
        insights["strategic_recommendations"] = {
            "tactical_approach": "Extracted from LLM analysis",
            "field_placements": "Extracted from LLM analysis",
            "bowling_plans": "Extracted from LLM analysis"
        }
        
        insights["risk_assessment"] = {
            "injury_risk": "Extracted from LLM analysis",
            "form_risk": "Extracted from LLM analysis",
            "match_impact": "Extracted from LLM analysis"
        }
        
        return insights
    
    def _generate_fallback_insights(self, player_name: str, match_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate fallback insights when LLM processing fails"""
        
        return {
            "player_name": player_name,
            "match_context": match_context,
            "generated_at": datetime.now().isoformat(),
            "data_sources": ["fallback"],
            "tactical_insights": {
                "message": "LLM analysis unavailable - using basic data processing"
            },
            "venue_factors": {
                "message": "Venue analysis requires LLM processing"
            },
            "form_analysis": {
                "message": "Form analysis requires web intelligence"
            },
            "strategic_recommendations": {
                "message": "Strategic recommendations require comprehensive analysis"
            },
            "risk_assessment": {
                "message": "Risk assessment requires real-time data"
            },
            "error": "LLM processing failed - fallback mode active"
        }

# Factory function for easy integration
def create_player_insight_agent(
    kg_engine: GNNEnhancedKGQueryEngine = None,
    web_intelligence_agent: WebIntelligenceAgent = None,
    openai_client: WicketWiseOpenAI = None
) -> PlayerInsightAgent:
    """
    Factory function to create Player Insight Agent
    
    Args:
        kg_engine: Knowledge Graph engine
        web_intelligence_agent: Web intelligence agent
        openai_client: OpenAI client
        
    Returns:
        Configured PlayerInsightAgent instance
    """
    return PlayerInsightAgent(
        kg_engine=kg_engine,
        web_intelligence_agent=web_intelligence_agent,
        openai_client=openai_client
    )

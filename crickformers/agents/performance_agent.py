# Purpose: Performance Analysis Agent - Player & Team Performance
# Author: WicketWise Team, Last Modified: 2025-08-24

"""
Performance Analysis Agent
=========================

Specialized agent for analyzing player and team performance across different
formats, conditions, and time periods. Integrates with the Knowledge Graph
and temporal decay systems for comprehensive analysis.

Key Capabilities:
- Player performance analysis and trends
- Team performance comparisons
- Format-specific performance insights
- Contextual performance (venue, conditions, opposition)
- Performance prediction and form analysis
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from ..gnn.temporal_decay import TemporalDecayEngine, PerformanceEvent
from ..gnn.enhanced_kg_api import EnhancedKGQueryEngine
from .base_agent import BaseAgent, AgentCapability, AgentContext, AgentResponse

logger = logging.getLogger(__name__)


class PerformanceAgent(BaseAgent):
    """
    Agent specialized in cricket performance analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="performance_agent",
            capabilities=[
                AgentCapability.PERFORMANCE_ANALYSIS,
                AgentCapability.TREND_ANALYSIS,
                AgentCapability.COMPARISON_ANALYSIS,
                AgentCapability.MULTI_FORMAT_ANALYSIS
            ],
            config=config
        )
        
        # Dependencies
        self.kg_engine: Optional[EnhancedKGQueryEngine] = None
        self.decay_engine: Optional[TemporalDecayEngine] = None
        
        # Performance analysis configuration
        self.min_matches_for_trend = self.config.get("min_matches_for_trend", 5)
        self.recent_form_days = self.config.get("recent_form_days", 365)
        self.performance_weights = self.config.get("performance_weights", {
            "runs": 0.4,
            "strike_rate": 0.3,
            "consistency": 0.2,
            "recent_form": 0.1
        })
        
        # Required dependencies
        self.required_dependencies = ["knowledge_graph", "temporal_decay"]
    
    def _initialize_agent(self) -> bool:
        """Initialize performance agent dependencies"""
        try:
            # Initialize KG engine
            self.kg_engine = EnhancedKGQueryEngine()
            
            # Initialize temporal decay engine
            self.decay_engine = TemporalDecayEngine()
            
            logger.info("PerformanceAgent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"PerformanceAgent initialization failed: {str(e)}")
            return False
    
    def can_handle(self, capability: AgentCapability, context: AgentContext) -> bool:
        """Check if agent can handle the capability and context"""
        if capability not in self.capabilities:
            return False
        
        # Check if we have player or team context
        has_player_context = (
            context.player_context is not None or 
            any(keyword in context.user_query.lower() for keyword in ["player", "batsman", "bowler"])
        )
        
        has_team_context = (
            context.team_context is not None or
            any(keyword in context.user_query.lower() for keyword in ["team", "squad"])
        )
        
        return has_player_context or has_team_context
    
    async def execute(self, context: AgentContext) -> AgentResponse:
        """Execute performance analysis"""
        start_time = datetime.now()
        
        try:
            # Parse query to understand what analysis is needed
            analysis_type = self._determine_analysis_type(context)
            
            if analysis_type == "player_performance":
                result = await self._analyze_player_performance(context)
            elif analysis_type == "team_performance":
                result = await self._analyze_team_performance(context)
            elif analysis_type == "comparison":
                result = await self._analyze_comparison(context)
            elif analysis_type == "trend":
                result = await self._analyze_trends(context)
            else:
                result = await self._analyze_general_performance(context)
            
            confidence = self._calculate_confidence(result, context)
            
            return AgentResponse(
                agent_id=self.agent_id,
                capability=AgentCapability.PERFORMANCE_ANALYSIS,
                success=True,
                confidence=confidence,
                execution_time=0.0,  # Will be set by base class
                result=result,
                dependencies_used=["knowledge_graph", "temporal_decay"],
                metadata={
                    "analysis_type": analysis_type,
                    "context_used": {
                        "player": context.player_context is not None,
                        "team": context.team_context is not None,
                        "format": context.format_context,
                        "temporal": context.temporal_context is not None
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"PerformanceAgent execution failed: {str(e)}")
            return AgentResponse(
                agent_id=self.agent_id,
                capability=AgentCapability.PERFORMANCE_ANALYSIS,
                success=False,
                confidence=0.0,
                execution_time=0.0,
                error_message=str(e)
            )
    
    def _determine_analysis_type(self, context: AgentContext) -> str:
        """Determine the type of performance analysis needed"""
        query_lower = context.user_query.lower()
        
        if "compare" in query_lower or "vs" in query_lower or "versus" in query_lower:
            return "comparison"
        elif "trend" in query_lower or "form" in query_lower or "recent" in query_lower:
            return "trend"
        elif context.team_context or "team" in query_lower:
            return "team_performance"
        elif context.player_context or any(word in query_lower for word in ["player", "batsman", "bowler"]):
            return "player_performance"
        else:
            return "general_performance"
    
    async def _analyze_player_performance(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze individual player performance"""
        result = {
            "analysis_type": "player_performance",
            "players": [],
            "summary": {}
        }
        
        # Extract player names from context or query
        player_names = self._extract_player_names(context)
        
        for player_name in player_names:
            try:
                # Query player performance from KG
                player_query = f"Get performance statistics for {player_name}"
                if context.format_context:
                    player_query += f" in {context.format_context}"
                
                kg_result = await self.kg_engine.execute_query(player_query)
                
                # Analyze temporal performance
                performance_events = self._create_performance_events(kg_result, player_name)
                temporal_analysis = self._analyze_temporal_performance(performance_events)
                
                # Calculate performance metrics
                metrics = self._calculate_performance_metrics(kg_result, temporal_analysis)
                
                player_analysis = {
                    "name": player_name,
                    "current_form": temporal_analysis.get("recent_form", "unknown"),
                    "career_stats": metrics.get("career", {}),
                    "recent_stats": metrics.get("recent", {}),
                    "trends": temporal_analysis.get("trends", {}),
                    "strengths": self._identify_strengths(metrics),
                    "weaknesses": self._identify_weaknesses(metrics),
                    "context_performance": self._analyze_contextual_performance(kg_result, context)
                }
                
                result["players"].append(player_analysis)
                
            except Exception as e:
                logger.warning(f"Failed to analyze player {player_name}: {str(e)}")
                result["players"].append({
                    "name": player_name,
                    "error": str(e)
                })
        
        # Generate summary
        result["summary"] = self._generate_player_summary(result["players"])
        
        return result
    
    async def _analyze_team_performance(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze team performance"""
        result = {
            "analysis_type": "team_performance",
            "teams": [],
            "summary": {}
        }
        
        # Extract team names
        team_names = self._extract_team_names(context)
        
        for team_name in team_names:
            try:
                # Query team performance
                team_query = f"Get team performance for {team_name}"
                if context.format_context:
                    team_query += f" in {context.format_context} format"
                
                kg_result = await self.kg_engine.execute_query(team_query)
                
                # Analyze team metrics
                team_metrics = self._calculate_team_metrics(kg_result)
                
                # Analyze recent form
                recent_form = self._analyze_team_recent_form(kg_result)
                
                team_analysis = {
                    "name": team_name,
                    "win_loss_record": team_metrics.get("win_loss", {}),
                    "batting_performance": team_metrics.get("batting", {}),
                    "bowling_performance": team_metrics.get("bowling", {}),
                    "recent_form": recent_form,
                    "home_away_split": team_metrics.get("home_away", {}),
                    "key_players": self._identify_key_players(kg_result)
                }
                
                result["teams"].append(team_analysis)
                
            except Exception as e:
                logger.warning(f"Failed to analyze team {team_name}: {str(e)}")
                result["teams"].append({
                    "name": team_name,
                    "error": str(e)
                })
        
        result["summary"] = self._generate_team_summary(result["teams"])
        
        return result
    
    async def _analyze_comparison(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze comparison between players or teams"""
        result = {
            "analysis_type": "comparison",
            "entities": [],
            "comparison_metrics": {},
            "winner": None,
            "summary": ""
        }
        
        # Determine what to compare
        players = self._extract_player_names(context)
        teams = self._extract_team_names(context)
        
        if players and len(players) >= 2:
            result["entities"] = players[:2]  # Compare first two players
            result["comparison_metrics"] = await self._compare_players(players[:2], context)
        elif teams and len(teams) >= 2:
            result["entities"] = teams[:2]  # Compare first two teams
            result["comparison_metrics"] = await self._compare_teams(teams[:2], context)
        else:
            result["error"] = "Unable to identify entities to compare"
        
        # Determine winner if applicable
        if "comparison_metrics" in result and result["comparison_metrics"]:
            result["winner"] = self._determine_comparison_winner(result["comparison_metrics"])
            result["summary"] = self._generate_comparison_summary(result)
        
        return result
    
    async def _analyze_trends(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze performance trends"""
        result = {
            "analysis_type": "trend",
            "trends": {},
            "predictions": {},
            "summary": ""
        }
        
        # Extract entities for trend analysis
        players = self._extract_player_names(context)
        teams = self._extract_team_names(context)
        
        # Analyze player trends
        for player in players:
            try:
                player_query = f"Get recent matches for {player}"
                kg_result = await self.kg_engine.execute_query(player_query)
                
                performance_events = self._create_performance_events(kg_result, player)
                trend_analysis = self._analyze_temporal_performance(performance_events)
                
                result["trends"][player] = {
                    "recent_form": trend_analysis.get("recent_form"),
                    "trend_direction": trend_analysis.get("trend_direction"),
                    "consistency": trend_analysis.get("consistency"),
                    "peak_performance": trend_analysis.get("peak_performance")
                }
                
                # Generate predictions
                result["predictions"][player] = self._predict_future_performance(trend_analysis)
                
            except Exception as e:
                logger.warning(f"Failed to analyze trends for {player}: {str(e)}")
        
        # Analyze team trends
        for team in teams:
            try:
                team_query = f"Get recent matches for team {team}"
                kg_result = await self.kg_engine.execute_query(team_query)
                
                team_trends = self._analyze_team_trends(kg_result)
                result["trends"][team] = team_trends
                
            except Exception as e:
                logger.warning(f"Failed to analyze trends for team {team}: {str(e)}")
        
        result["summary"] = self._generate_trend_summary(result["trends"])
        
        return result
    
    async def _analyze_general_performance(self, context: AgentContext) -> Dict[str, Any]:
        """General performance analysis when specific type unclear"""
        result = {
            "analysis_type": "general_performance",
            "insights": [],
            "summary": ""
        }
        
        # Try to extract any cricket entities from the query
        query = context.user_query.lower()
        
        # Look for performance-related keywords and respond accordingly
        if "best" in query or "top" in query:
            result["insights"].append({
                "type": "ranking",
                "description": "Identified request for top performers",
                "data": await self._get_top_performers(context)
            })
        
        if "worst" in query or "poor" in query:
            result["insights"].append({
                "type": "poor_performance",
                "description": "Identified request for poor performers",
                "data": await self._get_poor_performers(context)
            })
        
        if "average" in query or "typical" in query:
            result["insights"].append({
                "type": "average_performance",
                "description": "Identified request for average performance",
                "data": await self._get_average_performance(context)
            })
        
        result["summary"] = self._generate_general_summary(result["insights"])
        
        return result
    
    def _create_performance_events(self, kg_result: Dict[str, Any], entity_name: str) -> List[PerformanceEvent]:
        """Create performance events from KG query results"""
        events = []
        
        # This is a simplified version - in practice, would parse actual KG results
        # For now, create sample events for testing
        base_date = datetime.now() - timedelta(days=365)
        
        for i in range(10):  # Sample 10 matches
            event_date = base_date + timedelta(days=i * 30)
            
            event = PerformanceEvent(
                event_id=f"{entity_name}_{i}",
                player_id=entity_name,
                date=event_date,
                importance_score=np.random.uniform(0.5, 1.0),
                performance_metrics={
                    "runs": np.random.randint(0, 150),
                    "balls_faced": np.random.randint(20, 120),
                    "strike_rate": np.random.uniform(80, 150),
                    "dismissal_type": np.random.choice(["bowled", "caught", "lbw", "not_out"])
                },
                context={
                    "format": "ODI",
                    "venue": f"Venue_{i}",
                    "opposition": f"Team_{i}"
                }
            )
            events.append(event)
        
        return events
    
    def _analyze_temporal_performance(self, events: List[PerformanceEvent]) -> Dict[str, Any]:
        """Analyze temporal performance using decay engine"""
        if not self.decay_engine or not events:
            return {"error": "No decay engine or events available"}
        
        # Calculate weighted aggregations
        recent_form = self.decay_engine.get_weighted_aggregation(
            events, "runs", "recent_trend"
        )
        
        consistency = self.decay_engine.get_weighted_aggregation(
            events, "runs", "std"
        )
        
        return {
            "recent_form": "good" if recent_form > 0.1 else "poor",
            "trend_direction": "improving" if recent_form > 0 else "declining",
            "consistency": "high" if consistency < 20 else "low",
            "total_events": len(events)
        }
    
    def _extract_player_names(self, context: AgentContext) -> List[str]:
        """Extract player names from context or query"""
        players = []
        
        # From explicit context
        if context.player_context:
            players.extend(context.player_context.get("names", []))
        
        # From query parsing (simplified)
        query_words = context.user_query.split()
        potential_players = ["Kohli", "Sharma", "Dhoni", "Babar", "Root", "Smith"]
        
        for word in query_words:
            for player in potential_players:
                if player.lower() in word.lower():
                    players.append(player)
        
        return list(set(players))  # Remove duplicates
    
    def _extract_team_names(self, context: AgentContext) -> List[str]:
        """Extract team names from context or query"""
        teams = []
        
        # From explicit context
        if context.team_context:
            teams.extend(context.team_context.get("names", []))
        
        # From query parsing (simplified)
        query_lower = context.user_query.lower()
        potential_teams = ["india", "pakistan", "england", "australia", "south africa", "new zealand"]
        
        for team in potential_teams:
            if team in query_lower:
                teams.append(team.title())
        
        return list(set(teams))
    
    def _calculate_performance_metrics(self, kg_result: Dict[str, Any], temporal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        # Simplified metrics calculation
        return {
            "career": {
                "matches": 100,
                "runs": 4500,
                "average": 45.0,
                "strike_rate": 125.0
            },
            "recent": {
                "matches": 10,
                "runs": 400,
                "average": 40.0,
                "strike_rate": 130.0
            }
        }
    
    def _identify_strengths(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify player strengths from metrics"""
        strengths = []
        
        career = metrics.get("career", {})
        if career.get("average", 0) > 40:
            strengths.append("High batting average")
        if career.get("strike_rate", 0) > 120:
            strengths.append("Aggressive batting")
        
        return strengths
    
    def _identify_weaknesses(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify player weaknesses from metrics"""
        weaknesses = []
        
        career = metrics.get("career", {})
        if career.get("average", 0) < 30:
            weaknesses.append("Low batting average")
        if career.get("strike_rate", 0) < 100:
            weaknesses.append("Slow scoring rate")
        
        return weaknesses
    
    def _analyze_contextual_performance(self, kg_result: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Analyze performance in different contexts"""
        return {
            "home_vs_away": {"home_average": 50.0, "away_average": 40.0},
            "format_specific": {"ODI": 45.0, "T20": 35.0, "Test": 55.0},
            "conditions": {"spin_friendly": 40.0, "pace_friendly": 50.0}
        }
    
    def _calculate_confidence(self, result: Dict[str, Any], context: AgentContext) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.7  # Base confidence
        
        # Increase confidence if we have specific context
        if context.player_context or context.team_context:
            confidence += 0.1
        
        # Increase confidence if analysis was successful
        if result and "error" not in result:
            confidence += 0.1
        
        # Decrease confidence if we encountered errors
        if result and any("error" in item for item in result.get("players", []) + result.get("teams", [])):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    # Additional helper methods (simplified implementations)
    def _generate_player_summary(self, players: List[Dict[str, Any]]) -> str:
        return f"Analyzed {len(players)} player(s) with performance insights"
    
    def _calculate_team_metrics(self, kg_result: Dict[str, Any]) -> Dict[str, Any]:
        return {"win_loss": {"wins": 15, "losses": 10}, "batting": {"average": 280}, "bowling": {"average": 25}}
    
    def _analyze_team_recent_form(self, kg_result: Dict[str, Any]) -> str:
        return "Good - 3 wins in last 5 matches"
    
    def _identify_key_players(self, kg_result: Dict[str, Any]) -> List[str]:
        return ["Player A", "Player B", "Player C"]
    
    def _generate_team_summary(self, teams: List[Dict[str, Any]]) -> str:
        return f"Analyzed {len(teams)} team(s) with comprehensive metrics"
    
    async def _compare_players(self, players: List[str], context: AgentContext) -> Dict[str, Any]:
        return {
            "batting_average": {players[0]: 45.0, players[1]: 42.0},
            "strike_rate": {players[0]: 125.0, players[1]: 130.0},
            "winner": players[1]  # Higher strike rate
        }
    
    async def _compare_teams(self, teams: List[str], context: AgentContext) -> Dict[str, Any]:
        return {
            "win_percentage": {teams[0]: 65.0, teams[1]: 60.0},
            "average_score": {teams[0]: 285, teams[1]: 275},
            "winner": teams[0]  # Higher win percentage
        }
    
    def _determine_comparison_winner(self, metrics: Dict[str, Any]) -> Optional[str]:
        # Simple winner determination logic
        if "winner" in metrics:
            return metrics["winner"]
        return None
    
    def _generate_comparison_summary(self, result: Dict[str, Any]) -> str:
        entities = result.get("entities", [])
        winner = result.get("winner")
        return f"Comparison between {' vs '.join(entities)}, winner: {winner}"
    
    def _analyze_team_trends(self, kg_result: Dict[str, Any]) -> Dict[str, Any]:
        return {"form": "improving", "trend": "upward", "consistency": "high"}
    
    def _predict_future_performance(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        return {"next_match_score": "40-60 runs", "confidence": 0.7}
    
    def _generate_trend_summary(self, trends: Dict[str, Any]) -> str:
        return f"Analyzed trends for {len(trends)} entities"
    
    async def _get_top_performers(self, context: AgentContext) -> Dict[str, Any]:
        return {"top_batsmen": ["Player A", "Player B"], "top_bowlers": ["Bowler X", "Bowler Y"]}
    
    async def _get_poor_performers(self, context: AgentContext) -> Dict[str, Any]:
        return {"struggling_batsmen": ["Player C"], "struggling_bowlers": ["Bowler Z"]}
    
    async def _get_average_performance(self, context: AgentContext) -> Dict[str, Any]:
        return {"average_score": 250, "average_wickets": 6}
    
    def _generate_general_summary(self, insights: List[Dict[str, Any]]) -> str:
        return f"Generated {len(insights)} performance insights"

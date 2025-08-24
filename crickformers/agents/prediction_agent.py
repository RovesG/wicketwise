# Purpose: Prediction Agent - Match Outcome & Performance Predictions
# Author: WicketWise Team, Last Modified: 2025-08-24

"""
Prediction Agent
===============

Specialized agent for cricket match predictions, outcome forecasting,
and performance projections. Integrates with the MoE system and 
betting intelligence for comprehensive predictive analytics.

Key Capabilities:
- Match outcome predictions
- Player performance forecasts
- Score predictions and projections
- Win probability calculations
- Tournament outcome predictions
- Real-time prediction updates
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..model.mixture_of_experts import MixtureOfExperts, ModelType
from ..orchestration.moe_orchestrator import MoEOrchestrator
from ..gnn.enhanced_kg_api import EnhancedKGQueryEngine
from .base_agent import BaseAgent, AgentCapability, AgentContext, AgentResponse

logger = logging.getLogger(__name__)


class PredictionAgent(BaseAgent):
    """
    Agent specialized in cricket predictions and forecasting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="prediction_agent",
            capabilities=[
                AgentCapability.MATCH_PREDICTION,
                AgentCapability.TREND_ANALYSIS,
                AgentCapability.REAL_TIME_PROCESSING
            ],
            config=config
        )
        
        # Dependencies
        self.moe_orchestrator: Optional[MoEOrchestrator] = None
        self.kg_engine: Optional[EnhancedKGQueryEngine] = None
        
        # Prediction configuration
        self.prediction_types = [
            "match_winner", "total_score", "player_performance", 
            "tournament_winner", "series_outcome"
        ]
        
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        # Model preferences for different prediction types
        self.model_preferences = {
            "match_winner": ModelType.FAST,  # Quick decisions needed
            "total_score": ModelType.SLOW,  # Complex calculation
            "player_performance": ModelType.SLOW,  # Detailed analysis
            "real_time": ModelType.FAST  # Speed critical
        }
        
        # Required dependencies
        self.required_dependencies = ["moe_orchestrator", "knowledge_graph"]
    
    def _initialize_agent(self) -> bool:
        """Initialize prediction agent dependencies"""
        try:
            # Initialize MoE orchestrator
            self.moe_orchestrator = MoEOrchestrator()
            if not self.moe_orchestrator.initialize():
                logger.error("Failed to initialize MoE orchestrator")
                return False
            
            # Initialize KG engine
            self.kg_engine = EnhancedKGQueryEngine()
            
            logger.info("PredictionAgent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"PredictionAgent initialization failed: {str(e)}")
            return False
    
    def can_handle(self, capability: AgentCapability, context: AgentContext) -> bool:
        """Check if agent can handle the capability and context"""
        if capability not in self.capabilities:
            return False
        
        # Check for prediction keywords in query
        prediction_keywords = [
            "predict", "forecast", "outcome", "winner", "result",
            "probability", "chances", "likely", "expect", "projection"
        ]
        
        query_lower = context.user_query.lower()
        return any(keyword in query_lower for keyword in prediction_keywords)
    
    async def execute(self, context: AgentContext) -> AgentResponse:
        """Execute prediction analysis"""
        try:
            # Determine prediction type
            prediction_type = self._determine_prediction_type(context)
            
            # Gather relevant data
            prediction_data = await self._gather_prediction_data(context, prediction_type)
            
            # Generate predictions using appropriate models
            if prediction_type == "match_winner":
                result = await self._predict_match_winner(context, prediction_data)
            elif prediction_type == "total_score":
                result = await self._predict_total_score(context, prediction_data)
            elif prediction_type == "player_performance":
                result = await self._predict_player_performance(context, prediction_data)
            elif prediction_type == "tournament_winner":
                result = await self._predict_tournament_winner(context, prediction_data)
            elif prediction_type == "series_outcome":
                result = await self._predict_series_outcome(context, prediction_data)
            else:
                result = await self._predict_general_outcome(context, prediction_data)
            
            # Calculate overall confidence
            confidence = self._calculate_prediction_confidence(result, prediction_data)
            
            return AgentResponse(
                agent_id=self.agent_id,
                capability=AgentCapability.MATCH_PREDICTION,
                success=True,
                confidence=confidence,
                execution_time=0.0,
                result=result,
                dependencies_used=["moe_orchestrator", "knowledge_graph"],
                metadata={
                    "prediction_type": prediction_type,
                    "data_quality": self._assess_data_quality(prediction_data),
                    "model_used": result.get("model_info", {}).get("type", "unknown")
                }
            )
            
        except Exception as e:
            logger.error(f"PredictionAgent execution failed: {str(e)}")
            return AgentResponse(
                agent_id=self.agent_id,
                capability=AgentCapability.MATCH_PREDICTION,
                success=False,
                confidence=0.0,
                execution_time=0.0,
                error_message=str(e)
            )
    
    def _determine_prediction_type(self, context: AgentContext) -> str:
        """Determine the type of prediction needed"""
        query_lower = context.user_query.lower()
        
        if any(word in query_lower for word in ["winner", "win", "victory", "defeat"]):
            if "tournament" in query_lower or "cup" in query_lower:
                return "tournament_winner"
            elif "series" in query_lower:
                return "series_outcome"
            else:
                return "match_winner"
        elif any(word in query_lower for word in ["score", "runs", "total", "target"]):
            return "total_score"
        elif any(word in query_lower for word in ["player", "batsman", "bowler", "performance"]):
            return "player_performance"
        elif "tournament" in query_lower:
            return "tournament_winner"
        elif "series" in query_lower:
            return "series_outcome"
        else:
            return "general_outcome"
    
    async def _gather_prediction_data(self, context: AgentContext, prediction_type: str) -> Dict[str, Any]:
        """Gather relevant data for prediction"""
        data = {
            "teams": [],
            "players": [],
            "historical_data": {},
            "current_form": {},
            "conditions": {},
            "head_to_head": {},
            "venue_stats": {}
        }
        
        try:
            # Extract teams and players from context
            data["teams"] = self._extract_teams_from_context(context)
            data["players"] = self._extract_players_from_context(context)
            
            # Query historical data from KG
            if data["teams"]:
                for team in data["teams"]:
                    team_query = f"Get recent performance data for {team}"
                    team_data = await self.kg_engine.execute_query(team_query)
                    data["historical_data"][team] = team_data
            
            # Get head-to-head records
            if len(data["teams"]) >= 2:
                h2h_query = f"Get head to head record between {data['teams'][0]} and {data['teams'][1]}"
                data["head_to_head"] = await self.kg_engine.execute_query(h2h_query)
            
            # Get venue statistics if available
            if context.match_context and "venue" in context.match_context:
                venue = context.match_context["venue"]
                venue_query = f"Get venue statistics for {venue}"
                data["venue_stats"] = await self.kg_engine.execute_query(venue_query)
            
        except Exception as e:
            logger.warning(f"Failed to gather some prediction data: {str(e)}")
        
        return data
    
    async def _predict_match_winner(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict match winner using MoE system"""
        result = {
            "prediction_type": "match_winner",
            "predictions": {},
            "confidence_levels": {},
            "factors": {},
            "model_info": {}
        }
        
        try:
            # Prepare input for MoE system
            teams = data.get("teams", [])
            if len(teams) < 2:
                teams = ["Team A", "Team B"]  # Default for testing
            
            # Create feature vector for prediction
            features = self._create_match_features(data, context)
            
            # Get prediction from MoE system
            model_type = self.model_preferences.get("match_winner", ModelType.FAST)
            moe_result = await self._get_moe_prediction(features, model_type, "match_winner")
            
            # Process MoE results
            if moe_result and moe_result.get("predictions"):
                predictions = moe_result["predictions"]
                
                # Convert to team probabilities
                team_a_prob = predictions.get("team_a_win_prob", 0.5)
                team_b_prob = 1.0 - team_a_prob
                
                result["predictions"] = {
                    teams[0]: team_a_prob,
                    teams[1]: team_b_prob
                }
                
                # Determine winner
                winner = teams[0] if team_a_prob > team_b_prob else teams[1]
                winner_prob = max(team_a_prob, team_b_prob)
                
                result["predicted_winner"] = winner
                result["win_probability"] = winner_prob
                
                # Confidence levels
                if winner_prob > self.confidence_thresholds["high"]:
                    result["confidence_level"] = "high"
                elif winner_prob > self.confidence_thresholds["medium"]:
                    result["confidence_level"] = "medium"
                else:
                    result["confidence_level"] = "low"
                
                # Key factors
                result["factors"] = {
                    "recent_form": self._analyze_recent_form(data),
                    "head_to_head": self._analyze_head_to_head(data),
                    "venue_advantage": self._analyze_venue_advantage(data),
                    "team_strength": self._analyze_team_strength(data)
                }
                
                result["model_info"] = {
                    "type": model_type.value,
                    "latency_ms": moe_result.get("metadata", {}).get("latency_ms", 0),
                    "model_confidence": moe_result.get("metadata", {}).get("confidence", 0.5)
                }
            
        except Exception as e:
            logger.error(f"Match winner prediction failed: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    async def _predict_total_score(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict total score for innings/match"""
        result = {
            "prediction_type": "total_score",
            "score_predictions": {},
            "range_predictions": {},
            "factors": {},
            "model_info": {}
        }
        
        try:
            # Create features for score prediction
            features = self._create_score_features(data, context)
            
            # Use slow model for complex score calculations
            model_type = self.model_preferences.get("total_score", ModelType.SLOW)
            moe_result = await self._get_moe_prediction(features, model_type, "total_score")
            
            if moe_result and moe_result.get("predictions"):
                predictions = moe_result["predictions"]
                
                # Extract score predictions
                predicted_score = predictions.get("total_runs", 250)
                score_variance = predictions.get("score_variance", 30)
                
                result["score_predictions"] = {
                    "most_likely": int(predicted_score),
                    "optimistic": int(predicted_score + score_variance),
                    "conservative": int(predicted_score - score_variance)
                }
                
                # Score ranges
                result["range_predictions"] = {
                    "low": (int(predicted_score - 2 * score_variance), int(predicted_score - score_variance)),
                    "medium": (int(predicted_score - score_variance), int(predicted_score + score_variance)),
                    "high": (int(predicted_score + score_variance), int(predicted_score + 2 * score_variance))
                }
                
                # Contributing factors
                result["factors"] = {
                    "pitch_conditions": self._analyze_pitch_impact(data),
                    "weather_impact": self._analyze_weather_impact(data),
                    "batting_strength": self._analyze_batting_strength(data),
                    "bowling_quality": self._analyze_bowling_quality(data)
                }
                
                result["model_info"] = {
                    "type": model_type.value,
                    "latency_ms": moe_result.get("metadata", {}).get("latency_ms", 0),
                    "model_confidence": moe_result.get("metadata", {}).get("confidence", 0.5)
                }
            
        except Exception as e:
            logger.error(f"Total score prediction failed: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    async def _predict_player_performance(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict individual player performance"""
        result = {
            "prediction_type": "player_performance",
            "player_predictions": {},
            "performance_ranges": {},
            "factors": {},
            "model_info": {}
        }
        
        try:
            players = data.get("players", [])
            if not players:
                # Extract from query if not in data
                players = self._extract_players_from_query(context.user_query)
            
            for player in players:
                # Create player-specific features
                player_features = self._create_player_features(player, data, context)
                
                # Use slow model for detailed player analysis
                model_type = self.model_preferences.get("player_performance", ModelType.SLOW)
                moe_result = await self._get_moe_prediction(player_features, model_type, "player_performance")
                
                if moe_result and moe_result.get("predictions"):
                    predictions = moe_result["predictions"]
                    
                    # Extract player performance predictions
                    result["player_predictions"][player] = {
                        "runs": predictions.get("runs", 35),
                        "strike_rate": predictions.get("strike_rate", 120),
                        "probability_50_plus": predictions.get("fifty_prob", 0.3),
                        "probability_100_plus": predictions.get("century_prob", 0.1),
                        "dismissal_probability": predictions.get("dismissal_prob", 0.8)
                    }
                    
                    # Performance ranges
                    result["performance_ranges"][player] = {
                        "runs_range": (
                            max(0, predictions.get("runs", 35) - 20),
                            predictions.get("runs", 35) + 30
                        ),
                        "strike_rate_range": (
                            max(50, predictions.get("strike_rate", 120) - 30),
                            predictions.get("strike_rate", 120) + 40
                        )
                    }
            
            # Overall factors affecting player performance
            result["factors"] = {
                "form_analysis": self._analyze_player_form(data),
                "opposition_analysis": self._analyze_opposition_strength(data),
                "conditions_impact": self._analyze_conditions_impact(data),
                "pressure_factors": self._analyze_pressure_factors(context, data)
            }
            
        except Exception as e:
            logger.error(f"Player performance prediction failed: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    async def _predict_tournament_winner(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict tournament winner"""
        result = {
            "prediction_type": "tournament_winner",
            "winner_probabilities": {},
            "top_contenders": [],
            "dark_horses": [],
            "factors": {},
            "model_info": {}
        }
        
        try:
            # Get tournament teams
            tournament_teams = self._extract_tournament_teams(context, data)
            
            # Create tournament features
            features = self._create_tournament_features(data, context)
            
            # Use slow model for complex tournament analysis
            model_type = ModelType.SLOW
            moe_result = await self._get_moe_prediction(features, model_type, "tournament_winner")
            
            if moe_result and moe_result.get("predictions"):
                predictions = moe_result["predictions"]
                
                # Distribute probabilities among teams
                team_probs = self._distribute_tournament_probabilities(tournament_teams, predictions)
                result["winner_probabilities"] = team_probs
                
                # Identify top contenders and dark horses
                sorted_teams = sorted(team_probs.items(), key=lambda x: x[1], reverse=True)
                result["top_contenders"] = [team for team, prob in sorted_teams[:3]]
                result["dark_horses"] = [team for team, prob in sorted_teams if 0.05 < prob < 0.15]
                
                result["factors"] = {
                    "team_rankings": self._analyze_team_rankings(data),
                    "recent_tournament_form": self._analyze_tournament_form(data),
                    "squad_depth": self._analyze_squad_depth(data),
                    "experience_factor": self._analyze_experience(data)
                }
            
        except Exception as e:
            logger.error(f"Tournament winner prediction failed: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    async def _predict_series_outcome(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict series outcome"""
        result = {
            "prediction_type": "series_outcome",
            "series_winner": None,
            "match_by_match": [],
            "series_score_prediction": None,
            "factors": {},
            "model_info": {}
        }
        
        try:
            # Analyze series structure
            series_info = self._extract_series_info(context, data)
            
            # Create series features
            features = self._create_series_features(data, context, series_info)
            
            # Get series prediction
            model_type = ModelType.SLOW
            moe_result = await self._get_moe_prediction(features, model_type, "series_outcome")
            
            if moe_result and moe_result.get("predictions"):
                predictions = moe_result["predictions"]
                
                # Extract series predictions
                result["series_winner"] = predictions.get("series_winner", "Team A")
                result["series_score_prediction"] = predictions.get("series_score", "3-2")
                
                # Match-by-match predictions if applicable
                num_matches = series_info.get("num_matches", 5)
                for i in range(num_matches):
                    match_prob = predictions.get(f"match_{i+1}_prob", 0.5)
                    result["match_by_match"].append({
                        "match": i + 1,
                        "predicted_winner": "Team A" if match_prob > 0.5 else "Team B",
                        "confidence": abs(match_prob - 0.5) * 2
                    })
                
                result["factors"] = {
                    "home_advantage": self._analyze_home_advantage(data, series_info),
                    "series_momentum": self._analyze_series_momentum(data),
                    "squad_rotation": self._analyze_squad_rotation(data),
                    "format_expertise": self._analyze_format_expertise(data, context)
                }
            
        except Exception as e:
            logger.error(f"Series outcome prediction failed: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    async def _predict_general_outcome(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """General prediction when specific type unclear"""
        result = {
            "prediction_type": "general_outcome",
            "insights": [],
            "probabilities": {},
            "recommendations": {},
            "factors": {}
        }
        
        try:
            # Provide general insights based on available data
            result["insights"] = [
                "Multiple factors influence cricket outcomes",
                "Recent form is a key indicator",
                "Conditions play a significant role",
                "Head-to-head records provide valuable context"
            ]
            
            # General probability assessments
            result["probabilities"] = {
                "close_match": 0.6,
                "dominant_performance": 0.3,
                "upset_result": 0.1
            }
            
            result["recommendations"] = {
                "analysis_focus": "Consider recent form and conditions",
                "key_factors": "Monitor team news and weather",
                "prediction_confidence": "Medium due to limited specificity"
            }
            
        except Exception as e:
            logger.error(f"General prediction failed: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    async def _get_moe_prediction(self, features: Dict[str, Any], model_type: ModelType, prediction_type: str) -> Optional[Dict[str, Any]]:
        """Get prediction from MoE system"""
        try:
            if not self.moe_orchestrator:
                return None
            
            # Convert features to format expected by MoE system
            inference_request = {
                "features": features,
                "model_preference": model_type,
                "prediction_type": prediction_type,
                "require_explanation": True
            }
            
            # Get prediction from MoE orchestrator
            moe_result = self.moe_orchestrator.predict(inference_request)
            
            if moe_result.get("status") == "success":
                return moe_result
            else:
                logger.warning(f"MoE prediction failed: {moe_result.get('error', 'Unknown error')}")
                return None
            
        except Exception as e:
            logger.error(f"MoE prediction error: {str(e)}")
            return None
    
    def _calculate_prediction_confidence(self, result: Dict[str, Any], data: Dict[str, Any]) -> float:
        """Calculate overall confidence in the prediction"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on data availability
        if data.get("historical_data"):
            confidence += 0.1
        if data.get("head_to_head"):
            confidence += 0.1
        if data.get("venue_stats"):
            confidence += 0.05
        
        # Increase confidence based on model confidence
        model_confidence = result.get("model_info", {}).get("model_confidence", 0.5)
        confidence += (model_confidence - 0.5) * 0.3
        
        # Adjust based on prediction type
        prediction_type = result.get("prediction_type", "")
        if prediction_type in ["match_winner", "total_score"]:
            confidence += 0.05  # More confident in these predictions
        
        return max(0.0, min(1.0, confidence))
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> str:
        """Assess the quality of prediction data"""
        score = 0
        
        if data.get("teams"):
            score += 1
        if data.get("historical_data"):
            score += 2
        if data.get("head_to_head"):
            score += 1
        if data.get("venue_stats"):
            score += 1
        
        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        else:
            return "low"
    
    # Feature creation helper methods
    def _create_match_features(self, data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Create feature vector for match prediction"""
        return {
            "team_a_recent_wins": 3,
            "team_b_recent_wins": 2,
            "head_to_head_advantage": 0.6,
            "venue_advantage": 0.1,
            "format": context.format_context or "ODI",
            "conditions_factor": 0.5
        }
    
    def _create_score_features(self, data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Create features for score prediction"""
        return {
            "pitch_rating": 0.7,
            "weather_impact": 0.2,
            "batting_strength": 0.8,
            "bowling_quality": 0.7,
            "venue_avg_score": 250,
            "format": context.format_context or "ODI"
        }
    
    def _create_player_features(self, player: str, data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Create features for player performance prediction"""
        return {
            "player_name": player,
            "recent_form": 0.7,
            "opposition_strength": 0.6,
            "conditions_suitability": 0.8,
            "pressure_rating": 0.5,
            "format": context.format_context or "ODI"
        }
    
    # Additional helper methods (simplified implementations)
    def _extract_teams_from_context(self, context: AgentContext) -> List[str]:
        """Extract team names from context"""
        if context.team_context and "names" in context.team_context:
            return context.team_context["names"]
        
        # Simple extraction from query
        query_lower = context.user_query.lower()
        teams = []
        potential_teams = ["india", "australia", "england", "pakistan", "south africa"]
        
        for team in potential_teams:
            if team in query_lower:
                teams.append(team.title())
        
        return teams[:2]  # Return first two teams found
    
    def _extract_players_from_context(self, context: AgentContext) -> List[str]:
        """Extract player names from context"""
        if context.player_context and "names" in context.player_context:
            return context.player_context["names"]
        
        return self._extract_players_from_query(context.user_query)
    
    def _extract_players_from_query(self, query: str) -> List[str]:
        """Extract player names from query text"""
        # Simplified player name extraction
        potential_players = ["Kohli", "Sharma", "Dhoni", "Babar", "Root", "Smith"]
        players = []
        
        for player in potential_players:
            if player.lower() in query.lower():
                players.append(player)
        
        return players
    
    # Analysis helper methods (simplified implementations)
    def _analyze_recent_form(self, data: Dict[str, Any]) -> Dict[str, str]:
        return {"team_a": "good", "team_b": "average"}
    
    def _analyze_head_to_head(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"advantage": "team_a", "recent_meetings": "3-2"}
    
    def _analyze_venue_advantage(self, data: Dict[str, Any]) -> Dict[str, str]:
        return {"home_team": "team_a", "advantage_level": "moderate"}
    
    def _analyze_team_strength(self, data: Dict[str, Any]) -> Dict[str, float]:
        return {"team_a": 0.8, "team_b": 0.7}
    
    def _analyze_pitch_impact(self, data: Dict[str, Any]) -> str:
        return "batting_friendly"
    
    def _analyze_weather_impact(self, data: Dict[str, Any]) -> str:
        return "minimal"
    
    def _analyze_batting_strength(self, data: Dict[str, Any]) -> str:
        return "strong"
    
    def _analyze_bowling_quality(self, data: Dict[str, Any]) -> str:
        return "good"
    
    def _analyze_player_form(self, data: Dict[str, Any]) -> Dict[str, str]:
        return {"recent_trend": "improving", "consistency": "high"}
    
    def _analyze_opposition_strength(self, data: Dict[str, Any]) -> str:
        return "moderate"
    
    def _analyze_conditions_impact(self, data: Dict[str, Any]) -> str:
        return "favorable"
    
    def _analyze_pressure_factors(self, context: AgentContext, data: Dict[str, Any]) -> str:
        return "medium"
    
    def _extract_tournament_teams(self, context: AgentContext, data: Dict[str, Any]) -> List[str]:
        return ["Team A", "Team B", "Team C", "Team D", "Team E", "Team F", "Team G", "Team H"]
    
    def _create_tournament_features(self, data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        return {"tournament_type": "world_cup", "num_teams": 8, "format": "knockout"}
    
    def _distribute_tournament_probabilities(self, teams: List[str], predictions: Dict[str, Any]) -> Dict[str, float]:
        # Simple probability distribution
        probs = {}
        base_prob = 1.0 / len(teams)
        
        for i, team in enumerate(teams):
            # Vary probabilities slightly
            probs[team] = base_prob + (0.1 * np.sin(i))
        
        # Normalize to sum to 1.0
        total = sum(probs.values())
        return {team: prob/total for team, prob in probs.items()}
    
    def _analyze_team_rankings(self, data: Dict[str, Any]) -> Dict[str, int]:
        return {"team_a": 1, "team_b": 3, "team_c": 2}
    
    def _analyze_tournament_form(self, data: Dict[str, Any]) -> Dict[str, str]:
        return {"recent_performance": "good", "key_players": "available"}
    
    def _analyze_squad_depth(self, data: Dict[str, Any]) -> str:
        return "strong"
    
    def _analyze_experience(self, data: Dict[str, Any]) -> str:
        return "high"
    
    def _extract_series_info(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"num_matches": 5, "format": "ODI", "home_team": "Team A"}
    
    def _create_series_features(self, data: Dict[str, Any], context: AgentContext, series_info: Dict[str, Any]) -> Dict[str, Any]:
        return {"series_length": series_info.get("num_matches", 5), "format": series_info.get("format", "ODI")}
    
    def _analyze_home_advantage(self, data: Dict[str, Any], series_info: Dict[str, Any]) -> str:
        return "moderate"
    
    def _analyze_series_momentum(self, data: Dict[str, Any]) -> str:
        return "neutral"
    
    def _analyze_squad_rotation(self, data: Dict[str, Any]) -> str:
        return "balanced"
    
    def _analyze_format_expertise(self, data: Dict[str, Any], context: AgentContext) -> str:
        return "high"

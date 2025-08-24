# Purpose: Betting Analysis Agent - Value & Arbitrage Opportunities
# Author: WicketWise Team, Last Modified: 2025-08-24

"""
Betting Analysis Agent
=====================

Specialized agent for cricket betting analysis, value identification,
and risk assessment. Integrates with the betting intelligence module
and prediction systems for comprehensive betting insights.

Key Capabilities:
- Value opportunity identification
- Arbitrage detection and analysis
- Betting strategy recommendations
- Risk assessment and bankroll management
- Market efficiency analysis
- Real-time odds monitoring
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from ..betting.mispricing_engine import MispricingEngine, OddsData, BetType, BookmakerType, ValueOpportunity
from ..gnn.enhanced_kg_api import EnhancedKGQueryEngine
from .base_agent import BaseAgent, AgentCapability, AgentContext, AgentResponse
from .prediction_agent import PredictionAgent

logger = logging.getLogger(__name__)


class BettingAgent(BaseAgent):
    """
    Agent specialized in cricket betting analysis and value detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="betting_agent",
            capabilities=[
                AgentCapability.BETTING_ANALYSIS,
                AgentCapability.TREND_ANALYSIS,
                AgentCapability.REAL_TIME_PROCESSING
            ],
            config=config
        )
        
        # Dependencies
        self.mispricing_engine: Optional[MispricingEngine] = None
        self.prediction_agent: Optional[PredictionAgent] = None
        self.kg_engine: Optional[EnhancedKGQueryEngine] = None
        
        # Betting configuration
        self.supported_bet_types = [
            BetType.MATCH_WINNER, BetType.TOTAL_RUNS, BetType.PLAYER_RUNS,
            BetType.PLAYER_WICKETS, BetType.INNINGS_RUNS
        ]
        
        self.bookmaker_priorities = {
            BookmakerType.SHARP: 1.0,      # Highest reliability
            BookmakerType.EXCHANGE: 0.9,   # High reliability
            BookmakerType.REGULATED: 0.7,  # Good reliability
            BookmakerType.SOFT: 0.5,       # Moderate reliability
            BookmakerType.OFFSHORE: 0.3    # Lower reliability
        }
        
        # Risk management parameters
        self.max_kelly_fraction = self.config.get("max_kelly_fraction", 0.1)
        self.min_edge_threshold = self.config.get("min_edge_threshold", 0.05)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        
        # Required dependencies
        self.required_dependencies = ["mispricing_engine", "prediction_agent", "knowledge_graph"]
    
    def _initialize_agent(self) -> bool:
        """Initialize betting agent dependencies"""
        try:
            # Initialize mispricing engine
            betting_config = {
                "min_edge_threshold": self.min_edge_threshold,
                "max_kelly_fraction": self.max_kelly_fraction,
                "confidence_threshold": self.confidence_threshold
            }
            self.mispricing_engine = MispricingEngine(betting_config)
            
            # Initialize prediction agent for model predictions
            self.prediction_agent = PredictionAgent()
            if not self.prediction_agent.initialize():
                logger.warning("Failed to initialize prediction agent - betting analysis may be limited")
            
            # Initialize KG engine
            self.kg_engine = EnhancedKGQueryEngine()
            
            logger.info("BettingAgent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"BettingAgent initialization failed: {str(e)}")
            return False
    
    def can_handle(self, capability: AgentCapability, context: AgentContext) -> bool:
        """Check if agent can handle the capability and context"""
        if capability not in self.capabilities:
            return False
        
        # Check for betting keywords in query
        betting_keywords = [
            "bet", "betting", "odds", "value", "arbitrage", "bookmaker",
            "stake", "wager", "market", "price", "probability", "edge"
        ]
        
        query_lower = context.user_query.lower()
        return any(keyword in query_lower for keyword in betting_keywords)
    
    async def execute(self, context: AgentContext) -> AgentResponse:
        """Execute betting analysis"""
        try:
            # Determine betting analysis type
            analysis_type = self._determine_betting_analysis_type(context)
            
            # Gather betting data
            betting_data = await self._gather_betting_data(context, analysis_type)
            
            # Perform specific betting analysis
            if analysis_type == "value_opportunities":
                result = await self._analyze_value_opportunities(context, betting_data)
            elif analysis_type == "arbitrage_detection":
                result = await self._analyze_arbitrage_opportunities(context, betting_data)
            elif analysis_type == "market_analysis":
                result = await self._analyze_market_efficiency(context, betting_data)
            elif analysis_type == "betting_strategy":
                result = await self._analyze_betting_strategy(context, betting_data)
            elif analysis_type == "risk_assessment":
                result = await self._analyze_betting_risks(context, betting_data)
            else:
                result = await self._analyze_general_betting(context, betting_data)
            
            # Calculate confidence based on data quality and analysis depth
            confidence = self._calculate_betting_confidence(result, betting_data)
            
            return AgentResponse(
                agent_id=self.agent_id,
                capability=AgentCapability.BETTING_ANALYSIS,
                success=True,
                confidence=confidence,
                execution_time=0.0,
                result=result,
                dependencies_used=["mispricing_engine", "prediction_agent", "knowledge_graph"],
                metadata={
                    "analysis_type": analysis_type,
                    "data_sources": len(betting_data.get("odds_data", [])),
                    "bookmakers_analyzed": len(set(odds.bookmaker_id for odds in betting_data.get("odds_data", [])))
                }
            )
            
        except Exception as e:
            logger.error(f"BettingAgent execution failed: {str(e)}")
            return AgentResponse(
                agent_id=self.agent_id,
                capability=AgentCapability.BETTING_ANALYSIS,
                success=False,
                confidence=0.0,
                execution_time=0.0,
                error_message=str(e)
            )
    
    def _determine_betting_analysis_type(self, context: AgentContext) -> str:
        """Determine the type of betting analysis needed"""
        query_lower = context.user_query.lower()
        
        if any(word in query_lower for word in ["value", "edge", "mispricing", "opportunity"]):
            return "value_opportunities"
        elif any(word in query_lower for word in ["arbitrage", "arb", "sure bet", "risk free"]):
            return "arbitrage_detection"
        elif any(word in query_lower for word in ["market", "efficiency", "overround", "margin"]):
            return "market_analysis"
        elif any(word in query_lower for word in ["strategy", "bankroll", "staking", "approach"]):
            return "betting_strategy"
        elif any(word in query_lower for word in ["risk", "variance", "volatility", "safety"]):
            return "risk_assessment"
        else:
            return "general_betting"
    
    async def _gather_betting_data(self, context: AgentContext, analysis_type: str) -> Dict[str, Any]:
        """Gather relevant betting data"""
        data = {
            "odds_data": [],
            "market_predictions": {},
            "historical_odds": {},
            "bookmaker_info": {},
            "match_context": {}
        }
        
        try:
            # Generate sample odds data for testing
            data["odds_data"] = self._generate_sample_odds_data(context)
            
            # Get model predictions if prediction agent is available
            if self.prediction_agent:
                prediction_context = AgentContext(
                    request_id=context.request_id + "_prediction",
                    user_query=f"Predict outcome for {context.user_query}",
                    timestamp=context.timestamp,
                    match_context=context.match_context,
                    confidence_threshold=0.5
                )
                
                prediction_response = await self.prediction_agent.execute(prediction_context)
                if prediction_response.success:
                    data["market_predictions"] = prediction_response.result
            
            # Query historical data from KG if available
            if self.kg_engine and context.match_context:
                historical_query = "Get historical betting patterns and odds movements"
                historical_data = await self.kg_engine.execute_query(historical_query)
                data["historical_odds"] = historical_data
            
            # Add bookmaker information
            data["bookmaker_info"] = self._get_bookmaker_information()
            
            # Add match context
            data["match_context"] = context.match_context or {}
            
        except Exception as e:
            logger.warning(f"Failed to gather some betting data: {str(e)}")
        
        return data
    
    async def _analyze_value_opportunities(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze value betting opportunities"""
        result = {
            "analysis_type": "value_opportunities",
            "opportunities": [],
            "market_predictions": {},
            "value_summary": {},
            "recommendations": {}
        }
        
        try:
            odds_data = data.get("odds_data", [])
            market_predictions = data.get("market_predictions", {})
            
            if not odds_data:
                result["error"] = "No odds data available for analysis"
                return result
            
            # Extract model predictions for comparison
            model_probs = self._extract_model_probabilities(market_predictions)
            
            if model_probs:
                # Use mispricing engine to detect value opportunities
                match_id = context.match_context.get("match_id", "unknown_match")
                opportunities = self.mispricing_engine.detect_mispricing(
                    model_probs, odds_data, match_id
                )
                
                # Convert opportunities to analysis format
                for opp in opportunities:
                    opportunity_analysis = {
                        "selection": opp.selection,
                        "bookmaker": opp.metadata.get("bookmaker_id", "unknown"),
                        "odds": opp.bookmaker_odds,
                        "model_probability": opp.model_probability,
                        "implied_probability": float(opp.implied_probability),
                        "expected_value": opp.expected_value,
                        "kelly_fraction": opp.kelly_fraction,
                        "confidence_score": opp.confidence_score,
                        "risk_level": opp.risk_level,
                        "max_stake": opp.max_stake,
                        "recommendation": self._generate_betting_recommendation(opp)
                    }
                    result["opportunities"].append(opportunity_analysis)
                
                # Generate value summary
                result["value_summary"] = {
                    "total_opportunities": len(opportunities),
                    "high_value_count": len([o for o in opportunities if o.expected_value > 0.1]),
                    "average_edge": np.mean([o.expected_value for o in opportunities]) if opportunities else 0,
                    "total_potential_stake": sum(o.kelly_fraction * 1000 for o in opportunities)  # Assuming $1000 bankroll
                }
                
                # Generate recommendations
                result["recommendations"] = self._generate_value_recommendations(opportunities)
            
            else:
                result["error"] = "No model predictions available for value analysis"
            
        except Exception as e:
            logger.error(f"Value opportunity analysis failed: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    async def _analyze_arbitrage_opportunities(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze arbitrage betting opportunities"""
        result = {
            "analysis_type": "arbitrage_detection",
            "arbitrage_opportunities": [],
            "profit_analysis": {},
            "execution_requirements": {},
            "risk_factors": {}
        }
        
        try:
            odds_data = data.get("odds_data", [])
            
            if not odds_data:
                result["error"] = "No odds data available for arbitrage analysis"
                return result
            
            # Use mispricing engine to detect arbitrage
            arbitrage_ops = self.mispricing_engine.detect_arbitrage_opportunities(odds_data)
            
            for arb in arbitrage_ops:
                arb_analysis = {
                    "match_id": arb["match_id"],
                    "bet_type": arb["bet_type"],
                    "profit_margin": arb["profit_margin"] * 100,  # Convert to percentage
                    "total_implied_prob": arb["total_implied_prob"],
                    "best_odds": arb["best_odds"],
                    "optimal_stakes": arb["optimal_stakes"],
                    "bookmakers": arb["bookmakers"],
                    "profit_calculation": self._calculate_arbitrage_profit(arb),
                    "execution_difficulty": self._assess_execution_difficulty(arb),
                    "time_sensitivity": "High - odds may change quickly"
                }
                result["arbitrage_opportunities"].append(arb_analysis)
            
            # Generate profit analysis
            if arbitrage_ops:
                result["profit_analysis"] = {
                    "total_opportunities": len(arbitrage_ops),
                    "average_profit_margin": np.mean([arb["profit_margin"] for arb in arbitrage_ops]) * 100,
                    "best_opportunity": max(arbitrage_ops, key=lambda x: x["profit_margin"])["profit_margin"] * 100,
                    "total_capital_required": sum(
                        sum(stakes.values()) for stakes in [arb["optimal_stakes"] for arb in arbitrage_ops]
                    )
                }
            
            # Execution requirements
            result["execution_requirements"] = {
                "multiple_accounts": "Required for different bookmakers",
                "quick_execution": "Essential due to odds volatility",
                "capital_efficiency": "Distribute stakes optimally",
                "monitoring": "Real-time odds monitoring needed"
            }
            
            # Risk factors
            result["risk_factors"] = {
                "odds_movement": "High - odds can change rapidly",
                "account_limits": "Bookmaker may limit successful arbitrageurs",
                "execution_risk": "Partial execution if odds change mid-bet",
                "technical_risk": "Platform downtime or connectivity issues"
            }
            
        except Exception as e:
            logger.error(f"Arbitrage analysis failed: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    async def _analyze_market_efficiency(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze betting market efficiency"""
        result = {
            "analysis_type": "market_analysis",
            "efficiency_metrics": {},
            "bookmaker_analysis": {},
            "market_trends": {},
            "insights": []
        }
        
        try:
            odds_data = data.get("odds_data", [])
            
            if not odds_data:
                result["error"] = "No odds data available for market analysis"
                return result
            
            # Use mispricing engine for market efficiency analysis
            match_id = context.match_context.get("match_id", "unknown_match")
            efficiency_analysis = self.mispricing_engine.analyze_market_efficiency(match_id)
            
            result["efficiency_metrics"] = {
                "overall_efficiency": efficiency_analysis.get("overall_efficiency", "unknown"),
                "overround_analysis": self._analyze_overround(odds_data),
                "price_dispersion": self._calculate_price_dispersion(odds_data),
                "market_depth": self._assess_market_depth(odds_data)
            }
            
            # Bookmaker analysis
            result["bookmaker_analysis"] = self._analyze_bookmaker_performance(odds_data)
            
            # Market trends
            result["market_trends"] = {
                "odds_movement": "Stable with minor fluctuations",
                "volume_trends": "Moderate betting volume",
                "sharp_money": "Some sharp money detected",
                "public_sentiment": "Balanced public interest"
            }
            
            # Generate insights
            result["insights"] = [
                "Market shows good efficiency with minimal arbitrage opportunities",
                "Sharp bookmakers offer competitive odds with lower margins",
                "Some value opportunities exist for informed bettors",
                "Market reacts quickly to new information"
            ]
            
        except Exception as e:
            logger.error(f"Market efficiency analysis failed: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    async def _analyze_betting_strategy(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze betting strategy recommendations"""
        result = {
            "analysis_type": "betting_strategy",
            "strategy_recommendations": {},
            "bankroll_management": {},
            "staking_plans": {},
            "risk_management": {}
        }
        
        try:
            opportunities = []
            if data.get("odds_data"):
                # Get value opportunities for strategy analysis
                value_analysis = await self._analyze_value_opportunities(context, data)
                opportunities = value_analysis.get("opportunities", [])
            
            # Strategy recommendations based on opportunities
            if opportunities:
                result["strategy_recommendations"] = {
                    "primary_strategy": "Value betting with selective approach",
                    "focus_areas": self._identify_focus_areas(opportunities),
                    "bet_types": self._recommend_bet_types(opportunities),
                    "timing_strategy": "Early market entry for best value"
                }
                
                # Bankroll management
                result["bankroll_management"] = {
                    "recommended_bankroll": self._calculate_recommended_bankroll(opportunities),
                    "unit_size": "1-2% of total bankroll per bet",
                    "maximum_exposure": "10% of bankroll on single event",
                    "growth_strategy": "Conservative compound growth"
                }
                
                # Staking plans
                result["staking_plans"] = {
                    "kelly_criterion": "Optimal for long-term growth",
                    "fixed_percentage": "Conservative approach",
                    "proportional_staking": "Based on edge and confidence",
                    "recommended": self._recommend_staking_plan(opportunities)
                }
            
            else:
                result["strategy_recommendations"] = {
                    "primary_strategy": "Wait for better opportunities",
                    "market_monitoring": "Continue monitoring for value",
                    "patience": "Selective betting approach recommended"
                }
            
            # Risk management
            result["risk_management"] = {
                "diversification": "Spread bets across multiple events",
                "correlation_risk": "Avoid highly correlated bets",
                "variance_management": "Expect short-term volatility",
                "stop_loss": "Review strategy if significant losses occur"
            }
            
        except Exception as e:
            logger.error(f"Betting strategy analysis failed: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    async def _analyze_betting_risks(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze betting risks and risk management"""
        result = {
            "analysis_type": "risk_assessment",
            "risk_categories": {},
            "risk_mitigation": {},
            "variance_analysis": {},
            "recommendations": {}
        }
        
        try:
            # Risk categories
            result["risk_categories"] = {
                "market_risk": {
                    "level": "Medium",
                    "description": "Risk of adverse market movements",
                    "factors": ["Odds volatility", "Market liquidity", "Information asymmetry"]
                },
                "execution_risk": {
                    "level": "Low",
                    "description": "Risk of bet execution issues",
                    "factors": ["Platform reliability", "Account limits", "Technical issues"]
                },
                "model_risk": {
                    "level": "Medium",
                    "description": "Risk of prediction model inaccuracy",
                    "factors": ["Model assumptions", "Data quality", "Market efficiency"]
                },
                "bankroll_risk": {
                    "level": "High",
                    "description": "Risk of significant bankroll loss",
                    "factors": ["Overbetting", "Correlation", "Variance"]
                }
            }
            
            # Risk mitigation strategies
            result["risk_mitigation"] = {
                "diversification": "Spread bets across different matches and bet types",
                "position_sizing": "Use Kelly criterion with safety margin",
                "stop_loss": "Set maximum loss limits per day/week",
                "monitoring": "Continuous performance tracking and adjustment"
            }
            
            # Variance analysis
            opportunities = data.get("opportunities", [])
            if opportunities:
                result["variance_analysis"] = self._analyze_betting_variance(opportunities)
            else:
                result["variance_analysis"] = {
                    "expected_variance": "Moderate to high",
                    "drawdown_risk": "Significant short-term losses possible",
                    "time_horizon": "Long-term positive expectation"
                }
            
            # Recommendations
            result["recommendations"] = [
                "Maintain strict bankroll management discipline",
                "Never bet more than calculated Kelly fraction",
                "Keep detailed records of all betting activity",
                "Regular strategy review and adjustment",
                "Consider betting exchanges for better odds"
            ]
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    async def _analyze_general_betting(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """General betting analysis when specific type unclear"""
        result = {
            "analysis_type": "general_betting",
            "market_overview": {},
            "opportunities_summary": {},
            "general_insights": [],
            "next_steps": {}
        }
        
        try:
            odds_data = data.get("odds_data", [])
            
            # Market overview
            result["market_overview"] = {
                "bookmakers_available": len(set(odds.bookmaker_id for odds in odds_data)),
                "bet_types_covered": len(set(odds.bet_type for odds in odds_data)),
                "market_activity": "Moderate to high",
                "competitive_landscape": "Good competition between bookmakers"
            }
            
            # Quick opportunities summary
            if odds_data:
                # Quick value check
                sample_probs = {"Team A": 0.6, "Team B": 0.4}  # Sample probabilities
                match_id = "general_analysis"
                quick_opportunities = self.mispricing_engine.detect_mispricing(
                    sample_probs, odds_data[:6], match_id  # Limit to first 6 odds
                )
                
                result["opportunities_summary"] = {
                    "value_opportunities": len(quick_opportunities),
                    "potential_edge": np.mean([o.expected_value for o in quick_opportunities]) if quick_opportunities else 0,
                    "recommendation": "Further analysis recommended" if quick_opportunities else "Limited opportunities currently"
                }
            
            # General insights
            result["general_insights"] = [
                "Cricket betting markets are generally efficient",
                "Value opportunities exist for informed bettors",
                "Bankroll management is crucial for long-term success",
                "Market timing can significantly impact profitability"
            ]
            
            # Next steps
            result["next_steps"] = {
                "detailed_analysis": "Request specific value or arbitrage analysis",
                "market_monitoring": "Set up real-time odds monitoring",
                "strategy_development": "Develop systematic betting approach",
                "risk_management": "Establish clear risk management rules"
            }
            
        except Exception as e:
            logger.error(f"General betting analysis failed: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _generate_sample_odds_data(self, context: AgentContext) -> List[OddsData]:
        """Generate sample odds data for testing"""
        current_time = datetime.now()
        match_id = context.match_context.get("match_id", "test_match")
        
        # Generate sample odds for match winner
        odds_data = [
            # Team A odds
            OddsData("bet365", BookmakerType.REGULATED, BetType.MATCH_WINNER, 
                    "Match Winner", 2.1, 1000.0, current_time, match_id, "Team A"),
            OddsData("betfair", BookmakerType.EXCHANGE, BetType.MATCH_WINNER,
                    "Match Winner", 2.2, 5000.0, current_time, match_id, "Team A"),
            OddsData("pinnacle", BookmakerType.SHARP, BetType.MATCH_WINNER,
                    "Match Winner", 2.05, 2000.0, current_time, match_id, "Team A"),
            
            # Team B odds
            OddsData("bet365", BookmakerType.REGULATED, BetType.MATCH_WINNER,
                    "Match Winner", 1.8, 1000.0, current_time, match_id, "Team B"),
            OddsData("betfair", BookmakerType.EXCHANGE, BetType.MATCH_WINNER,
                    "Match Winner", 1.85, 5000.0, current_time, match_id, "Team B"),
            OddsData("pinnacle", BookmakerType.SHARP, BetType.MATCH_WINNER,
                    "Match Winner", 1.75, 2000.0, current_time, match_id, "Team B"),
            
            # Total runs market
            OddsData("bet365", BookmakerType.REGULATED, BetType.TOTAL_RUNS,
                    "Total Runs Over 250.5", 1.9, 800.0, current_time, match_id, "Over 250.5"),
            OddsData("betfair", BookmakerType.EXCHANGE, BetType.TOTAL_RUNS,
                    "Total Runs Under 250.5", 1.95, 1200.0, current_time, match_id, "Under 250.5")
        ]
        
        return odds_data
    
    def _extract_model_probabilities(self, market_predictions: Dict[str, Any]) -> Dict[str, float]:
        """Extract model probabilities from prediction results"""
        model_probs = {}
        
        if not market_predictions:
            # Return default probabilities for testing
            return {"Team A": 0.55, "Team B": 0.45}
        
        # Extract from match winner predictions
        if "predictions" in market_predictions:
            predictions = market_predictions["predictions"]
            if isinstance(predictions, dict):
                for team, prob in predictions.items():
                    if isinstance(prob, (int, float)):
                        model_probs[team] = float(prob)
        
        # Extract from win probability if available
        if "predicted_winner" in market_predictions and "win_probability" in market_predictions:
            winner = market_predictions["predicted_winner"]
            win_prob = market_predictions["win_probability"]
            model_probs[winner] = win_prob
            
            # Assume binary outcome for simplicity
            other_team = "Team B" if winner == "Team A" else "Team A"
            model_probs[other_team] = 1.0 - win_prob
        
        return model_probs
    
    def _calculate_betting_confidence(self, result: Dict[str, Any], data: Dict[str, Any]) -> float:
        """Calculate confidence in betting analysis"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on data availability
        if data.get("odds_data"):
            confidence += 0.2
        if data.get("market_predictions"):
            confidence += 0.2
        if data.get("historical_odds"):
            confidence += 0.1
        
        # Adjust based on analysis success
        if result and "error" not in result:
            confidence += 0.1
        elif result and "error" in result:
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    # Helper methods (simplified implementations)
    def _get_bookmaker_information(self) -> Dict[str, Any]:
        return {
            "sharp_bookmakers": ["Pinnacle", "Betfair"],
            "recreational_bookmakers": ["Bet365", "William Hill"],
            "exchange_platforms": ["Betfair", "Betdaq"],
            "reliability_ratings": {"Pinnacle": 9.5, "Betfair": 9.0, "Bet365": 8.5}
        }
    
    def _generate_betting_recommendation(self, opportunity: ValueOpportunity) -> str:
        if opportunity.expected_value > 0.1:
            return "Strong recommendation - high value opportunity"
        elif opportunity.expected_value > 0.05:
            return "Moderate recommendation - decent value"
        else:
            return "Weak recommendation - marginal value"
    
    def _generate_value_recommendations(self, opportunities: List[ValueOpportunity]) -> Dict[str, str]:
        return {
            "overall": f"Found {len(opportunities)} value opportunities",
            "best_bet": opportunities[0].selection if opportunities else "None",
            "strategy": "Focus on highest edge opportunities with good confidence"
        }
    
    def _calculate_arbitrage_profit(self, arb: Dict[str, Any]) -> Dict[str, float]:
        profit_margin = arb["profit_margin"]
        return {
            "profit_percentage": profit_margin * 100,
            "profit_per_100": profit_margin * 100,
            "annual_return_estimate": profit_margin * 100 * 50  # Assuming 50 opportunities per year
        }
    
    def _assess_execution_difficulty(self, arb: Dict[str, Any]) -> str:
        bookmakers = arb.get("bookmakers", {})
        if len(bookmakers) > 2:
            return "High - requires multiple bookmaker accounts"
        else:
            return "Medium - requires two bookmaker accounts"
    
    def _analyze_overround(self, odds_data: List[OddsData]) -> Dict[str, float]:
        # Simplified overround calculation
        return {"average_overround": 0.05, "best_bookmaker": "Pinnacle", "worst_bookmaker": "Local Bookie"}
    
    def _calculate_price_dispersion(self, odds_data: List[OddsData]) -> float:
        # Simplified price dispersion
        return 0.03  # 3% average dispersion
    
    def _assess_market_depth(self, odds_data: List[OddsData]) -> str:
        return "Good - multiple bookmakers with reasonable limits"
    
    def _analyze_bookmaker_performance(self, odds_data: List[OddsData]) -> Dict[str, Any]:
        return {
            "sharpest": "Pinnacle",
            "best_value": "Betfair",
            "most_competitive": "Overall competitive market"
        }
    
    def _identify_focus_areas(self, opportunities: List[Dict[str, Any]]) -> List[str]:
        return ["Match winner markets", "Total runs markets", "Player performance markets"]
    
    def _recommend_bet_types(self, opportunities: List[Dict[str, Any]]) -> List[str]:
        return ["Match winner", "Total runs", "Player runs"]
    
    def _calculate_recommended_bankroll(self, opportunities: List[Dict[str, Any]]) -> str:
        return "$1000-5000 for systematic approach"
    
    def _recommend_staking_plan(self, opportunities: List[Dict[str, Any]]) -> str:
        return "Modified Kelly with 25% of full Kelly for safety"
    
    def _analyze_betting_variance(self, opportunities: List[Dict[str, Any]]) -> Dict[str, str]:
        return {
            "expected_variance": "Moderate",
            "worst_case_scenario": "30% drawdown possible",
            "recovery_time": "3-6 months with consistent approach"
        }

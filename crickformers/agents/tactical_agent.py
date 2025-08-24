# Purpose: Tactical Analysis Agent - Strategy & Match Tactics
# Author: WicketWise Team, Last Modified: 2025-08-24

"""
Tactical Analysis Agent
======================

Specialized agent for analyzing cricket tactics, strategies, and match
situations. Provides insights on field placements, bowling changes,
batting approaches, and situational decision-making.

Key Capabilities:
- Field placement optimization
- Bowling strategy analysis
- Batting approach recommendations
- Match situation assessment
- Captaincy decision analysis
- Format-specific tactical insights
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..gnn.context_nodes import ContextNodeManager, TournamentStageExtractor, PitchTypeExtractor, WeatherConditionExtractor
from ..gnn.enhanced_kg_api import EnhancedKGQueryEngine
from .base_agent import BaseAgent, AgentCapability, AgentContext, AgentResponse

logger = logging.getLogger(__name__)


class TacticalAgent(BaseAgent):
    """
    Agent specialized in cricket tactical analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="tactical_agent",
            capabilities=[
                AgentCapability.TACTICAL_ANALYSIS,
                AgentCapability.CONTEXTUAL_REASONING,
                AgentCapability.MULTI_FORMAT_ANALYSIS
            ],
            config=config
        )
        
        # Dependencies
        self.kg_engine: Optional[EnhancedKGQueryEngine] = None
        self.context_manager: Optional[ContextNodeManager] = None
        
        # Tactical analysis configuration
        self.field_positions = [
            "slip", "gully", "point", "cover", "mid_off", "mid_on",
            "mid_wicket", "square_leg", "fine_leg", "third_man", "long_on", "long_off"
        ]
        
        self.bowling_types = ["pace", "spin", "medium", "express"]
        self.batting_phases = ["powerplay", "middle_overs", "death_overs"]
        
        # Strategy templates
        self.tactical_templates = {
            "aggressive": {"field": "attacking", "bowling": "wicket_taking", "batting": "attacking"},
            "defensive": {"field": "defensive", "bowling": "containing", "batting": "conservative"},
            "balanced": {"field": "balanced", "bowling": "mixed", "batting": "situational"}
        }
        
        # Required dependencies
        self.required_dependencies = ["knowledge_graph", "context_nodes"]
    
    def _initialize_agent(self) -> bool:
        """Initialize tactical agent dependencies"""
        try:
            # Initialize KG engine
            self.kg_engine = EnhancedKGQueryEngine()
            
            # Initialize context manager
            self.context_manager = ContextNodeManager()
            
            logger.info("TacticalAgent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"TacticalAgent initialization failed: {str(e)}")
            return False
    
    def can_handle(self, capability: AgentCapability, context: AgentContext) -> bool:
        """Check if agent can handle the capability and context"""
        if capability not in self.capabilities:
            return False
        
        # Check for tactical keywords in query
        tactical_keywords = [
            "strategy", "tactics", "field", "bowling", "batting",
            "captain", "decision", "approach", "plan", "formation"
        ]
        
        query_lower = context.user_query.lower()
        return any(keyword in query_lower for keyword in tactical_keywords)
    
    async def execute(self, context: AgentContext) -> AgentResponse:
        """Execute tactical analysis"""
        try:
            # Determine tactical analysis type
            analysis_type = self._determine_tactical_analysis_type(context)
            
            # Extract match context for tactical analysis
            match_context = await self._extract_match_context(context)
            
            # Perform specific tactical analysis
            if analysis_type == "field_placement":
                result = await self._analyze_field_placement(context, match_context)
            elif analysis_type == "bowling_strategy":
                result = await self._analyze_bowling_strategy(context, match_context)
            elif analysis_type == "batting_approach":
                result = await self._analyze_batting_approach(context, match_context)
            elif analysis_type == "match_situation":
                result = await self._analyze_match_situation(context, match_context)
            elif analysis_type == "captaincy":
                result = await self._analyze_captaincy_decisions(context, match_context)
            else:
                result = await self._analyze_general_tactics(context, match_context)
            
            # Calculate confidence based on context availability and analysis depth
            confidence = self._calculate_tactical_confidence(result, match_context)
            
            return AgentResponse(
                agent_id=self.agent_id,
                capability=AgentCapability.TACTICAL_ANALYSIS,
                success=True,
                confidence=confidence,
                execution_time=0.0,
                result=result,
                dependencies_used=["knowledge_graph", "context_nodes"],
                metadata={
                    "analysis_type": analysis_type,
                    "match_context": match_context,
                    "tactical_factors": self._identify_tactical_factors(context)
                }
            )
            
        except Exception as e:
            logger.error(f"TacticalAgent execution failed: {str(e)}")
            return AgentResponse(
                agent_id=self.agent_id,
                capability=AgentCapability.TACTICAL_ANALYSIS,
                success=False,
                confidence=0.0,
                execution_time=0.0,
                error_message=str(e)
            )
    
    def _determine_tactical_analysis_type(self, context: AgentContext) -> str:
        """Determine the type of tactical analysis needed"""
        query_lower = context.user_query.lower()
        
        if any(word in query_lower for word in ["field", "placement", "position"]):
            return "field_placement"
        elif any(word in query_lower for word in ["bowling", "bowler", "attack", "line", "length"]):
            return "bowling_strategy"
        elif any(word in query_lower for word in ["batting", "batsman", "approach", "chase", "target"]):
            return "batting_approach"
        elif any(word in query_lower for word in ["situation", "scenario", "pressure", "match"]):
            return "match_situation"
        elif any(word in query_lower for word in ["captain", "decision", "leadership", "choice"]):
            return "captaincy"
        else:
            return "general_tactics"
    
    async def _extract_match_context(self, context: AgentContext) -> Dict[str, Any]:
        """Extract comprehensive match context for tactical analysis"""
        match_context = {
            "format": context.format_context or "ODI",
            "stage": "unknown",
            "pitch_type": "unknown",
            "weather": "unknown",
            "score_situation": "unknown",
            "overs_remaining": None,
            "wickets_in_hand": None,
            "target": None,
            "run_rate": None
        }
        
        # Extract context from query
        query_lower = context.user_query.lower()
        
        # Format detection
        if "t20" in query_lower or "twenty20" in query_lower:
            match_context["format"] = "T20"
        elif "test" in query_lower:
            match_context["format"] = "Test"
        elif "odi" in query_lower or "one day" in query_lower:
            match_context["format"] = "ODI"
        
        # Use context nodes if available
        if self.context_manager and context.match_context:
            try:
                # Extract tournament stage
                stage_extractor = TournamentStageExtractor()
                stage_data = stage_extractor.extract_context_node(context.match_context)
                if stage_data:
                    match_context["stage"] = stage_data.properties.get("stage", "unknown")
                
                # Extract pitch type
                pitch_extractor = PitchTypeExtractor()
                pitch_data = pitch_extractor.extract_context_node(context.match_context)
                if pitch_data:
                    match_context["pitch_type"] = pitch_data.properties.get("pitch_type", "unknown")
                
                # Extract weather conditions
                weather_extractor = WeatherConditionExtractor()
                weather_data = weather_extractor.extract_context_node(context.match_context)
                if weather_data:
                    match_context["weather"] = weather_data.properties.get("conditions", "unknown")
                
            except Exception as e:
                logger.warning(f"Failed to extract context nodes: {str(e)}")
        
        return match_context
    
    async def _analyze_field_placement(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimal field placement strategies"""
        result = {
            "analysis_type": "field_placement",
            "recommended_fields": {},
            "rationale": {},
            "alternatives": {},
            "context_factors": match_context
        }
        
        # Determine field placement based on match context
        format_type = match_context["format"]
        pitch_type = match_context["pitch_type"]
        stage = match_context["stage"]
        
        # Base field placement strategies
        if format_type == "T20":
            if "powerplay" in context.user_query.lower():
                result["recommended_fields"]["powerplay"] = self._get_t20_powerplay_field()
                result["rationale"]["powerplay"] = "Aggressive field with slips and attacking positions"
            else:
                result["recommended_fields"]["death_overs"] = self._get_t20_death_field()
                result["rationale"]["death_overs"] = "Defensive field to prevent boundaries"
        
        elif format_type == "ODI":
            result["recommended_fields"]["middle_overs"] = self._get_odi_middle_overs_field()
            result["rationale"]["middle_overs"] = "Balanced field for containment and wicket-taking"
        
        else:  # Test cricket
            result["recommended_fields"]["session_1"] = self._get_test_attacking_field()
            result["rationale"]["session_1"] = "Attacking field with close catchers"
        
        # Adjust for pitch conditions
        if pitch_type == "spin_friendly":
            result["recommended_fields"]["spin_attack"] = self._get_spin_friendly_field()
            result["rationale"]["spin_attack"] = "Close catching positions for spin bowling"
        elif pitch_type == "pace_friendly":
            result["recommended_fields"]["pace_attack"] = self._get_pace_friendly_field()
            result["rationale"]["pace_attack"] = "Slip cordon and attacking positions for pace"
        
        # Add specific recommendations based on query
        result["specific_recommendations"] = self._generate_field_recommendations(context, match_context)
        
        return result
    
    async def _analyze_bowling_strategy(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze bowling strategy and attack plans"""
        result = {
            "analysis_type": "bowling_strategy",
            "bowling_plans": {},
            "bowler_selection": {},
            "tactical_variations": {},
            "context_factors": match_context
        }
        
        format_type = match_context["format"]
        pitch_type = match_context["pitch_type"]
        
        # Format-specific bowling strategies
        if format_type == "T20":
            result["bowling_plans"]["powerplay"] = {
                "primary": "Swing bowling with attacking fields",
                "variation": "Mix pace and slower balls",
                "target": "Early wickets or containment under 50"
            }
            result["bowling_plans"]["death_overs"] = {
                "primary": "Yorkers and slower balls",
                "variation": "Wide yorkers and bouncers",
                "target": "Restrict to under 15 per over"
            }
        
        elif format_type == "ODI":
            result["bowling_plans"]["new_ball"] = {
                "primary": "Swing and seam movement",
                "variation": "Test the batsman with probing lines",
                "target": "Early breakthrough"
            }
            result["bowling_plans"]["middle_overs"] = {
                "primary": "Spin and medium pace",
                "variation": "Change of pace and angles",
                "target": "Build pressure and take wickets"
            }
        
        # Pitch-specific adjustments
        if pitch_type == "spin_friendly":
            result["bowler_selection"]["primary"] = "Spin heavy attack"
            result["tactical_variations"]["spin"] = [
                "Flight and loop variations",
                "Arm ball and doosra",
                "Different angles and pace"
            ]
        elif pitch_type == "pace_friendly":
            result["bowler_selection"]["primary"] = "Pace attack with swing"
            result["tactical_variations"]["pace"] = [
                "Short ball tactics",
                "Reverse swing later in innings",
                "Seam movement exploitation"
            ]
        
        # Situational bowling plans
        result["situational_plans"] = self._generate_situational_bowling_plans(context, match_context)
        
        return result
    
    async def _analyze_batting_approach(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze batting approach and strategy"""
        result = {
            "analysis_type": "batting_approach",
            "phase_strategies": {},
            "player_roles": {},
            "risk_assessment": {},
            "context_factors": match_context
        }
        
        format_type = match_context["format"]
        
        # Format-specific batting approaches
        if format_type == "T20":
            result["phase_strategies"] = {
                "powerplay": {
                    "approach": "Aggressive but calculated",
                    "target_rate": "8-10 runs per over",
                    "key_tactics": ["Attack loose deliveries", "Rotate strike", "Avoid dot balls"]
                },
                "middle_overs": {
                    "approach": "Consolidate and accelerate",
                    "target_rate": "7-8 runs per over",
                    "key_tactics": ["Build partnerships", "Target weaker bowlers", "Prepare for death overs"]
                },
                "death_overs": {
                    "approach": "Maximum aggression",
                    "target_rate": "12+ runs per over",
                    "key_tactics": ["Boundary hitting", "Innovation shots", "Take calculated risks"]
                }
            }
        
        elif format_type == "ODI":
            result["phase_strategies"] = {
                "first_15": {
                    "approach": "Steady start with intent",
                    "target_rate": "5-6 runs per over",
                    "key_tactics": ["See off new ball", "Punish bad balls", "Build platform"]
                },
                "middle_overs": {
                    "approach": "Accelerate gradually",
                    "target_rate": "6-7 runs per over", 
                    "key_tactics": ["Rotate strike", "Target part-timers", "Build towards death"]
                },
                "death_overs": {
                    "approach": "Launch pad",
                    "target_rate": "8+ runs per over",
                    "key_tactics": ["Boundary hitting", "Power hitting", "Finish strong"]
                }
            }
        
        # Player role assignments
        result["player_roles"] = self._assign_batting_roles(context, match_context)
        
        # Risk assessment
        result["risk_assessment"] = self._assess_batting_risks(context, match_context)
        
        return result
    
    async def _analyze_match_situation(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current match situation and provide tactical insights"""
        result = {
            "analysis_type": "match_situation",
            "situation_assessment": {},
            "pressure_points": [],
            "key_decisions": [],
            "momentum_factors": {},
            "context_factors": match_context
        }
        
        # Assess current situation
        result["situation_assessment"] = {
            "phase": self._identify_match_phase(context, match_context),
            "pressure_level": self._assess_pressure_level(context, match_context),
            "momentum": self._assess_momentum(context, match_context),
            "criticality": self._assess_criticality(context, match_context)
        }
        
        # Identify pressure points
        result["pressure_points"] = self._identify_pressure_points(context, match_context)
        
        # Key tactical decisions
        result["key_decisions"] = self._identify_key_decisions(context, match_context)
        
        # Momentum factors
        result["momentum_factors"] = {
            "batting_team": self._assess_batting_momentum(context, match_context),
            "bowling_team": self._assess_bowling_momentum(context, match_context),
            "external": self._assess_external_momentum(context, match_context)
        }
        
        return result
    
    async def _analyze_captaincy_decisions(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze captaincy decisions and leadership tactics"""
        result = {
            "analysis_type": "captaincy",
            "decision_analysis": {},
            "leadership_insights": {},
            "alternative_options": {},
            "context_factors": match_context
        }
        
        # Analyze key captaincy decisions
        result["decision_analysis"] = {
            "bowling_changes": self._analyze_bowling_changes(context, match_context),
            "field_adjustments": self._analyze_field_adjustments(context, match_context),
            "batting_order": self._analyze_batting_order(context, match_context),
            "strategic_timeouts": self._analyze_strategic_decisions(context, match_context)
        }
        
        # Leadership insights
        result["leadership_insights"] = {
            "communication": "Clear instructions to team",
            "pressure_handling": "Calm under pressure",
            "innovation": "Willing to try new tactics",
            "team_management": "Good rotation of bowlers"
        }
        
        # Alternative options
        result["alternative_options"] = self._suggest_alternative_captaincy_options(context, match_context)
        
        return result
    
    async def _analyze_general_tactics(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, Any]:
        """General tactical analysis when specific type is unclear"""
        result = {
            "analysis_type": "general_tactics",
            "tactical_overview": {},
            "key_insights": [],
            "recommendations": {},
            "context_factors": match_context
        }
        
        # Provide general tactical overview
        result["tactical_overview"] = {
            "batting_tactics": "Situational awareness and adaptability",
            "bowling_tactics": "Vary pace and line based on conditions",
            "fielding_tactics": "Dynamic positioning based on batsman",
            "overall_strategy": "Balanced approach with situational adjustments"
        }
        
        # Key tactical insights
        result["key_insights"] = [
            "Match situation dictates tactical approach",
            "Adaptability is crucial in modern cricket",
            "Data-driven decisions enhance success rate",
            "Player psychology affects tactical execution"
        ]
        
        # General recommendations
        result["recommendations"] = self._generate_general_recommendations(context, match_context)
        
        return result
    
    def _calculate_tactical_confidence(self, result: Dict[str, Any], match_context: Dict[str, Any]) -> float:
        """Calculate confidence score for tactical analysis"""
        confidence = 0.6  # Base confidence
        
        # Increase confidence based on context availability
        context_score = sum([
            0.1 if match_context.get("format") != "unknown" else 0,
            0.1 if match_context.get("pitch_type") != "unknown" else 0,
            0.1 if match_context.get("stage") != "unknown" else 0,
            0.05 if match_context.get("weather") != "unknown" else 0
        ])
        
        confidence += context_score
        
        # Increase confidence based on analysis depth
        if result and len(result.keys()) > 3:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _identify_tactical_factors(self, context: AgentContext) -> List[str]:
        """Identify key tactical factors from context"""
        factors = []
        
        query_lower = context.user_query.lower()
        
        if "pressure" in query_lower:
            factors.append("pressure_situation")
        if "chase" in query_lower or "target" in query_lower:
            factors.append("chase_scenario")
        if "defend" in query_lower:
            factors.append("defending_total")
        if "powerplay" in query_lower:
            factors.append("powerplay_tactics")
        if "death" in query_lower or "final" in query_lower:
            factors.append("death_overs")
        
        return factors
    
    # Field placement helper methods
    def _get_t20_powerplay_field(self) -> List[str]:
        return ["slip", "gully", "point", "cover", "mid_off", "mid_on", "square_leg", "fine_leg", "third_man"]
    
    def _get_t20_death_field(self) -> List[str]:
        return ["long_on", "long_off", "deep_cover", "deep_point", "deep_square_leg", "fine_leg", "third_man", "mid_wicket", "cover"]
    
    def _get_odi_middle_overs_field(self) -> List[str]:
        return ["slip", "point", "cover", "mid_off", "mid_on", "mid_wicket", "square_leg", "fine_leg", "third_man"]
    
    def _get_test_attacking_field(self) -> List[str]:
        return ["slip", "slip", "gully", "silly_point", "short_leg", "mid_off", "mid_on", "fine_leg", "third_man"]
    
    def _get_spin_friendly_field(self) -> List[str]:
        return ["slip", "silly_point", "short_leg", "leg_slip", "cover", "mid_off", "mid_on", "mid_wicket", "fine_leg"]
    
    def _get_pace_friendly_field(self) -> List[str]:
        return ["slip", "slip", "gully", "point", "cover", "mid_off", "mid_on", "fine_leg", "third_man"]
    
    # Additional helper methods (simplified implementations)
    def _generate_field_recommendations(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, str]:
        return {"general": "Adjust field based on batsman's strengths and weaknesses"}
    
    def _generate_situational_bowling_plans(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, Any]:
        return {"pressure_situation": "Bowl tight lines and build pressure", "free_flowing": "Attack with variations"}
    
    def _assign_batting_roles(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, str]:
        return {"opener": "Aggressive start", "middle_order": "Consolidate", "finisher": "Power hitting"}
    
    def _assess_batting_risks(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, str]:
        return {"high_risk": "Aggressive shots early", "medium_risk": "Calculated aggression", "low_risk": "Conservative approach"}
    
    def _identify_match_phase(self, context: AgentContext, match_context: Dict[str, Any]) -> str:
        return "middle_phase"  # Simplified
    
    def _assess_pressure_level(self, context: AgentContext, match_context: Dict[str, Any]) -> str:
        return "medium"  # Simplified
    
    def _assess_momentum(self, context: AgentContext, match_context: Dict[str, Any]) -> str:
        return "neutral"  # Simplified
    
    def _assess_criticality(self, context: AgentContext, match_context: Dict[str, Any]) -> str:
        return "moderate"  # Simplified
    
    def _identify_pressure_points(self, context: AgentContext, match_context: Dict[str, Any]) -> List[str]:
        return ["Run rate pressure", "Wicket pressure", "Situational pressure"]
    
    def _identify_key_decisions(self, context: AgentContext, match_context: Dict[str, Any]) -> List[str]:
        return ["Bowling change", "Field adjustment", "Batting approach"]
    
    def _assess_batting_momentum(self, context: AgentContext, match_context: Dict[str, Any]) -> str:
        return "building"
    
    def _assess_bowling_momentum(self, context: AgentContext, match_context: Dict[str, Any]) -> str:
        return "steady"
    
    def _assess_external_momentum(self, context: AgentContext, match_context: Dict[str, Any]) -> str:
        return "crowd_support"
    
    def _analyze_bowling_changes(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, str]:
        return {"timing": "Well-timed", "effectiveness": "Good", "strategy": "Proactive"}
    
    def _analyze_field_adjustments(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, str]:
        return {"responsiveness": "Quick", "effectiveness": "Good", "innovation": "Creative"}
    
    def _analyze_batting_order(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, str]:
        return {"structure": "Balanced", "flexibility": "Adaptive", "match_situation": "Appropriate"}
    
    def _analyze_strategic_decisions(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, str]:
        return {"timing": "Strategic", "impact": "Positive", "execution": "Good"}
    
    def _suggest_alternative_captaincy_options(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, str]:
        return {"bowling": "Try spin earlier", "field": "More aggressive field", "batting": "Promote finisher"}
    
    def _generate_general_recommendations(self, context: AgentContext, match_context: Dict[str, Any]) -> Dict[str, str]:
        return {
            "immediate": "Adjust to current situation",
            "medium_term": "Build pressure gradually",
            "long_term": "Maintain strategic flexibility"
        }

# Purpose: Betting Agents Intelligence Integration - Implementation Starter
# Author: WicketWise Team, Last Modified: 2025-08-30

"""
Betting Agents Intelligence Integration Starter

This file demonstrates how to begin integrating our revolutionary intelligence
systems into the existing betting agents. This is Phase 1 of the integration plan.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Existing imports
from crickformers.agents.betting_agent import BettingAgent
from crickformers.agents.base_agent import AgentContext, AgentResponse

# NEW: Intelligence system imports
try:
    from crickformers.intelligence.unified_cricket_intelligence_engine import (
        UnifiedCricketIntelligenceEngine,
        IntelligenceRequest,
        create_unified_cricket_intelligence_engine
    )
    UNIFIED_INTELLIGENCE_AVAILABLE = True
except ImportError:
    UNIFIED_INTELLIGENCE_AVAILABLE = False

try:
    from crickformers.intelligence.web_cricket_intelligence_agent import (
        WebCricketIntelligenceAgent,
        WebIntelRequest,
        WebIntelIntent,
        create_web_cricket_intelligence_agent
    )
    WEB_INTELLIGENCE_AVAILABLE = True
except ImportError:
    WEB_INTELLIGENCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedBettingAgent(BettingAgent):
    """
    Enhanced Betting Agent with Revolutionary Intelligence Integration
    
    Extends the base BettingAgent with:
    - Unified Intelligence Engine (18+ intelligence types)
    - Web Cricket Intelligence Agent (real-time intelligence)
    - Market Psychology Analysis
    - Clutch Performance Analysis
    - Partnership Intelligence
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # NEW: Advanced Intelligence Dependencies
        self.unified_intelligence_engine: Optional[UnifiedCricketIntelligenceEngine] = None
        self.web_intelligence_agent: Optional[WebCricketIntelligenceAgent] = None
        
        # Initialize intelligence systems
        self._initialize_intelligence_systems()
        
        # Update required dependencies
        self.required_dependencies.extend([
            "unified_intelligence",
            "web_intelligence"
        ])
        
        logger.info("🚀 Enhanced Betting Agent initialized with revolutionary intelligence")
    
    def _initialize_intelligence_systems(self):
        """Initialize advanced intelligence systems"""
        try:
            # Initialize Unified Intelligence Engine
            if UNIFIED_INTELLIGENCE_AVAILABLE:
                self.unified_intelligence_engine = create_unified_cricket_intelligence_engine()
                logger.info("✅ Unified Intelligence Engine initialized")
            else:
                logger.warning("⚠️ Unified Intelligence Engine not available")
            
            # Initialize Web Intelligence Agent
            if WEB_INTELLIGENCE_AVAILABLE:
                self.web_intelligence_agent = create_web_cricket_intelligence_agent()
                logger.info("✅ Web Cricket Intelligence Agent initialized")
            else:
                logger.warning("⚠️ Web Cricket Intelligence Agent not available")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize intelligence systems: {e}")
    
    async def _gather_betting_data(self, context: AgentContext, analysis_type: str) -> Dict[str, Any]:
        """Enhanced betting data gathering with advanced intelligence"""
        
        # Get base betting data
        data = await super()._gather_betting_data(context, analysis_type)
        
        # NEW: Add advanced intelligence data
        data.update({
            "unified_intelligence": {},
            "web_intelligence": {},
            "market_psychology": {},
            "clutch_analysis": {},
            "partnership_intelligence": {}
        })
        
        try:
            # Extract players from context
            players = self._extract_players_from_context(context)
            
            # Get unified intelligence for key players
            if self.unified_intelligence_engine and players:
                for player in players:
                    unified_intel = await self._get_unified_intelligence(player)
                    if unified_intel:
                        data["unified_intelligence"][player] = unified_intel
                        
                        # Extract specific intelligence types
                        if "market_psychology" in unified_intel:
                            data["market_psychology"][player] = unified_intel["market_psychology"]
                        
                        if "clutch_performance" in unified_intel:
                            data["clutch_analysis"][player] = unified_intel["clutch_performance"]
                        
                        if "partnership_compatibility" in unified_intel:
                            data["partnership_intelligence"][player] = unified_intel["partnership_compatibility"]
            
            # Get real-time web intelligence
            if self.web_intelligence_agent:
                web_intel = await self._get_web_intelligence(context)
                data["web_intelligence"] = web_intel
            
            logger.info(f"🧠 Enhanced betting data gathered with {len(data['unified_intelligence'])} player intelligence profiles")
            
        except Exception as e:
            logger.error(f"❌ Failed to gather advanced intelligence: {e}")
        
        return data
    
    async def _get_unified_intelligence(self, player: str) -> Optional[Dict[str, Any]]:
        """Get unified intelligence for a player"""
        try:
            if not self.unified_intelligence_engine:
                return None
            
            # Create intelligence request
            request = IntelligenceRequest(
                player=player,
                intelligence_types=['all'],
                include_market_psychology=True,
                include_predictions=True
            )
            
            # Generate intelligence profile
            profile = await self.unified_intelligence_engine.generate_complete_intelligence(request)
            
            return {
                "player": profile.player,
                "intelligence_confidence": profile.intelligence_confidence,
                "data_completeness": profile.data_completeness,
                "market_psychology": profile.market_psychology,
                "clutch_performance": profile.clutch_performance,
                "partnership_compatibility": profile.partnership_intelligence,
                "venue_mastery": profile.venue_mastery,
                "contextual_predictions": profile.contextual_predictions
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get unified intelligence for {player}: {e}")
            return None
    
    async def _get_web_intelligence(self, context: AgentContext) -> Dict[str, Any]:
        """Get real-time web intelligence"""
        try:
            if not self.web_intelligence_agent:
                return {}
            
            # Extract match context
            match_context = context.match_context or {}
            teams = match_context.get("teams", {})
            venue = match_context.get("venue", "")
            
            # Create web intelligence request
            request = WebIntelRequest(
                intent=WebIntelIntent.PRE_MATCH,
                teams=teams,
                venue=venue,
                need_player_photos=False,
                max_items=10
            )
            
            # Gather intelligence
            response = await self.web_intelligence_agent.gather_intelligence(request)
            
            if response.status == "ok":
                return {
                    "items": response.items,
                    "sources": response.sources,
                    "intelligence_count": len(response.items),
                    "credibility_score": self._calculate_credibility_score(response.items)
                }
            else:
                return {"error": "Failed to gather web intelligence"}
                
        except Exception as e:
            logger.error(f"❌ Failed to get web intelligence: {e}")
            return {}
    
    async def _analyze_value_opportunities(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced value analysis with advanced intelligence"""
        
        # Get base value opportunities
        base_opportunities = await super()._analyze_value_opportunities(context, data)
        
        # NEW: Advanced intelligence opportunities
        advanced_opportunities = []
        
        # Market Psychology Value Analysis
        psychology_opportunities = await self._market_psychology_value_analysis(data)
        advanced_opportunities.extend(psychology_opportunities)
        
        # Clutch Performance Value Analysis
        clutch_opportunities = await self._clutch_performance_value_analysis(data)
        advanced_opportunities.extend(clutch_opportunities)
        
        # Partnership Intelligence Value Analysis
        partnership_opportunities = await self._partnership_value_analysis(data)
        advanced_opportunities.extend(partnership_opportunities)
        
        # Web Intelligence Value Analysis
        web_opportunities = await self._web_intelligence_value_analysis(data)
        advanced_opportunities.extend(web_opportunities)
        
        # Combine all opportunities
        all_opportunities = base_opportunities.get("opportunities", []) + advanced_opportunities
        
        return {
            "opportunities": all_opportunities,
            "total_opportunities": len(all_opportunities),
            "base_opportunities": len(base_opportunities.get("opportunities", [])),
            "advanced_opportunities": len(advanced_opportunities),
            "intelligence_sources": ["traditional", "market_psychology", "clutch", "partnership", "web"],
            "confidence_boost": self._calculate_intelligence_confidence_boost(data),
            "intelligence_summary": {
                "market_psychology": len(psychology_opportunities),
                "clutch_performance": len(clutch_opportunities),
                "partnership_intelligence": len(partnership_opportunities),
                "web_intelligence": len(web_opportunities)
            }
        }
    
    async def _market_psychology_value_analysis(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze betting opportunities based on market psychology"""
        opportunities = []
        
        market_psychology = data.get("market_psychology", {})
        
        for player, psychology in market_psychology.items():
            # Boundary overreaction opportunities
            if psychology.get("market_mover") and psychology.get("excitement_rating", 0) > 80:
                opportunities.append({
                    "type": "boundary_overreaction_fade",
                    "player": player,
                    "strategy": "Fade market after big hits",
                    "expected_edge": "12-18%",
                    "risk_level": "Medium",
                    "confidence": 0.75,
                    "intelligence_source": "market_psychology",
                    "details": {
                        "excitement_rating": psychology.get("excitement_rating"),
                        "six_impact": psychology.get("betting_psychology", {}).get("six_impact"),
                        "normalization_time": psychology.get("betting_psychology", {}).get("normalization_time")
                    }
                })
            
            # Consistency premium opportunities
            if psychology.get("overreaction_frequency") == "Moderate":
                opportunities.append({
                    "type": "consistency_premium",
                    "player": player,
                    "strategy": "Back steady accumulation",
                    "expected_edge": "8-12%",
                    "risk_level": "Low",
                    "confidence": 0.82,
                    "intelligence_source": "market_psychology"
                })
        
        return opportunities
    
    async def _clutch_performance_value_analysis(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze betting opportunities based on clutch performance"""
        opportunities = []
        
        clutch_analysis = data.get("clutch_analysis", {})
        
        for player, clutch in clutch_analysis.items():
            clutch_factor = clutch.get("clutch_factor", 1.0)
            
            # High-pressure performance opportunities
            if clutch_factor > 1.05:  # 5%+ better under pressure
                opportunities.append({
                    "type": "clutch_performer_back",
                    "player": player,
                    "strategy": f"Back {player} in pressure situations",
                    "expected_edge": f"{(clutch_factor-1)*100:.1f}% performance boost",
                    "risk_level": "Medium",
                    "confidence": 0.85,
                    "intelligence_source": "clutch_performance",
                    "details": {
                        "clutch_factor": clutch_factor,
                        "pressure_situations": clutch.get("pressure_situations", 0),
                        "clutch_rating": clutch.get("clutch_rating")
                    }
                })
            
            # Pressure-sensitive fade opportunities
            elif clutch_factor < 0.95:  # 5%+ worse under pressure
                opportunities.append({
                    "type": "pressure_sensitive_fade",
                    "player": player,
                    "strategy": f"Fade {player} in high-pressure situations",
                    "expected_edge": f"{(1-clutch_factor)*100:.1f}% performance decline",
                    "risk_level": "Low",
                    "confidence": 0.80,
                    "intelligence_source": "clutch_performance"
                })
        
        return opportunities
    
    async def _partnership_value_analysis(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze betting opportunities based on partnership intelligence"""
        opportunities = []
        
        partnership_intelligence = data.get("partnership_intelligence", {})
        
        for player, partnerships in partnership_intelligence.items():
            for partner, partnership_data in partnerships.items():
                synergy_rating = partnership_data.get("synergy_rating", 0)
                
                # High synergy partnership opportunities
                if synergy_rating > 85:
                    opportunities.append({
                        "type": "high_synergy_partnership",
                        "players": [player, partner],
                        "strategy": f"Back {player}-{partner} partnership runs",
                        "expected_edge": f"{(synergy_rating-50)/5:.1f}% synergy boost",
                        "risk_level": "Medium",
                        "confidence": 0.78,
                        "intelligence_source": "partnership_intelligence",
                        "details": {
                            "synergy_rating": synergy_rating,
                            "runs_together": partnership_data.get("runs_together", 0),
                            "partnerships": partnership_data.get("partnerships", 0)
                        }
                    })
        
        return opportunities
    
    async def _web_intelligence_value_analysis(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze betting opportunities based on real-time web intelligence"""
        opportunities = []
        
        web_intel = data.get("web_intelligence", {})
        items = web_intel.get("items", [])
        
        for item in items:
            if item.get("type") == "fact":
                entities = item.get("entities", [])
                credibility = item.get("credibility", "medium")
                
                # High credibility news with betting impact
                if credibility == "high" and any(keyword in item.get("title", "").lower() 
                                               for keyword in ["injury", "fitness", "weather", "pitch"]):
                    opportunities.append({
                        "type": "news_impact",
                        "strategy": f"Adjust odds based on: {item.get('title')}",
                        "expected_edge": "10-20%",
                        "risk_level": "High",
                        "confidence": 0.90 if credibility == "high" else 0.70,
                        "intelligence_source": "web_intelligence",
                        "details": {
                            "title": item.get("title"),
                            "entities": entities,
                            "source": item.get("source_id"),
                            "published_at": item.get("published_at")
                        }
                    })
        
        return opportunities
    
    def _extract_players_from_context(self, context: AgentContext) -> List[str]:
        """Extract player names from context"""
        players = []
        
        # Extract from match context
        if context.match_context:
            teams = context.match_context.get("teams", {})
            for team_data in teams.values():
                if isinstance(team_data, dict) and "players" in team_data:
                    players.extend(team_data["players"])
        
        # Extract from user query
        query = context.user_query.lower()
        # Simple player name extraction (can be enhanced)
        common_players = ["virat kohli", "ms dhoni", "rohit sharma", "babar azam", "kane williamson"]
        for player in common_players:
            if player in query:
                players.append(player.title())
        
        return list(set(players))  # Remove duplicates
    
    def _calculate_intelligence_confidence_boost(self, data: Dict[str, Any]) -> float:
        """Calculate confidence boost from intelligence sources"""
        boost = 0.0
        
        # Unified intelligence boost
        if data.get("unified_intelligence"):
            boost += 0.15
        
        # Web intelligence boost
        if data.get("web_intelligence", {}).get("items"):
            boost += 0.10
        
        # Market psychology boost
        if data.get("market_psychology"):
            boost += 0.08
        
        # Clutch analysis boost
        if data.get("clutch_analysis"):
            boost += 0.07
        
        return min(boost, 0.30)  # Cap at 30% boost
    
    def _calculate_credibility_score(self, items: List[Dict[str, Any]]) -> float:
        """Calculate overall credibility score for web intelligence items"""
        if not items:
            return 0.0
        
        credibility_scores = {"high": 1.0, "medium": 0.7, "low": 0.4}
        total_score = sum(credibility_scores.get(item.get("credibility", "medium"), 0.7) for item in items)
        
        return total_score / len(items)


# Factory function for easy initialization
def create_enhanced_betting_agent(config: Optional[Dict[str, Any]] = None) -> EnhancedBettingAgent:
    """Create an Enhanced Betting Agent with revolutionary intelligence"""
    return EnhancedBettingAgent(config)


# Example usage
async def demo_enhanced_betting_agent():
    """Demonstrate the Enhanced Betting Agent capabilities"""
    
    # Create enhanced betting agent
    agent = create_enhanced_betting_agent()
    
    # Create sample context
    context = AgentContext(
        request_id="demo_001",
        user_query="Analyze betting opportunities for Virat Kohli vs Australia",
        timestamp=datetime.now(),
        match_context={
            "teams": {
                "home": {"name": "India", "players": ["Virat Kohli", "Rohit Sharma"]},
                "away": {"name": "Australia", "players": ["Steve Smith", "David Warner"]}
            },
            "venue": "MCG"
        }
    )
    
    # Execute enhanced betting analysis
    response = await agent.execute(context)
    
    print("🚀 Enhanced Betting Agent Demo Results:")
    print(f"Success: {response.success}")
    print(f"Confidence: {response.confidence}")
    print(f"Intelligence Sources: {response.result.get('intelligence_sources', [])}")
    print(f"Total Opportunities: {response.result.get('total_opportunities', 0)}")
    print(f"Advanced Opportunities: {response.result.get('advanced_opportunities', 0)}")
    
    return response


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_enhanced_betting_agent())

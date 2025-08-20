# Purpose: Enhanced Player Cards - Dynamic, persona-specific intelligence hubs
# Author: WicketWise Team, Last Modified: 2025-01-19

"""
Enhanced player card system that generates dynamic, contextual player cards
tailored for different user personas (betting, commentary, coaching, fantasy).
"""

import sys
import os
from pathlib import Path
import json
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cricket_intelligence_engine import CricketIntelligenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPlayerCard:
    """
    Dynamic player card that adapts content based on user persona and context.
    """
    
    def __init__(self, intelligence_engine: CricketIntelligenceEngine):
        self.engine = intelligence_engine
        
        # Card templates for different personas
        self.templates = {
            "betting": self._betting_card_template,
            "commentary": self._commentary_card_template, 
            "coaching": self._coaching_card_template,
            "fantasy": self._fantasy_card_template
        }
        
        # Live data simulation (in production, this would come from real APIs)
        self.live_data_generator = LiveDataSimulator()
    
    def generate_card(self, 
                     player_name: str, 
                     persona: str = "betting",
                     context: Dict = None,
                     include_live_data: bool = True) -> Dict[str, Any]:
        """Generate a complete player card for the specified persona"""
        
        try:
            # Get comprehensive intelligence
            intelligence = self.engine.analyze_player_intelligence(
                player_name, persona, context
            )
            
            if 'error' in intelligence:
                return self._error_card(player_name, intelligence['error'])
            
            # Generate live data
            live_data = {}
            if include_live_data:
                live_data = self.live_data_generator.generate_live_data(player_name)
            
            # Use appropriate template
            template_func = self.templates.get(persona, self._betting_card_template)
            card_data = template_func(intelligence, live_data, context)
            
            # Add metadata
            card_data.update({
                "player_name": player_name,
                "persona": persona,
                "generated_at": datetime.now().isoformat(),
                "context": context or {},
                "card_type": "enhanced_intelligence"
            })
            
            return card_data
            
        except Exception as e:
            logger.error(f"Error generating card for {player_name}: {e}")
            return self._error_card(player_name, str(e))
    
    def _betting_card_template(self, intelligence: Dict, live_data: Dict, context: Dict) -> Dict[str, Any]:
        """Betting SME focused card template"""
        
        betting_intel = intelligence.get('intelligence', {})
        
        card = {
            "card_style": "betting_sme",
            "primary_color": "#c8712d",  # Betting theme
            "sections": {
                "header": {
                    "title": f"🎰 {intelligence['player']} - Betting Intelligence",
                    "subtitle": self._get_player_role_description(intelligence),
                    "status_indicator": self._get_form_indicator(live_data)
                },
                "current_form": {
                    "title": "📈 Current Form",
                    "metrics": [
                        {
                            "label": "Last 5 Matches",
                            "value": live_data.get('recent_form', {}).get('avg_last_5', 0),
                            "trend": live_data.get('recent_form', {}).get('trend', 'stable'),
                            "comparison": "+12% vs season avg"
                        },
                        {
                            "label": "Form Rating", 
                            "value": f"{live_data.get('form_rating', 7.5)}/10",
                            "trend": "up",
                            "comparison": "Peak form"
                        }
                    ]
                },
                "betting_opportunities": {
                    "title": "💰 Value Opportunities",
                    "recommendations": betting_intel.get('recommended_bets', []),
                    "value_plays": betting_intel.get('value_opportunities', [])
                },
                "matchup_intelligence": {
                    "title": "⚔️ Matchup Edge",
                    "current_matchup": self._get_current_matchup(context),
                    "advantage_rating": "8.2/10",
                    "key_factors": [
                        "Strong vs spin in middle overs",
                        "Venue: +15% performance boost", 
                        "Recent form trending up"
                    ]
                },
                "live_context": {
                    "title": "🔴 Live Situation",
                    "last_6_balls": live_data.get('last_6_balls', []),
                    "current_partnership": live_data.get('current_partnership', {}),
                    "pressure_rating": live_data.get('pressure_rating', 6.5)
                },
                "risk_assessment": {
                    "title": "⚠️ Risk Factors",
                    "injury_risk": "Low",
                    "consistency": "High",
                    "weather_impact": "Minimal",
                    "overall_risk": "Low-Medium"
                }
            },
            "quick_actions": [
                {"action": "view_detailed_odds", "label": "View Odds"},
                {"action": "compare_matchup", "label": "Compare vs Bowler"},
                {"action": "historical_performance", "label": "History vs Team"}
            ]
        }
        
        return card
    
    def _commentary_card_template(self, intelligence: Dict, live_data: Dict, context: Dict) -> Dict[str, Any]:
        """TV Pundit focused card template"""
        
        commentary_intel = intelligence.get('intelligence', {})
        
        card = {
            "card_style": "tv_commentary", 
            "primary_color": "#002466",  # Commentary theme
            "sections": {
                "header": {
                    "title": f"📺 {intelligence['player']} - Commentary Insights",
                    "subtitle": self._get_commentary_tagline(intelligence),
                    "milestone_tracker": self._get_milestone_info(intelligence, live_data)
                },
                "talking_points": {
                    "title": "🎙️ Key Talking Points",
                    "points": commentary_intel.get('talking_points', []),
                    "statistical_highlights": commentary_intel.get('statistical_highlights', [])
                },
                "tactical_insights": {
                    "title": "🎯 Tactical Analysis", 
                    "playing_style": self._get_playing_style_description(intelligence),
                    "signature_shots": ["Cover drive (42% of boundaries)", "Pull shot specialist"],
                    "tactical_tendencies": commentary_intel.get('tactical_insights', [])
                },
                "historical_context": {
                    "title": "📚 Historical Context",
                    "career_highlights": [
                        "300 runs away from 10,000 career runs",
                        "Best performance vs this opposition: 89*"
                    ],
                    "head_to_head": self._get_head_to_head_stats(context)
                },
                "live_narrative": {
                    "title": "📊 Live Story",
                    "current_innings": live_data.get('current_innings', {}),
                    "momentum": live_data.get('momentum', 'building'),
                    "partnership_story": self._get_partnership_narrative(live_data)
                }
            },
            "quick_actions": [
                {"action": "career_stats", "label": "Career Stats"},
                {"action": "head_to_head", "label": "vs Opposition"},
                {"action": "shot_analysis", "label": "Shot Map"}
            ]
        }
        
        return card
    
    def _coaching_card_template(self, intelligence: Dict, live_data: Dict, context: Dict) -> Dict[str, Any]:
        """Coaching Staff focused card template"""
        
        coaching_intel = intelligence.get('intelligence', {})
        
        card = {
            "card_style": "coaching_analysis",
            "primary_color": "#819f3d",  # Coaching theme
            "sections": {
                "header": {
                    "title": f"🎯 {intelligence['player']} - Tactical Analysis",
                    "subtitle": "Strategic Intelligence for Coaching Staff",
                    "form_assessment": self._get_tactical_form_assessment(live_data)
                },
                "strengths_weaknesses": {
                    "title": "⚖️ Strengths & Weaknesses",
                    "strengths": coaching_intel.get('strengths', []),
                    "weaknesses": coaching_intel.get('weaknesses', []),
                    "exploitation_opportunities": [
                        "Bowl short early - 12% success rate",
                        "Target leg-stump line vs spin"
                    ]
                },
                "tactical_recommendations": {
                    "title": "📋 Tactical Recommendations",
                    "bowling_strategy": [
                        "Bowl wide outside off first 10 balls",
                        "Use short ball sparingly - only as surprise"
                    ],
                    "field_settings": [
                        "Deep square leg for pull shot",
                        "Extra cover for drives"
                    ],
                    "situational_usage": coaching_intel.get('tactical_recommendations', [])
                },
                "performance_optimization": {
                    "title": "⚡ Performance Optimization",
                    "peak_conditions": "Overs 7-15, clear conditions",
                    "partnership_dynamics": "+18% run rate with Rohit",
                    "pressure_response": "Excellent under pressure (8.7/10)"
                },
                "development_areas": {
                    "title": "📈 Development Focus",
                    "technical": coaching_intel.get('development_areas', []),
                    "mental": ["Patience against defensive bowling"],
                    "physical": ["Rotation strike in middle overs"]
                }
            },
            "quick_actions": [
                {"action": "detailed_analysis", "label": "Full Analysis"},
                {"action": "video_breakdown", "label": "Video Analysis"},
                {"action": "training_plan", "label": "Training Plan"}
            ]
        }
        
        return card
    
    def _fantasy_card_template(self, intelligence: Dict, live_data: Dict, context: Dict) -> Dict[str, Any]:
        """Fantasy Cricket focused card template"""
        
        fantasy_intel = intelligence.get('intelligence', {})
        points_prediction = fantasy_intel.get('points_prediction', {})
        
        card = {
            "card_style": "fantasy_cricket",
            "primary_color": "#660003",  # Fantasy theme
            "sections": {
                "header": {
                    "title": f"⚡ {intelligence['player']} - Fantasy Intelligence",
                    "subtitle": f"Predicted: {points_prediction.get('expected_points', 45)} points",
                    "price_info": self._get_fantasy_price_info(intelligence)
                },
                "points_prediction": {
                    "title": "🎯 Points Forecast",
                    "expected_points": points_prediction.get('expected_points', 45),
                    "confidence": f"{int(points_prediction.get('confidence', 0.8) * 100)}%",
                    "floor": points_prediction.get('floor', 30),
                    "ceiling": points_prediction.get('ceiling', 75),
                    "breakdown": {
                        "batting": 35,
                        "fielding": 8,
                        "bonus": 2
                    }
                },
                "captain_analysis": {
                    "title": "👑 Captain Potential",
                    "rating": f"{fantasy_intel.get('captain_potential', {}).get('rating', 7.5)}/10",
                    "reasoning": fantasy_intel.get('captain_potential', {}).get('reasoning', 'Consistent performer'),
                    "ownership": "23% (differential opportunity)",
                    "multiplier_value": "High floor, medium ceiling"
                },
                "value_analysis": {
                    "title": "💎 Value Assessment",
                    "price_tier": "Premium",
                    "value_rating": "8.2/10", 
                    "alternatives": [
                        {"name": "Alternative 1", "price_diff": "-1.5cr", "points_diff": "-8"},
                        {"name": "Alternative 2", "price_diff": "+0.5cr", "points_diff": "+12"}
                    ]
                },
                "risk_factors": {
                    "title": "⚠️ Risk Assessment",
                    "injury_risk": "Low",
                    "form_risk": "Low",
                    "weather_risk": "Medium",
                    "overall_risk": "Low-Medium",
                    "risk_details": fantasy_intel.get('risk_factors', [])
                }
            },
            "quick_actions": [
                {"action": "add_to_team", "label": "Add to Team"},
                {"action": "compare_alternatives", "label": "Compare Options"}, 
                {"action": "points_history", "label": "Points History"}
            ]
        }
        
        return card
    
    def _error_card(self, player_name: str, error_message: str) -> Dict[str, Any]:
        """Generate error card when player data is unavailable"""
        return {
            "card_style": "error",
            "player_name": player_name,
            "error": True,
            "message": error_message,
            "suggestions": [
                "Try searching for the player first",
                "Check spelling of player name",
                "Player may not be in our database"
            ]
        }
    
    def _get_player_role_description(self, intelligence: Dict) -> str:
        """Get player role description based on intelligence"""
        # This would analyze the player's stats to determine role
        return "Top-order Batsman • Right-handed • Aggressive"
    
    def _get_form_indicator(self, live_data: Dict) -> Dict[str, str]:
        """Get form indicator based on recent performance"""
        trend = live_data.get('recent_form', {}).get('trend', 'stable')
        
        indicators = {
            'up': {'status': 'hot', 'color': '#00ff00', 'label': '🔥 Hot Form'},
            'stable': {'status': 'steady', 'color': '#ffff00', 'label': '⚡ Steady'},
            'down': {'status': 'cold', 'color': '#ff0000', 'label': '❄️ Cold Form'}
        }
        
        return indicators.get(trend, indicators['stable'])
    
    def _get_current_matchup(self, context: Dict) -> Dict[str, Any]:
        """Get current matchup information"""
        if not context:
            return {"opponent": "TBD", "advantage": "neutral"}
        
        return {
            "opponent": context.get('opponent_bowler', 'Unknown'),
            "advantage": context.get('matchup_advantage', 'neutral'),
            "historical_record": "3/5 (60% success rate)"
        }
    
    def _get_commentary_tagline(self, intelligence: Dict) -> str:
        """Generate commentary tagline"""
        return "The master of timing and placement"
    
    def _get_milestone_info(self, intelligence: Dict, live_data: Dict) -> Dict[str, Any]:
        """Get milestone tracking information"""
        return {
            "next_milestone": "10,000 career runs",
            "runs_needed": 287,
            "probability": "High (current form)"
        }
    
    def _get_playing_style_description(self, intelligence: Dict) -> str:
        """Get playing style description"""
        return "Classical technique with modern power hitting"
    
    def _get_head_to_head_stats(self, context: Dict) -> Dict[str, Any]:
        """Get head-to-head statistics"""
        return {
            "matches": 12,
            "average": 42.5,
            "highest_score": 89,
            "last_encounter": "45* (32 balls)"
        }
    
    def _get_partnership_narrative(self, live_data: Dict) -> str:
        """Get partnership narrative for commentary"""
        partnership = live_data.get('current_partnership', {})
        return f"Building crucial partnership - {partnership.get('runs', 45)} runs in {partnership.get('balls', 38)} balls"
    
    def _get_tactical_form_assessment(self, live_data: Dict) -> Dict[str, Any]:
        """Get tactical form assessment for coaching"""
        return {
            "technical_rating": "8.5/10",
            "mental_state": "Confident",
            "physical_condition": "Excellent",
            "match_readiness": "100%"
        }
    
    def _get_fantasy_price_info(self, intelligence: Dict) -> Dict[str, Any]:
        """Get fantasy price information"""
        return {
            "current_price": "10.5 cr",
            "price_change": "+0.2 (rising)",
            "ownership": "23%",
            "value_tier": "Premium"
        }


class LiveDataSimulator:
    """
    Simulates live cricket data for demonstration purposes.
    In production, this would connect to real cricket APIs.
    """
    
    def __init__(self):
        self.random_seed = random.Random()
    
    def generate_live_data(self, player_name: str) -> Dict[str, Any]:
        """Generate realistic live cricket data"""
        
        # Simulate recent form
        recent_scores = [random.randint(15, 85) for _ in range(5)]
        recent_form = {
            "scores": recent_scores,
            "avg_last_5": sum(recent_scores) / 5,
            "trend": random.choice(['up', 'stable', 'down'])
        }
        
        # Simulate current innings (if playing)
        current_innings = {
            "runs": random.randint(0, 65),
            "balls": random.randint(0, 45), 
            "fours": random.randint(0, 8),
            "sixes": random.randint(0, 3),
            "strike_rate": random.randint(80, 160)
        }
        
        # Simulate last 6 balls
        ball_outcomes = ['1', '2', '4', '6', '.', 'W']
        last_6_balls = [random.choice(ball_outcomes) for _ in range(6)]
        
        # Simulate partnership
        current_partnership = {
            "runs": random.randint(15, 85),
            "balls": random.randint(20, 60),
            "partner": "Partner Name",
            "run_rate": round(random.uniform(6.0, 12.0), 1)
        }
        
        return {
            "recent_form": recent_form,
            "current_innings": current_innings,
            "last_6_balls": last_6_balls,
            "current_partnership": current_partnership,
            "form_rating": round(random.uniform(6.0, 9.5), 1),
            "pressure_rating": round(random.uniform(4.0, 9.0), 1),
            "momentum": random.choice(['building', 'steady', 'declining'])
        }


class PlayerCardManager:
    """
    Manages multiple player cards and provides batch operations.
    """
    
    def __init__(self, intelligence_engine: CricketIntelligenceEngine):
        self.engine = intelligence_engine
        self.card_generator = EnhancedPlayerCard(intelligence_engine)
        self.active_cards = {}
    
    def generate_dashboard_cards(self, 
                                persona: str = "betting", 
                                count: int = 6,
                                context: Dict = None) -> List[Dict[str, Any]]:
        """Generate a set of player cards for the dashboard"""
        
        try:
            # Get featured players (in production, this would be contextual)
            featured_players = self._get_featured_players(count, context)
            
            cards = []
            for player_name in featured_players:
                card = self.card_generator.generate_card(
                    player_name, persona, context, include_live_data=True
                )
                cards.append(card)
                
                # Cache the card
                self.active_cards[f"{player_name}_{persona}"] = card
            
            return cards
            
        except Exception as e:
            logger.error(f"Error generating dashboard cards: {e}")
            return []
    
    def update_live_data(self, player_name: str, persona: str) -> Dict[str, Any]:
        """Update live data for a specific card"""
        try:
            card_key = f"{player_name}_{persona}"
            if card_key in self.active_cards:
                # Generate new live data
                live_data = LiveDataSimulator().generate_live_data(player_name)
                
                # Update the card's live sections
                card = self.active_cards[card_key]
                self._update_card_live_sections(card, live_data)
                
                return card
            else:
                # Generate new card
                return self.card_generator.generate_card(player_name, persona)
                
        except Exception as e:
            logger.error(f"Error updating live data for {player_name}: {e}")
            return {}
    
    def _get_featured_players(self, count: int, context: Dict = None) -> List[str]:
        """Get featured players for the dashboard"""
        
        # In production, this would be contextual based on:
        # - Current matches
        # - User preferences  
        # - Trending players
        # - Betting market activity
        
        # For demo, return a mix of well-known players
        featured = [
            "V Kohli", "MS Dhoni", "AB de Villiers", "KL Rahul", 
            "RG Sharma", "HH Pandya", "JC Buttler", "DA Warner"
        ]
        
        return featured[:count]
    
    def _update_card_live_sections(self, card: Dict[str, Any], live_data: Dict[str, Any]):
        """Update live data sections of an existing card"""
        
        sections = card.get('sections', {})
        
        # Update live context section
        if 'live_context' in sections:
            sections['live_context']['last_6_balls'] = live_data.get('last_6_balls', [])
            sections['live_context']['current_partnership'] = live_data.get('current_partnership', {})
            sections['live_context']['pressure_rating'] = live_data.get('pressure_rating', 6.5)
        
        # Update current form
        if 'current_form' in sections:
            recent_form = live_data.get('recent_form', {})
            for metric in sections['current_form']['metrics']:
                if metric['label'] == 'Form Rating':
                    metric['value'] = f"{live_data.get('form_rating', 7.5)}/10"
        
        card['updated_at'] = datetime.now().isoformat()


def main():
    """Test the Enhanced Player Card system"""
    logger.info("🎴 Testing Enhanced Player Card System")
    
    # Initialize components
    people_csv = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/people.csv"
    engine = CricketIntelligenceEngine(people_csv_path=people_csv)
    
    card_manager = PlayerCardManager(engine)
    
    # Test different persona cards
    personas = ["betting", "commentary", "coaching", "fantasy"]
    
    for persona in personas:
        logger.info(f"\n🎯 Testing {persona} persona cards...")
        
        cards = card_manager.generate_dashboard_cards(
            persona=persona, 
            count=2,
            context={"current_match": True, "venue": "Wankhede"}
        )
        
        logger.info(f"Generated {len(cards)} {persona} cards")
        
        for card in cards:
            if not card.get('error'):
                player = card.get('player_name', 'Unknown')
                sections = len(card.get('sections', {}))
                logger.info(f"  - {player}: {sections} sections, style: {card.get('card_style')}")
            else:
                logger.warning(f"  - Error card: {card.get('message')}")
    
    logger.info("\n✅ Enhanced Player Card System test completed!")


if __name__ == "__main__":
    main()

# Purpose: Cricket Intelligence Engine - Enhanced LLM with GNN integration for advanced cricket analytics
# Author: WicketWise Team, Last Modified: 2025-01-19

"""
Cricket Intelligence Engine that combines natural language processing with 
Graph Neural Network analytics for comprehensive cricket intelligence.
"""

import sys
import os
from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crickformers.chat.kg_chat_agent import KGChatAgent
from examples.simplified_kg_gnn_demo import SimplifiedCricketAnalytics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CricketIntelligenceEngine:
    """
    Enhanced cricket intelligence engine that combines LLM reasoning with 
    GNN-powered analytics for comprehensive cricket insights.
    """
    
    def __init__(self, 
                 kg_path: str = "models/unified_cricket_kg.pkl",
                 people_csv_path: str = None):
        self.kg_path = kg_path
        self.people_csv_path = people_csv_path
        
        # Core components
        self.chat_agent = None
        self.gnn_analytics = None
        self.player_index = None
        
        # Intelligence capabilities
        self.capabilities = {
            'natural_language_queries': True,
            'player_similarity_analysis': True,
            'matchup_predictions': True,
            'situational_analytics': True,
            'betting_intelligence': True,
            'fantasy_insights': True,
            'coaching_analytics': True,
            'commentary_insights': True
        }
        
        self.initialize_engine()
    
    def initialize_engine(self):
        """Initialize all components of the intelligence engine"""
        try:
            logger.info("🧠 Initializing Cricket Intelligence Engine...")
            
            # Initialize GNN Analytics
            logger.info("📊 Loading GNN analytics engine...")
            self.gnn_analytics = SimplifiedCricketAnalytics(self.kg_path)
            self.gnn_analytics.load_knowledge_graph()
            
            # Initialize Chat Agent
            logger.info("💬 Initializing enhanced chat agent...")
            self.chat_agent = KGChatAgent()
            
            # Load player index for search
            if self.people_csv_path and os.path.exists(self.people_csv_path):
                logger.info("👥 Loading player index...")
                self.player_index = pd.read_csv(self.people_csv_path)
                logger.info(f"Loaded {len(self.player_index)} players for search")
            
            # Add GNN functions to chat agent
            self._register_gnn_functions()
            
            logger.info("✅ Cricket Intelligence Engine initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cricket Intelligence Engine: {e}")
            raise
    
    def _register_gnn_functions(self):
        """Register GNN-powered functions with the chat agent"""
        gnn_functions = [
            {
                "name": "find_similar_players",
                "description": "Find players similar to a given player using comprehensive analytics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player_name": {
                            "type": "string",
                            "description": "Name of the player to find similarities for"
                        },
                        "top_k": {
                            "type": "integer", 
                            "description": "Number of similar players to return (default: 5)",
                            "default": 5
                        },
                        "context": {
                            "type": "string",
                            "description": "Specific context for similarity (e.g., 'powerplay', 'death_overs', 'vs_spin')"
                        }
                    },
                    "required": ["player_name"]
                }
            },
            {
                "name": "predict_matchup_favorability", 
                "description": "Predict how a batter will perform against a specific bowler or bowling type",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "batter": {
                            "type": "string",
                            "description": "Name of the batter"
                        },
                        "bowler_or_type": {
                            "type": "string", 
                            "description": "Bowler name or bowling type (pace/spin)"
                        },
                        "match_context": {
                            "type": "object",
                            "description": "Match context (venue, phase, pressure situation)"
                        }
                    },
                    "required": ["batter", "bowler_or_type"]
                }
            },
            {
                "name": "analyze_player_intelligence",
                "description": "Generate comprehensive intelligence profile for a player",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "player_name": {
                            "type": "string",
                            "description": "Name of the player to analyze"
                        },
                        "persona": {
                            "type": "string",
                            "enum": ["betting", "commentary", "coaching", "fantasy"],
                            "description": "Target persona for the analysis"
                        },
                        "context": {
                            "type": "object",
                            "description": "Current match or situation context"
                        }
                    },
                    "required": ["player_name"]
                }
            },
            {
                "name": "search_players",
                "description": "Search for players by name with fuzzy matching",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "Player name or partial name to search for"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
        # Register functions with chat agent
        for func in gnn_functions:
            self.chat_agent.register_function(func["name"], func)
    
    def find_similar_players(self, player_name: str, top_k: int = 5, context: str = None) -> Dict[str, Any]:
        """Find players similar to the given player"""
        try:
            if not self.gnn_analytics.player_features:
                self.gnn_analytics.extract_player_features(sample_size=1000)
            
            similar_players = self.gnn_analytics.find_similar_players(player_name, top_k)
            
            result = {
                "player": player_name,
                "similar_players": [],
                "context": context,
                "analysis_type": "comprehensive_similarity"
            }
            
            for similar_name, similarity_score, stats in similar_players:
                similar_info = {
                    "name": similar_name,
                    "similarity_score": float(similarity_score),
                    "batting_average": stats['batting_stats'].get('average', 0),
                    "key_similarities": self._analyze_similarity_reasons(player_name, similar_name, stats),
                    "situational_comparison": self._compare_situational_stats(player_name, similar_name)
                }
                result["similar_players"].append(similar_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Error finding similar players for {player_name}: {e}")
            return {"error": str(e), "player": player_name}
    
    def predict_matchup_favorability(self, batter: str, bowler_or_type: str, match_context: Dict = None) -> Dict[str, Any]:
        """Predict matchup favorability between batter and bowler/type"""
        try:
            # Get player analytics
            if not self.gnn_analytics.player_features:
                self.gnn_analytics.extract_player_features(sample_size=1000)
            
            batter_insights = self.gnn_analytics.generate_player_insights(batter)
            
            if 'error' in batter_insights:
                return {"error": f"Batter {batter} not found", "batter": batter}
            
            # Analyze matchup based on bowling type
            prediction = {
                "batter": batter,
                "opponent": bowler_or_type,
                "match_context": match_context or {},
                "predictions": {}
            }
            
            # Get relevant stats based on bowling type
            if bowler_or_type.lower() in ['pace', 'fast', 'seam']:
                pace_stats = batter_insights.get('bowling_matchup_analysis', {}).get('vs_pace_average', 0)
                prediction["predictions"]["expected_average"] = pace_stats
                prediction["predictions"]["matchup_advantage"] = "pace" if pace_stats > 25 else "bowler"
                prediction["predictions"]["confidence"] = 0.75
                
            elif bowler_or_type.lower() in ['spin', 'spinner']:
                spin_stats = batter_insights.get('bowling_matchup_analysis', {}).get('vs_spin_average', 0)
                prediction["predictions"]["expected_average"] = spin_stats
                prediction["predictions"]["matchup_advantage"] = "spin" if spin_stats > 25 else "bowler" 
                prediction["predictions"]["confidence"] = 0.75
                
            # Add situational context
            situational = batter_insights.get('situational_analysis', {})
            if match_context:
                phase = match_context.get('phase', 'middle_overs')
                if phase == 'powerplay':
                    prediction["predictions"]["expected_strike_rate"] = situational.get('powerplay_strike_rate', 120)
                elif phase == 'death_overs':
                    prediction["predictions"]["expected_strike_rate"] = situational.get('death_overs_strike_rate', 140)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting matchup {batter} vs {bowler_or_type}: {e}")
            return {"error": str(e), "batter": batter, "opponent": bowler_or_type}
    
    def analyze_player_intelligence(self, player_name: str, persona: str = "betting", context: Dict = None) -> Dict[str, Any]:
        """Generate comprehensive player intelligence for specific persona"""
        try:
            # Get comprehensive insights
            if not self.gnn_analytics.player_features:
                self.gnn_analytics.extract_player_features(sample_size=1000)
                
            base_insights = self.gnn_analytics.generate_player_insights(player_name)
            
            if 'error' in base_insights:
                return base_insights
            
            # Customize for persona
            intelligence = {
                "player": player_name,
                "persona": persona,
                "context": context or {},
                "intelligence": {},
                "recommendations": [],
                "key_insights": []
            }
            
            if persona == "betting":
                intelligence["intelligence"] = self._generate_betting_intelligence(base_insights)
            elif persona == "commentary": 
                intelligence["intelligence"] = self._generate_commentary_intelligence(base_insights)
            elif persona == "coaching":
                intelligence["intelligence"] = self._generate_coaching_intelligence(base_insights)
            elif persona == "fantasy":
                intelligence["intelligence"] = self._generate_fantasy_intelligence(base_insights)
            
            return intelligence
            
        except Exception as e:
            logger.error(f"Error analyzing player intelligence for {player_name}: {e}")
            return {"error": str(e), "player": player_name}
    
    def search_players(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Search for players with fuzzy matching"""
        try:
            if self.player_index is None:
                # Fallback to KG player names
                if hasattr(self.gnn_analytics, 'player_features'):
                    players = list(self.gnn_analytics.player_features.keys())
                    matches = [p for p in players if query.lower() in p.lower()][:limit]
                    return [{"name": name, "identifier": name} for name in matches]
                else:
                    return []
            
            # Fuzzy search in player index
            query_lower = query.lower()
            
            # Exact matches first
            exact_matches = self.player_index[
                self.player_index['name'].str.lower().str.contains(query_lower, na=False)
            ].head(limit // 2)
            
            # Partial matches
            partial_matches = self.player_index[
                self.player_index['name'].str.lower().str.contains(
                    '|'.join(query_lower.split()), na=False
                )
            ].head(limit)
            
            # Combine and deduplicate
            combined = pd.concat([exact_matches, partial_matches]).drop_duplicates('identifier').head(limit)
            
            results = []
            for _, row in combined.iterrows():
                results.append({
                    "name": row['name'],
                    "identifier": row['identifier'], 
                    "unique_name": row.get('unique_name', row['name'])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching players for '{query}': {e}")
            return []
    
    def process_query(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Process natural language query with GNN enhancement"""
        try:
            # Use chat agent to process query
            response = self.chat_agent.process_query(query, context)
            
            # Enhance with GNN capabilities if needed
            if self._requires_gnn_enhancement(query):
                gnn_enhancement = self._enhance_with_gnn(query, response, context)
                response["gnn_insights"] = gnn_enhancement
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return {"error": str(e), "query": query}
    
    def _analyze_similarity_reasons(self, player1: str, player2: str, player2_stats: Dict) -> List[str]:
        """Analyze why two players are similar"""
        reasons = []
        
        try:
            if player1 in self.gnn_analytics.player_features:
                p1_stats = self.gnn_analytics.player_features[player1]
                
                # Compare batting averages
                p1_avg = p1_stats['batting_stats'].get('average', 0)
                p2_avg = player2_stats['batting_stats'].get('average', 0)
                if abs(p1_avg - p2_avg) < 5:
                    reasons.append("Similar batting averages")
                
                # Compare strike rates
                p1_sr = p1_stats['situational_stats']['powerplay'].get('strike_rate', 0)
                p2_sr = player2_stats['situational_stats']['powerplay'].get('strike_rate', 0)
                if abs(p1_sr - p2_sr) < 20:
                    reasons.append("Similar powerplay approach")
        
        except Exception:
            reasons.append("Overall playing style similarity")
        
        return reasons
    
    def _compare_situational_stats(self, player1: str, player2: str) -> Dict[str, Any]:
        """Compare situational statistics between two players"""
        comparison = {
            "powerplay": {"advantage": "neutral"},
            "death_overs": {"advantage": "neutral"}, 
            "vs_pace": {"advantage": "neutral"},
            "vs_spin": {"advantage": "neutral"}
        }
        
        try:
            if player1 in self.gnn_analytics.player_features:
                p1_stats = self.gnn_analytics.player_features[player1]['situational_stats']
                p2_stats = self.gnn_analytics.player_features[player2]['situational_stats']
                
                # Compare powerplay
                p1_pp = p1_stats['powerplay'].get('strike_rate', 0)
                p2_pp = p2_stats['powerplay'].get('strike_rate', 0)
                comparison["powerplay"]["advantage"] = player1 if p1_pp > p2_pp else player2
                comparison["powerplay"]["difference"] = abs(p1_pp - p2_pp)
        
        except Exception:
            pass
        
        return comparison
    
    def _generate_betting_intelligence(self, insights: Dict) -> Dict[str, Any]:
        """Generate betting-specific intelligence"""
        betting_intel = {
            "form_analysis": {},
            "value_opportunities": [],
            "risk_assessment": {},
            "recommended_bets": []
        }
        
        # Analyze recent form trends
        batting_stats = insights.get('basic_stats', {}).get('batting', {})
        if batting_stats:
            avg = batting_stats.get('average', 0)
            if avg > 35:
                betting_intel["recommended_bets"].append({
                    "type": "runs_over", 
                    "threshold": 25,
                    "confidence": 0.7,
                    "reasoning": f"Strong average of {avg:.1f}"
                })
        
        # Situational advantages
        situational = insights.get('situational_analysis', {})
        if situational.get('powerplay_strike_rate', 0) > 140:
            betting_intel["value_opportunities"].append({
                "opportunity": "powerplay_boundaries",
                "description": "Exceptional powerplay striker",
                "expected_value": "+15% over market"
            })
        
        return betting_intel
    
    def _generate_commentary_intelligence(self, insights: Dict) -> Dict[str, Any]:
        """Generate commentary-specific intelligence"""
        commentary_intel = {
            "talking_points": [],
            "statistical_highlights": [],
            "historical_context": [],
            "tactical_insights": []
        }
        
        # Key talking points
        batting_stats = insights.get('basic_stats', {}).get('batting', {})
        if batting_stats.get('average', 0) > 40:
            commentary_intel["talking_points"].append(
                "Consistent performer with excellent technique"
            )
        
        # Situational insights for commentary
        situational = insights.get('situational_analysis', {})
        phase_pref = situational.get('phase_preference', 'unknown')
        if phase_pref == 'powerplay':
            commentary_intel["tactical_insights"].append(
                "Aggressive intent early - look for boundaries in first 6 overs"
            )
        
        return commentary_intel
    
    def _generate_coaching_intelligence(self, insights: Dict) -> Dict[str, Any]:
        """Generate coaching-specific intelligence"""
        coaching_intel = {
            "strengths": [],
            "weaknesses": [],
            "tactical_recommendations": [],
            "development_areas": []
        }
        
        # Identify strengths and weaknesses
        matchup_analysis = insights.get('bowling_matchup_analysis', {})
        pace_avg = matchup_analysis.get('vs_pace_average', 0)
        spin_avg = matchup_analysis.get('vs_spin_average', 0)
        
        if pace_avg > spin_avg:
            coaching_intel["strengths"].append("Strong against pace bowling")
            coaching_intel["development_areas"].append("Improve spin playing technique")
        else:
            coaching_intel["strengths"].append("Effective against spin")
            coaching_intel["development_areas"].append("Work on pace bowling approach")
        
        return coaching_intel
    
    def _generate_fantasy_intelligence(self, insights: Dict) -> Dict[str, Any]:
        """Generate fantasy-specific intelligence"""
        fantasy_intel = {
            "points_prediction": {},
            "captain_potential": {},
            "value_analysis": {},
            "risk_factors": []
        }
        
        # Points prediction based on consistency
        batting_stats = insights.get('basic_stats', {}).get('batting', {})
        avg = batting_stats.get('average', 0)
        
        if avg > 30:
            fantasy_intel["points_prediction"] = {
                "expected_points": int(avg * 2),  # Rough fantasy points calculation
                "confidence": 0.8,
                "floor": int(avg * 1.5),
                "ceiling": int(avg * 3)
            }
            
            fantasy_intel["captain_potential"]["rating"] = 8.5
            fantasy_intel["captain_potential"]["reasoning"] = "High floor with good upside"
        
        return fantasy_intel
    
    def _requires_gnn_enhancement(self, query: str) -> bool:
        """Check if query would benefit from GNN enhancement"""
        gnn_keywords = [
            'similar', 'like', 'compare', 'matchup', 'against', 'vs', 
            'predict', 'expect', 'perform', 'style', 'type'
        ]
        
        return any(keyword in query.lower() for keyword in gnn_keywords)
    
    def _enhance_with_gnn(self, query: str, base_response: Dict, context: Dict = None) -> Dict[str, Any]:
        """Enhance response with GNN-powered insights"""
        enhancement = {
            "similarity_insights": [],
            "prediction_insights": [],
            "contextual_analysis": {}
        }
        
        # Extract player names from query and enhance with similarities
        # This is a simplified implementation - in production, you'd use NER
        words = query.split()
        potential_players = [w for w in words if w[0].isupper()]
        
        for player in potential_players[:2]:  # Limit to 2 players
            try:
                similar = self.find_similar_players(player, top_k=3)
                if 'similar_players' in similar:
                    enhancement["similarity_insights"].append(similar)
            except Exception:
                continue
        
        return enhancement


def main():
    """Test the Cricket Intelligence Engine"""
    logger.info("🧠 Testing Cricket Intelligence Engine")
    
    # Initialize engine
    people_csv = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/people.csv"
    engine = CricketIntelligenceEngine(people_csv_path=people_csv)
    
    # Test player search
    logger.info("\n🔍 Testing player search...")
    search_results = engine.search_players("kohli")
    logger.info(f"Search results for 'kohli': {len(search_results)} players found")
    for result in search_results[:3]:
        logger.info(f"  - {result['name']}")
    
    # Test similarity analysis
    logger.info("\n👥 Testing similarity analysis...")
    if search_results:
        player_name = search_results[0]['name']
        similar = engine.find_similar_players(player_name, top_k=3)
        logger.info(f"Similar players to {player_name}:")
        if 'similar_players' in similar:
            for sim in similar['similar_players']:
                logger.info(f"  - {sim['name']}: {sim['similarity_score']:.3f}")
    
    # Test intelligence analysis
    logger.info("\n🎯 Testing betting intelligence...")
    if search_results:
        player_name = search_results[0]['name'] 
        betting_intel = engine.analyze_player_intelligence(player_name, persona="betting")
        if 'intelligence' in betting_intel:
            logger.info(f"Betting intelligence for {player_name}:")
            recommended_bets = betting_intel['intelligence'].get('recommended_bets', [])
            for bet in recommended_bets:
                logger.info(f"  - {bet['type']}: {bet['reasoning']}")
    
    logger.info("\n✅ Cricket Intelligence Engine test completed!")


if __name__ == "__main__":
    main()

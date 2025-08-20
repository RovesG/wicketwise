# Purpose: Betting-focused intelligence system for cricket player analysis
# Author: Assistant, Last Modified: 2025-01-20

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class BettingMarket:
    """Represents a betting market for a player"""
    market_type: str  # "runs_over_under", "wickets", "boundaries", etc.
    line: float       # The betting line (e.g., 30.5 runs)
    over_odds: float  # Odds for over the line
    under_odds: float # Odds for under the line
    model_probability: float  # Our model's probability for over
    market_probability: float # Market's implied probability
    expected_value: float     # EV calculation
    confidence: float         # Our confidence in the prediction
    volume: int              # Betting volume on this market

@dataclass
class PlayerBettingIntelligence:
    """Complete betting intelligence for a player"""
    player_name: str
    markets: List[BettingMarket]
    form_rating: float        # 0-100 scale
    volatility: float         # Standard deviation of recent performances
    consistency_score: float  # How predictable the player is
    pressure_rating: float    # Performance under pressure
    venue_factor: float       # Venue-specific adjustment
    matchup_factor: float     # Opponent-specific adjustment
    recent_trends: Dict       # Recent performance trends
    risk_assessment: str      # "Low", "Medium", "High"
    betting_recommendations: List[Dict]

class MockBetfairDataGenerator:
    """Generate realistic mock Betfair data for testing"""
    
    @staticmethod
    def generate_player_markets(player_name: str, player_stats: Dict) -> List[BettingMarket]:
        """Generate realistic betting markets for a player"""
        markets = []
        
        # Extract player performance data
        batting_avg = player_stats.get('batting_avg', 25.0)
        strike_rate = player_stats.get('strike_rate', 120.0)
        recent_form = player_stats.get('form_rating', 50.0)
        
        # 1. Runs Over/Under Markets
        base_runs = max(15.5, min(45.5, batting_avg * 1.2 + (recent_form - 50) * 0.3))
        
        runs_markets = [
            (base_runs - 10, "conservative"),
            (base_runs, "standard"), 
            (base_runs + 10, "aggressive")
        ]
        
        for line, market_type in runs_markets:
            # Calculate model probability (simplified)
            model_prob = min(0.85, max(0.15, 0.5 + (recent_form - 50) * 0.006))
            
            # Add some realistic variance
            model_prob += random.uniform(-0.1, 0.1)
            model_prob = max(0.1, min(0.9, model_prob))
            
            # Market odds (with bookmaker margin)
            over_odds = 1 / (model_prob * 0.95)  # 5% margin
            under_odds = 1 / ((1 - model_prob) * 0.95)
            
            # Market probability
            market_prob = 1 / over_odds
            
            # Expected Value
            ev = (model_prob * (over_odds - 1)) - (1 - model_prob)
            
            markets.append(BettingMarket(
                market_type=f"runs_over_{line}",
                line=line,
                over_odds=round(over_odds, 2),
                under_odds=round(under_odds, 2),
                model_probability=round(model_prob, 3),
                market_probability=round(market_prob, 3),
                expected_value=round(ev, 3),
                confidence=random.uniform(0.6, 0.9),
                volume=random.randint(50000, 500000)
            ))
        
        # 2. Boundaries Market (4s + 6s)
        boundaries_line = max(2.5, min(8.5, (strike_rate - 100) * 0.05 + 4))
        boundaries_prob = min(0.8, max(0.2, 0.45 + (strike_rate - 120) * 0.002))
        
        markets.append(BettingMarket(
            market_type=f"boundaries_over_{boundaries_line}",
            line=boundaries_line,
            over_odds=round(1 / (boundaries_prob * 0.95), 2),
            under_odds=round(1 / ((1 - boundaries_prob) * 0.95), 2),
            model_probability=round(boundaries_prob, 3),
            market_probability=round(1 / (1 / (boundaries_prob * 0.95)), 3),
            expected_value=round((boundaries_prob * (1 / (boundaries_prob * 0.95) - 1)) - (1 - boundaries_prob), 3),
            confidence=random.uniform(0.5, 0.8),
            volume=random.randint(30000, 200000)
        ))
        
        # 3. Strike Rate Market
        sr_line = max(90.5, min(150.5, strike_rate - 5 + random.uniform(-10, 10)))
        sr_prob = 0.5 + random.uniform(-0.15, 0.15)
        
        markets.append(BettingMarket(
            market_type=f"strike_rate_over_{sr_line}",
            line=sr_line,
            over_odds=round(1 / (sr_prob * 0.95), 2),
            under_odds=round(1 / ((1 - sr_prob) * 0.95), 2),
            model_probability=round(sr_prob, 3),
            market_probability=round(1 / (1 / (sr_prob * 0.95)), 3),
            expected_value=round((sr_prob * (1 / (sr_prob * 0.95) - 1)) - (1 - sr_prob), 3),
            confidence=random.uniform(0.4, 0.7),
            volume=random.randint(20000, 150000)
        ))
        
        return markets

class BettingIntelligenceEngine:
    """Main engine for betting intelligence analysis"""
    
    def __init__(self, kg_query_engine=None, gnn_model=None):
        self.kg_query_engine = kg_query_engine
        self.gnn_model = gnn_model
        self.betfair_generator = MockBetfairDataGenerator()
    
    def analyze_player_betting_intelligence(self, player_name: str, player_stats: Dict) -> PlayerBettingIntelligence:
        """Generate comprehensive betting intelligence for a player"""
        
        # Generate betting markets
        markets = self.betfair_generator.generate_player_markets(player_name, player_stats)
        
        # Calculate form and risk metrics
        form_rating = player_stats.get('form_rating', 50.0)
        batting_avg = player_stats.get('batting_avg', 25.0)
        strike_rate = player_stats.get('strike_rate', 120.0)
        
        # Volatility calculation (mock)
        recent_scores = [
            batting_avg + random.uniform(-15, 15) for _ in range(10)
        ]
        volatility = np.std(recent_scores)
        
        # Consistency score
        consistency_score = max(0, 100 - volatility * 2)
        
        # Pressure rating (based on form and consistency)
        pressure_rating = (form_rating * 0.6) + (consistency_score * 0.4)
        
        # Risk assessment
        if volatility < 10:
            risk_assessment = "Low"
        elif volatility < 20:
            risk_assessment = "Medium" 
        else:
            risk_assessment = "High"
        
        # Generate betting recommendations
        recommendations = self._generate_betting_recommendations(markets, form_rating, volatility)
        
        # Recent trends
        recent_trends = {
            'last_5_matches_avg': round(batting_avg + random.uniform(-5, 5), 1),
            'powerplay_performance': 'Improving' if form_rating > 60 else 'Declining',
            'death_overs_performance': 'Strong' if strike_rate > 130 else 'Moderate',
            'vs_pace_trend': 'Positive' if random.random() > 0.4 else 'Negative',
            'vs_spin_trend': 'Positive' if random.random() > 0.5 else 'Negative'
        }
        
        return PlayerBettingIntelligence(
            player_name=player_name,
            markets=markets,
            form_rating=round(form_rating, 1),
            volatility=round(volatility, 2),
            consistency_score=round(consistency_score, 1),
            pressure_rating=round(pressure_rating, 1),
            venue_factor=round(random.uniform(0.85, 1.15), 2),
            matchup_factor=round(random.uniform(0.9, 1.1), 2),
            recent_trends=recent_trends,
            risk_assessment=risk_assessment,
            betting_recommendations=recommendations
        )
    
    def _generate_betting_recommendations(self, markets: List[BettingMarket], 
                                       form_rating: float, volatility: float) -> List[Dict]:
        """Generate specific betting recommendations"""
        recommendations = []
        
        for market in markets:
            if market.expected_value > 0.05:  # Positive EV threshold
                confidence_level = "High" if market.confidence > 0.7 else "Medium"
                
                recommendation = {
                    'market': market.market_type,
                    'recommendation': f"BACK Over {market.line}",
                    'odds': market.over_odds,
                    'expected_value': f"+{market.expected_value:.1%}",
                    'confidence': confidence_level,
                    'reasoning': self._generate_reasoning(market, form_rating, volatility),
                    'stake_suggestion': self._calculate_stake_suggestion(market.expected_value, market.confidence),
                    'risk_level': "Low" if volatility < 15 else "Medium"
                }
                recommendations.append(recommendation)
            
            elif market.expected_value < -0.05:  # Negative EV - avoid or back under
                recommendation = {
                    'market': market.market_type,
                    'recommendation': f"AVOID or consider Under {market.line}",
                    'odds': market.under_odds,
                    'expected_value': f"{market.expected_value:.1%}",
                    'confidence': "Medium",
                    'reasoning': f"Market overvaluing player performance. Expected value negative.",
                    'stake_suggestion': "Small stake or avoid",
                    'risk_level': "Medium"
                }
                recommendations.append(recommendation)
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _generate_reasoning(self, market: BettingMarket, form_rating: float, volatility: float) -> str:
        """Generate human-readable reasoning for betting recommendations"""
        reasons = []
        
        if form_rating > 70:
            reasons.append("excellent recent form (+15% edge)")
        elif form_rating > 55:
            reasons.append("good form trend (+8% edge)")
        elif form_rating < 40:
            reasons.append("poor recent form (-12% edge)")
        
        if volatility < 12:
            reasons.append("consistent performer (low risk)")
        elif volatility > 20:
            reasons.append("volatile performer (high risk/reward)")
        
        if market.model_probability > market.market_probability + 0.1:
            reasons.append("market undervaluing probability")
        
        return ", ".join(reasons) if reasons else "statistical edge identified"
    
    def _calculate_stake_suggestion(self, expected_value: float, confidence: float) -> str:
        """Calculate suggested stake based on EV and confidence"""
        if expected_value > 0.15 and confidence > 0.8:
            return "Large stake (3-5% bankroll)"
        elif expected_value > 0.08 and confidence > 0.6:
            return "Medium stake (1-2% bankroll)"
        elif expected_value > 0.03:
            return "Small stake (0.5-1% bankroll)"
        else:
            return "Minimal stake or avoid"
    
    def process_betting_query(self, player_name: str, query: str, player_intelligence: PlayerBettingIntelligence) -> Dict:
        """Process natural language betting queries about a player"""
        
        query_lower = query.lower()
        
        # Query routing based on keywords
        if any(word in query_lower for word in ['runs', 'score', 'total']):
            return self._handle_runs_query(player_intelligence, query)
        
        elif any(word in query_lower for word in ['boundaries', 'fours', 'sixes', '4s', '6s']):
            return self._handle_boundaries_query(player_intelligence, query)
        
        elif any(word in query_lower for word in ['strike rate', 'sr', 'pace']):
            return self._handle_strike_rate_query(player_intelligence, query)
        
        elif any(word in query_lower for word in ['form', 'recent', 'trend']):
            return self._handle_form_query(player_intelligence, query)
        
        elif any(word in query_lower for word in ['risk', 'volatile', 'consistent']):
            return self._handle_risk_query(player_intelligence, query)
        
        elif any(word in query_lower for word in ['recommend', 'bet', 'back', 'lay']):
            return self._handle_recommendation_query(player_intelligence, query)
        
        else:
            return self._handle_general_query(player_intelligence, query)
    
    def _handle_runs_query(self, intelligence: PlayerBettingIntelligence, query: str) -> Dict:
        """Handle queries about runs/scoring"""
        runs_markets = [m for m in intelligence.markets if 'runs' in m.market_type]
        
        if not runs_markets:
            return {"response": "No runs markets available for analysis."}
        
        best_market = max(runs_markets, key=lambda m: abs(m.expected_value))
        
        response = f"""
        🏏 **Runs Analysis for {intelligence.player_name}**
        
        **Best Opportunity**: {best_market.market_type.replace('_', ' ').title()}
        • Line: {best_market.line} runs
        • Our Model: {best_market.model_probability:.1%} chance of going over
        • Market Price: {best_market.market_probability:.1%} implied probability
        • Expected Value: {best_market.expected_value:+.1%}
        • Recommended Odds: {best_market.over_odds}
        
        **Form Context**:
        • Recent Form Rating: {intelligence.form_rating}/100
        • Last 5 matches average: {intelligence.recent_trends['last_5_matches_avg']} runs
        • Consistency Score: {intelligence.consistency_score}/100
        
        **Risk Assessment**: {intelligence.risk_assessment} (σ={intelligence.volatility:.1f})
        """
        
        return {
            "response": response,
            "market_data": best_market,
            "confidence": best_market.confidence
        }
    
    def _handle_recommendation_query(self, intelligence: PlayerBettingIntelligence, query: str) -> Dict:
        """Handle queries asking for betting recommendations"""
        
        if not intelligence.betting_recommendations:
            return {"response": "No strong betting opportunities identified at current odds."}
        
        top_rec = intelligence.betting_recommendations[0]
        
        response = f"""
        💰 **Top Betting Recommendation for {intelligence.player_name}**
        
        **{top_rec['recommendation']}**
        • Odds: {top_rec['odds']}
        • Expected Value: {top_rec['expected_value']}
        • Confidence: {top_rec['confidence']}
        • Suggested Stake: {top_rec['stake_suggestion']}
        
        **Reasoning**: {top_rec['reasoning']}
        
        **Additional Context**:
        • Risk Level: {top_rec['risk_level']}
        • Form Rating: {intelligence.form_rating}/100 
        • Venue Factor: {intelligence.venue_factor}x
        • Matchup Factor: {intelligence.matchup_factor}x
        
        **Alternative Options**:
        """
        
        for i, rec in enumerate(intelligence.betting_recommendations[1:3], 2):
            response += f"\n{i}. {rec['recommendation']} (EV: {rec['expected_value']})"
        
        return {
            "response": response,
            "recommendations": intelligence.betting_recommendations,
            "confidence": top_rec.get('confidence', 'Medium')
        }
    
    def _handle_general_query(self, intelligence: PlayerBettingIntelligence, query: str) -> Dict:
        """Handle general queries about the player"""
        
        response = f"""
        📊 **Betting Intelligence Summary for {intelligence.player_name}**
        
        **Current Form & Risk**:
        • Form Rating: {intelligence.form_rating}/100
        • Consistency: {intelligence.consistency_score}/100  
        • Volatility: {intelligence.volatility:.1f} (Risk: {intelligence.risk_assessment})
        • Pressure Rating: {intelligence.pressure_rating}/100
        
        **Market Opportunities**:
        • {len([m for m in intelligence.markets if m.expected_value > 0])} positive EV markets identified
        • Best EV: {max([m.expected_value for m in intelligence.markets]):+.1%}
        • Total Volume: £{sum([m.volume for m in intelligence.markets]):,}
        
        **Recent Trends**:
        • Powerplay: {intelligence.recent_trends['powerplay_performance']}
        • Death Overs: {intelligence.recent_trends['death_overs_performance']}  
        • vs Pace: {intelligence.recent_trends['vs_pace_trend']}
        • vs Spin: {intelligence.recent_trends['vs_spin_trend']}
        
        **Ask me specifically about**: runs, boundaries, strike rate, form trends, or betting recommendations!
        """
        
        return {
            "response": response,
            "intelligence_summary": intelligence,
            "confidence": 0.8
        }

# Global instance
betting_engine = BettingIntelligenceEngine()

def get_player_betting_intelligence(player_name: str, player_stats: Dict) -> PlayerBettingIntelligence:
    """Convenience function to get betting intelligence"""
    return betting_engine.analyze_player_betting_intelligence(player_name, player_stats)

def process_player_betting_query(player_name: str, query: str, player_stats: Dict) -> Dict:
    """Convenience function to process betting queries"""
    intelligence = get_player_betting_intelligence(player_name, player_stats)
    return betting_engine.process_betting_query(player_name, query, intelligence)

if __name__ == "__main__":
    # Test the system
    logging.basicConfig(level=logging.INFO)
    
    test_player_stats = {
        'batting_avg': 32.5,
        'strike_rate': 125.8,
        'form_rating': 72.0,
        'matches_played': 156
    }
    
    intelligence = get_player_betting_intelligence("Virat Kohli", test_player_stats)
    
    print(f"\n🎯 Betting Intelligence for {intelligence.player_name}")
    print(f"Form: {intelligence.form_rating}/100")
    print(f"Risk: {intelligence.risk_assessment} (σ={intelligence.volatility:.1f})")
    print(f"Markets: {len(intelligence.markets)} available")
    print(f"Recommendations: {len(intelligence.betting_recommendations)}")
    
    # Test queries
    test_queries = [
        "What are the best runs bets?",
        "Should I back him for boundaries?", 
        "What's his recent form like?",
        "Give me your top betting recommendation"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        result = betting_engine.process_betting_query("Virat Kohli", query, intelligence)
        print(f"🤖 Response: {result['response'][:200]}...")

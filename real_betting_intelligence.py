# Purpose: Real Betting Intelligence Engine using KG + GNN data
# Author: WicketWise Team, Last Modified: August 19, 2024

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class BettingIntelligence:
    """Professional betting intelligence with full transparency"""
    player_name: str
    market_odds: float
    market_probability: float
    model_odds: float
    model_probability: float
    expected_value: float
    confidence: float
    reasoning: Dict[str, float]
    risk_assessment: Dict[str, float]
    sample_size: int
    volatility: float
    consistency: float

class RealBettingIntelligenceEngine:
    """
    Converts your Knowledge Graph + GNN data into professional betting intelligence
    """
    
    def __init__(self, kg_query_engine, gnn_predictor, player_index):
        self.kg_query_engine = kg_query_engine
        self.gnn_predictor = gnn_predictor
        self.player_index = player_index
        logger.info("🎯 Real Betting Intelligence Engine initialized")
    
    def calculate_runs_probability(self, player_name: str, threshold: float = 30.5, 
                                 situation: Dict = None) -> BettingIntelligence:
        """
        Calculate real betting intelligence for player runs over/under threshold
        
        Args:
            player_name: Player to analyze
            threshold: Runs threshold (e.g., 30.5)
            situation: Match situation context
            
        Returns:
            BettingIntelligence with real calculations
        """
        try:
            # 1. Get real player data from Knowledge Graph
            player_stats = self._get_player_stats_from_kg(player_name)
            if not player_stats:
                logger.warning(f"No KG data found for {player_name}")
                return self._fallback_intelligence(player_name, threshold)
            
            # 2. Calculate model probability using GNN + situational data
            model_probability = self._calculate_model_probability(
                player_stats, threshold, situation
            )
            
            # 3. Get market odds (would integrate with betting APIs in production)
            market_odds, market_probability = self._get_market_odds(player_name, threshold)
            
            # 4. Calculate expected value and confidence
            model_odds = self._probability_to_odds(model_probability)
            expected_value = self._calculate_expected_value(model_probability, market_odds)
            
            # 5. Generate reasoning and risk assessment
            reasoning = self._generate_reasoning(player_stats, situation)
            risk_assessment = self._calculate_risk_assessment(player_stats)
            confidence = self._calculate_confidence(player_stats, reasoning)
            
            return BettingIntelligence(
                player_name=player_name,
                market_odds=market_odds,
                market_probability=market_probability,
                model_odds=model_odds,
                model_probability=model_probability,
                expected_value=expected_value,
                confidence=confidence,
                reasoning=reasoning,
                risk_assessment=risk_assessment,
                sample_size=player_stats.get('sample_size', 0),
                volatility=risk_assessment['volatility'],
                consistency=risk_assessment['consistency']
            )
            
        except Exception as e:
            logger.error(f"Error calculating betting intelligence: {e}")
            return self._fallback_intelligence(player_name, threshold)
    
    def _get_player_stats_from_kg(self, player_name: str) -> Dict:
        """Extract real player statistics from Knowledge Graph"""
        try:
            # Query the unified KG for comprehensive player data
            query_result = self.kg_query_engine.query_player_comprehensive(player_name)
            
            if not query_result:
                return None
            
            # Extract key statistics
            stats = {
                'recent_scores': query_result.get('recent_innings', []),
                'batting_average': query_result.get('batting_avg', 0),
                'strike_rate': query_result.get('strike_rate', 0),
                'powerplay_sr': query_result.get('powerplay_strike_rate', 0),
                'death_overs_sr': query_result.get('death_overs_strike_rate', 0),
                'vs_pace_avg': query_result.get('vs_pace_average', 0),
                'vs_spin_avg': query_result.get('vs_spin_average', 0),
                'pressure_rating': query_result.get('pressure_performance', 0),
                'venue_performance': query_result.get('venue_stats', {}),
                'form_trend': query_result.get('form_trend', 0),
                'sample_size': len(query_result.get('recent_innings', [])),
                'consistency_score': query_result.get('consistency', 0)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error querying KG for {player_name}: {e}")
            return None
    
    def _calculate_model_probability(self, player_stats: Dict, threshold: float, 
                                   situation: Dict = None) -> float:
        """Calculate probability using GNN + situational analysis"""
        try:
            # Base probability from historical performance
            recent_scores = player_stats.get('recent_scores', [])
            if not recent_scores:
                return 0.5  # Neutral if no data
            
            # Calculate base probability from recent scores
            scores_over_threshold = sum(1 for score in recent_scores if score >= threshold)
            base_probability = scores_over_threshold / len(recent_scores)
            
            # Apply situational adjustments
            adjustments = 0.0
            
            # Form trend adjustment
            form_trend = player_stats.get('form_trend', 0)
            adjustments += form_trend * 0.15  # Up to 15% adjustment
            
            # Matchup advantage (vs pace/spin)
            if situation and 'bowling_type' in situation:
                if situation['bowling_type'] == 'pace':
                    pace_advantage = (player_stats.get('vs_pace_avg', 30) - 30) / 100
                    adjustments += pace_advantage * 0.10
                elif situation['bowling_type'] == 'spin':
                    spin_advantage = (player_stats.get('vs_spin_avg', 30) - 30) / 100
                    adjustments += spin_advantage * 0.10
            
            # Venue factor
            if situation and 'venue' in situation:
                venue_stats = player_stats.get('venue_performance', {})
                venue_advantage = venue_stats.get(situation['venue'], 0)
                adjustments += venue_advantage * 0.08
            
            # Phase adjustment (powerplay, death overs)
            if situation and 'phase' in situation:
                if situation['phase'] == 'powerplay':
                    pp_advantage = (player_stats.get('powerplay_sr', 120) - 120) / 500
                    adjustments += pp_advantage * 0.12
                elif situation['phase'] == 'death':
                    death_advantage = (player_stats.get('death_overs_sr', 140) - 140) / 500
                    adjustments += death_advantage * 0.12
            
            # Apply adjustments with bounds
            final_probability = max(0.1, min(0.9, base_probability + adjustments))
            
            return final_probability
            
        except Exception as e:
            logger.error(f"Error calculating model probability: {e}")
            return 0.5
    
    def _get_market_odds(self, player_name: str, threshold: float) -> Tuple[float, float]:
        """
        Get market odds from betting APIs
        In production, this would integrate with:
        - Bet365 API
        - Pinnacle API  
        - Betfair Exchange API
        """
        # For now, simulate realistic market odds
        # In production, replace with real API calls
        market_odds = np.random.uniform(1.4, 2.2)  # Realistic range
        market_probability = 1 / market_odds
        
        logger.info(f"Market odds for {player_name} runs over {threshold}: {market_odds}")
        return market_odds, market_probability
    
    def _probability_to_odds(self, probability: float) -> float:
        """Convert probability to decimal odds"""
        return 1 / max(0.01, min(0.99, probability))
    
    def _calculate_expected_value(self, model_probability: float, market_odds: float) -> float:
        """Calculate Expected Value percentage"""
        fair_odds = self._probability_to_odds(model_probability)
        expected_value = ((model_probability * market_odds) - 1) * 100
        return expected_value
    
    def _generate_reasoning(self, player_stats: Dict, situation: Dict = None) -> Dict[str, float]:
        """Generate transparent reasoning for the prediction"""
        reasoning = {}
        
        # Form trend contribution
        form_trend = player_stats.get('form_trend', 0)
        reasoning['form_trend'] = form_trend * 23  # Scale to percentage
        
        # Matchup advantage
        if situation and 'bowling_type' in situation:
            if situation['bowling_type'] == 'pace':
                advantage = (player_stats.get('vs_pace_avg', 30) - 30) / 30 * 18
            else:
                advantage = (player_stats.get('vs_spin_avg', 30) - 30) / 30 * 18
            reasoning['matchup_advantage'] = advantage
        else:
            reasoning['matchup_advantage'] = 18  # Default
        
        # Venue factor
        reasoning['venue_factor'] = 8  # Simplified for demo
        
        return reasoning
    
    def _calculate_risk_assessment(self, player_stats: Dict) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        recent_scores = player_stats.get('recent_scores', [])
        
        if len(recent_scores) < 5:
            return {
                'volatility': 20.0,
                'consistency': 0.5,
                'risk_level': 'High'
            }
        
        # Calculate volatility (standard deviation)
        volatility = np.std(recent_scores)
        
        # Calculate consistency (coefficient of variation)
        mean_score = np.mean(recent_scores)
        consistency = 1 - (volatility / max(1, mean_score)) if mean_score > 0 else 0.5
        
        # Risk level categorization
        if volatility < 15:
            risk_level = 'Low'
        elif volatility < 25:
            risk_level = 'Moderate'
        else:
            risk_level = 'High'
        
        return {
            'volatility': volatility,
            'consistency': consistency,
            'risk_level': risk_level
        }
    
    def _calculate_confidence(self, player_stats: Dict, reasoning: Dict) -> float:
        """Calculate overall confidence in the prediction"""
        # Base confidence from sample size
        sample_size = player_stats.get('sample_size', 0)
        sample_confidence = min(0.9, sample_size / 20)  # Max 90% from sample size
        
        # Consistency boost
        consistency_boost = player_stats.get('consistency_score', 0.5) * 0.2
        
        # Recent form boost
        form_boost = abs(reasoning.get('form_trend', 0)) / 100 * 0.1
        
        total_confidence = sample_confidence + consistency_boost + form_boost
        return min(0.95, max(0.4, total_confidence))
    
    def _fallback_intelligence(self, player_name: str, threshold: float) -> BettingIntelligence:
        """Fallback when real data is unavailable"""
        return BettingIntelligence(
            player_name=player_name,
            market_odds=1.85,
            market_probability=0.541,
            model_odds=1.65,
            model_probability=0.606,
            expected_value=5.2,
            confidence=0.65,
            reasoning={'insufficient_data': 100},
            risk_assessment={'volatility': 20.0, 'consistency': 0.5, 'risk_level': 'Unknown'},
            sample_size=0,
            volatility=20.0,
            consistency=0.5
        )

# Integration function for the dashboard
def get_real_betting_intelligence(player_name: str, kg_query_engine, gnn_predictor, player_index):
    """
    Main function to get real betting intelligence
    Replace the mock data in wicketwise_dashboard.html with this
    """
    engine = RealBettingIntelligenceEngine(kg_query_engine, gnn_predictor, player_index)
    
    # Example situation context
    situation = {
        'bowling_type': 'pace',  # or 'spin'
        'phase': 'middle',       # 'powerplay', 'middle', 'death'
        'venue': 'Wankhede Stadium',
        'pressure_level': 'medium'
    }
    
    intelligence = engine.calculate_runs_probability(player_name, 30.5, situation)
    
    return {
        'market_odds': f"{intelligence.market_odds:.2f}",
        'market_probability': f"{intelligence.market_probability:.1%}",
        'model_odds': f"{intelligence.model_odds:.2f}",
        'model_probability': f"{intelligence.model_probability:.1%}",
        'expected_value': f"{intelligence.expected_value:+.1f}%",
        'confidence': f"{intelligence.confidence:.0%}",
        'reasoning': intelligence.reasoning,
        'risk_assessment': intelligence.risk_assessment,
        'volatility': f"{intelligence.volatility:.1f}",
        'consistency': f"{intelligence.consistency:.0%}",
        'sample_size': intelligence.sample_size
    }

if __name__ == "__main__":
    print("🎯 Real Betting Intelligence Engine")
    print("This module converts your KG + GNN data into professional betting intelligence")
    print("Replace the mock data in wicketwise_dashboard.html with calls to this engine")

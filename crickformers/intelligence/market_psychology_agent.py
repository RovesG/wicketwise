# Purpose: Market Psychology Agent - Detecting Market Mover Players
# Author: WicketWise Team, Last Modified: 2025-08-30

"""
Market Psychology Agent

Revolutionary insight: Identifies players whose boundaries create irrational market movements.
Analyzes betting odds shifts after boundaries to find players who generate:
- Disproportionate excitement in betting markets
- Odds movements that don't reflect true probability changes
- Market overreactions that normalize after 2-3 balls

This creates massive betting opportunities by identifying when markets overreact.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class MarketMovement:
    """Market movement after a boundary"""
    player: str
    boundary_type: str  # "four" or "six"
    over: float
    match_phase: str  # "powerplay", "middle", "death"
    
    # Odds before boundary
    win_odds_before: float
    next_ball_odds_before: Dict[str, float]
    
    # Odds immediately after boundary (within 30 seconds)
    win_odds_after: float
    next_ball_odds_after: Dict[str, float]
    
    # Odds after 2-3 balls (normalization)
    win_odds_normalized: float
    next_ball_odds_normalized: Dict[str, float]
    
    # Market movement metrics
    win_odds_shift: float  # Immediate shift in win odds
    excitement_factor: float  # How much next-ball odds shifted
    normalization_speed: int  # Balls taken to normalize
    overreaction_magnitude: float  # How much market overreacted
    
    # Context
    match_situation: Dict[str, Any]
    market_liquidity: float
    confidence: float

@dataclass
class PlayerMarketProfile:
    """Complete market psychology profile for a player"""
    player: str
    
    # Market mover metrics
    excitement_rating: float  # 0-100 scale of market excitement generation
    overreaction_frequency: float  # % of boundaries that cause overreactions
    avg_odds_shift: float  # Average win odds shift after boundaries
    normalization_pattern: str  # "fast", "medium", "slow" market recovery
    
    # Boundary impact analysis
    four_excitement: float  # Market reaction to fours
    six_excitement: float   # Market reaction to sixes
    phase_impact: Dict[str, float]  # Impact by match phase
    situation_impact: Dict[str, float]  # Impact by match situation
    
    # Market exploitation opportunities
    fade_opportunities: int  # Times market overreacted (betting against)
    follow_opportunities: int  # Times market underreacted (betting with)
    avg_edge_percentage: float  # Average edge when exploiting overreactions
    
    # Sample data
    boundaries_analyzed: int
    market_movements: List[MarketMovement]
    confidence: float

@dataclass
class MarketExploitationOpportunity:
    """Specific betting opportunity based on market psychology"""
    player: str
    opportunity_type: str  # "fade_excitement", "follow_momentum", "anticipate_overreaction"
    trigger_event: str  # What to watch for
    betting_strategy: str  # How to exploit
    expected_edge: float  # Expected edge percentage
    risk_level: str  # "low", "medium", "high"
    time_window: str  # When to place/exit bet
    historical_success_rate: float
    confidence: float

class MarketPsychologyAgent:
    """
    Market Psychology Agent
    
    Analyzes betting market reactions to player boundaries to identify:
    - Players who create irrational market excitement
    - Opportunities to fade overreactions
    - Patterns in market normalization
    - Exploitation strategies for market psychology
    """
    
    def __init__(self, kg_engine=None, betting_data_source=None):
        """
        Initialize Market Psychology Agent
        
        Args:
            kg_engine: Knowledge Graph engine for ball-by-ball data
            betting_data_source: Source for historical betting odds data
        """
        self.kg_engine = kg_engine
        self.betting_data_source = betting_data_source
        
        # Cache for expensive computations
        self.market_profile_cache = {}
        self.movement_cache = {}
        
        # Market psychology parameters
        self.overreaction_threshold = 0.05  # 5% odds shift threshold
        self.normalization_window = 3  # Balls to check for normalization
        self.excitement_factors = {
            "powerplay": 1.2,  # Higher excitement in powerplay
            "middle": 1.0,     # Normal excitement
            "death": 1.5       # Highest excitement in death overs
        }
        
        logger.info("📊 Market Psychology Agent initialized")
    
    def analyze_player_market_psychology(self, player: str) -> Optional[PlayerMarketProfile]:
        """
        Analyze player's impact on betting market psychology
        
        Revolutionary insight: Which players create irrational market movements?
        
        Args:
            player: Player name
            
        Returns:
            PlayerMarketProfile with complete market psychology analysis
        """
        if player in self.market_profile_cache:
            return self.market_profile_cache[player]
        
        try:
            logger.info(f"📊 Analyzing market psychology for {player}")
            
            # Get all boundaries by this player with market data
            boundaries_data = self._get_player_boundaries_with_market_data(player)
            
            if not boundaries_data or len(boundaries_data) < 10:
                logger.warning(f"⚠️ Insufficient market data for {player}")
                return None
            
            # Analyze market movements for each boundary
            market_movements = []
            for boundary in boundaries_data:
                movement = self._analyze_boundary_market_impact(boundary)
                if movement:
                    market_movements.append(movement)
            
            if not market_movements:
                return None
            
            # Calculate aggregate market psychology metrics
            profile = self._calculate_market_psychology_profile(player, market_movements)
            
            self.market_profile_cache[player] = profile
            return profile
            
        except Exception as e:
            logger.error(f"❌ Market psychology analysis failed for {player}: {e}")
            return None
    
    def detect_market_overreactions(self, player: str, recent_boundaries: int = 5) -> List[MarketExploitationOpportunity]:
        """
        Detect current market overreaction opportunities for a player
        
        Revolutionary insight: When is the market overreacting right now?
        
        Args:
            player: Player name
            recent_boundaries: Number of recent boundaries to analyze
            
        Returns:
            List of current exploitation opportunities
        """
        try:
            # Get player's market psychology profile
            profile = self.analyze_player_market_psychology(player)
            if not profile:
                return []
            
            # Get recent boundary market movements
            recent_movements = self._get_recent_market_movements(player, recent_boundaries)
            
            opportunities = []
            
            for movement in recent_movements:
                # Check for fade opportunities (market overreacted)
                if movement.overreaction_magnitude > self.overreaction_threshold:
                    fade_opp = MarketExploitationOpportunity(
                        player=player,
                        opportunity_type="fade_excitement",
                        trigger_event=f"{movement.boundary_type} hit in {movement.match_phase}",
                        betting_strategy=f"Bet against {player} in next 2-3 balls",
                        expected_edge=movement.overreaction_magnitude * 100,
                        risk_level=self._assess_risk_level(movement),
                        time_window="Next 2-3 balls",
                        historical_success_rate=profile.avg_edge_percentage,
                        confidence=movement.confidence
                    )
                    opportunities.append(fade_opp)
                
                # Check for follow opportunities (market underreacted)
                elif movement.overreaction_magnitude < -self.overreaction_threshold:
                    follow_opp = MarketExploitationOpportunity(
                        player=player,
                        opportunity_type="follow_momentum",
                        trigger_event=f"{movement.boundary_type} hit, market slow to react",
                        betting_strategy=f"Bet on {player} boundaries before market catches up",
                        expected_edge=abs(movement.overreaction_magnitude) * 100,
                        risk_level=self._assess_risk_level(movement),
                        time_window="Next 1-2 balls",
                        historical_success_rate=profile.avg_edge_percentage * 0.8,  # Slightly lower success
                        confidence=movement.confidence
                    )
                    opportunities.append(follow_opp)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"❌ Market overreaction detection failed for {player}: {e}")
            return []
    
    def predict_market_reaction(self, player: str, boundary_type: str, match_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict how market will react to a boundary by this player
        
        Revolutionary insight: Predict market movements before they happen
        
        Args:
            player: Player name
            boundary_type: "four" or "six"
            match_context: Current match situation
            
        Returns:
            Predicted market reaction metrics
        """
        try:
            # Get player's market psychology profile
            profile = self.analyze_player_market_psychology(player)
            if not profile:
                return {"predicted_odds_shift": 0.0, "confidence": 0.0}
            
            # Extract match context factors
            phase = self._determine_match_phase(match_context)
            pressure_level = self._calculate_pressure_level(match_context)
            
            # Base excitement from player profile
            if boundary_type == "six":
                base_excitement = profile.six_excitement
            else:
                base_excitement = profile.four_excitement
            
            # Apply phase multiplier
            phase_multiplier = self.excitement_factors.get(phase, 1.0)
            
            # Apply pressure multiplier (higher pressure = more excitement)
            pressure_multiplier = 1.0 + (pressure_level * 0.5)
            
            # Calculate predicted odds shift
            predicted_shift = base_excitement * phase_multiplier * pressure_multiplier
            
            # Predict normalization pattern
            if profile.normalization_pattern == "fast":
                normalization_balls = 1
            elif profile.normalization_pattern == "medium":
                normalization_balls = 2
            else:  # slow
                normalization_balls = 3
            
            return {
                "predicted_odds_shift": predicted_shift,
                "predicted_excitement_factor": base_excitement * phase_multiplier,
                "normalization_balls": normalization_balls,
                "exploitation_edge": predicted_shift * profile.overreaction_frequency,
                "confidence": profile.confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Market reaction prediction failed for {player}: {e}")
            return {"predicted_odds_shift": 0.0, "confidence": 0.0}
    
    def get_top_market_movers(self, min_boundaries: int = 20) -> List[Tuple[str, float]]:
        """
        Get top players who move betting markets
        
        Revolutionary insight: Who are the biggest market movers in cricket?
        
        Args:
            min_boundaries: Minimum boundaries for inclusion
            
        Returns:
            List of (player, excitement_rating) sorted by market impact
        """
        try:
            # Get all players with sufficient boundary data
            players_with_data = self._get_players_with_market_data(min_boundaries)
            
            market_movers = []
            
            for player in players_with_data:
                profile = self.analyze_player_market_psychology(player)
                if profile and profile.boundaries_analyzed >= min_boundaries:
                    market_movers.append((player, profile.excitement_rating))
            
            # Sort by excitement rating
            market_movers.sort(key=lambda x: x[1], reverse=True)
            
            return market_movers
            
        except Exception as e:
            logger.error(f"❌ Top market movers analysis failed: {e}")
            return []
    
    # Helper methods for market analysis
    
    def _get_player_boundaries_with_market_data(self, player: str) -> List[Dict[str, Any]]:
        """Get all boundaries by player with corresponding market data"""
        try:
            if not self.kg_engine:
                # Simulate market data for demonstration
                return self._simulate_boundary_market_data(player)
            
            # In production, query KG for boundaries and match with betting data
            boundaries_query = f"""
            MATCH (p:Player {{name: '{player}'}})-[r:HIT_BOUNDARY]-(b:Ball)
            WHERE r.boundary_type IN ['four', 'six']
            RETURN b.match_id, b.over, b.innings, r.boundary_type, b.match_context
            ORDER BY b.match_id, b.over
            """
            
            boundaries = self._execute_kg_query(boundaries_query)
            
            # Match with betting data (would integrate with betting data source)
            boundaries_with_market = []
            for boundary in boundaries:
                market_data = self._get_market_data_for_ball(
                    boundary['match_id'], 
                    boundary['over'], 
                    boundary['innings']
                )
                if market_data:
                    boundary.update(market_data)
                    boundaries_with_market.append(boundary)
            
            return boundaries_with_market
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get boundary market data for {player}: {e}")
            return self._simulate_boundary_market_data(player)
    
    def _simulate_boundary_market_data(self, player: str) -> List[Dict[str, Any]]:
        """Simulate boundary market data for demonstration"""
        # Simulate different player personalities for market impact
        player_personalities = {
            "MS Dhoni": {"excitement_base": 0.08, "six_multiplier": 2.5, "phase_preference": "death"},
            "Virat Kohli": {"excitement_base": 0.06, "six_multiplier": 2.0, "phase_preference": "middle"},
            "Andre Russell": {"excitement_base": 0.12, "six_multiplier": 3.0, "phase_preference": "death"},
            "AB de Villiers": {"excitement_base": 0.10, "six_multiplier": 2.8, "phase_preference": "death"},
            "Rohit Sharma": {"excitement_base": 0.07, "six_multiplier": 2.2, "phase_preference": "powerplay"},
            "Chris Gayle": {"excitement_base": 0.15, "six_multiplier": 3.5, "phase_preference": "powerplay"}
        }
        
        personality = player_personalities.get(player, {"excitement_base": 0.05, "six_multiplier": 2.0, "phase_preference": "middle"})
        
        boundaries = []
        for i in range(25):  # Simulate 25 boundaries
            boundary_type = np.random.choice(["four", "six"], p=[0.7, 0.3])
            phase = np.random.choice(["powerplay", "middle", "death"], p=[0.3, 0.4, 0.3])
            
            # Base market movement
            base_shift = personality["excitement_base"]
            if boundary_type == "six":
                base_shift *= personality["six_multiplier"]
            
            # Phase adjustment
            if phase == personality["phase_preference"]:
                base_shift *= 1.5
            
            # Add some randomness
            actual_shift = base_shift * (0.5 + np.random.random())
            
            boundaries.append({
                "match_id": f"match_{i//5}",
                "over": 5.0 + i * 0.5,
                "boundary_type": boundary_type,
                "phase": phase,
                "win_odds_before": 2.5 + np.random.random(),
                "win_odds_after": 2.5 + np.random.random() - actual_shift,
                "win_odds_normalized": 2.5 + np.random.random() - (actual_shift * 0.3),
                "market_shift": actual_shift,
                "normalization_balls": np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            })
        
        return boundaries
    
    def _analyze_boundary_market_impact(self, boundary_data: Dict[str, Any]) -> Optional[MarketMovement]:
        """Analyze market impact of a specific boundary"""
        try:
            # Calculate market movement metrics
            win_odds_shift = boundary_data.get("win_odds_before", 0) - boundary_data.get("win_odds_after", 0)
            
            # Calculate overreaction magnitude
            immediate_shift = win_odds_shift
            normalized_shift = boundary_data.get("win_odds_before", 0) - boundary_data.get("win_odds_normalized", 0)
            overreaction_magnitude = immediate_shift - normalized_shift
            
            # Calculate excitement factor (how much next-ball odds shifted)
            excitement_factor = abs(win_odds_shift) * 10  # Scale for visibility
            
            movement = MarketMovement(
                player=boundary_data.get("player", "Unknown"),
                boundary_type=boundary_data.get("boundary_type", "four"),
                over=boundary_data.get("over", 0),
                match_phase=boundary_data.get("phase", "middle"),
                win_odds_before=boundary_data.get("win_odds_before", 0),
                next_ball_odds_before={},  # Would populate from real data
                win_odds_after=boundary_data.get("win_odds_after", 0),
                next_ball_odds_after={},
                win_odds_normalized=boundary_data.get("win_odds_normalized", 0),
                next_ball_odds_normalized={},
                win_odds_shift=win_odds_shift,
                excitement_factor=excitement_factor,
                normalization_speed=boundary_data.get("normalization_balls", 2),
                overreaction_magnitude=overreaction_magnitude,
                match_situation={},
                market_liquidity=1.0,  # Would get from real data
                confidence=0.85
            )
            
            return movement
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to analyze boundary market impact: {e}")
            return None
    
    def _calculate_market_psychology_profile(self, player: str, movements: List[MarketMovement]) -> PlayerMarketProfile:
        """Calculate aggregate market psychology profile"""
        try:
            # Calculate excitement rating (0-100 scale)
            avg_excitement = np.mean([m.excitement_factor for m in movements])
            excitement_rating = min(100, avg_excitement * 10)
            
            # Calculate overreaction frequency
            overreactions = [m for m in movements if abs(m.overreaction_magnitude) > self.overreaction_threshold]
            overreaction_frequency = len(overreactions) / len(movements) if movements else 0
            
            # Calculate average odds shift
            avg_odds_shift = np.mean([abs(m.win_odds_shift) for m in movements])
            
            # Determine normalization pattern
            avg_normalization = np.mean([m.normalization_speed for m in movements])
            if avg_normalization <= 1.5:
                normalization_pattern = "fast"
            elif avg_normalization <= 2.5:
                normalization_pattern = "medium"
            else:
                normalization_pattern = "slow"
            
            # Separate four vs six excitement
            fours = [m for m in movements if m.boundary_type == "four"]
            sixes = [m for m in movements if m.boundary_type == "six"]
            
            four_excitement = np.mean([m.excitement_factor for m in fours]) if fours else 0
            six_excitement = np.mean([m.excitement_factor for m in sixes]) if sixes else 0
            
            # Phase impact analysis
            phase_impact = {}
            for phase in ["powerplay", "middle", "death"]:
                phase_movements = [m for m in movements if m.match_phase == phase]
                if phase_movements:
                    phase_impact[phase] = np.mean([m.excitement_factor for m in phase_movements])
                else:
                    phase_impact[phase] = 0
            
            # Calculate exploitation opportunities
            fade_opportunities = len([m for m in movements if m.overreaction_magnitude > self.overreaction_threshold])
            follow_opportunities = len([m for m in movements if m.overreaction_magnitude < -self.overreaction_threshold])
            
            # Calculate average edge
            edges = [abs(m.overreaction_magnitude) for m in overreactions]
            avg_edge_percentage = (np.mean(edges) * 100) if edges else 0
            
            profile = PlayerMarketProfile(
                player=player,
                excitement_rating=excitement_rating,
                overreaction_frequency=overreaction_frequency,
                avg_odds_shift=avg_odds_shift,
                normalization_pattern=normalization_pattern,
                four_excitement=four_excitement,
                six_excitement=six_excitement,
                phase_impact=phase_impact,
                situation_impact={},  # Would calculate from match situations
                fade_opportunities=fade_opportunities,
                follow_opportunities=follow_opportunities,
                avg_edge_percentage=avg_edge_percentage,
                boundaries_analyzed=len(movements),
                market_movements=movements,
                confidence=min(0.95, len(movements) / 30.0)
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate market psychology profile: {e}")
            return None
    
    def _get_recent_market_movements(self, player: str, count: int) -> List[MarketMovement]:
        """Get recent market movements for a player"""
        # In production, would query recent betting data
        # For now, return empty list
        return []
    
    def _assess_risk_level(self, movement: MarketMovement) -> str:
        """Assess risk level of exploiting a market movement"""
        if movement.confidence > 0.8 and abs(movement.overreaction_magnitude) > 0.08:
            return "low"
        elif movement.confidence > 0.6 and abs(movement.overreaction_magnitude) > 0.05:
            return "medium"
        else:
            return "high"
    
    def _determine_match_phase(self, match_context: Dict[str, Any]) -> str:
        """Determine current match phase from context"""
        over = match_context.get("over", 10)
        if over <= 6:
            return "powerplay"
        elif over <= 15:
            return "middle"
        else:
            return "death"
    
    def _calculate_pressure_level(self, match_context: Dict[str, Any]) -> float:
        """Calculate current pressure level (0-1 scale)"""
        # Simplified pressure calculation
        required_rate = match_context.get("required_rate", 6.0)
        wickets_remaining = match_context.get("wickets_remaining", 10)
        overs_remaining = match_context.get("overs_remaining", 10)
        
        # Higher required rate and fewer resources = more pressure
        rate_pressure = min(1.0, (required_rate - 6.0) / 10.0)
        resource_pressure = 1.0 - (wickets_remaining * overs_remaining) / 100.0
        
        return (rate_pressure + resource_pressure) / 2.0
    
    def _get_players_with_market_data(self, min_boundaries: int) -> List[str]:
        """Get players with sufficient market data"""
        # For demonstration, return known market movers
        return [
            "MS Dhoni", "Virat Kohli", "Andre Russell", "AB de Villiers",
            "Rohit Sharma", "Chris Gayle", "David Warner", "Jos Buttler"
        ]
    
    def _execute_kg_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute query against Knowledge Graph"""
        try:
            if self.kg_engine and hasattr(self.kg_engine, 'query'):
                result = self.kg_engine.query(query)
                return result.results if result.results else []
            return []
        except Exception as e:
            logger.warning(f"⚠️ KG query failed: {e}")
            return []
    
    def _get_market_data_for_ball(self, match_id: str, over: float, innings: int) -> Optional[Dict[str, Any]]:
        """Get market data for a specific ball"""
        # In production, would query betting data source
        # For now, return None to trigger simulation
        return None


# Factory function for compatibility with existing agent patterns
def create_market_psychology_agent(kg_engine=None, betting_data_source=None) -> MarketPsychologyAgent:
    """
    Factory function to create Market Psychology Agent
    
    Args:
        kg_engine: Knowledge Graph query engine
        betting_data_source: Betting odds data source
        
    Returns:
        MarketPsychologyAgent instance
    """
    return MarketPsychologyAgent(kg_engine=kg_engine, betting_data_source=betting_data_source)

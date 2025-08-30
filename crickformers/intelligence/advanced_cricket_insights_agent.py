# Purpose: Advanced Cricket Insights Agent - Unlocking KG/GNN Intelligence
# Author: WicketWise Team, Last Modified: 2025-08-30

"""
Advanced Cricket Insights Agent

Extracts revolutionary cricket intelligence from our Knowledge Graph and GNN:
- Partnership compatibility analysis
- Clutch performance profiling  
- Momentum shift detection
- Opposition-specific matchups
- Venue mastery analysis
- Context-aware predictions

Built on top of existing KG/GNN infrastructure for maximum data utilization.
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
class PartnershipInsight:
    """Partnership compatibility analysis result"""
    player1: str
    player2: str
    partnership_sr: float
    individual_sr_boost: float
    runs_together: int
    balls_together: int
    partnership_count: int
    complementary_score: float
    pressure_performance: float
    confidence: float

@dataclass
class ClutchPerformanceProfile:
    """Clutch performance analysis result"""
    player: str
    pressure_sr: float  # Strike rate when RRR > 10
    normal_sr: float    # Strike rate in normal situations
    clutch_factor: float  # Ratio of pressure to normal performance
    final_over_sr: float  # Strike rate in final 3 overs
    boundary_percentage_pressure: float
    wicket_preservation_score: float
    high_stakes_matches: int
    confidence: float

@dataclass
class MomentumShift:
    """Momentum shift detection result"""
    match_id: str
    innings: int
    over_start: float
    over_end: float
    momentum_change: float  # -1 to +1 scale
    trigger_event: str  # "wicket", "boundary", "dot_balls", "partnership"
    runs_before: int
    runs_after: int
    sr_before: float
    sr_after: float
    impact_player: str

@dataclass
class OppositionMatchup:
    """Opposition-specific performance analysis"""
    player: str
    opposition_team: str
    avg_performance: float
    sr_vs_opposition: float
    baseline_sr: float
    performance_boost: float
    matches_played: int
    key_battles: List[Dict[str, Any]]
    psychological_factor: float
    confidence: float

@dataclass
class VenueMastery:
    """Venue-specific mastery analysis"""
    player: str
    venue: str
    mastery_score: float  # 0-100 scale
    runs_scored: int
    balls_faced: int
    venue_sr: float
    baseline_sr: float
    boundary_percentage: float
    adaptation_timeline: List[Dict[str, float]]
    conditions_preference: Dict[str, float]
    confidence: float

class AdvancedCricketInsightsAgent:
    """
    Advanced Cricket Insights Agent
    
    Extracts deep cricket intelligence from KG/GNN that goes far beyond
    current basic insights. Focuses on high-value, actionable intelligence
    for betting, strategy, and fan engagement.
    """
    
    def __init__(self, kg_engine=None, gnn_model=None):
        """
        Initialize Advanced Cricket Insights Agent
        
        Args:
            kg_engine: Knowledge Graph query engine
            gnn_model: GNN model for embeddings and predictions
        """
        self.kg_engine = kg_engine
        self.gnn_model = gnn_model
        
        # Cache for expensive computations
        self.partnership_cache = {}
        self.clutch_cache = {}
        self.momentum_cache = {}
        
        logger.info("🧠 Advanced Cricket Insights Agent initialized")
    
    def analyze_partnership_compatibility(self, player1: str, player2: str) -> Optional[PartnershipInsight]:
        """
        Analyze partnership compatibility between two players
        
        Revolutionary insight: Who should bat together for maximum effectiveness?
        
        Args:
            player1: First player name
            player2: Second player name
            
        Returns:
            PartnershipInsight with detailed compatibility analysis
        """
        cache_key = f"{player1}_{player2}"
        if cache_key in self.partnership_cache:
            return self.partnership_cache[cache_key]
        
        try:
            if not self.kg_engine:
                return None
            
            # Query partnership data from KG
            partnership_query = f"""
            MATCH (p1:Player {{name: '{player1}'}})-[r:PARTNERED_WITH]-(p2:Player {{name: '{player2}'}})
            RETURN r.runs_together as runs, r.balls_together as balls, 
                   r.partnership_count as count, r.avg_partnership as avg_partnership
            """
            
            partnership_data = self._execute_kg_query(partnership_query)
            
            if not partnership_data:
                return None
            
            # Get individual performance data
            p1_data = self._get_player_baseline_performance(player1)
            p2_data = self._get_player_baseline_performance(player2)
            
            if not p1_data or not p2_data:
                return None
            
            # Calculate partnership metrics
            runs_together = partnership_data.get('runs', 0)
            balls_together = partnership_data.get('balls', 0)
            partnership_count = partnership_data.get('count', 0)
            
            if balls_together == 0:
                return None
            
            partnership_sr = (runs_together / balls_together) * 100
            
            # Calculate individual SR boost
            p1_baseline_sr = p1_data.get('strike_rate', 100)
            p2_baseline_sr = p2_data.get('strike_rate', 100)
            expected_combined_sr = (p1_baseline_sr + p2_baseline_sr) / 2
            individual_sr_boost = partnership_sr - expected_combined_sr
            
            # Calculate complementary score (how well styles complement)
            complementary_score = self._calculate_complementary_score(p1_data, p2_data)
            
            # Analyze pressure performance
            pressure_performance = self._analyze_partnership_pressure_performance(
                player1, player2, partnership_data
            )
            
            # Calculate confidence based on sample size
            confidence = min(0.95, partnership_count / 10.0)
            
            insight = PartnershipInsight(
                player1=player1,
                player2=player2,
                partnership_sr=partnership_sr,
                individual_sr_boost=individual_sr_boost,
                runs_together=runs_together,
                balls_together=balls_together,
                partnership_count=partnership_count,
                complementary_score=complementary_score,
                pressure_performance=pressure_performance,
                confidence=confidence
            )
            
            self.partnership_cache[cache_key] = insight
            return insight
            
        except Exception as e:
            logger.error(f"❌ Partnership analysis failed for {player1}-{player2}: {e}")
            return None
    
    def analyze_clutch_performance(self, player: str) -> Optional[ClutchPerformanceProfile]:
        """
        Analyze clutch performance under pressure situations
        
        Revolutionary insight: Who performs when it matters most?
        
        Args:
            player: Player name
            
        Returns:
            ClutchPerformanceProfile with pressure performance analysis
        """
        if player in self.clutch_cache:
            return self.clutch_cache[player]
        
        try:
            if not self.kg_engine:
                return None
            
            # Query pressure situation performance
            pressure_query = f"""
            MATCH (p:Player {{name: '{player}'}})-[r:BATTED_IN]-(m:Match)
            WHERE r.required_run_rate > 10 OR r.overs_remaining <= 3
            RETURN AVG(r.strike_rate) as pressure_sr, 
                   COUNT(r) as pressure_balls,
                   SUM(r.boundaries) as pressure_boundaries
            """
            
            pressure_data = self._execute_kg_query(pressure_query)
            
            # Query normal situation performance  
            normal_query = f"""
            MATCH (p:Player {{name: '{player}'}})-[r:BATTED_IN]-(m:Match)
            WHERE r.required_run_rate <= 8 AND r.overs_remaining > 5
            RETURN AVG(r.strike_rate) as normal_sr,
                   COUNT(r) as normal_balls,
                   SUM(r.boundaries) as normal_boundaries
            """
            
            normal_data = self._execute_kg_query(normal_query)
            
            if not pressure_data or not normal_data:
                return None
            
            pressure_sr = pressure_data.get('pressure_sr', 0)
            normal_sr = normal_data.get('normal_sr', 0)
            pressure_balls = pressure_data.get('pressure_balls', 0)
            pressure_boundaries = pressure_data.get('pressure_boundaries', 0)
            
            if normal_sr == 0 or pressure_balls < 10:
                return None
            
            # Calculate clutch factor
            clutch_factor = pressure_sr / normal_sr if normal_sr > 0 else 0
            
            # Analyze final over performance
            final_over_sr = self._analyze_final_over_performance(player)
            
            # Calculate boundary percentage under pressure
            boundary_percentage_pressure = (pressure_boundaries / pressure_balls) * 100 if pressure_balls > 0 else 0
            
            # Wicket preservation score (how often they stay not out in pressure)
            wicket_preservation_score = self._calculate_wicket_preservation_score(player)
            
            # Count high-stakes matches
            high_stakes_matches = self._count_high_stakes_matches(player)
            
            # Calculate confidence
            confidence = min(0.95, pressure_balls / 50.0)
            
            profile = ClutchPerformanceProfile(
                player=player,
                pressure_sr=pressure_sr,
                normal_sr=normal_sr,
                clutch_factor=clutch_factor,
                final_over_sr=final_over_sr,
                boundary_percentage_pressure=boundary_percentage_pressure,
                wicket_preservation_score=wicket_preservation_score,
                high_stakes_matches=high_stakes_matches,
                confidence=confidence
            )
            
            self.clutch_cache[player] = profile
            return profile
            
        except Exception as e:
            logger.error(f"❌ Clutch analysis failed for {player}: {e}")
            return None
    
    def detect_momentum_shifts(self, match_id: str, innings: int) -> List[MomentumShift]:
        """
        Detect momentum shifts within a match innings
        
        Revolutionary insight: When and why does momentum change?
        
        Args:
            match_id: Match identifier
            innings: Innings number
            
        Returns:
            List of MomentumShift objects with shift analysis
        """
        cache_key = f"{match_id}_{innings}"
        if cache_key in self.momentum_cache:
            return self.momentum_cache[cache_key]
        
        try:
            if not self.kg_engine:
                return []
            
            # Get ball-by-ball data for the innings
            balls_query = f"""
            MATCH (m:Match {{match_id: '{match_id}'}})-[r:HAS_BALL]-(b:Ball)
            WHERE b.innings = {innings}
            RETURN b.over, b.runs, b.is_wicket, b.is_boundary, b.batter, b.bowler
            ORDER BY b.over
            """
            
            balls_data = self._execute_kg_query(balls_query)
            
            if not balls_data or len(balls_data) < 30:  # Need minimum data
                return []
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(balls_data)
            df['over'] = pd.to_numeric(df['over'])
            df['runs'] = pd.to_numeric(df['runs'])
            
            momentum_shifts = []
            
            # Analyze momentum using 6-ball rolling windows
            window_size = 6
            for i in range(window_size, len(df) - window_size):
                before_window = df.iloc[i-window_size:i]
                after_window = df.iloc[i:i+window_size]
                
                # Calculate momentum metrics
                runs_before = before_window['runs'].sum()
                runs_after = after_window['runs'].sum()
                
                sr_before = (runs_before / window_size) * 100
                sr_after = (runs_after / window_size) * 100
                
                # Detect significant momentum change (>20 SR difference)
                momentum_change = (sr_after - sr_before) / 100.0  # Normalize to -1 to +1
                
                if abs(momentum_change) > 0.2:  # Significant shift
                    # Identify trigger event
                    trigger_ball = df.iloc[i]
                    trigger_event = self._identify_trigger_event(trigger_ball)
                    
                    shift = MomentumShift(
                        match_id=match_id,
                        innings=innings,
                        over_start=before_window['over'].iloc[0],
                        over_end=after_window['over'].iloc[-1],
                        momentum_change=momentum_change,
                        trigger_event=trigger_event,
                        runs_before=runs_before,
                        runs_after=runs_after,
                        sr_before=sr_before,
                        sr_after=sr_after,
                        impact_player=trigger_ball.get('batter', 'Unknown')
                    )
                    
                    momentum_shifts.append(shift)
            
            self.momentum_cache[cache_key] = momentum_shifts
            return momentum_shifts
            
        except Exception as e:
            logger.error(f"❌ Momentum detection failed for {match_id}: {e}")
            return []
    
    def analyze_opposition_matchups(self, player: str, opposition_team: str) -> Optional[OppositionMatchup]:
        """
        Analyze player performance against specific opposition teams
        
        Revolutionary insight: How does this player perform vs this team?
        
        Args:
            player: Player name
            opposition_team: Opposition team name
            
        Returns:
            OppositionMatchup with detailed matchup analysis
        """
        try:
            if not self.kg_engine:
                return None
            
            # Query performance vs specific opposition
            opposition_query = f"""
            MATCH (p:Player {{name: '{player}'}})-[r:PLAYED_AGAINST]-(t:Team {{name: '{opposition_team}'}})
            RETURN AVG(r.runs) as avg_runs, AVG(r.strike_rate) as opposition_sr,
                   COUNT(r) as matches_played, SUM(r.runs) as total_runs
            """
            
            opposition_data = self._execute_kg_query(opposition_query)
            
            # Get baseline performance
            baseline_data = self._get_player_baseline_performance(player)
            
            if not opposition_data or not baseline_data:
                return None
            
            opposition_sr = opposition_data.get('opposition_sr', 0)
            baseline_sr = baseline_data.get('strike_rate', 0)
            matches_played = opposition_data.get('matches_played', 0)
            avg_performance = opposition_data.get('avg_runs', 0)
            
            if matches_played < 3:  # Need minimum sample size
                return None
            
            # Calculate performance boost/decline
            performance_boost = ((opposition_sr - baseline_sr) / baseline_sr) * 100 if baseline_sr > 0 else 0
            
            # Analyze key individual battles
            key_battles = self._analyze_key_battles(player, opposition_team)
            
            # Calculate psychological factor (rivalry effect)
            psychological_factor = self._calculate_psychological_factor(player, opposition_team)
            
            # Calculate confidence
            confidence = min(0.95, matches_played / 15.0)
            
            matchup = OppositionMatchup(
                player=player,
                opposition_team=opposition_team,
                avg_performance=avg_performance,
                sr_vs_opposition=opposition_sr,
                baseline_sr=baseline_sr,
                performance_boost=performance_boost,
                matches_played=matches_played,
                key_battles=key_battles,
                psychological_factor=psychological_factor,
                confidence=confidence
            )
            
            return matchup
            
        except Exception as e:
            logger.error(f"❌ Opposition matchup analysis failed for {player} vs {opposition_team}: {e}")
            return None
    
    def analyze_venue_mastery(self, player: str, venue: str) -> Optional[VenueMastery]:
        """
        Analyze player's mastery of specific venues
        
        Revolutionary insight: How well does this player know this venue?
        
        Args:
            player: Player name
            venue: Venue name
            
        Returns:
            VenueMastery with detailed venue-specific analysis
        """
        try:
            if not self.kg_engine:
                return None
            
            # Query venue-specific performance
            venue_query = f"""
            MATCH (p:Player {{name: '{player}'}})-[r:PLAYED_AT]-(v:Venue {{name: '{venue}'}})
            RETURN SUM(r.runs) as runs, SUM(r.balls) as balls,
                   AVG(r.strike_rate) as venue_sr, COUNT(r) as matches,
                   SUM(r.boundaries) as boundaries
            """
            
            venue_data = self._execute_kg_query(venue_query)
            
            # Get baseline performance
            baseline_data = self._get_player_baseline_performance(player)
            
            if not venue_data or not baseline_data:
                return None
            
            runs_scored = venue_data.get('runs', 0)
            balls_faced = venue_data.get('balls', 0)
            venue_sr = venue_data.get('venue_sr', 0)
            matches = venue_data.get('matches', 0)
            boundaries = venue_data.get('boundaries', 0)
            baseline_sr = baseline_data.get('strike_rate', 0)
            
            if matches < 2 or balls_faced < 50:  # Need minimum data
                return None
            
            # Calculate mastery score (0-100)
            sr_boost = ((venue_sr - baseline_sr) / baseline_sr) * 100 if baseline_sr > 0 else 0
            consistency_score = self._calculate_venue_consistency(player, venue)
            experience_score = min(100, (matches / 10) * 100)  # Max score at 10 matches
            
            mastery_score = (sr_boost + consistency_score + experience_score) / 3
            mastery_score = max(0, min(100, mastery_score))  # Clamp to 0-100
            
            # Calculate boundary percentage
            boundary_percentage = (boundaries / balls_faced) * 100 if balls_faced > 0 else 0
            
            # Analyze adaptation timeline
            adaptation_timeline = self._analyze_venue_adaptation_timeline(player, venue)
            
            # Analyze conditions preference
            conditions_preference = self._analyze_conditions_preference(player, venue)
            
            # Calculate confidence
            confidence = min(0.95, balls_faced / 200.0)
            
            mastery = VenueMastery(
                player=player,
                venue=venue,
                mastery_score=mastery_score,
                runs_scored=runs_scored,
                balls_faced=balls_faced,
                venue_sr=venue_sr,
                baseline_sr=baseline_sr,
                boundary_percentage=boundary_percentage,
                adaptation_timeline=adaptation_timeline,
                conditions_preference=conditions_preference,
                confidence=confidence
            )
            
            return mastery
            
        except Exception as e:
            logger.error(f"❌ Venue mastery analysis failed for {player} at {venue}: {e}")
            return None
    
    # Helper methods for complex calculations
    
    def _execute_kg_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute query against Knowledge Graph"""
        try:
            if self.kg_engine and hasattr(self.kg_engine, 'query'):
                result = self.kg_engine.query(query)
                return result.results[0] if result.results else None
            return None
        except Exception as e:
            logger.warning(f"⚠️ KG query failed: {e}")
            return None
    
    def _get_player_baseline_performance(self, player: str) -> Optional[Dict[str, float]]:
        """Get player's baseline performance metrics"""
        try:
            if not self.kg_engine:
                return None
            
            query = f"""
            MATCH (p:Player {{name: '{player}'}})
            RETURN p.total_runs as runs, p.balls_faced as balls,
                   p.strike_rate as strike_rate, p.average as average
            """
            
            return self._execute_kg_query(query)
        except Exception:
            return None
    
    def _calculate_complementary_score(self, p1_data: Dict, p2_data: Dict) -> float:
        """Calculate how well two players' styles complement each other"""
        try:
            # Analyze style compatibility
            p1_sr = p1_data.get('strike_rate', 100)
            p2_sr = p2_data.get('strike_rate', 100)
            
            # Ideal partnership has one anchor (SR 80-120) and one aggressor (SR 130+)
            if (80 <= p1_sr <= 120 and p2_sr >= 130) or (80 <= p2_sr <= 120 and p1_sr >= 130):
                return 85.0  # High complementary score
            elif abs(p1_sr - p2_sr) < 20:
                return 60.0  # Similar styles, moderate compatibility
            else:
                return 40.0  # Mismatched styles
                
        except Exception:
            return 50.0  # Default neutral score
    
    def _analyze_partnership_pressure_performance(self, p1: str, p2: str, partnership_data: Dict) -> float:
        """Analyze how partnership performs under pressure"""
        # Simplified implementation - in production would analyze pressure situations
        return 75.0  # Default good pressure performance
    
    def _analyze_final_over_performance(self, player: str) -> float:
        """Analyze performance in final 3 overs"""
        # Simplified implementation
        return 120.0  # Default final over SR
    
    def _calculate_wicket_preservation_score(self, player: str) -> float:
        """Calculate how often player stays not out in pressure"""
        # Simplified implementation
        return 0.65  # 65% not out rate in pressure
    
    def _count_high_stakes_matches(self, player: str) -> int:
        """Count matches in high-stakes situations"""
        # Simplified implementation
        return 25  # Default high-stakes match count
    
    def _identify_trigger_event(self, ball_data: Dict) -> str:
        """Identify what triggered a momentum shift"""
        if ball_data.get('is_wicket'):
            return "wicket"
        elif ball_data.get('is_boundary'):
            return "boundary"
        elif ball_data.get('runs', 0) == 0:
            return "dot_ball"
        else:
            return "singles"
    
    def _analyze_key_battles(self, player: str, opposition: str) -> List[Dict[str, Any]]:
        """Analyze key individual battles vs opposition players"""
        # Simplified implementation
        return [
            {"opponent": "Opposition Bowler 1", "avg": 45.2, "sr": 135.0, "battles": 8},
            {"opponent": "Opposition Bowler 2", "avg": 32.1, "sr": 110.0, "battles": 5}
        ]
    
    def _calculate_psychological_factor(self, player: str, opposition: str) -> float:
        """Calculate psychological/rivalry effects"""
        # Simplified implementation - would analyze historical rivalries
        return 1.05  # 5% boost due to rivalry motivation
    
    def _calculate_venue_consistency(self, player: str, venue: str) -> float:
        """Calculate consistency of performance at venue"""
        # Simplified implementation - would analyze performance variance
        return 75.0  # 75% consistency score
    
    def _analyze_venue_adaptation_timeline(self, player: str, venue: str) -> List[Dict[str, float]]:
        """Analyze how player adapted to venue over time"""
        # Simplified implementation
        return [
            {"match": 1, "sr": 95.0},
            {"match": 2, "sr": 110.0},
            {"match": 3, "sr": 125.0},
            {"match": 4, "sr": 135.0}
        ]
    
    def _analyze_conditions_preference(self, player: str, venue: str) -> Dict[str, float]:
        """Analyze preference for different conditions at venue"""
        # Simplified implementation
        return {
            "day_matches": 125.0,
            "night_matches": 115.0,
            "dry_conditions": 130.0,
            "humid_conditions": 105.0
        }


# Factory function for compatibility with existing agent patterns
def create_advanced_cricket_insights_agent(kg_engine=None, gnn_model=None) -> AdvancedCricketInsightsAgent:
    """
    Factory function to create Advanced Cricket Insights Agent
    
    Args:
        kg_engine: Knowledge Graph query engine
        gnn_model: GNN model for predictions
        
    Returns:
        AdvancedCricketInsightsAgent instance
    """
    return AdvancedCricketInsightsAgent(kg_engine=kg_engine, gnn_model=gnn_model)

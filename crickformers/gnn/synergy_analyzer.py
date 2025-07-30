# Purpose: Player synergy analysis and has_synergy_with edge generation
# Author: Shamus Rae, Last Modified: 2024-01-15

"""
This module implements player synergy analysis to identify and quantify
partnerships and collaborations in cricket. It computes synergy scores
for batting pairs, bowler-fielder combinations, and bowler-captain
relationships, then creates weighted has_synergy_with edges in the
knowledge graph.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging
import networkx as nx
from itertools import combinations

logger = logging.getLogger(__name__)


@dataclass
class SynergyConfig:
    """Configuration for synergy analysis."""
    
    # Minimum thresholds for synergy computation
    min_batting_partnerships: int = 5      # Minimum partnerships to analyze
    min_bowling_overs: int = 10            # Minimum overs bowled together
    min_fielding_dismissals: int = 3       # Minimum shared dismissals
    min_captain_overs: int = 20            # Minimum overs under captain
    
    # Synergy score thresholds
    batting_synergy_threshold: float = 0.6  # Minimum batting synergy score
    bowling_synergy_threshold: float = 0.5  # Minimum bowling synergy score
    fielding_synergy_threshold: float = 0.4 # Minimum fielding synergy score
    captain_synergy_threshold: float = 0.5  # Minimum captain synergy score
    
    # Weight factors for different synergy types
    batting_weight: float = 1.0
    bowling_weight: float = 0.8
    fielding_weight: float = 0.6
    captain_weight: float = 0.7
    
    # Edge creation parameters
    max_synergy_edges_per_player: int = 10  # Limit edges per player
    synergy_decay_factor: float = 0.9       # Decay for older partnerships


@dataclass
class BattingSynergyMetrics:
    """Metrics for batting partnership synergy."""
    
    partnerships_count: int
    total_runs: int
    total_balls: int
    average_partnership: float
    strike_rotation_rate: float    # Rate of strike rotation
    run_rate: float               # Partnership run rate
    non_dismissal_correlation: float  # How often both survive
    boundary_synergy: float       # Combined boundary scoring
    pressure_performance: float   # Performance under pressure


@dataclass
class BowlingFieldingSynergyMetrics:
    """Metrics for bowler-fielder synergy."""
    
    shared_dismissals: int
    total_overs_together: int
    wicket_rate: float           # Wickets per over when together
    catch_success_rate: float    # Successful catches by fielder off bowler
    run_saving_efficiency: float # Runs saved through good fielding
    pressure_wickets: int        # Wickets in pressure situations


@dataclass
class CaptainBowlerSynergyMetrics:
    """Metrics for captain-bowler synergy."""
    
    overs_under_captain: int
    wickets_under_captain: int
    economy_under_captain: float
    field_setting_effectiveness: float  # Success with captain's field settings
    pressure_situation_success: float  # Success in pressure moments
    tactical_adaptability: float       # Adaptation to match situations


class SynergyScoreCalculator:
    """Calculates synergy scores between players."""
    
    def __init__(self, config: SynergyConfig = None):
        self.config = config or SynergyConfig()
    
    def calculate_batting_synergy(
        self,
        player1: str,
        player2: str,
        match_data: pd.DataFrame
    ) -> Tuple[float, BattingSynergyMetrics]:
        """
        Calculate batting synergy score between two players.
        
        Args:
            player1: First batter ID
            player2: Second batter ID
            match_data: Ball-by-ball match data
        
        Returns:
            Tuple of (synergy_score, metrics)
        """
        # Filter data for partnerships between these players
        partnerships = self._extract_batting_partnerships(player1, player2, match_data)
        
        if len(partnerships) < self.config.min_batting_partnerships:
            return 0.0, BattingSynergyMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate partnership metrics
        total_runs = sum(p['runs'] for p in partnerships)
        total_balls = sum(p['balls'] for p in partnerships)
        partnerships_count = len(partnerships)
        
        if total_balls == 0:
            return 0.0, BattingSynergyMetrics(partnerships_count, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Core metrics
        average_partnership = total_runs / partnerships_count
        run_rate = (total_runs / total_balls) * 6.0  # Runs per over
        
        # Advanced synergy metrics
        strike_rotation_rate = self._calculate_strike_rotation_rate(partnerships)
        non_dismissal_correlation = self._calculate_non_dismissal_correlation(partnerships)
        boundary_synergy = self._calculate_boundary_synergy(partnerships)
        pressure_performance = self._calculate_pressure_performance(partnerships, match_data)
        
        # Compute overall synergy score
        synergy_score = self._compute_batting_synergy_score(
            run_rate, strike_rotation_rate, non_dismissal_correlation,
            boundary_synergy, pressure_performance, partnerships_count
        )
        
        metrics = BattingSynergyMetrics(
            partnerships_count=partnerships_count,
            total_runs=total_runs,
            total_balls=total_balls,
            average_partnership=average_partnership,
            strike_rotation_rate=strike_rotation_rate,
            run_rate=run_rate,
            non_dismissal_correlation=non_dismissal_correlation,
            boundary_synergy=boundary_synergy,
            pressure_performance=pressure_performance
        )
        
        return synergy_score, metrics
    
    def calculate_bowling_fielding_synergy(
        self,
        bowler: str,
        fielder: str,
        match_data: pd.DataFrame
    ) -> Tuple[float, BowlingFieldingSynergyMetrics]:
        """
        Calculate synergy between bowler and fielder.
        
        Args:
            bowler: Bowler ID
            fielder: Fielder ID
            match_data: Ball-by-ball match data
        
        Returns:
            Tuple of (synergy_score, metrics)
        """
        # Check if required columns exist
        required_cols = ['bowler', 'fielder']
        if len(match_data) == 0 or not all(col in match_data.columns for col in required_cols):
            return 0.0, BowlingFieldingSynergyMetrics(0, 0, 0.0, 0.0, 0.0, 0)
        
        # Filter data for bowler-fielder combinations
        bowler_fielder_data = match_data[
            (match_data['bowler'] == bowler) & 
            (match_data['fielder'] == fielder)
        ]
        
        # Also consider general fielding when this fielder is on field
        bowler_data = match_data[match_data['bowler'] == bowler]
        
        if len(bowler_fielder_data) == 0:
            return 0.0, BowlingFieldingSynergyMetrics(0, 0, 0.0, 0.0, 0.0, 0)
        
        # Calculate dismissal metrics
        shared_dismissals = len(bowler_fielder_data[bowler_fielder_data['wicket_type'].notna()])
        total_overs_together = len(bowler_fielder_data) / 6.0
        
        if total_overs_together < self.config.min_bowling_overs / 6.0:
            return 0.0, BowlingFieldingSynergyMetrics(shared_dismissals, int(total_overs_together * 6), 0.0, 0.0, 0.0, 0)
        
        # Core metrics
        wicket_rate = shared_dismissals / total_overs_together if total_overs_together > 0 else 0.0
        catch_success_rate = self._calculate_catch_success_rate(bowler_fielder_data)
        run_saving_efficiency = self._calculate_run_saving_efficiency(bowler_fielder_data, bowler_data)
        pressure_wickets = self._count_pressure_wickets(bowler_fielder_data)
        
        # Compute synergy score
        synergy_score = self._compute_bowling_fielding_synergy_score(
            wicket_rate, catch_success_rate, run_saving_efficiency, 
            shared_dismissals, pressure_wickets
        )
        
        metrics = BowlingFieldingSynergyMetrics(
            shared_dismissals=shared_dismissals,
            total_overs_together=int(total_overs_together * 6),
            wicket_rate=wicket_rate,
            catch_success_rate=catch_success_rate,
            run_saving_efficiency=run_saving_efficiency,
            pressure_wickets=pressure_wickets
        )
        
        return synergy_score, metrics
    
    def calculate_captain_bowler_synergy(
        self,
        captain: str,
        bowler: str,
        match_data: pd.DataFrame
    ) -> Tuple[float, CaptainBowlerSynergyMetrics]:
        """
        Calculate synergy between captain and bowler.
        
        Args:
            captain: Captain ID
            bowler: Bowler ID
            match_data: Ball-by-ball match data
        
        Returns:
            Tuple of (synergy_score, metrics)
        """
        # Check if required columns exist
        required_cols = ['bowler', 'captain']
        if len(match_data) == 0 or not all(col in match_data.columns for col in required_cols):
            return 0.0, CaptainBowlerSynergyMetrics(0, 0, 0.0, 0.0, 0.0, 0.0)
        
        # Filter data for bowler under this captain
        captain_bowler_data = match_data[
            (match_data['bowler'] == bowler) & 
            (match_data['captain'] == captain)
        ]
        
        # Compare with bowler under other captains
        other_captain_data = match_data[
            (match_data['bowler'] == bowler) & 
            (match_data['captain'] != captain) &
            (match_data['captain'].notna())
        ]
        
        if len(captain_bowler_data) < self.config.min_captain_overs:
            return 0.0, CaptainBowlerSynergyMetrics(0, 0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate performance metrics
        overs_under_captain = len(captain_bowler_data) / 6.0
        wickets_under_captain = len(captain_bowler_data[captain_bowler_data['wicket_type'].notna()])
        runs_conceded = captain_bowler_data['runs_scored'].sum()
        economy_under_captain = (runs_conceded / overs_under_captain) if overs_under_captain > 0 else 0.0
        
        # Advanced metrics
        field_setting_effectiveness = self._calculate_field_setting_effectiveness(captain_bowler_data)
        pressure_situation_success = self._calculate_pressure_situation_success(captain_bowler_data)
        tactical_adaptability = self._calculate_tactical_adaptability(captain_bowler_data, other_captain_data)
        
        # Compute synergy score
        synergy_score = self._compute_captain_bowler_synergy_score(
            wickets_under_captain, economy_under_captain, field_setting_effectiveness,
            pressure_situation_success, tactical_adaptability, overs_under_captain
        )
        
        metrics = CaptainBowlerSynergyMetrics(
            overs_under_captain=int(overs_under_captain * 6),
            wickets_under_captain=wickets_under_captain,
            economy_under_captain=economy_under_captain,
            field_setting_effectiveness=field_setting_effectiveness,
            pressure_situation_success=pressure_situation_success,
            tactical_adaptability=tactical_adaptability
        )
        
        return synergy_score, metrics
    
    def _extract_batting_partnerships(
        self,
        player1: str,
        player2: str,
        match_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Extract partnership data between two batters."""
        partnerships = []
        
        # Check if required columns exist
        required_cols = ['match_id', 'innings', 'batter', 'non_striker', 'runs_scored']
        if len(match_data) == 0 or not all(col in match_data.columns for col in required_cols):
            return partnerships
        
        # Group by match and innings
        for (match_id, innings), group in match_data.groupby(['match_id', 'innings']):
            # Find partnerships where both players bat together
            partnership_balls = group[
                ((group['batter'] == player1) & (group['non_striker'] == player2)) |
                ((group['batter'] == player2) & (group['non_striker'] == player1))
            ]
            
            if len(partnership_balls) > 0:
                partnership_runs = partnership_balls['runs_scored'].sum()
                partnership_balls_count = len(partnership_balls)
                boundaries = len(partnership_balls[partnership_balls['runs_scored'] >= 4])
                
                # Check how partnership ended
                last_ball = partnership_balls.iloc[-1]
                ended_by_dismissal = pd.notna(last_ball.get('wicket_type'))
                dismissed_player = last_ball.get('batter') if ended_by_dismissal else None
                
                partnerships.append({
                    'match_id': match_id,
                    'innings': innings,
                    'runs': partnership_runs,
                    'balls': partnership_balls_count,
                    'boundaries': boundaries,
                    'ended_by_dismissal': ended_by_dismissal,
                    'dismissed_player': dismissed_player,
                    'partnership_data': partnership_balls
                })
        
        return partnerships
    
    def _calculate_strike_rotation_rate(self, partnerships: List[Dict[str, Any]]) -> float:
        """Calculate how often strike is rotated in partnerships."""
        total_rotations = 0
        total_opportunities = 0
        
        for partnership in partnerships:
            partnership_data = partnership['partnership_data']
            
            # Count strike rotations (odd runs scored)
            odd_runs = partnership_data[partnership_data['runs_scored'] % 2 == 1]
            rotations = len(odd_runs)
            
            # Opportunities are non-boundary balls
            opportunities = len(partnership_data[partnership_data['runs_scored'] < 4])
            
            total_rotations += rotations
            total_opportunities += opportunities
        
        return total_rotations / total_opportunities if total_opportunities > 0 else 0.0
    
    def _calculate_non_dismissal_correlation(self, partnerships: List[Dict[str, Any]]) -> float:
        """Calculate how often both batters survive partnerships."""
        if not partnerships:
            return 0.0
        
        survived_partnerships = sum(1 for p in partnerships if not p['ended_by_dismissal'])
        return survived_partnerships / len(partnerships)
    
    def _calculate_boundary_synergy(self, partnerships: List[Dict[str, Any]]) -> float:
        """Calculate combined boundary scoring effectiveness."""
        total_boundaries = sum(p['boundaries'] for p in partnerships)
        total_balls = sum(p['balls'] for p in partnerships)
        
        return total_boundaries / total_balls if total_balls > 0 else 0.0
    
    def _calculate_pressure_performance(
        self,
        partnerships: List[Dict[str, Any]],
        match_data: pd.DataFrame
    ) -> float:
        """Calculate performance under pressure situations."""
        pressure_partnerships = []
        
        for partnership in partnerships:
            match_id = partnership['match_id']
            innings = partnership['innings']
            
            # Determine if this was a pressure situation
            match_balls = match_data[
                (match_data['match_id'] == match_id) & 
                (match_data['innings'] == innings)
            ]
            
            # Pressure indicators: death overs, low run rate, recent wickets
            partnership_data = partnership['partnership_data']
            avg_over = partnership_data['over'].mean() if 'over' in partnership_data.columns else 10.0
            
            # Check for wickets if column exists
            wicket_count = 0
            if 'wicket_type' in match_balls.columns:
                wicket_count = len(match_balls[match_balls['wicket_type'].notna()])
            
            is_pressure = (
                avg_over >= 15 or  # Death overs
                partnership['runs'] / partnership['balls'] * 6 < 6.0 or  # Low run rate
                wicket_count >= 3  # Multiple wickets
            )
            
            if is_pressure:
                pressure_partnerships.append(partnership)
        
        if not pressure_partnerships:
            return 0.5  # Neutral score if no pressure situations
        
        # Calculate performance in pressure vs normal situations
        pressure_run_rate = sum(p['runs'] for p in pressure_partnerships) / sum(p['balls'] for p in pressure_partnerships) * 6
        normal_partnerships = [p for p in partnerships if p not in pressure_partnerships]
        
        if normal_partnerships:
            normal_run_rate = sum(p['runs'] for p in normal_partnerships) / sum(p['balls'] for p in normal_partnerships) * 6
            return min(1.0, pressure_run_rate / normal_run_rate) if normal_run_rate > 0 else 0.5
        else:
            return min(1.0, pressure_run_rate / 6.0)  # Compare to average T20 run rate
    
    def _compute_batting_synergy_score(
        self,
        run_rate: float,
        strike_rotation_rate: float,
        non_dismissal_correlation: float,
        boundary_synergy: float,
        pressure_performance: float,
        partnerships_count: int
    ) -> float:
        """Compute overall batting synergy score."""
        # Normalize components to 0-1 scale
        run_rate_score = min(1.0, run_rate / 10.0)  # Normalize against 10 RPO
        rotation_score = strike_rotation_rate
        survival_score = non_dismissal_correlation
        boundary_score = min(1.0, boundary_synergy * 10)  # Scale boundary rate
        pressure_score = pressure_performance
        
        # Weight the components
        synergy_score = (
            0.25 * run_rate_score +
            0.20 * rotation_score +
            0.20 * survival_score +
            0.20 * boundary_score +
            0.15 * pressure_score
        )
        
        # Apply partnership count bonus
        count_bonus = min(0.1, partnerships_count / 50)  # Up to 10% bonus
        
        return min(1.0, synergy_score + count_bonus)
    
    def _calculate_catch_success_rate(self, bowler_fielder_data: pd.DataFrame) -> float:
        """Calculate catch success rate for bowler-fielder combination."""
        catch_opportunities = bowler_fielder_data[
            bowler_fielder_data['wicket_type'].isin(['caught', 'caught and bowled'])
        ]
        
        if len(catch_opportunities) == 0:
            return 0.0
        
        successful_catches = len(catch_opportunities[catch_opportunities['wicket_type'] == 'caught'])
        return successful_catches / len(catch_opportunities)
    
    def _calculate_run_saving_efficiency(
        self,
        bowler_fielder_data: pd.DataFrame,
        bowler_data: pd.DataFrame
    ) -> float:
        """Calculate run saving efficiency when fielder is involved."""
        if len(bowler_fielder_data) == 0 or len(bowler_data) == 0:
            return 0.0
        
        # Compare runs per ball when fielder is involved vs overall
        runs_with_fielder = bowler_fielder_data['runs_scored'].mean()
        runs_overall = bowler_data['runs_scored'].mean()
        
        if runs_overall == 0:
            return 0.0
        
        # Higher efficiency means fewer runs conceded
        efficiency = max(0.0, 1.0 - (runs_with_fielder / runs_overall))
        return min(1.0, efficiency)
    
    def _count_pressure_wickets(self, bowler_fielder_data: pd.DataFrame) -> int:
        """Count wickets taken in pressure situations."""
        wickets = bowler_fielder_data[bowler_fielder_data['wicket_type'].notna()]
        
        # Pressure situations: death overs, powerplay, or high-scoring overs
        pressure_wickets = wickets[
            (wickets['over'] >= 16) |  # Death overs
            (wickets['over'] <= 6) |   # Powerplay
            (wickets.groupby('over')['runs_scored'].transform('sum') >= 12)  # High-scoring overs
        ]
        
        return len(pressure_wickets)
    
    def _compute_bowling_fielding_synergy_score(
        self,
        wicket_rate: float,
        catch_success_rate: float,
        run_saving_efficiency: float,
        shared_dismissals: int,
        pressure_wickets: int
    ) -> float:
        """Compute bowling-fielding synergy score."""
        # Normalize components
        wicket_score = min(1.0, wicket_rate / 2.0)  # Normalize against 2 wickets per over
        catch_score = catch_success_rate
        efficiency_score = run_saving_efficiency
        dismissal_score = min(1.0, shared_dismissals / 10.0)  # Normalize against 10 dismissals
        pressure_score = min(1.0, pressure_wickets / 5.0)  # Normalize against 5 pressure wickets
        
        # Weight the components
        synergy_score = (
            0.30 * wicket_score +
            0.25 * catch_score +
            0.20 * efficiency_score +
            0.15 * dismissal_score +
            0.10 * pressure_score
        )
        
        return synergy_score
    
    def _calculate_field_setting_effectiveness(self, captain_bowler_data: pd.DataFrame) -> float:
        """Calculate effectiveness of field settings."""
        if len(captain_bowler_data) == 0:
            return 0.0
        
        # Proxy: fewer boundaries and more wickets indicate good field settings
        boundaries = len(captain_bowler_data[captain_bowler_data['runs_scored'] >= 4])
        wickets = len(captain_bowler_data[captain_bowler_data['wicket_type'].notna()])
        total_balls = len(captain_bowler_data)
        
        boundary_prevention = 1.0 - (boundaries / total_balls) if total_balls > 0 else 0.0
        wicket_creation = wickets / total_balls if total_balls > 0 else 0.0
        
        effectiveness = 0.6 * boundary_prevention + 0.4 * min(1.0, wicket_creation * 10)  # Scale and cap wicket rate
        return min(1.0, effectiveness)  # Ensure [0,1] range
    
    def _calculate_pressure_situation_success(self, captain_bowler_data: pd.DataFrame) -> float:
        """Calculate success in pressure situations under captain."""
        if len(captain_bowler_data) == 0:
            return 0.0
        
        # Pressure situations: death overs, powerplay
        pressure_balls = captain_bowler_data[
            (captain_bowler_data['over'] >= 16) | (captain_bowler_data['over'] <= 6)
        ]
        
        if len(pressure_balls) == 0:
            return 0.5  # Neutral if no pressure situations
        
        # Success metrics in pressure
        pressure_wickets = len(pressure_balls[pressure_balls['wicket_type'].notna()])
        pressure_runs = pressure_balls['runs_scored'].sum()
        pressure_overs = len(pressure_balls) / 6.0
        
        wicket_success = min(1.0, pressure_wickets / pressure_overs) if pressure_overs > 0 else 0.0
        economy_success = max(0.0, 1.0 - (pressure_runs / pressure_overs / 8.0)) if pressure_overs > 0 else 0.0  # Against 8 RPO
        
        return min(1.0, 0.6 * wicket_success + 0.4 * economy_success)
    
    def _calculate_tactical_adaptability(
        self,
        captain_bowler_data: pd.DataFrame,
        other_captain_data: pd.DataFrame
    ) -> float:
        """Calculate tactical adaptability under different captains."""
        if len(captain_bowler_data) == 0:
            return 0.0
        
        # Compare performance under this captain vs others
        current_economy = (captain_bowler_data['runs_scored'].sum() / len(captain_bowler_data) * 6) if len(captain_bowler_data) > 0 else 0.0
        current_wicket_rate = len(captain_bowler_data[captain_bowler_data['wicket_type'].notna()]) / len(captain_bowler_data) * 6
        
        if len(other_captain_data) > 0:
            other_economy = (other_captain_data['runs_scored'].sum() / len(other_captain_data) * 6)
            other_wicket_rate = len(other_captain_data[other_captain_data['wicket_type'].notna()]) / len(other_captain_data) * 6
            
            # Better performance under this captain indicates good synergy
            economy_improvement = max(0.0, (other_economy - current_economy) / other_economy) if other_economy > 0 else 0.0
            wicket_improvement = (current_wicket_rate - other_wicket_rate) / max(0.1, other_wicket_rate) if other_wicket_rate > 0 else 0.0
            
            return min(1.0, (economy_improvement + max(0.0, wicket_improvement)) / 2.0)
        else:
            # No comparison data, use absolute performance
            economy_score = max(0.0, 1.0 - current_economy / 8.0)  # Against 8 RPO
            wicket_score = min(1.0, current_wicket_rate / 2.0)  # Against 2 wickets per over
            
            return (economy_score + wicket_score) / 2.0
    
    def _compute_captain_bowler_synergy_score(
        self,
        wickets: int,
        economy: float,
        field_effectiveness: float,
        pressure_success: float,
        adaptability: float,
        overs: float
    ) -> float:
        """Compute captain-bowler synergy score."""
        # Normalize components
        wicket_score = min(1.0, (wickets / overs) / 2.0) if overs > 0 else 0.0  # Against 2 wickets per over
        economy_score = max(0.0, 1.0 - economy / 8.0)  # Against 8 RPO
        field_score = field_effectiveness
        pressure_score = pressure_success
        adapt_score = adaptability
        
        # Weight the components
        synergy_score = (
            0.25 * wicket_score +
            0.25 * economy_score +
            0.20 * field_score +
            0.15 * pressure_score +
            0.15 * adapt_score
        )
        
        return synergy_score


class SynergyGraphBuilder:
    """Builds has_synergy_with edges in the knowledge graph."""
    
    def __init__(self, config: SynergyConfig = None):
        self.config = config or SynergyConfig()
        self.calculator = SynergyScoreCalculator(config)
    
    def add_synergy_edges(
        self,
        graph: nx.Graph,
        match_data: pd.DataFrame,
        player_roles: Optional[Dict[str, str]] = None
    ) -> nx.Graph:
        """
        Add has_synergy_with edges to the knowledge graph.
        
        Args:
            graph: NetworkX graph with cricket data
            match_data: Ball-by-ball match data
            player_roles: Optional player role information
        
        Returns:
            Updated graph with synergy edges
        """
        logger.info("Computing player synergies and adding has_synergy_with edges")
        
        # Get all players from the graph
        players = [node for node, data in graph.nodes(data=True) 
                  if data.get('node_type') == 'player']
        
        if len(players) < 2:
            logger.warning("Not enough players in graph for synergy analysis")
            return graph
        
        # Compute different types of synergies
        synergy_edges = []
        
        # 1. Batting synergies
        batting_synergies = self._compute_batting_synergies(players, match_data)
        synergy_edges.extend(batting_synergies)
        
        # 2. Bowling-Fielding synergies
        bowling_fielding_synergies = self._compute_bowling_fielding_synergies(players, match_data)
        synergy_edges.extend(bowling_fielding_synergies)
        
        # 3. Captain-Bowler synergies
        captain_bowler_synergies = self._compute_captain_bowler_synergies(players, match_data)
        synergy_edges.extend(captain_bowler_synergies)
        
        # Filter and rank synergies
        filtered_synergies = self._filter_and_rank_synergies(synergy_edges)
        
        # Add edges to graph
        edges_added = self._add_synergy_edges_to_graph(graph, filtered_synergies)
        
        logger.info(f"Added {edges_added} has_synergy_with edges to graph")
        return graph
    
    def _compute_batting_synergies(
        self,
        players: List[str],
        match_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Compute batting synergies between player pairs."""
        synergies = []
        
        # Check if required columns exist
        required_cols = ['batter', 'non_striker']
        if len(match_data) == 0 or not all(col in match_data.columns for col in required_cols):
            return synergies
        
        # Get all potential batting pairs
        batting_pairs = []
        for player1, player2 in combinations(players, 2):
            # Check if they've batted together
            partnership_data = match_data[
                ((match_data['batter'] == player1) & (match_data['non_striker'] == player2)) |
                ((match_data['batter'] == player2) & (match_data['non_striker'] == player1))
            ]
            
            if len(partnership_data) >= self.config.min_batting_partnerships:
                batting_pairs.append((player1, player2))
        
        logger.info(f"Found {len(batting_pairs)} potential batting partnerships")
        
        # Calculate synergy for each pair
        for player1, player2 in batting_pairs:
            synergy_score, metrics = self.calculator.calculate_batting_synergy(
                player1, player2, match_data
            )
            
            if synergy_score >= self.config.batting_synergy_threshold:
                synergies.append({
                    'player1': player1,
                    'player2': player2,
                    'synergy_type': 'batting',
                    'synergy_score': synergy_score,
                    'weight': synergy_score * self.config.batting_weight,
                    'metrics': metrics,
                    'edge_attributes': {
                        'edge_type': 'has_synergy_with',
                        'synergy_type': 'batting_partnership',
                        'synergy_score': synergy_score,
                        'weight': synergy_score * self.config.batting_weight,
                        'partnerships_count': metrics.partnerships_count,
                        'average_partnership': metrics.average_partnership,
                        'run_rate': metrics.run_rate,
                        'strike_rotation_rate': metrics.strike_rotation_rate,
                        'non_dismissal_correlation': metrics.non_dismissal_correlation,
                        'pressure_performance': metrics.pressure_performance
                    }
                })
        
        return synergies
    
    def _compute_bowling_fielding_synergies(
        self,
        players: List[str],
        match_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Compute bowling-fielding synergies."""
        synergies = []
        
        # Check if required columns exist
        required_cols = ['bowler', 'fielder']
        if len(match_data) == 0 or not all(col in match_data.columns for col in required_cols):
            return synergies
        
        # Get bowlers and fielders
        bowlers = set(match_data['bowler'].dropna().unique())
        fielders = set(match_data['fielder'].dropna().unique())
        
        # Find bowler-fielder combinations
        for bowler in bowlers:
            if bowler not in players:
                continue
                
            for fielder in fielders:
                if fielder not in players or fielder == bowler:
                    continue
                
                # Check if they've worked together
                combination_data = match_data[
                    (match_data['bowler'] == bowler) & 
                    (match_data['fielder'] == fielder)
                ]
                
                if len(combination_data) >= self.config.min_fielding_dismissals:
                    synergy_score, metrics = self.calculator.calculate_bowling_fielding_synergy(
                        bowler, fielder, match_data
                    )
                    
                    if synergy_score >= self.config.fielding_synergy_threshold:
                        synergies.append({
                            'player1': bowler,
                            'player2': fielder,
                            'synergy_type': 'bowling_fielding',
                            'synergy_score': synergy_score,
                            'weight': synergy_score * self.config.bowling_weight,
                            'metrics': metrics,
                            'edge_attributes': {
                                'edge_type': 'has_synergy_with',
                                'synergy_type': 'bowling_fielding',
                                'synergy_score': synergy_score,
                                'weight': synergy_score * self.config.bowling_weight,
                                'shared_dismissals': metrics.shared_dismissals,
                                'wicket_rate': metrics.wicket_rate,
                                'catch_success_rate': metrics.catch_success_rate,
                                'run_saving_efficiency': metrics.run_saving_efficiency,
                                'pressure_wickets': metrics.pressure_wickets
                            }
                        })
        
        return synergies
    
    def _compute_captain_bowler_synergies(
        self,
        players: List[str],
        match_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Compute captain-bowler synergies."""
        synergies = []
        
        # Check if required columns exist
        required_cols = ['captain', 'bowler']
        if len(match_data) == 0 or not all(col in match_data.columns for col in required_cols):
            return synergies
        
        # Get captains and bowlers
        captains = set(match_data['captain'].dropna().unique())
        bowlers = set(match_data['bowler'].dropna().unique())
        
        # Find captain-bowler combinations
        for captain in captains:
            if captain not in players:
                continue
                
            for bowler in bowlers:
                if bowler not in players or bowler == captain:
                    continue
                
                # Check if bowler has bowled under this captain
                combination_data = match_data[
                    (match_data['captain'] == captain) & 
                    (match_data['bowler'] == bowler)
                ]
                
                if len(combination_data) >= self.config.min_captain_overs:
                    synergy_score, metrics = self.calculator.calculate_captain_bowler_synergy(
                        captain, bowler, match_data
                    )
                    
                    if synergy_score >= self.config.captain_synergy_threshold:
                        synergies.append({
                            'player1': captain,
                            'player2': bowler,
                            'synergy_type': 'captain_bowler',
                            'synergy_score': synergy_score,
                            'weight': synergy_score * self.config.captain_weight,
                            'metrics': metrics,
                            'edge_attributes': {
                                'edge_type': 'has_synergy_with',
                                'synergy_type': 'captain_bowler',
                                'synergy_score': synergy_score,
                                'weight': synergy_score * self.config.captain_weight,
                                'overs_under_captain': metrics.overs_under_captain,
                                'wickets_under_captain': metrics.wickets_under_captain,
                                'economy_under_captain': metrics.economy_under_captain,
                                'field_setting_effectiveness': metrics.field_setting_effectiveness,
                                'pressure_situation_success': metrics.pressure_situation_success,
                                'tactical_adaptability': metrics.tactical_adaptability
                            }
                        })
        
        return synergies
    
    def _filter_and_rank_synergies(
        self,
        synergy_edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter and rank synergies to avoid too many edges per player."""
        # Group by player
        player_synergies = defaultdict(list)
        
        for synergy in synergy_edges:
            player1 = synergy['player1']
            player2 = synergy['player2']
            
            player_synergies[player1].append(synergy)
            player_synergies[player2].append(synergy)
        
        # Convert back to list of synergy dictionaries, avoiding duplicates
        unique_synergies = []
        seen_combinations = set()
        
        for synergy in synergy_edges:
            synergy_key = tuple(sorted([synergy['player1'], synergy['player2']]))
            combination_key = (synergy_key, synergy['synergy_type'])
            
            if combination_key not in seen_combinations:
                unique_synergies.append(synergy)
                seen_combinations.add(combination_key)
        
        # Sort by weight and apply per-player limits
        unique_synergies.sort(key=lambda x: x['weight'], reverse=True)
        
        # Apply per-player edge limits
        player_edge_counts = defaultdict(int)
        filtered_synergies = []
        
        for synergy in unique_synergies:
            player1 = synergy['player1']
            player2 = synergy['player2']
            
            # Check if adding this edge would exceed limits
            if (player_edge_counts[player1] < self.config.max_synergy_edges_per_player and
                player_edge_counts[player2] < self.config.max_synergy_edges_per_player):
                filtered_synergies.append(synergy)
                player_edge_counts[player1] += 1
                player_edge_counts[player2] += 1
        
        return filtered_synergies
    
    def _add_synergy_edges_to_graph(
        self,
        graph: nx.Graph,
        synergies: List[Dict[str, Any]]
    ) -> int:
        """Add synergy edges to the graph."""
        edges_added = 0
        
        for synergy in synergies:
            player1 = synergy['player1']
            player2 = synergy['player2']
            
            # Check if both players exist in graph
            if player1 not in graph.nodes or player2 not in graph.nodes:
                continue
            
            # Check if edge already exists
            if graph.has_edge(player1, player2):
                # Update existing edge with synergy information
                existing_attrs = graph[player1][player2]
                
                # Add synergy attributes
                for key, value in synergy['edge_attributes'].items():
                    existing_attrs[f"synergy_{key}"] = value
            else:
                # Add new edge
                graph.add_edge(player1, player2, **synergy['edge_attributes'])
                edges_added += 1
        
        return edges_added


def add_synergy_edges_to_graph(
    graph: nx.Graph,
    match_data: pd.DataFrame,
    config: SynergyConfig = None,
    player_roles: Optional[Dict[str, str]] = None
) -> nx.Graph:
    """
    Main function to add synergy edges to cricket knowledge graph.
    
    Args:
        graph: NetworkX graph with cricket data
        match_data: Ball-by-ball match data
        config: Synergy analysis configuration
        player_roles: Optional player role information
    
    Returns:
        Updated graph with has_synergy_with edges
    """
    builder = SynergyGraphBuilder(config)
    return builder.add_synergy_edges(graph, match_data, player_roles)
# Purpose: Advanced Query Engine for Unified Cricket Knowledge Graph
# Author: WicketWise Team, Last Modified: 2025-08-17

import networkx as nx
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@contextmanager
def timeout(seconds: int):
    """Context manager for query timeouts"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Query timed out")
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)


class QueryTimeoutError(Exception):
    pass


class UnifiedKGQueryEngine:
    """
    Advanced query engine for the unified cricket knowledge graph
    
    Supports:
    - Complete player profiles (batting + bowling + fielding)
    - Situational analysis (vs spinners, death overs, venues)
    - Advanced cricket analytics
    - Ball-by-ball granular queries
    - Performance comparisons with context
    """
    
    def __init__(self, graph_path: str = "models/unified_cricket_kg.pkl"):
        self.graph_path = Path(graph_path)
        self.graph: Optional[nx.Graph] = None
        self.max_results = 1000
        self.query_timeout = 15  # seconds
        
        # Load the graph
        self._load_graph()
        
        # Cache frequently used data
        self._cache_graph_metadata()
    
    def _load_graph(self) -> None:
        """Load the unified knowledge graph from pickle file"""
        try:
            if not self.graph_path.exists():
                logger.error(f"Unified knowledge graph not found at {self.graph_path}")
                self.graph = nx.Graph()  # Empty graph as fallback
                return
                
            with open(self.graph_path, 'rb') as f:
                self.graph = pickle.load(f)
                
            logger.info(f"Loaded unified knowledge graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Failed to load unified knowledge graph: {e}")
            self.graph = nx.Graph()  # Empty graph as fallback
    
    def _cache_graph_metadata(self) -> None:
        """Cache frequently accessed graph metadata"""
        if not self.graph:
            self.metadata = {}
            return
            
        node_types = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            node_types[data.get('type', 'unknown')] += 1
        
        self.metadata = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': dict(node_types),
            'graph_info': self.graph.graph
        }
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the unified knowledge graph"""
        return self.metadata.copy()
    
    def find_player_node(self, player_name: str) -> Optional[str]:
        """Find a player node by name with intelligent matching"""
        if not self.graph:
            return None
            
        player_name_lower = player_name.lower()
        
        # Common name mappings for cricket players
        name_mappings = {
            'virat kohli': ['V Kohli', 'Virat Kohli', 'Kohli'],
            'ms dhoni': ['M Dhoni', 'MS Dhoni', 'Dhoni', 'Mahendra Singh Dhoni'],
            'rohit sharma': ['R Sharma', 'Rohit Sharma', 'Sharma'],
            'sachin tendulkar': ['S Tendulkar', 'Sachin Tendulkar', 'Tendulkar'],
            'kapil dev': ['K Dev', 'Kapil Dev'],
            'rahul dravid': ['R Dravid', 'Rahul Dravid', 'Dravid'],
            'sourav ganguly': ['S Ganguly', 'Sourav Ganguly', 'Ganguly'],
            'anil kumble': ['A Kumble', 'Anil Kumble', 'Kumble']
        }
        
        # Check mapped names first
        if player_name_lower in name_mappings:
            for variant in name_mappings[player_name_lower]:
                if variant in self.graph.nodes:
                    return variant
        
        # Try exact match
        for node, data in self.graph.nodes(data=True):
            if (data.get('type') == 'player' and 
                str(node).lower() == player_name_lower):
                return node
        
        # Try partial match (last name)
        if ' ' in player_name_lower:
            last_name = player_name_lower.split()[-1]
            for node, data in self.graph.nodes(data=True):
                if (data.get('type') == 'player' and 
                    last_name in str(node).lower()):
                    return node
        
        # Try partial match (any part)
        for node, data in self.graph.nodes(data=True):
            if (data.get('type') == 'player' and 
                player_name_lower in str(node).lower()):
                return node
        
        return None
    
    def get_complete_player_profile(self, player: str) -> Dict[str, Any]:
        """
        Get complete player profile including batting, bowling, and situational stats
        
        Args:
            player: Player name
            
        Returns:
            Complete player profile with all statistics
        """
        try:
            with timeout(self.query_timeout):
                player_node = self.find_player_node(player)
                if not player_node:
                    return {"error": f"Player '{player}' not found in knowledge graph"}
                
                # Get player data
                player_data = self.graph.nodes[player_node]
                
                # Build comprehensive profile
                profile = {
                    "player": str(player_node),
                    "player_id": player_data.get('player_id', player_node.lower().replace(' ', '_')),
                    "primary_role": player_data.get('primary_role', 'unknown'),
                    "hand": player_data.get('hand', 'right'),
                    "bowling_style": player_data.get('bowling_style'),
                    "matches_played": player_data.get('matches_played', 0),
                    "teams": player_data.get('teams', []),
                    
                    # Core statistics
                    "batting_stats": player_data.get('batting_stats', {}),
                    "bowling_stats": player_data.get('bowling_stats', {}),
                    
                    # Situational analysis
                    "vs_pace": player_data.get('vs_pace', {}),
                    "vs_spin": player_data.get('vs_spin', {}),
                    "in_powerplay": player_data.get('in_powerplay', {}),
                    "in_death_overs": player_data.get('in_death_overs', {}),
                    
                    # Venue performance
                    "venues_played": self._get_player_venues(player_node),
                    "venue_performance": self._get_venue_performance(player_node),
                    
                    # Advanced insights
                    "strengths": self._analyze_player_strengths(player_data),
                    "style_analysis": self._analyze_playing_style(player_data)
                }
                
                return profile
                
        except QueryTimeoutError:
            return {"error": "Query timed out - please try a more specific search"}
        except Exception as e:
            logger.error(f"Error getting player profile: {e}")
            return {"error": f"Failed to get player profile: {str(e)}"}
    
    def compare_players_advanced(self, player1: str, player2: str, 
                               context: Optional[str] = None) -> Dict[str, Any]:
        """
        Advanced player comparison with situational context
        
        Args:
            player1: First player name
            player2: Second player name
            context: Optional context (vs_spin, death_overs, powerplay, etc.)
            
        Returns:
            Detailed comparison with insights
        """
        try:
            with timeout(self.query_timeout):
                # Get both player profiles
                profile1 = self.get_complete_player_profile(player1)
                profile2 = self.get_complete_player_profile(player2)
                
                if "error" in profile1:
                    return profile1
                if "error" in profile2:
                    return profile2
                
                # Extract comparison data based on context
                if context == "vs_spin":
                    stats1 = profile1.get("vs_spin", {})
                    stats2 = profile2.get("vs_spin", {})
                    context_label = "Against Spinners"
                elif context == "death_overs":
                    stats1 = profile1.get("in_death_overs", {})
                    stats2 = profile2.get("in_death_overs", {})
                    context_label = "In Death Overs"
                elif context == "powerplay":
                    stats1 = profile1.get("in_powerplay", {})
                    stats2 = profile2.get("in_powerplay", {})
                    context_label = "In Powerplay"
                else:
                    stats1 = profile1.get("batting_stats", {})
                    stats2 = profile2.get("batting_stats", {})
                    context_label = "Overall Career"
                
                comparison = {
                    "player1": {
                        "name": profile1["player"],
                        "role": profile1["primary_role"],
                        "stats": stats1
                    },
                    "player2": {
                        "name": profile2["player"],
                        "role": profile2["primary_role"],
                        "stats": stats2
                    },
                    "context": context_label,
                    "comparison_metrics": self._generate_comparison_metrics(stats1, stats2),
                    "head_to_head_summary": self._generate_comparison_summary(profile1, profile2, stats1, stats2)
                }
                
                return comparison
                
        except QueryTimeoutError:
            return {"error": "Query timed out - please try a more specific search"}
        except Exception as e:
            logger.error(f"Error comparing players: {e}")
            return {"error": f"Failed to compare players: {str(e)}"}
    
    def get_situational_analysis(self, player: str, situation: str, 
                               venue: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed situational analysis for a player
        
        Args:
            player: Player name
            situation: Situation type (vs_spin, vs_pace, death_overs, powerplay)
            venue: Optional venue filter
            
        Returns:
            Situational analysis with insights
        """
        try:
            with timeout(self.query_timeout):
                profile = self.get_complete_player_profile(player)
                if "error" in profile:
                    return profile
                
                # Get situational stats
                if situation == "vs_spin":
                    stats = profile.get("vs_spin", {})
                    context = "Against Spin Bowling"
                elif situation == "vs_pace":
                    stats = profile.get("vs_pace", {})
                    context = "Against Pace Bowling"
                elif situation == "death_overs":
                    stats = profile.get("in_death_overs", {})
                    context = "In Death Overs (16-20)"
                elif situation == "powerplay":
                    stats = profile.get("in_powerplay", {})
                    context = "In Powerplay (1-6)"
                else:
                    return {"error": f"Unknown situation: {situation}"}
                
                # Venue-specific analysis if requested
                venue_stats = {}
                if venue:
                    venue_performance = profile.get("venue_performance", {})
                    venue_stats = venue_performance.get(venue, {})
                
                analysis = {
                    "player": profile["player"],
                    "situation": context,
                    "venue": venue,
                    "stats": stats,
                    "venue_stats": venue_stats if venue else None,
                    "analysis": self._generate_situational_insights(stats, situation),
                    "comparison_to_overall": self._compare_to_overall_stats(stats, profile.get("batting_stats", {}))
                }
                
                return analysis
                
        except QueryTimeoutError:
            return {"error": "Query timed out - please try a more specific search"}
        except Exception as e:
            logger.error(f"Error in situational analysis: {e}")
            return {"error": f"Failed to perform situational analysis: {str(e)}"}
    
    def find_best_performers(self, context: str, min_balls: int = 100, 
                           limit: int = 10) -> Dict[str, Any]:
        """
        Find best performers in specific contexts
        
        Args:
            context: Performance context (vs_spin, death_overs, powerplay, etc.)
            min_balls: Minimum balls faced for qualification
            limit: Maximum number of results
            
        Returns:
            List of top performers with statistics
        """
        try:
            with timeout(self.query_timeout):
                performers = []
                
                for node, data in self.graph.nodes(data=True):
                    if data.get('type') != 'player':
                        continue
                    
                    # Get contextual stats
                    if context == "vs_spin":
                        stats = data.get('vs_spin', {})
                    elif context == "vs_pace":
                        stats = data.get('vs_pace', {})
                    elif context == "death_overs":
                        stats = data.get('in_death_overs', {})
                    elif context == "powerplay":
                        stats = data.get('in_powerplay', {})
                    else:
                        stats = data.get('batting_stats', {})
                    
                    if not stats or stats.get('balls', 0) < min_balls:
                        continue
                    
                    # Calculate performance metric (weighted average and strike rate)
                    average = stats.get('average', 0)
                    strike_rate = stats.get('strike_rate', 0)
                    balls = stats.get('balls', 0)
                    
                    # Performance score (weighted combination)
                    performance_score = (average * 0.6) + (strike_rate * 0.4)
                    
                    performers.append({
                        "player": str(node),
                        "primary_role": data.get('primary_role', 'unknown'),
                        "stats": stats,
                        "performance_score": performance_score
                    })
                
                # Sort by performance score
                performers.sort(key=lambda x: x['performance_score'], reverse=True)
                
                return {
                    "context": context,
                    "min_qualification": f"{min_balls} balls",
                    "total_qualified": len(performers),
                    "top_performers": performers[:limit]
                }
                
        except QueryTimeoutError:
            return {"error": "Query timed out - please try a more specific search"}
        except Exception as e:
            logger.error(f"Error finding best performers: {e}")
            return {"error": f"Failed to find best performers: {str(e)}"}
    
    def _get_player_venues(self, player_node: str) -> List[str]:
        """Get list of venues where player has performed"""
        venues = []
        for neighbor in self.graph.neighbors(player_node):
            neighbor_data = self.graph.nodes[neighbor]
            if neighbor_data.get('type') == 'venue':
                venues.append(str(neighbor))
        return venues[:10]  # Limit to top 10
    
    def _get_venue_performance(self, player_node: str) -> Dict[str, Dict]:
        """Get venue-specific performance statistics"""
        venue_performance = {}
        for neighbor in self.graph.neighbors(player_node):
            neighbor_data = self.graph.nodes[neighbor]
            if neighbor_data.get('type') == 'venue':
                edge_data = self.graph[player_node][neighbor]
                venue_performance[str(neighbor)] = {
                    "balls": edge_data.get('balls', 0),
                    "runs": edge_data.get('runs', 0),
                    "average": edge_data.get('runs', 0) / max(edge_data.get('dismissals', 1), 1),
                    "strike_rate": edge_data.get('runs', 0) / max(edge_data.get('balls', 1), 1) * 100
                }
        return venue_performance
    
    def _analyze_player_strengths(self, player_data: Dict) -> List[str]:
        """Analyze player's key strengths"""
        strengths = []
        
        batting_stats = player_data.get('batting_stats', {})
        vs_pace = player_data.get('vs_pace', {})
        vs_spin = player_data.get('vs_spin', {})
        powerplay = player_data.get('in_powerplay', {})
        death_overs = player_data.get('in_death_overs', {})
        
        # Analyze strike rates
        if batting_stats.get('strike_rate', 0) > 130:
            strengths.append("Aggressive batsman with high strike rate")
        elif batting_stats.get('strike_rate', 0) > 110:
            strengths.append("Good strike rate in T20 format")
        
        # Analyze vs bowling types
        pace_sr = vs_pace.get('strike_rate', 0)
        spin_sr = vs_spin.get('strike_rate', 0)
        
        if pace_sr > spin_sr + 20:
            strengths.append("Strong against pace bowling")
        elif spin_sr > pace_sr + 20:
            strengths.append("Excellent player of spin")
        
        # Analyze situational performance
        if death_overs.get('strike_rate', 0) > 140:
            strengths.append("Outstanding death overs finisher")
        
        if powerplay.get('strike_rate', 0) > 130:
            strengths.append("Explosive powerplay batsman")
        
        return strengths[:5]  # Top 5 strengths
    
    def _analyze_playing_style(self, player_data: Dict) -> Dict[str, str]:
        """Analyze player's playing style"""
        batting_stats = player_data.get('batting_stats', {})
        
        style = {}
        
        # Aggression level
        strike_rate = batting_stats.get('strike_rate', 0)
        if strike_rate > 140:
            style['aggression'] = "Highly aggressive"
        elif strike_rate > 120:
            style['aggression'] = "Moderately aggressive"
        else:
            style['aggression'] = "Conservative"
        
        # Consistency
        average = batting_stats.get('average', 0)
        if average > 40:
            style['consistency'] = "Highly consistent"
        elif average > 25:
            style['consistency'] = "Reasonably consistent"
        else:
            style['consistency'] = "Inconsistent"
        
        # Boundary hitting
        boundaries = batting_stats.get('boundaries', 0)
        balls = batting_stats.get('balls', 1)
        boundary_rate = boundaries / balls * 100 if balls > 0 else 0
        
        if boundary_rate > 15:
            style['boundary_hitting'] = "Frequent boundary hitter"
        elif boundary_rate > 10:
            style['boundary_hitting'] = "Good boundary hitter"
        else:
            style['boundary_hitting'] = "Selective boundary hitter"
        
        return style
    
    def _generate_comparison_metrics(self, stats1: Dict, stats2: Dict) -> Dict[str, Any]:
        """Generate comparison metrics between two stat sets"""
        metrics = {}
        
        for metric in ['runs', 'balls', 'average', 'strike_rate', 'boundaries', 'sixes']:
            val1 = stats1.get(metric, 0)
            val2 = stats2.get(metric, 0)
            
            if val1 > val2:
                winner = "player1"
                difference = val1 - val2
                percentage = (difference / val2 * 100) if val2 > 0 else float('inf')
            elif val2 > val1:
                winner = "player2"
                difference = val2 - val1
                percentage = (difference / val1 * 100) if val1 > 0 else float('inf')
            else:
                winner = "tie"
                difference = 0
                percentage = 0
            
            metrics[metric] = {
                "player1_value": val1,
                "player2_value": val2,
                "winner": winner,
                "difference": difference,
                "percentage_difference": percentage
            }
        
        return metrics
    
    def _generate_comparison_summary(self, profile1: Dict, profile2: Dict, 
                                   stats1: Dict, stats2: Dict) -> str:
        """Generate a natural language comparison summary"""
        p1_name = profile1["player"]
        p2_name = profile2["player"]
        
        # Compare key metrics
        avg1 = stats1.get('average', 0)
        avg2 = stats2.get('average', 0)
        sr1 = stats1.get('strike_rate', 0)
        sr2 = stats2.get('strike_rate', 0)
        
        summary_parts = []
        
        if avg1 > avg2:
            summary_parts.append(f"{p1_name} has a higher average ({avg1:.1f} vs {avg2:.1f})")
        elif avg2 > avg1:
            summary_parts.append(f"{p2_name} has a higher average ({avg2:.1f} vs {avg1:.1f})")
        
        if sr1 > sr2:
            summary_parts.append(f"{p1_name} has a better strike rate ({sr1:.1f} vs {sr2:.1f})")
        elif sr2 > sr1:
            summary_parts.append(f"{p2_name} has a better strike rate ({sr2:.1f} vs {sr1:.1f})")
        
        return ". ".join(summary_parts) if summary_parts else "Both players have similar performance metrics"
    
    def _generate_situational_insights(self, stats: Dict, situation: str) -> List[str]:
        """Generate insights for situational performance"""
        insights = []
        
        if not stats:
            return ["Insufficient data for analysis"]
        
        average = stats.get('average', 0)
        strike_rate = stats.get('strike_rate', 0)
        balls = stats.get('balls', 0)
        
        if balls < 50:
            insights.append(f"Limited sample size ({balls} balls) - analysis may not be conclusive")
        
        if situation == "vs_spin":
            if strike_rate > 120:
                insights.append("Strong player of spin bowling with good strike rate")
            elif strike_rate < 90:
                insights.append("Struggles against spin bowling - low strike rate")
        
        elif situation == "death_overs":
            if strike_rate > 140:
                insights.append("Excellent death overs batsman - can accelerate when needed")
            elif average > 30:
                insights.append("Reliable finisher with good average in pressure situations")
        
        elif situation == "powerplay":
            if strike_rate > 130:
                insights.append("Aggressive powerplay batsman - takes advantage of field restrictions")
        
        return insights
    
    def _compare_to_overall_stats(self, situational_stats: Dict, overall_stats: Dict) -> Dict[str, Any]:
        """Compare situational stats to overall career stats"""
        if not situational_stats or not overall_stats:
            return {}
        
        comparison = {}
        
        for metric in ['average', 'strike_rate']:
            situational_val = situational_stats.get(metric, 0)
            overall_val = overall_stats.get(metric, 0)
            
            if overall_val > 0:
                difference = ((situational_val - overall_val) / overall_val) * 100
                comparison[f"{metric}_difference"] = {
                    "situational": situational_val,
                    "overall": overall_val,
                    "percentage_change": difference,
                    "better_in_situation": difference > 0
                }
        
        return comparison


# Backwards compatibility wrapper
class KGQueryEngine(UnifiedKGQueryEngine):
    """Backwards compatibility wrapper for existing code"""
    
    def get_player_stats(self, player: str, format_filter: Optional[str] = None, 
                        venue_filter: Optional[str] = None) -> Dict[str, Any]:
        """Legacy method - returns complete profile"""
        return self.get_complete_player_profile(player)
    
    def compare_players(self, player1: str, player2: str) -> Dict[str, Any]:
        """Legacy method - uses advanced comparison"""
        return self.compare_players_advanced(player1, player2)

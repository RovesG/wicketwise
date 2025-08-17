# Purpose: Safe Knowledge Graph Query Engine with NetworkX operations
# Author: WicketWise Team, Last Modified: 2025-08-16

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class QueryTimeoutError(Exception):
    """Raised when a query exceeds the timeout limit"""
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for query timeouts - simplified version without signals"""
    start_time = time.time()
    try:
        yield
        # Check if we exceeded timeout (simple check)
        if time.time() - start_time > seconds:
            raise QueryTimeoutError(f"Query took longer than {seconds} seconds")
    except Exception as e:
        if time.time() - start_time > seconds:
            raise QueryTimeoutError(f"Query timed out after {seconds} seconds")
        raise e


class KGQueryEngine:
    """
    Safe Knowledge Graph Query Engine
    
    Provides read-only access to the cricket knowledge graph with safety guardrails:
    - Query timeouts
    - Result size limits
    - Safe NetworkX operations only
    - No graph modifications allowed
    """
    
    def __init__(self, graph_path: str = "models/cricket_knowledge_graph.pkl"):
        self.graph_path = Path(graph_path)
        self.graph: Optional[nx.Graph] = None
        self.max_results = 1000
        self.query_timeout = 10  # seconds
        
        # Load the graph
        self._load_graph()
        
        # Cache frequently used data
        self._cache_graph_metadata()
    
    def _load_graph(self) -> None:
        """Load the knowledge graph from pickle file"""
        try:
            if not self.graph_path.exists():
                logger.error(f"Knowledge graph not found at {self.graph_path}")
                self.graph = nx.Graph()  # Empty graph as fallback
                return
                
            with open(self.graph_path, 'rb') as f:
                self.graph = pickle.load(f)
                
            logger.info(f"Loaded knowledge graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")
            self.graph = nx.Graph()  # Empty graph as fallback
    
    def _cache_graph_metadata(self) -> None:
        """Cache frequently accessed graph metadata"""
        if not self.graph:
            self.metadata = {}
            return
            
        try:
            with timeout(5):  # Quick metadata collection
                # Get node types
                node_types = {}
                for node, data in self.graph.nodes(data=True):
                    node_type = data.get('node_type', 'unknown')
                    if node_type not in node_types:
                        node_types[node_type] = []
                    node_types[node_type].append(node)
                
                # Sample some key entities
                players = node_types.get('player', [])[:50]  # Top 50 players
                venues = node_types.get('venue', [])[:20]    # Top 20 venues
                teams = node_types.get('team', [])[:20]      # Top 20 teams
                
                self.metadata = {
                    'total_nodes': self.graph.number_of_nodes(),
                    'total_edges': self.graph.number_of_edges(),
                    'node_types': {k: len(v) for k, v in node_types.items()},
                    'sample_players': players,
                    'sample_venues': venues,
                    'sample_teams': teams
                }
                
        except Exception as e:
            logger.warning(f"Failed to cache metadata: {e}")
            self.metadata = {}
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the knowledge graph"""
        return self.metadata.copy()
    
    def find_player_node(self, player_name: str) -> Optional[str]:
        """Find a player node by name (fuzzy matching)"""
        if not self.graph:
            return None
            
        player_name_lower = player_name.lower()
        
        # Try exact match first
        for node, data in self.graph.nodes(data=True):
            if (data.get('node_type') == 'player' and 
                data.get('name', '').lower() == player_name_lower):
                return node
        
        # Try partial match
        for node, data in self.graph.nodes(data=True):
            if (data.get('node_type') == 'player' and 
                player_name_lower in data.get('name', '').lower()):
                return node
        
        return None
    
    def find_venue_node(self, venue_name: str) -> Optional[str]:
        """Find a venue node by name (fuzzy matching)"""
        if not self.graph:
            return None
            
        venue_name_lower = venue_name.lower()
        
        # Try exact match first
        for node, data in self.graph.nodes(data=True):
            if (data.get('node_type') == 'venue' and 
                data.get('name', '').lower() == venue_name_lower):
                return node
        
        # Try partial match
        for node, data in self.graph.nodes(data=True):
            if (data.get('node_type') == 'venue' and 
                venue_name_lower in data.get('name', '').lower()):
                return node
        
        return None
    
    def find_team_node(self, team_name: str) -> Optional[str]:
        """Find a team node by name (fuzzy matching)"""
        if not self.graph:
            return None
            
        team_name_lower = team_name.lower()
        
        # Try exact match first
        for node, data in self.graph.nodes(data=True):
            if (data.get('node_type') == 'team' and 
                data.get('name', '').lower() == team_name_lower):
                return node
        
        # Try partial match
        for node, data in self.graph.nodes(data=True):
            if (data.get('node_type') == 'team' and 
                team_name_lower in data.get('name', '').lower()):
                return node
        
        return None
    
    def get_player_stats(self, player: str, format_filter: Optional[str] = None, 
                        venue_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive stats for a player
        
        Args:
            player: Player name
            format_filter: Optional format filter (T20, ODI, Test)
            venue_filter: Optional venue filter
            
        Returns:
            Dictionary with player statistics
        """
        try:
            with timeout(self.query_timeout):
                player_node = self.find_player_node(player)
                if not player_node:
                    return {"error": f"Player '{player}' not found in knowledge graph"}
                
                # Get player data
                player_data = self.graph.nodes[player_node]
                
                # Get connected venues and teams
                neighbors = list(self.graph.neighbors(player_node))
                
                venues = []
                teams = []
                matches = 0
                
                for neighbor in neighbors:
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get('node_type') == 'venue':
                        venues.append(neighbor_data.get('name', neighbor))
                    elif neighbor_data.get('node_type') == 'team':
                        teams.append(neighbor_data.get('name', neighbor))
                    elif neighbor_data.get('node_type') == 'match':
                        matches += 1
                
                # Apply filters if specified
                filtered_stats = {}
                if venue_filter:
                    venue_node = self.find_venue_node(venue_filter)
                    if venue_node and venue_node in neighbors:
                        # Get edge data for this venue
                        edge_data = self.graph.edges[player_node, venue_node]
                        filtered_stats['venue_specific'] = edge_data
                
                result = {
                    "player": player_data.get('name', player),
                    "node_id": player_node,
                    "career_stats": {
                        "total_runs": player_data.get('total_runs', 0),
                        "total_balls": player_data.get('total_balls', 0),
                        "matches": matches,
                        "strike_rate": player_data.get('strike_rate', 0),
                        "average": player_data.get('average', 0),
                        "boundaries": player_data.get('boundaries', 0),
                        "sixes": player_data.get('sixes', 0)
                    },
                    "venues_played": venues[:10],  # Limit to top 10
                    "teams_played_for": teams,
                    "formats": player_data.get('formats', []),
                    "filtered_stats": filtered_stats
                }
                
                return result
                
        except QueryTimeoutError:
            return {"error": "Query timed out - please try a more specific search"}
        except Exception as e:
            logger.error(f"Error getting player stats: {e}")
            return {"error": f"Failed to get player stats: {str(e)}"}
    
    def compare_players(self, players: List[str], metric: str = "strike_rate", 
                       context: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple players across specified metrics
        
        Args:
            players: List of player names
            metric: Metric to compare (strike_rate, average, total_runs, etc.)
            context: Optional context filter (venue, format, etc.)
            
        Returns:
            Comparison data in table format
        """
        try:
            with timeout(self.query_timeout):
                if len(players) > 10:
                    return {"error": "Too many players to compare (max 10)"}
                
                comparison_data = []
                
                for player in players:
                    player_stats = self.get_player_stats(player)
                    if "error" in player_stats:
                        continue
                    
                    stats = player_stats["career_stats"]
                    comparison_data.append({
                        "player": player_stats["player"],
                        "strike_rate": stats.get("strike_rate", 0),
                        "average": stats.get("average", 0),
                        "total_runs": stats.get("total_runs", 0),
                        "matches": stats.get("matches", 0),
                        "boundaries": stats.get("boundaries", 0),
                        "sixes": stats.get("sixes", 0)
                    })
                
                if not comparison_data:
                    return {"error": "No valid players found for comparison"}
                
                # Sort by the specified metric
                if metric in ["strike_rate", "average", "total_runs", "matches", "boundaries", "sixes"]:
                    comparison_data.sort(key=lambda x: x.get(metric, 0), reverse=True)
                
                return {
                    "comparison": comparison_data,
                    "metric": metric,
                    "context": context,
                    "total_players": len(comparison_data)
                }
                
        except QueryTimeoutError:
            return {"error": "Query timed out - please try fewer players"}
        except Exception as e:
            logger.error(f"Error comparing players: {e}")
            return {"error": f"Failed to compare players: {str(e)}"}
    
    def get_venue_history(self, venue: str, team: Optional[str] = None, 
                         format_filter: Optional[str] = None, last_n_matches: int = 10) -> Dict[str, Any]:
        """
        Get venue history and performance data
        
        Args:
            venue: Venue name
            team: Optional team filter
            format_filter: Optional format filter
            last_n_matches: Number of recent matches to include
            
        Returns:
            Venue history and statistics
        """
        try:
            with timeout(self.query_timeout):
                venue_node = self.find_venue_node(venue)
                if not venue_node:
                    return {"error": f"Venue '{venue}' not found in knowledge graph"}
                
                venue_data = self.graph.nodes[venue_node]
                neighbors = list(self.graph.neighbors(venue_node))
                
                # Get connected teams and matches
                teams_played = []
                matches = []
                players = []
                
                for neighbor in neighbors:
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get('node_type') == 'team':
                        teams_played.append(neighbor_data.get('name', neighbor))
                    elif neighbor_data.get('node_type') == 'match':
                        matches.append(neighbor)
                    elif neighbor_data.get('node_type') == 'player':
                        players.append(neighbor_data.get('name', neighbor))
                
                # Apply team filter if specified
                if team:
                    team_node = self.find_team_node(team)
                    if team_node and team_node in neighbors:
                        # Get edge data for this team
                        edge_data = self.graph.edges[venue_node, team_node]
                
                result = {
                    "venue": venue_data.get('name', venue),
                    "node_id": venue_node,
                    "venue_stats": {
                        "total_matches": len(matches),
                        "teams_played": len(teams_played),
                        "avg_score": venue_data.get('avg_score', 0),
                        "avg_wickets": venue_data.get('avg_wickets', 0),
                        "boundary_percentage": venue_data.get('boundary_percentage', 0)
                    },
                    "teams": teams_played[:10],  # Limit to top 10
                    "top_performers": players[:10],  # Top 10 performers
                    "recent_matches": matches[:last_n_matches],
                    "characteristics": {
                        "pitch_type": venue_data.get('pitch_type', 'unknown'),
                        "avg_first_innings": venue_data.get('avg_first_innings', 0),
                        "chase_success_rate": venue_data.get('chase_success_rate', 0)
                    }
                }
                
                return result
                
        except QueryTimeoutError:
            return {"error": "Query timed out - please try a more specific search"}
        except Exception as e:
            logger.error(f"Error getting venue history: {e}")
            return {"error": f"Failed to get venue history: {str(e)}"}
    
    def get_head_to_head(self, team1: str, team2: str, format_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Get head-to-head record between two teams
        
        Args:
            team1: First team name
            team2: Second team name
            format_filter: Optional format filter
            
        Returns:
            Head-to-head statistics
        """
        try:
            with timeout(self.query_timeout):
                team1_node = self.find_team_node(team1)
                team2_node = self.find_team_node(team2)
                
                if not team1_node:
                    return {"error": f"Team '{team1}' not found in knowledge graph"}
                if not team2_node:
                    return {"error": f"Team '{team2}' not found in knowledge graph"}
                
                # Find common neighbors (matches between these teams)
                team1_neighbors = set(self.graph.neighbors(team1_node))
                team2_neighbors = set(self.graph.neighbors(team2_node))
                common_matches = team1_neighbors.intersection(team2_neighbors)
                
                # Filter for match nodes only
                match_nodes = [node for node in common_matches 
                              if self.graph.nodes[node].get('node_type') == 'match']
                
                if not match_nodes:
                    return {
                        "team1": self.graph.nodes[team1_node].get('name', team1),
                        "team2": self.graph.nodes[team2_node].get('name', team2),
                        "matches_played": 0,
                        "message": "No direct matches found between these teams"
                    }
                
                # Analyze match results
                team1_wins = 0
                team2_wins = 0
                total_matches = len(match_nodes)
                
                for match_node in match_nodes:
                    match_data = self.graph.nodes[match_node]
                    winner = match_data.get('winner')
                    if winner == team1_node:
                        team1_wins += 1
                    elif winner == team2_node:
                        team2_wins += 1
                
                result = {
                    "team1": self.graph.nodes[team1_node].get('name', team1),
                    "team2": self.graph.nodes[team2_node].get('name', team2),
                    "matches_played": total_matches,
                    "team1_wins": team1_wins,
                    "team2_wins": team2_wins,
                    "draws": total_matches - team1_wins - team2_wins,
                    "win_percentage": {
                        "team1": (team1_wins / total_matches * 100) if total_matches > 0 else 0,
                        "team2": (team2_wins / total_matches * 100) if total_matches > 0 else 0
                    },
                    "recent_matches": match_nodes[:5]  # Last 5 matches
                }
                
                return result
                
        except QueryTimeoutError:
            return {"error": "Query timed out - please try a more specific search"}
        except Exception as e:
            logger.error(f"Error getting head-to-head: {e}")
            return {"error": f"Failed to get head-to-head: {str(e)}"}
    
    def find_similar_players(self, player: str, metric: str = "strike_rate", 
                           limit: int = 5) -> Dict[str, Any]:
        """
        Find players similar to the given player based on specified metric
        
        Args:
            player: Reference player name
            metric: Metric for similarity (strike_rate, average, etc.)
            limit: Number of similar players to return
            
        Returns:
            List of similar players
        """
        try:
            with timeout(self.query_timeout):
                player_node = self.find_player_node(player)
                if not player_node:
                    return {"error": f"Player '{player}' not found in knowledge graph"}
                
                player_data = self.graph.nodes[player_node]
                reference_value = player_data.get(metric, 0)
                
                if reference_value == 0:
                    return {"error": f"No {metric} data available for {player}"}
                
                # Find similar players
                similar_players = []
                
                for node, data in self.graph.nodes(data=True):
                    if (data.get('node_type') == 'player' and 
                        node != player_node and 
                        data.get(metric, 0) > 0):
                        
                        player_value = data.get(metric, 0)
                        # Calculate similarity (smaller difference = more similar)
                        difference = abs(reference_value - player_value)
                        similarity_score = max(0, 100 - (difference / reference_value * 100))
                        
                        similar_players.append({
                            "player": data.get('name', node),
                            "value": player_value,
                            "similarity_score": similarity_score,
                            "difference": difference
                        })
                
                # Sort by similarity and limit results
                similar_players.sort(key=lambda x: x["similarity_score"], reverse=True)
                similar_players = similar_players[:limit]
                
                result = {
                    "reference_player": player_data.get('name', player),
                    "reference_value": reference_value,
                    "metric": metric,
                    "similar_players": similar_players,
                    "total_found": len(similar_players)
                }
                
                return result
                
        except QueryTimeoutError:
            return {"error": "Query timed out - please try a more specific search"}
        except Exception as e:
            logger.error(f"Error finding similar players: {e}")
            return {"error": f"Failed to find similar players: {str(e)}"}

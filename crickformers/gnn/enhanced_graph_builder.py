# Purpose: Enhanced cricket knowledge graph builder with comprehensive relationships
# Author: WicketWise Team, Last Modified: 2024-12-07

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EnhancedGraphBuilder:
    """
    Enhanced cricket knowledge graph builder that creates comprehensive
    relationships between players, venues, teams, and match contexts.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_stats = defaultdict(dict)
        self.edge_stats = defaultdict(dict)
        
    def build_from_dataframe(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build a comprehensive cricket knowledge graph from ball-by-ball data.
        
        Args:
            df: DataFrame with ball-by-ball cricket data
            
        Returns:
            NetworkX DiGraph with comprehensive cricket relationships
        """
        logger.info(f"Building cricket knowledge graph from {len(df):,} balls")
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # New scalable pipeline using vectorized aggregations
        from .schema_resolver import resolve_schema
        from .kg_aggregator import aggregate_core, compute_partnerships
        from .scalable_graph_builder import build_graph_from_aggregates

        mapping = resolve_schema(df, use_llm=False)

        aggs = aggregate_core(df, mapping)
        # Partnerships (approximate if non-striker missing)
        try:
            partnerships = compute_partnerships(df, mapping)
            if not partnerships.empty:
                aggs["partnerships"] = partnerships
        except Exception:
            pass

        self.graph = build_graph_from_aggregates(aggs)
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _add_player_nodes(self, df: pd.DataFrame):
        """Add player nodes with attributes."""
        # Get unique players
        batters = df['batter'].dropna().unique()
        bowlers = df['bowler'].dropna().unique()
        
        # Add batter nodes
        for batter in batters:
            batter_stats = self._calculate_batter_stats(df, batter)
            self.graph.add_node(
                batter,
                type='batter',
                role='batter',
                **batter_stats
            )
        
        # Add bowler nodes
        for bowler in bowlers:
            bowler_stats = self._calculate_bowler_stats(df, bowler)
            self.graph.add_node(
                bowler,
                type='bowler',
                role='bowler',
                **bowler_stats
            )
    
    def _add_venue_nodes(self, df: pd.DataFrame):
        """Add venue nodes with attributes."""
        venues = df['venue'].dropna().unique()
        
        for venue in venues:
            venue_stats = self._calculate_venue_stats(df, venue)
            self.graph.add_node(
                venue,
                type='venue',
                **venue_stats
            )
    
    def _add_team_nodes(self, df: pd.DataFrame):
        """Add team nodes with attributes."""
        teams = set()
        if 'team_batting' in df.columns:
            teams.update(df['team_batting'].dropna().unique())
        if 'team_bowling' in df.columns:
            teams.update(df['team_bowling'].dropna().unique())
        
        for team in teams:
            team_stats = self._calculate_team_stats(df, team)
            self.graph.add_node(
                team,
                type='team',
                **team_stats
            )
    
    def _add_match_nodes(self, df: pd.DataFrame):
        """Add match nodes with attributes."""
        matches = df['match_id'].dropna().unique()
        
        for match_id in matches:
            match_data = df[df['match_id'] == match_id]
            match_stats = self._calculate_match_stats(match_data)
            self.graph.add_node(
                match_id,
                type='match',
                **match_stats
            )
    
    def _add_player_relationships(self, df: pd.DataFrame):
        """Add player-to-player relationships."""
        # Batter vs Bowler relationships
        for _, row in df.iterrows():
            batter = row.get('batter')
            bowler = row.get('bowler')
            
            if pd.notna(batter) and pd.notna(bowler):
                # Add or update batter-bowler edge
                if self.graph.has_edge(batter, bowler):
                    edge_data = self.graph[batter][bowler]
                    edge_data['balls_faced'] = edge_data.get('balls_faced', 0) + 1
                    edge_data['runs_scored'] = edge_data.get('runs_scored', 0) + row.get('runs_scored', 0)
                    if row.get('is_wicket', False):
                        edge_data['dismissals'] = edge_data.get('dismissals', 0) + 1
                else:
                    self.graph.add_edge(
                        batter, bowler,
                        edge_type='batter_vs_bowler',
                        balls_faced=1,
                        runs_scored=row.get('runs_scored', 0),
                        dismissals=1 if row.get('is_wicket', False) else 0
                    )
    
    def _add_venue_relationships(self, df: pd.DataFrame):
        """Add venue-related relationships."""
        for _, row in df.iterrows():
            venue = row.get('venue')
            batter = row.get('batter')
            bowler = row.get('bowler')
            
            if pd.notna(venue):
                # Player-venue relationships
                if pd.notna(batter):
                    self._add_or_update_edge(
                        batter, venue, 'plays_at_venue',
                        balls_faced=1,
                        runs_scored=row.get('runs_scored', 0)
                    )
                
                if pd.notna(bowler):
                    self._add_or_update_edge(
                        bowler, venue, 'bowls_at_venue',
                        balls_bowled=1,
                        runs_conceded=row.get('runs_scored', 0),
                        wickets=1 if row.get('is_wicket', False) else 0
                    )
    
    def _add_performance_relationships(self, df: pd.DataFrame):
        """Add performance-based relationships."""
        # High-scoring partnerships
        for match_id in df['match_id'].unique():
            match_data = df[df['match_id'] == match_id]
            
            for innings in match_data['innings'].unique():
                innings_data = match_data[match_data['innings'] == innings]
                
                # Track partnerships
                current_batters = set()
                partnership_runs = 0
                
                for _, row in innings_data.iterrows():
                    batter = row.get('batter')
                    
                    if pd.notna(batter):
                        current_batters.add(batter)
                        partnership_runs += row.get('runs_scored', 0)
                        
                        # If wicket, record partnership
                        if row.get('is_wicket', False) and len(current_batters) >= 2:
                            batters_list = list(current_batters)
                            for i in range(len(batters_list)):
                                for j in range(i + 1, len(batters_list)):
                                    self._add_or_update_edge(
                                        batters_list[i], batters_list[j],
                                        'partnership',
                                        runs=partnership_runs,
                                        partnerships=1
                                    )
                            
                            # Reset for next partnership
                            current_batters = {batter}  # Keep current batter
                            partnership_runs = 0
    
    def _add_tactical_relationships(self, df: pd.DataFrame):
        """Add tactical relationships based on match situations."""
        # Phase-based performance
        for _, row in df.iterrows():
            over = row.get('over', 0)
            batter = row.get('batter')
            bowler = row.get('bowler')
            
            # Determine phase
            if over < 6:
                phase = 'powerplay'
            elif over < 16:
                phase = 'middle_overs'
            else:
                phase = 'death_overs'
            
            # Add phase node if not exists
            if not self.graph.has_node(phase):
                self.graph.add_node(phase, type='phase')
            
            # Connect players to phases
            if pd.notna(batter):
                self._add_or_update_edge(
                    batter, phase, 'performs_in_phase',
                    balls_faced=1,
                    runs_scored=row.get('runs_scored', 0)
                )
            
            if pd.notna(bowler):
                self._add_or_update_edge(
                    bowler, phase, 'bowls_in_phase',
                    balls_bowled=1,
                    runs_conceded=row.get('runs_scored', 0),
                    wickets=1 if row.get('is_wicket', False) else 0
                )
    
    def _add_or_update_edge(self, source: str, target: str, edge_type: str, **kwargs):
        """Add or update an edge with cumulative statistics."""
        if self.graph.has_edge(source, target):
            edge_data = self.graph[source][target]
            for key, value in kwargs.items():
                edge_data[key] = edge_data.get(key, 0) + value
        else:
            self.graph.add_edge(source, target, edge_type=edge_type, **kwargs)
    
    def _calculate_batter_stats(self, df: pd.DataFrame, batter: str) -> Dict[str, Any]:
        """Calculate comprehensive batter statistics."""
        batter_data = df[df['batter'] == batter]
        
        if len(batter_data) == 0:
            return {}
        
        total_runs = batter_data['runs_scored'].sum()
        balls_faced = len(batter_data)
        dismissals = batter_data['is_wicket'].sum()
        boundaries = (batter_data['runs_scored'] >= 4).sum()
        sixes = (batter_data['runs_scored'] == 6).sum()
        
        return {
            'total_runs': total_runs,
            'balls_faced': balls_faced,
            'dismissals': dismissals,
            'average': total_runs / max(dismissals, 1),
            'strike_rate': (total_runs / balls_faced) * 100 if balls_faced > 0 else 0,
            'boundary_rate': boundaries / balls_faced if balls_faced > 0 else 0,
            'six_rate': sixes / balls_faced if balls_faced > 0 else 0,
            'matches_played': batter_data['match_id'].nunique()
        }
    
    def _calculate_bowler_stats(self, df: pd.DataFrame, bowler: str) -> Dict[str, Any]:
        """Calculate comprehensive bowler statistics."""
        bowler_data = df[df['bowler'] == bowler]
        
        if len(bowler_data) == 0:
            return {}
        
        runs_conceded = bowler_data['runs_scored'].sum()
        balls_bowled = len(bowler_data)
        wickets = bowler_data['is_wicket'].sum()
        
        return {
            'runs_conceded': runs_conceded,
            'balls_bowled': balls_bowled,
            'wickets': wickets,
            'average': runs_conceded / max(wickets, 1),
            'economy': (runs_conceded / balls_bowled) * 6 if balls_bowled > 0 else 0,
            'strike_rate': balls_bowled / max(wickets, 1),
            'wicket_rate': wickets / balls_bowled if balls_bowled > 0 else 0,
            'matches_played': bowler_data['match_id'].nunique()
        }
    
    def _calculate_venue_stats(self, df: pd.DataFrame, venue: str) -> Dict[str, Any]:
        """Calculate venue statistics."""
        venue_data = df[df['venue'] == venue]
        
        if len(venue_data) == 0:
            return {}
        
        # Calculate innings totals
        innings_totals = venue_data.groupby(['match_id', 'innings'])['runs_scored'].sum()
        
        return {
            'matches_played': venue_data['match_id'].nunique(),
            'balls_played': len(venue_data),
            'avg_score': innings_totals.mean(),
            'highest_score': innings_totals.max(),
            'lowest_score': innings_totals.min(),
            'boundary_rate': (venue_data['runs_scored'] >= 4).sum() / len(venue_data),
            'wicket_rate': venue_data['is_wicket'].sum() / len(venue_data)
        }
    
    def _calculate_team_stats(self, df: pd.DataFrame, team: str) -> Dict[str, Any]:
        """Calculate team statistics."""
        team_data = df[
            (df['team_batting'] == team) | (df['team_bowling'] == team)
        ]
        
        if len(team_data) == 0:
            return {}
        
        batting_data = df[df['team_batting'] == team]
        bowling_data = df[df['team_bowling'] == team]
        
        return {
            'matches_played': team_data['match_id'].nunique(),
            'runs_scored': batting_data['runs_scored'].sum(),
            'runs_conceded': bowling_data['runs_scored'].sum(),
            'wickets_lost': batting_data['is_wicket'].sum(),
            'wickets_taken': bowling_data['is_wicket'].sum(),
            'balls_batted': len(batting_data),
            'balls_bowled': len(bowling_data)
        }
    
    def _calculate_match_stats(self, match_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate match statistics."""
        innings_scores = match_data.groupby('innings')['runs_scored'].sum()
        
        return {
            'total_balls': len(match_data),
            'total_runs': match_data['runs_scored'].sum(),
            'total_wickets': match_data['is_wicket'].sum(),
            'innings_count': match_data['innings'].nunique(),
            'first_innings_score': innings_scores.get(1, 0),
            'second_innings_score': innings_scores.get(2, 0),
            'venue': match_data['venue'].iloc[0] if 'venue' in match_data.columns else None,
            'competition': match_data['competition_name'].iloc[0] if 'competition_name' in match_data.columns else None
        }
    
    def _calculate_node_statistics(self, df: pd.DataFrame):
        """Calculate additional node statistics and centrality measures."""
        # Calculate centrality measures
        centrality_measures = {
            'degree_centrality': nx.degree_centrality(self.graph),
            'betweenness_centrality': nx.betweenness_centrality(self.graph),
            'closeness_centrality': nx.closeness_centrality(self.graph),
            'pagerank': nx.pagerank(self.graph)
        }
        
        # Add centrality measures to nodes
        for node in self.graph.nodes():
            for measure, values in centrality_measures.items():
                self.graph.nodes[node][measure] = values.get(node, 0)
    
    def get_player_embeddings(self, embedding_dim: int = 128) -> Dict[str, np.ndarray]:
        """
        Generate player embeddings based on graph structure and statistics.
        
        Args:
            embedding_dim: Dimension of embeddings
            
        Returns:
            Dictionary mapping player names to embedding vectors
        """
        embeddings = {}
        
        # Get all player nodes
        players = [node for node in self.graph.nodes() 
                  if self.graph.nodes[node].get('type') in ['batter', 'bowler']]
        
        for player in players:
            # Create embedding based on node attributes and graph structure
            node_attrs = self.graph.nodes[player]
            
            # Statistical features
            stats_features = []
            if node_attrs.get('type') == 'batter':
                stats_features = [
                    node_attrs.get('average', 0) / 100,  # Normalize
                    node_attrs.get('strike_rate', 0) / 200,  # Normalize
                    node_attrs.get('boundary_rate', 0),
                    node_attrs.get('six_rate', 0)
                ]
            else:  # bowler
                stats_features = [
                    node_attrs.get('average', 0) / 50,  # Normalize
                    node_attrs.get('economy', 0) / 15,  # Normalize
                    node_attrs.get('wicket_rate', 0) * 10,  # Scale up
                    node_attrs.get('strike_rate', 0) / 50  # Normalize
                ]
            
            # Graph structure features
            graph_features = [
                node_attrs.get('degree_centrality', 0),
                node_attrs.get('betweenness_centrality', 0),
                node_attrs.get('closeness_centrality', 0),
                node_attrs.get('pagerank', 0)
            ]
            
            # Combine features
            combined_features = stats_features + graph_features
            
            # Pad or truncate to desired dimension
            if len(combined_features) < embedding_dim:
                # Pad with random values
                padding = np.random.normal(0, 0.1, embedding_dim - len(combined_features))
                embedding = np.array(combined_features + padding.tolist())
            else:
                embedding = np.array(combined_features[:embedding_dim])
            
            embeddings[player] = embedding
        
        return embeddings
    
    def get_venue_embeddings(self, embedding_dim: int = 64) -> Dict[str, np.ndarray]:
        """Generate venue embeddings."""
        embeddings = {}
        
        venues = [node for node in self.graph.nodes() 
                 if self.graph.nodes[node].get('type') == 'venue']
        
        for venue in venues:
            node_attrs = self.graph.nodes[venue]
            
            # Venue features
            venue_features = [
                node_attrs.get('avg_score', 0) / 200,  # Normalize
                node_attrs.get('boundary_rate', 0),
                node_attrs.get('wicket_rate', 0) * 10,  # Scale up
                node_attrs.get('matches_played', 0) / 100  # Normalize
            ]
            
            # Graph structure features
            graph_features = [
                node_attrs.get('degree_centrality', 0),
                node_attrs.get('betweenness_centrality', 0),
                node_attrs.get('pagerank', 0)
            ]
            
            # Combine and pad
            combined_features = venue_features + graph_features
            
            if len(combined_features) < embedding_dim:
                padding = np.random.normal(0, 0.1, embedding_dim - len(combined_features))
                embedding = np.array(combined_features + padding.tolist())
            else:
                embedding = np.array(combined_features[:embedding_dim])
            
            embeddings[venue] = embedding
        
        return embeddings
    
    def export_graph(self, filepath: str):
        """Export graph to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)
        logger.info(f"Graph exported to {filepath}")
    
    def load_graph(self, filepath: str):
        """Load graph from file."""
        import pickle
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)
        logger.info(f"Graph loaded from {filepath}")
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the graph."""
        node_types = defaultdict(int)
        edge_types = defaultdict(int)
        
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            node_types[node_type] += 1
        
        for edge in self.graph.edges():
            edge_type = self.graph.edges[edge].get('edge_type', 'unknown')
            edge_types[edge_type] += 1
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': dict(node_types),
            'edge_types': dict(edge_types),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph)
        } 
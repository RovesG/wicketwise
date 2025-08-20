# Purpose: Simplified demonstration of enhanced KG features without complex GNN layers
# Author: WicketWise Team, Last Modified: 2025-01-18

"""
Simplified demonstration that focuses on the rich feature extraction from the
comprehensive knowledge graph, showing the analytical power without complex GNN layers.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import pickle
import json
import logging
from typing import Dict, List, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crickformers.gnn.enhanced_kg_gnn import SituationalFeatureExtractor, EnhancedFeatureConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplifiedCricketAnalytics:
    """
    Simplified cricket analytics that demonstrates the power of comprehensive
    KG features without complex GNN training requirements.
    """
    
    def __init__(self, kg_path: str = "models/unified_cricket_kg.pkl"):
        self.kg_path = kg_path
        self.kg = None
        self.feature_extractor = None
        self.player_features = {}
        self.player_names = []
        
    def load_knowledge_graph(self):
        """Load the comprehensive knowledge graph"""
        try:
            with open(self.kg_path, 'rb') as f:
                self.kg = pickle.load(f)
            logger.info(f"Loaded KG: {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges")
            
            # Initialize feature extractor
            self.feature_extractor = SituationalFeatureExtractor(
                EnhancedFeatureConfig(
                    use_situational_stats=True,
                    use_venue_performance=True,
                    use_pressure_stats=True,
                    use_bowling_matchups=True,
                    feature_dim=128,
                    normalize_features=True
                )
            )
            
            self._analyze_kg_features()
            
        except FileNotFoundError:
            logger.error(f"Knowledge graph not found at {self.kg_path}")
            logger.info("Please run the unified KG builder first!")
            raise
    
    def _analyze_kg_features(self):
        """Analyze the rich features available in the knowledge graph"""
        feature_stats = {
            'players_with_batting_stats': 0,
            'players_with_bowling_stats': 0,
            'players_with_situational_stats': 0,
            'players_with_venue_stats': 0,
            'players_with_pressure_stats': 0,
            'players_with_pace_spin_stats': 0
        }
        
        player_count = 0
        for node, attrs in self.kg.nodes(data=True):
            if attrs.get('type') == 'player':
                player_count += 1
                
                if 'batting_stats' in attrs and attrs['batting_stats']:
                    feature_stats['players_with_batting_stats'] += 1
                if 'bowling_stats' in attrs and attrs['bowling_stats']:
                    feature_stats['players_with_bowling_stats'] += 1
                if 'in_powerplay' in attrs or 'in_death_overs' in attrs:
                    feature_stats['players_with_situational_stats'] += 1
                if 'by_venue' in attrs and attrs['by_venue']:
                    feature_stats['players_with_venue_stats'] += 1
                if 'under_pressure' in attrs:
                    feature_stats['players_with_pressure_stats'] += 1
                if 'vs_pace' in attrs or 'vs_spin' in attrs:
                    feature_stats['players_with_pace_spin_stats'] += 1
        
        logger.info("Knowledge Graph Feature Analysis:")
        logger.info(f"  Total players: {player_count:,}")
        for stat, value in feature_stats.items():
            percentage = (value / player_count * 100) if player_count > 0 else 0
            logger.info(f"  {stat}: {value:,} ({percentage:.1f}%)")
    
    def extract_player_features(self, sample_size: int = 1000):
        """Extract comprehensive features for a sample of players"""
        logger.info(f"Extracting comprehensive features for {sample_size} players...")
        
        player_nodes = [(node, attrs) for node, attrs in self.kg.nodes(data=True) 
                       if attrs.get('type') == 'player']
        
        # Sample players for analysis
        if len(player_nodes) > sample_size:
            import random
            player_nodes = random.sample(player_nodes, sample_size)
        
        self.player_names = []
        feature_vectors = []
        
        for player_name, attrs in player_nodes:
            try:
                # Extract comprehensive features
                features = self.feature_extractor.extract_player_features(player_name, attrs)
                
                self.player_names.append(player_name)
                feature_vectors.append(features)
                self.player_features[player_name] = {
                    'features': features,
                    'batting_stats': attrs.get('batting_stats', {}),
                    'bowling_stats': attrs.get('bowling_stats', {}),
                    'situational_stats': {
                        'powerplay': attrs.get('in_powerplay', {}),
                        'death_overs': attrs.get('in_death_overs', {}),
                        'vs_pace': attrs.get('vs_pace', {}),
                        'vs_spin': attrs.get('vs_spin', {}),
                        'under_pressure': attrs.get('under_pressure', {})
                    },
                    'venue_performance': attrs.get('by_venue', {})
                }
                
            except Exception as e:
                logger.warning(f"Failed to extract features for {player_name}: {e}")
                continue
        
        self.feature_matrix = np.array(feature_vectors)
        logger.info(f"Extracted features for {len(self.player_names)} players")
        logger.info(f"Feature matrix shape: {self.feature_matrix.shape}")
        
        return self.feature_matrix
    
    def find_similar_players(self, player_name: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Find players most similar to the given player using comprehensive features"""
        if player_name not in self.player_features:
            logger.error(f"Player {player_name} not found in extracted features")
            return []
        
        target_features = self.player_features[player_name]['features'].reshape(1, -1)
        
        # Compute similarities with all other players
        similarities = cosine_similarity(target_features, self.feature_matrix)[0]
        
        # Get top-k similar players (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        similar_players = []
        for idx in similar_indices:
            similar_name = self.player_names[idx]
            similarity_score = similarities[idx]
            similar_stats = self.player_features[similar_name]
            
            similar_players.append((similar_name, similarity_score, similar_stats))
        
        return similar_players
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze which features are most discriminative"""
        if self.feature_matrix.size == 0:
            return {}
        
        # Compute feature variance (higher variance = more discriminative)
        feature_variance = np.var(self.feature_matrix, axis=0)
        
        # Feature names (simplified mapping)
        feature_categories = {
            'basic_batting': list(range(0, 5)),
            'basic_bowling': list(range(5, 10)),
            'powerplay_performance': list(range(10, 15)),
            'death_overs_performance': list(range(15, 20)),
            'pace_vs_performance': list(range(20, 24)),
            'spin_vs_performance': list(range(24, 28)),
            'pressure_performance': list(range(28, 34)),
            'venue_performance': list(range(34, 38)),
            'role_embeddings': list(range(38, 54)),
            'style_embeddings': list(range(54, 86)),
            'other_features': list(range(86, 128))
        }
        
        category_importance = {}
        for category, indices in feature_categories.items():
            valid_indices = [i for i in indices if i < len(feature_variance)]
            if valid_indices:
                category_importance[category] = np.mean(feature_variance[valid_indices])
            else:
                category_importance[category] = 0.0
        
        return category_importance
    
    def generate_player_insights(self, player_name: str) -> Dict[str, Any]:
        """Generate comprehensive insights for a specific player"""
        if player_name not in self.player_features:
            return {'error': f'Player {player_name} not found'}
        
        player_data = self.player_features[player_name]
        
        insights = {
            'player_name': player_name,
            'basic_stats': {
                'batting': player_data['batting_stats'],
                'bowling': player_data['bowling_stats']
            },
            'situational_analysis': {},
            'venue_analysis': {},
            'similar_players': []
        }
        
        # Situational analysis
        situational = player_data['situational_stats']
        if situational['powerplay'] and situational['death_overs']:
            pp_sr = situational['powerplay'].get('strike_rate', 0)
            death_sr = situational['death_overs'].get('strike_rate', 0)
            
            insights['situational_analysis'] = {
                'powerplay_strike_rate': pp_sr,
                'death_overs_strike_rate': death_sr,
                'phase_preference': 'powerplay' if pp_sr > death_sr else 'death_overs',
                'adaptability_score': abs(pp_sr - death_sr)  # Lower = more adaptable
            }
        
        # Bowling matchup analysis
        if situational['vs_pace'] and situational['vs_spin']:
            pace_avg = situational['vs_pace'].get('average', 0)
            spin_avg = situational['vs_spin'].get('average', 0)
            
            insights['bowling_matchup_analysis'] = {
                'vs_pace_average': pace_avg,
                'vs_spin_average': spin_avg,
                'strength': 'vs_pace' if pace_avg > spin_avg else 'vs_spin',
                'matchup_difference': abs(pace_avg - spin_avg)
            }
        
        # Venue analysis
        venue_data = player_data['venue_performance']
        if venue_data:
            venue_averages = [(venue, stats.get('average', 0)) for venue, stats in venue_data.items()]
            venue_averages.sort(key=lambda x: x[1], reverse=True)
            
            insights['venue_analysis'] = {
                'venues_played': len(venue_data),
                'best_venue': venue_averages[0] if venue_averages else None,
                'worst_venue': venue_averages[-1] if venue_averages else None,
                'venue_consistency': np.std([avg for _, avg in venue_averages]) if len(venue_averages) > 1 else 0
            }
        
        # Find similar players
        similar = self.find_similar_players(player_name, top_k=5)
        insights['similar_players'] = [
            {
                'name': name,
                'similarity': float(score),
                'batting_average': stats['batting_stats'].get('average', 0),
                'key_similarity': self._identify_similarity_reason(player_data, stats)
            }
            for name, score, stats in similar
        ]
        
        return insights
    
    def _identify_similarity_reason(self, player1: Dict, player2: Dict) -> str:
        """Identify the main reason two players are similar"""
        reasons = []
        
        # Compare batting averages
        p1_bat_avg = player1['batting_stats'].get('average', 0)
        p2_bat_avg = player2['batting_stats'].get('average', 0)
        if abs(p1_bat_avg - p2_bat_avg) < 5:
            reasons.append('similar_batting_average')
        
        # Compare powerplay performance
        p1_pp = player1['situational_stats']['powerplay'].get('strike_rate', 0)
        p2_pp = player2['situational_stats']['powerplay'].get('strike_rate', 0)
        if abs(p1_pp - p2_pp) < 20:
            reasons.append('similar_powerplay_style')
        
        # Compare pace vs spin preference
        p1_pace = player1['situational_stats']['vs_pace'].get('average', 0)
        p1_spin = player1['situational_stats']['vs_spin'].get('average', 0)
        p2_pace = player2['situational_stats']['vs_pace'].get('average', 0)
        p2_spin = player2['situational_stats']['vs_spin'].get('average', 0)
        
        if (p1_pace > p1_spin) == (p2_pace > p2_spin):
            reasons.append('similar_bowling_matchup_preference')
        
        return ', '.join(reasons) if reasons else 'overall_playing_style'
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analytics report"""
        if not self.player_features:
            self.extract_player_features(sample_size=500)  # Smaller sample for demo
        
        # Feature importance analysis
        feature_importance = self.analyze_feature_importance()
        
        # Sample player insights
        sample_players = list(self.player_features.keys())[:10]
        player_insights = {}
        
        for player in sample_players:
            try:
                insights = self.generate_player_insights(player)
                if 'error' not in insights:
                    player_insights[player] = insights
            except Exception as e:
                logger.warning(f"Failed to generate insights for {player}: {e}")
        
        # Overall statistics
        all_batting_avgs = []
        all_powerplay_srs = []
        all_death_srs = []
        
        for player_data in self.player_features.values():
            bat_avg = player_data['batting_stats'].get('average', 0)
            if bat_avg > 0:
                all_batting_avgs.append(bat_avg)
            
            pp_sr = player_data['situational_stats']['powerplay'].get('strike_rate', 0)
            if pp_sr > 0:
                all_powerplay_srs.append(pp_sr)
            
            death_sr = player_data['situational_stats']['death_overs'].get('strike_rate', 0)
            if death_sr > 0:
                all_death_srs.append(death_sr)
        
        return {
            'feature_importance': feature_importance,
            'sample_player_insights': player_insights,
            'overall_statistics': {
                'batting_average_distribution': {
                    'mean': np.mean(all_batting_avgs) if all_batting_avgs else 0,
                    'std': np.std(all_batting_avgs) if all_batting_avgs else 0,
                    'min': np.min(all_batting_avgs) if all_batting_avgs else 0,
                    'max': np.max(all_batting_avgs) if all_batting_avgs else 0
                },
                'powerplay_strike_rate_distribution': {
                    'mean': np.mean(all_powerplay_srs) if all_powerplay_srs else 0,
                    'std': np.std(all_powerplay_srs) if all_powerplay_srs else 0
                },
                'death_overs_strike_rate_distribution': {
                    'mean': np.mean(all_death_srs) if all_death_srs else 0,
                    'std': np.std(all_death_srs) if all_death_srs else 0
                }
            },
            'analysis_summary': {
                'total_players_analyzed': len(self.player_features),
                'feature_dimensions': self.feature_matrix.shape[1] if hasattr(self, 'feature_matrix') else 0,
                'most_discriminative_feature_category': max(feature_importance.items(), key=lambda x: x[1])[0] if feature_importance else 'unknown'
            }
        }


def main():
    """Demonstrate the comprehensive KG feature extraction and analysis"""
    logger.info("🎯 Simplified Enhanced Cricket Knowledge Graph Analytics Demo")
    logger.info("=" * 70)
    
    # Initialize the analytics system
    analytics = SimplifiedCricketAnalytics()
    
    try:
        # Load the comprehensive knowledge graph
        logger.info("📊 Loading comprehensive knowledge graph...")
        analytics.load_knowledge_graph()
        
        # Extract features for analysis
        logger.info("🔍 Extracting comprehensive features...")
        analytics.extract_player_features(sample_size=200)  # Smaller sample for demo
        
        # Generate comprehensive report
        logger.info("📈 Generating comprehensive analytics report...")
        report = analytics.generate_comprehensive_report()
        
        # Display results
        logger.info("\n" + "=" * 70)
        logger.info("🎯 COMPREHENSIVE KG ANALYTICS RESULTS")
        logger.info("=" * 70)
        
        # Feature importance
        logger.info("\n📊 Feature Importance Analysis:")
        feature_importance = report['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        for feature, importance in sorted_features[:8]:  # Top 8
            logger.info(f"   {feature.replace('_', ' ').title()}: {importance:.4f}")
        
        # Overall statistics
        stats = report['overall_statistics']
        logger.info(f"\n📈 Overall Statistics:")
        logger.info(f"   Average Batting Average: {stats['batting_average_distribution']['mean']:.2f} ± {stats['batting_average_distribution']['std']:.2f}")
        logger.info(f"   Average Powerplay Strike Rate: {stats['powerplay_strike_rate_distribution']['mean']:.2f} ± {stats['powerplay_strike_rate_distribution']['std']:.2f}")
        logger.info(f"   Average Death Overs Strike Rate: {stats['death_overs_strike_rate_distribution']['mean']:.2f} ± {stats['death_overs_strike_rate_distribution']['std']:.2f}")
        
        # Sample player insights
        logger.info(f"\n🏏 Sample Player Insights:")
        sample_insights = report['sample_player_insights']
        
        for i, (player, insights) in enumerate(list(sample_insights.items())[:3]):
            logger.info(f"\n   Player {i+1}: {player}")
            
            # Batting stats
            batting = insights['basic_stats']['batting']
            if batting:
                logger.info(f"     Batting: {batting.get('runs', 0)} runs, {batting.get('average', 0):.2f} avg")
            
            # Situational analysis
            situational = insights.get('situational_analysis', {})
            if situational:
                logger.info(f"     Powerplay SR: {situational.get('powerplay_strike_rate', 0):.1f}")
                logger.info(f"     Death Overs SR: {situational.get('death_overs_strike_rate', 0):.1f}")
                logger.info(f"     Phase Preference: {situational.get('phase_preference', 'N/A')}")
            
            # Similar players
            similar = insights.get('similar_players', [])
            if similar:
                logger.info(f"     Most Similar: {similar[0]['name']} (similarity: {similar[0]['similarity']:.3f})")
        
        # Summary
        summary = report['analysis_summary']
        logger.info(f"\n📊 Analysis Summary:")
        logger.info(f"   Players Analyzed: {summary['total_players_analyzed']:,}")
        logger.info(f"   Feature Dimensions: {summary['feature_dimensions']}")
        logger.info(f"   Most Discriminative: {summary['most_discriminative_feature_category'].replace('_', ' ').title()}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Comprehensive KG Analytics Demo Completed Successfully!")
        logger.info("🎯 The system successfully extracted and analyzed rich features including:")
        logger.info("   • Situational performance (powerplay vs death overs)")
        logger.info("   • Bowling matchup analysis (pace vs spin)")
        logger.info("   • Pressure situation performance")
        logger.info("   • Venue-specific performance patterns")
        logger.info("   • Player similarity analysis based on comprehensive features")
        logger.info("\n🚀 This demonstrates the analytical power of the comprehensive knowledge graph!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

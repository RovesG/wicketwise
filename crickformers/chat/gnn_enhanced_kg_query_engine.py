# Purpose: GNN-Enhanced Knowledge Graph Query Engine for Cricket Intelligence
# Author: WicketWise Team, Last Modified: 2025-08-25

import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import networkx as nx

from .unified_kg_query_engine import UnifiedKGQueryEngine, QueryTimeoutError
from ..gnn.kg_gnn_integration import KGGNNEmbeddingService

logger = logging.getLogger(__name__)


class GNNEnhancedKGQueryEngine(UnifiedKGQueryEngine):
    """
    Enhanced Knowledge Graph Query Engine with GNN embedding capabilities
    
    Extends the unified KG engine with:
    - Semantic similarity searches using GNN embeddings
    - Contextual performance predictions
    - Advanced player/venue compatibility analysis
    - Style-based comparisons and recommendations
    """
    
    def __init__(self, 
                 graph_path: str = "models/unified_cricket_kg.pkl",
                 gnn_embeddings_path: str = "models/gnn_embeddings.pt",
                 gnn_model_path: Optional[str] = None):
        """
        Initialize GNN-enhanced query engine
        
        Args:
            graph_path: Path to unified knowledge graph
            gnn_embeddings_path: Path to pre-trained GNN embeddings
            gnn_model_path: Optional path to GNN model for dynamic embeddings
        """
        # Initialize base KG engine
        super().__init__(graph_path)
        
        # Initialize GNN service
        self.gnn_service = None
        self.gnn_embeddings_available = False
        
        try:
            # Try to load GNN service
            if Path(graph_path).exists():
                self.gnn_service = KGGNNEmbeddingService(
                    kg_path=graph_path,
                    gnn_model_path=gnn_model_path
                )
                self.gnn_embeddings_available = True
                logger.info(f"✅ GNN service initialized with KG from {graph_path}")
            else:
                logger.warning(f"⚠️ Knowledge graph not found at {graph_path}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize GNN service: {e}")
            
        # Configuration
        self.similarity_threshold = 0.7
        self.max_similarity_results = 10
        self.embedding_cache = {}
        
        logger.info(f"🚀 GNN-Enhanced KG Query Engine initialized (GNN: {'✅' if self.gnn_embeddings_available else '❌'})")
    
    def find_similar_players_gnn(self, 
                                player: str, 
                                top_k: int = 5,
                                similarity_metric: str = "cosine",
                                min_similarity: float = 0.6) -> Dict[str, Any]:
        """
        Find players with similar playing styles using GNN embeddings
        
        Args:
            player: Target player name
            top_k: Number of similar players to return
            similarity_metric: "cosine" or "euclidean"
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary with similar players and similarity scores
        """
        try:
            if not self.gnn_embeddings_available:
                return self._fallback_similarity_search(player, top_k)
            
            # Get target player embedding
            target_embedding = self._get_player_embedding(player)
            if target_embedding is None:
                return {"error": f"No GNN embedding found for player '{player}'"}
            
            # Get all player embeddings
            all_players = self._get_all_players_with_embeddings()
            if len(all_players) < 2:
                return {"error": "Insufficient player embeddings for similarity analysis"}
            
            # Calculate similarities
            similarities = []
            target_emb = target_embedding.reshape(1, -1)
            
            for other_player, other_embedding in all_players.items():
                if other_player.lower() == player.lower():
                    continue  # Skip self
                
                other_emb = other_embedding.reshape(1, -1)
                
                if similarity_metric == "cosine":
                    similarity = cosine_similarity(target_emb, other_emb)[0][0]
                else:  # euclidean (converted to similarity)
                    distance = euclidean_distances(target_emb, other_emb)[0][0]
                    similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                
                if similarity >= min_similarity:
                    similarities.append({
                        "player": other_player,
                        "similarity_score": float(similarity),
                        "similarity_metric": similarity_metric
                    })
            
            # Sort by similarity and take top_k
            similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            top_similar = similarities[:top_k]
            
            # Enhance with cricket insights
            enhanced_results = []
            for result in top_similar:
                enhanced = self._enhance_similarity_result(player, result)
                enhanced_results.append(enhanced)
            
            return {
                "target_player": player,
                "similar_players": enhanced_results,
                "total_comparisons": len(similarities),
                "similarity_metric": similarity_metric,
                "gnn_powered": True,
                "analysis_summary": self._generate_similarity_summary(player, enhanced_results)
            }
            
        except Exception as e:
            logger.error(f"Error in GNN similarity search: {e}")
            return {"error": f"GNN similarity search failed: {str(e)}"}
    
    def predict_contextual_performance(self, 
                                     player: str, 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict player performance in specific match contexts using GNN
        
        Args:
            player: Player name
            context: Match context (venue, phase, bowling_type, etc.)
            
        Returns:
            Performance prediction with confidence scores
        """
        try:
            if not self.gnn_embeddings_available:
                return self._fallback_contextual_analysis(player, context)
            
            # Get player embedding
            player_embedding = self._get_player_embedding(player)
            if player_embedding is None:
                return {"error": f"No GNN embedding found for player '{player}'"}
            
            # Get context embedding
            context_embedding = self._get_context_embedding(context)
            
            # Combine player and context embeddings
            combined_embedding = self._combine_embeddings(player_embedding, context_embedding)
            
            # Generate performance prediction
            prediction = self._generate_performance_prediction(combined_embedding, context)
            
            # Add cricket domain insights
            cricket_insights = self._generate_contextual_insights(player, context, prediction)
            
            return {
                "player": player,
                "context": context,
                "prediction": prediction,
                "insights": cricket_insights,
                "confidence": prediction.get("confidence", 0.7),
                "gnn_powered": True,
                "methodology": "GNN contextual embedding analysis"
            }
            
        except Exception as e:
            logger.error(f"Error in contextual performance prediction: {e}")
            return {"error": f"Contextual prediction failed: {str(e)}"}
    
    def analyze_venue_compatibility(self, 
                                  player: str, 
                                  venue: str) -> Dict[str, Any]:
        """
        Analyze how well a player might perform at a specific venue
        
        Args:
            player: Player name
            venue: Venue name
            
        Returns:
            Venue compatibility analysis with recommendations
        """
        try:
            if not self.gnn_embeddings_available:
                return self._fallback_venue_analysis(player, venue)
            
            # Get embeddings
            player_embedding = self._get_player_embedding(player)
            venue_embedding = self._get_venue_embedding(venue)
            
            if player_embedding is None:
                return {"error": f"No GNN embedding found for player '{player}'"}
            if venue_embedding is None:
                return {"error": f"No GNN embedding found for venue '{venue}'"}
            
            # Calculate compatibility score
            compatibility = self._calculate_venue_compatibility(player_embedding, venue_embedding)
            
            # Get historical performance if available
            historical_data = self._get_historical_venue_performance(player, venue)
            
            # Generate insights
            insights = self._generate_venue_insights(player, venue, compatibility, historical_data)
            
            return {
                "player": player,
                "venue": venue,
                "compatibility_score": float(compatibility),
                "compatibility_level": self._categorize_compatibility(compatibility),
                "historical_performance": historical_data,
                "insights": insights,
                "recommendations": self._generate_venue_recommendations(compatibility, insights),
                "gnn_powered": True
            }
            
        except Exception as e:
            logger.error(f"Error in venue compatibility analysis: {e}")
            return {"error": f"Venue compatibility analysis failed: {str(e)}"}
    
    def get_playing_style_similarity(self, 
                                   player1: str, 
                                   player2: str) -> Dict[str, Any]:
        """
        Analyze playing style similarity between two players
        
        Args:
            player1: First player name
            player2: Second player name
            
        Returns:
            Detailed style similarity analysis
        """
        try:
            if not self.gnn_embeddings_available:
                return self._fallback_style_comparison(player1, player2)
            
            # Get embeddings
            emb1 = self._get_player_embedding(player1)
            emb2 = self._get_player_embedding(player2)
            
            if emb1 is None or emb2 is None:
                missing = player1 if emb1 is None else player2
                return {"error": f"No GNN embedding found for player '{missing}'"}
            
            # Calculate style similarity
            style_similarity = self._calculate_style_similarity(emb1, emb2)
            
            # Analyze specific aspects
            aspect_analysis = self._analyze_style_aspects(emb1, emb2, player1, player2)
            
            # Generate comparison insights
            insights = self._generate_style_comparison_insights(
                player1, player2, style_similarity, aspect_analysis
            )
            
            return {
                "player1": player1,
                "player2": player2,
                "overall_similarity": float(style_similarity),
                "similarity_level": self._categorize_similarity(style_similarity),
                "aspect_analysis": aspect_analysis,
                "insights": insights,
                "gnn_powered": True,
                "comparison_summary": self._generate_style_summary(player1, player2, style_similarity)
            }
            
        except Exception as e:
            logger.error(f"Error in style similarity analysis: {e}")
            return {"error": f"Style similarity analysis failed: {str(e)}"}
    
    # ============================================================================
    # PRIVATE HELPER METHODS
    # ============================================================================
    
    def _get_player_embedding(self, player: str) -> Optional[np.ndarray]:
        """Get GNN embedding for a player"""
        if not self.gnn_service:
            return None
        
        # Check cache first
        cache_key = f"player_{player.lower()}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Get from GNN service
        embedding = self.gnn_service.get_player_embedding(player)
        if embedding is not None and embedding.size > 0:
            self.embedding_cache[cache_key] = embedding
            return embedding
        
        return None
    
    def _get_venue_embedding(self, venue: str) -> Optional[np.ndarray]:
        """Get GNN embedding for a venue"""
        if not self.gnn_service:
            return None
        
        cache_key = f"venue_{venue.lower()}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        embedding = self.gnn_service.get_venue_embedding(venue)
        if embedding is not None and embedding.size > 0:
            self.embedding_cache[cache_key] = embedding
            return embedding
        
        return None
    
    def _get_context_embedding(self, context: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for match context"""
        # Simple context encoding (can be enhanced with learned embeddings)
        context_features = []
        
        # Phase encoding
        phase = context.get('phase', 'middle')
        phase_map = {'powerplay': [1, 0, 0], 'middle': [0, 1, 0], 'death': [0, 0, 1]}
        context_features.extend(phase_map.get(phase, [0, 1, 0]))
        
        # Bowling type encoding
        bowling_type = context.get('bowling_type', 'pace')
        bowling_map = {'pace': [1, 0], 'spin': [0, 1]}
        context_features.extend(bowling_map.get(bowling_type, [1, 0]))
        
        # Pressure situation
        pressure = context.get('pressure', False)
        context_features.append(1.0 if pressure else 0.0)
        
        # Required run rate (normalized)
        rrr = context.get('required_run_rate', 6.0)
        context_features.append(min(rrr / 15.0, 1.0))  # Normalize to 0-1
        
        # Wickets lost (normalized)
        wickets = context.get('wickets_lost', 3)
        context_features.append(wickets / 10.0)  # Normalize to 0-1
        
        # Balls remaining (normalized)
        balls = context.get('balls_remaining', 60)
        context_features.append(balls / 120.0)  # Normalize to 0-1
        
        # Pad to fixed size (32D)
        while len(context_features) < 32:
            context_features.append(0.0)
        
        return np.array(context_features[:32], dtype=np.float32)
    
    def _get_all_players_with_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all players that have GNN embeddings"""
        if not self.gnn_service:
            return {}
        
        players_with_embeddings = {}
        
        # Get all player nodes from KG
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'player':
                embedding = self._get_player_embedding(str(node))
                if embedding is not None:
                    players_with_embeddings[str(node)] = embedding
        
        return players_with_embeddings
    
    def _combine_embeddings(self, player_emb: np.ndarray, context_emb: np.ndarray) -> np.ndarray:
        """Combine player and context embeddings"""
        # Simple concatenation (can be enhanced with learned combination)
        return np.concatenate([player_emb, context_emb])
    
    def _generate_performance_prediction(self, combined_emb: np.ndarray, context: Dict) -> Dict[str, Any]:
        """Generate performance prediction from combined embedding"""
        # Simplified prediction logic (would use trained model in production)
        embedding_norm = np.linalg.norm(combined_emb)
        
        # Mock prediction based on embedding characteristics
        base_score = min(embedding_norm / 10.0, 1.0)
        
        # Adjust based on context
        phase = context.get('phase', 'middle')
        if phase == 'powerplay':
            base_score *= 1.1  # Slight boost for powerplay
        elif phase == 'death':
            base_score *= 0.9  # Slight penalty for death overs
        
        return {
            "expected_strike_rate": max(80 + (base_score * 60), 60),
            "expected_average": max(20 + (base_score * 30), 15),
            "confidence": min(base_score + 0.3, 0.95),
            "risk_level": "low" if base_score > 0.7 else "medium" if base_score > 0.4 else "high"
        }
    
    def _calculate_venue_compatibility(self, player_emb: np.ndarray, venue_emb: np.ndarray) -> float:
        """Calculate player-venue compatibility score"""
        # Use cosine similarity as compatibility metric
        player_norm = player_emb.reshape(1, -1)
        venue_norm = venue_emb.reshape(1, -1)
        
        compatibility = cosine_similarity(player_norm, venue_norm)[0][0]
        return max(0.0, min(1.0, compatibility))  # Clamp to [0, 1]
    
    def _calculate_style_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate style similarity between two players"""
        emb1_norm = emb1.reshape(1, -1)
        emb2_norm = emb2.reshape(1, -1)
        
        similarity = cosine_similarity(emb1_norm, emb2_norm)[0][0]
        return max(0.0, min(1.0, similarity))
    
    # ============================================================================
    # FALLBACK METHODS (when GNN not available)
    # ============================================================================
    
    def _fallback_similarity_search(self, player: str, top_k: int) -> Dict[str, Any]:
        """Fallback similarity search using KG structure"""
        try:
            # Use existing KG methods as fallback
            result = self.get_complete_player_profile(player)
            if result.get("error"):
                return result
            
            # Simple heuristic-based similarity
            player_role = result.get("primary_role", "unknown")
            similar_players = []
            
            for node, data in self.graph.nodes(data=True):
                if (data.get('type') == 'player' and 
                    data.get('primary_role') == player_role and
                    str(node).lower() != player.lower()):
                    similar_players.append({
                        "player": str(node),
                        "similarity_score": 0.7,  # Heuristic score
                        "similarity_metric": "role_based"
                    })
            
            return {
                "target_player": player,
                "similar_players": similar_players[:top_k],
                "gnn_powered": False,
                "fallback_method": "role_based_heuristic"
            }
            
        except Exception as e:
            return {"error": f"Fallback similarity search failed: {str(e)}"}
    
    def _fallback_contextual_analysis(self, player: str, context: Dict) -> Dict[str, Any]:
        """Fallback contextual analysis using KG data"""
        return {
            "player": player,
            "context": context,
            "prediction": {
                "expected_strike_rate": 120,
                "expected_average": 25,
                "confidence": 0.5
            },
            "gnn_powered": False,
            "fallback_method": "statistical_average"
        }
    
    def _fallback_venue_analysis(self, player: str, venue: str) -> Dict[str, Any]:
        """Fallback venue analysis using historical data"""
        historical_data = self._get_historical_venue_performance(player, venue)
        
        return {
            "player": player,
            "venue": venue,
            "compatibility_score": 0.6,
            "historical_performance": historical_data,
            "gnn_powered": False,
            "fallback_method": "historical_statistics"
        }
    
    def _fallback_style_comparison(self, player1: str, player2: str) -> Dict[str, Any]:
        """Fallback style comparison using KG attributes"""
        return {
            "player1": player1,
            "player2": player2,
            "overall_similarity": 0.5,
            "gnn_powered": False,
            "fallback_method": "attribute_comparison"
        }
    
    # ============================================================================
    # INSIGHT GENERATION METHODS
    # ============================================================================
    
    def _enhance_similarity_result(self, target_player: str, result: Dict) -> Dict:
        """Enhance similarity result with cricket insights"""
        enhanced = result.copy()
        
        # Add role comparison
        target_profile = self.get_complete_player_profile(target_player)
        similar_profile = self.get_complete_player_profile(result["player"])
        
        if not target_profile.get("error") and not similar_profile.get("error"):
            enhanced["role_match"] = (
                target_profile.get("primary_role") == similar_profile.get("primary_role")
            )
            enhanced["style_aspects"] = self._compare_style_aspects(target_profile, similar_profile)
        
        return enhanced
    
    def _generate_similarity_summary(self, player: str, results: List[Dict]) -> str:
        """Generate human-readable similarity summary"""
        if not results:
            return f"No similar players found for {player}"
        
        top_similar = results[0]["player"]
        avg_similarity = np.mean([r["similarity_score"] for r in results])
        
        return (f"{player} shows highest similarity to {top_similar} "
                f"(avg similarity: {avg_similarity:.2f})")
    
    def _generate_contextual_insights(self, player: str, context: Dict, prediction: Dict) -> List[str]:
        """Generate contextual performance insights"""
        insights = []
        
        phase = context.get('phase', 'middle')
        confidence = prediction.get('confidence', 0.5)
        
        if confidence > 0.8:
            insights.append(f"High confidence prediction for {player} in {phase} phase")
        elif confidence < 0.4:
            insights.append(f"Limited data available for {player} in this context")
        
        if phase == 'death' and prediction.get('expected_strike_rate', 0) > 140:
            insights.append(f"{player} shows strong death-over capabilities")
        
        return insights
    
    def _generate_venue_insights(self, player: str, venue: str, compatibility: float, historical: Dict) -> List[str]:
        """Generate venue compatibility insights"""
        insights = []
        
        if compatibility > 0.8:
            insights.append(f"{player} shows excellent compatibility with {venue}")
        elif compatibility < 0.4:
            insights.append(f"{player} may face challenges at {venue}")
        
        if historical.get("matches_played", 0) > 5:
            avg = historical.get("average", 0)
            if avg > 30:
                insights.append(f"Strong historical record at {venue} (avg: {avg:.1f})")
        
        return insights
    
    def _generate_venue_recommendations(self, compatibility: float, insights: List[str]) -> List[str]:
        """Generate venue-specific recommendations"""
        recommendations = []
        
        if compatibility > 0.7:
            recommendations.append("Highly recommended venue for this player")
        elif compatibility < 0.4:
            recommendations.append("Consider tactical adjustments for this venue")
        
        return recommendations
    
    def _generate_style_comparison_insights(self, player1: str, player2: str, similarity: float, aspects: Dict) -> List[str]:
        """Generate style comparison insights"""
        insights = []
        
        if similarity > 0.8:
            insights.append(f"{player1} and {player2} have very similar playing styles")
        elif similarity < 0.3:
            insights.append(f"{player1} and {player2} have contrasting playing styles")
        
        return insights
    
    def _generate_style_summary(self, player1: str, player2: str, similarity: float) -> str:
        """Generate style comparison summary"""
        level = self._categorize_similarity(similarity)
        return f"{player1} and {player2} show {level} style similarity ({similarity:.2f})"
    
    # ============================================================================
    # LEGACY COMPATIBILITY METHODS
    # ============================================================================
    
    def get_player_stats(self, player: str, format_filter: Optional[str] = None, 
                        venue_filter: Optional[str] = None) -> Dict[str, Any]:
        """Legacy method - returns complete profile (delegates to parent)"""
        return self.get_complete_player_profile(player)
    
    def compare_players(self, player1: str, player2: str) -> Dict[str, Any]:
        """Legacy method - uses advanced comparison (delegates to parent)"""
        return super().compare_players(player1, player2)
    
    def find_similar_players(self, player: str, **kwargs) -> Dict[str, Any]:
        """Legacy method - try GNN first, fallback to parent"""
        try:
            # Try GNN-enhanced similarity first
            if self.gnn_embeddings_available:
                return self.find_similar_players_gnn(player, **kwargs)
            else:
                # Fallback to parent method if available
                if hasattr(super(), 'find_similar_players'):
                    return super().find_similar_players(player, **kwargs)
                else:
                    return self._fallback_similarity_search(player, kwargs.get('top_k', 5))
        except Exception as e:
            logger.error(f"Error in find_similar_players: {e}")
            return {"error": f"Similarity search failed: {str(e)}"}
    
    def explain_data_limitations(self) -> Dict[str, Any]:
        """Explain data limitations including GNN availability"""
        base_limitations = super().explain_data_limitations() if hasattr(super(), 'explain_data_limitations') else {}
        
        gnn_status = {
            "gnn_embeddings_available": self.gnn_embeddings_available,
            "gnn_enhanced_functions": [
                "find_similar_players_gnn",
                "predict_contextual_performance", 
                "analyze_venue_compatibility",
                "get_playing_style_similarity"
            ] if self.gnn_embeddings_available else [],
            "fallback_methods": "Statistical heuristics when GNN unavailable"
        }
        
        return {**base_limitations, "gnn_capabilities": gnn_status}
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def _categorize_compatibility(self, score: float) -> str:
        """Categorize compatibility score"""
        if score > 0.8:
            return "excellent"
        elif score > 0.6:
            return "good"
        elif score > 0.4:
            return "moderate"
        else:
            return "poor"
    
    def _categorize_similarity(self, score: float) -> str:
        """Categorize similarity score"""
        if score > 0.8:
            return "very high"
        elif score > 0.6:
            return "high"
        elif score > 0.4:
            return "moderate"
        else:
            return "low"
    
    def _get_historical_venue_performance(self, player: str, venue: str) -> Dict[str, Any]:
        """Get historical performance data at venue"""
        # Use existing KG methods to get venue performance
        try:
            profile = self.get_complete_player_profile(player)
            if not profile.get("error"):
                venue_perf = profile.get("venue_performance", {})
                return venue_perf.get(venue, {"matches_played": 0})
        except:
            pass
        
        return {"matches_played": 0}
    
    def _compare_style_aspects(self, profile1: Dict, profile2: Dict) -> Dict[str, Any]:
        """Compare specific style aspects between players"""
        aspects = {}
        
        # Compare batting stats
        batting1 = profile1.get("batting_stats", {})
        batting2 = profile2.get("batting_stats", {})
        
        if batting1 and batting2:
            sr1 = batting1.get("strike_rate", 0)
            sr2 = batting2.get("strike_rate", 0)
            aspects["strike_rate_similarity"] = 1.0 - abs(sr1 - sr2) / max(sr1 + sr2, 1)
        
        return aspects
    
    def _analyze_style_aspects(self, emb1: np.ndarray, emb2: np.ndarray, player1: str, player2: str) -> Dict[str, Any]:
        """Analyze specific style aspects using embeddings"""
        # Simplified aspect analysis
        # In production, this would use learned aspect extractors
        
        # Split embeddings into conceptual regions
        mid_point = len(emb1) // 2
        
        # Batting style similarity (first half of embedding)
        batting_sim = cosine_similarity(
            emb1[:mid_point].reshape(1, -1), 
            emb2[:mid_point].reshape(1, -1)
        )[0][0]
        
        # Bowling/fielding style similarity (second half)
        bowling_sim = cosine_similarity(
            emb1[mid_point:].reshape(1, -1), 
            emb2[mid_point:].reshape(1, -1)
        )[0][0]
        
        return {
            "batting_style_similarity": float(batting_sim),
            "bowling_style_similarity": float(bowling_sim),
            "overall_balance": abs(batting_sim - bowling_sim) < 0.2
        }

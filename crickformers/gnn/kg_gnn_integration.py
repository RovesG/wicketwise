# Purpose: Integration module for enhanced KG-GNN with existing training pipeline
# Author: WicketWise Team, Last Modified: 2025-01-18

"""
Integration module that connects the enhanced KG-GNN with the existing
WicketWise training pipeline and model architecture.
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import pickle

from .enhanced_kg_gnn import EnhancedKGGNNTrainer, EnhancedFeatureConfig, EnhancedCricketGNN
from ..model.crickformer_model import CrickformerModel
from ..inference.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class KGGNNEmbeddingService(EmbeddingService):
    """
    Enhanced embedding service that uses the comprehensive KG-GNN
    for player and situational embeddings.
    """
    
    def __init__(self, 
                 kg_path: str,
                 gnn_model_path: Optional[str] = None,
                 feature_config: Optional[EnhancedFeatureConfig] = None):
        super().__init__()
        
        self.kg_path = kg_path
        self.gnn_model_path = gnn_model_path
        self.feature_config = feature_config or EnhancedFeatureConfig()
        
        # Load knowledge graph and model
        self.kg = None
        self.gnn_model = None
        self.gnn_trainer = None
        self.player_embeddings = {}
        self.venue_embeddings = {}
        
        self._load_kg_and_model()
    
    def _load_kg_and_model(self):
        """Load the knowledge graph and trained GNN model"""
        try:
            # Load knowledge graph
            with open(self.kg_path, 'rb') as f:
                self.kg = pickle.load(f)
            logger.info(f"Loaded KG: {self.kg.number_of_nodes()} nodes")
            
            # Initialize GNN trainer
            self.gnn_trainer = EnhancedKGGNNTrainer(
                unified_kg=self.kg,
                feature_config=self.feature_config
            )
            
            # Load or train model
            if self.gnn_model_path and Path(self.gnn_model_path).exists():
                self._load_trained_model()
            else:
                logger.info("No trained model found, training new model...")
                self._train_new_model()
            
        except Exception as e:
            logger.error(f"Failed to load KG and model: {e}")
            raise
    
    def _load_trained_model(self):
        """Load a pre-trained GNN model"""
        try:
            checkpoint = torch.load(self.gnn_model_path, map_location='cpu')
            
            # Prepare data to get feature dimensions
            data, node_feature_dims = self.gnn_trainer.prepare_hetero_data()
            
            # Create and load model
            self.gnn_model = self.gnn_trainer.create_model(node_feature_dims)
            self.gnn_model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info("Loaded pre-trained GNN model")
            
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            self._train_new_model()
    
    def _train_new_model(self):
        """Train a new GNN model"""
        logger.info("Training new enhanced KG-GNN model...")
        results = self.gnn_trainer.train(num_epochs=100)
        
        self.gnn_model = results['model']
        self.data = results['data']
        
        # Save the trained model
        if self.gnn_model_path:
            self._save_model(results)
        
        logger.info("New GNN model trained successfully")
    
    def _save_model(self, training_results: Dict[str, Any]):
        """Save the trained model"""
        try:
            checkpoint = {
                'model_state_dict': training_results['model'].state_dict(),
                'node_feature_dims': training_results['node_feature_dims'],
                'training_losses': training_results['training_losses'],
                'feature_config': self.feature_config.__dict__
            }
            
            torch.save(checkpoint, self.gnn_model_path)
            logger.info(f"Model saved to {self.gnn_model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def get_player_embedding(self, player_name: str, 
                           context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Get enhanced player embedding that includes situational context.
        
        Args:
            player_name: Name of the player
            context: Situational context (powerplay, venue, pressure, etc.)
            
        Returns:
            Enhanced player embedding vector
        """
        if self.gnn_model is None:
            logger.error("GNN model not loaded")
            return np.zeros(128)  # Default embedding size
        
        try:
            # Generate embeddings if not cached
            if not self.player_embeddings:
                self._generate_all_embeddings()
            
            if player_name in self.player_embeddings:
                base_embedding = self.player_embeddings[player_name]
                
                # Add contextual information if provided
                if context:
                    context_embedding = self._get_context_embedding(context)
                    # Combine base and context embeddings
                    enhanced_embedding = np.concatenate([base_embedding, context_embedding])
                    return enhanced_embedding
                else:
                    return base_embedding
            else:
                logger.warning(f"Player {player_name} not found in embeddings")
                return np.zeros(128)
                
        except Exception as e:
            logger.error(f"Error getting player embedding: {e}")
            return np.zeros(128)
    
    def get_venue_embedding(self, venue_name: str) -> np.ndarray:
        """Get venue embedding from the GNN"""
        if not self.venue_embeddings:
            self._generate_all_embeddings()
        
        return self.venue_embeddings.get(venue_name, np.zeros(64))
    
    def _generate_all_embeddings(self):
        """Generate embeddings for all players and venues"""
        if self.gnn_model is None:
            return
        
        self.gnn_model.eval()
        with torch.no_grad():
            if not hasattr(self, 'data'):
                self.data, _ = self.gnn_trainer.prepare_hetero_data()
            
            results = self.gnn_model(self.data)
            embeddings = results['embeddings']
            
            # Extract player embeddings
            if 'player' in embeddings:
                player_nodes = [node for node, attrs in self.kg.nodes(data=True) 
                               if attrs.get('type') == 'player']
                player_embs = embeddings['player'].cpu().numpy()
                
                for i, player in enumerate(player_nodes):
                    if i < len(player_embs):
                        self.player_embeddings[player] = player_embs[i]
            
            # Extract venue embeddings
            if 'venue' in embeddings:
                venue_nodes = [node for node, attrs in self.kg.nodes(data=True) 
                              if attrs.get('type') == 'venue']
                venue_embs = embeddings['venue'].cpu().numpy()
                
                for i, venue in enumerate(venue_nodes):
                    if i < len(venue_embs):
                        self.venue_embeddings[venue] = venue_embs[i]
        
        logger.info(f"Generated embeddings for {len(self.player_embeddings)} players "
                   f"and {len(self.venue_embeddings)} venues")
    
    def _get_context_embedding(self, context: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for situational context"""
        features = []
        
        # Phase context
        phase = context.get('phase', 'middle')
        features.extend([
            1.0 if phase == 'powerplay' else 0.0,
            1.0 if phase == 'middle' else 0.0,
            1.0 if phase == 'death' else 0.0
        ])
        
        # Bowling type
        bowling_type = context.get('bowling_type', 'pace')
        features.extend([
            1.0 if bowling_type == 'pace' else 0.0,
            1.0 if bowling_type == 'spin' else 0.0
        ])
        
        # Pressure situation
        features.append(1.0 if context.get('pressure', False) else 0.0)
        
        # Match situation
        features.extend([
            context.get('required_run_rate', 6.0) / 15.0,
            context.get('wickets_lost', 3) / 10.0,
            context.get('balls_remaining', 60) / 120.0,
            context.get('target_score', 160) / 250.0
        ])
        
        # Pad to 32 dimensions for context
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32], dtype=np.float32)
    
    def predict_performance(self, player_name: str, 
                          context: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict player performance in given context using enhanced embeddings.
        
        Args:
            player_name: Name of the player
            context: Situational context
            
        Returns:
            Performance predictions
        """
        if self.gnn_model is None:
            return {'performance_score': 0.5}
        
        try:
            # Get player embedding with context
            player_embedding = self.get_player_embedding(player_name, context)
            
            # Convert to tensor
            player_tensor = torch.tensor(player_embedding[:128], dtype=torch.float32).unsqueeze(0)
            context_tensor = torch.tensor(self._get_context_embedding(context), dtype=torch.float32).unsqueeze(0)
            
            # Predict using the model
            self.gnn_model.eval()
            with torch.no_grad():
                # Create input for performance predictor
                combined_input = torch.cat([player_tensor, context_tensor], dim=1)
                performance_score = torch.sigmoid(self.gnn_model.performance_predictor(combined_input))
            
            return {
                'performance_score': performance_score.item(),
                'confidence': 0.8  # Could be computed from model uncertainty
            }
            
        except Exception as e:
            logger.error(f"Error predicting performance: {e}")
            return {'performance_score': 0.5, 'confidence': 0.5}
    
    def predict_matchup(self, batter: str, bowler: str, 
                       context: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict batter vs bowler matchup using comprehensive KG features.
        
        Args:
            batter: Batter name
            bowler: Bowler name
            context: Match context
            
        Returns:
            Matchup favorability predictions
        """
        if self.gnn_model is None:
            return {'favorable': 0.33, 'neutral': 0.34, 'unfavorable': 0.33}
        
        try:
            # Get embeddings
            batter_emb = self.get_player_embedding(batter, context)
            bowler_emb = self.get_player_embedding(bowler, context)
            
            # Convert to tensors
            batter_tensor = torch.tensor(batter_emb[:128], dtype=torch.float32).unsqueeze(0)
            bowler_tensor = torch.tensor(bowler_emb[:128], dtype=torch.float32).unsqueeze(0)
            context_tensor = torch.tensor(self._get_context_embedding(context), dtype=torch.float32).unsqueeze(0)
            
            # Predict matchup
            self.gnn_model.eval()
            with torch.no_grad():
                matchup_probs = self.gnn_model.predict_matchup(batter_tensor, bowler_tensor, context_tensor)
                probs = matchup_probs[0].tolist()
            
            return {
                'favorable': probs[0],
                'neutral': probs[1], 
                'unfavorable': probs[2]
            }
            
        except Exception as e:
            logger.error(f"Error predicting matchup: {e}")
            return {'favorable': 0.33, 'neutral': 0.34, 'unfavorable': 0.33}


class EnhancedCrickformerModel(CrickformerModel):
    """
    Enhanced Crickformer model that integrates KG-GNN embeddings
    for superior contextual understanding.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize KG-GNN embedding service
        kg_path = config.get('kg_path', 'models/unified_cricket_kg.pkl')
        gnn_model_path = config.get('gnn_model_path', 'models/enhanced_kg_gnn.pth')
        
        self.kg_embedding_service = KGGNNEmbeddingService(
            kg_path=kg_path,
            gnn_model_path=gnn_model_path
        )
        
        # Enhanced embedding dimensions
        self.player_emb_dim = 160  # 128 base + 32 context
        self.venue_emb_dim = 64
        
        # Update embedding layers
        self._update_embedding_layers()
    
    def _update_embedding_layers(self):
        """Update embedding layers to use KG-GNN embeddings"""
        # Replace player embedding layer
        self.player_embedding = nn.Identity()  # Will use KG embeddings directly
        
        # Replace venue embedding layer  
        self.venue_embedding = nn.Identity()  # Will use KG embeddings directly
        
        # Update input projection to handle new embedding dimensions
        self.input_projection = nn.Linear(
            self.player_emb_dim + self.venue_emb_dim + self.config.get('context_dim', 64),
            self.config.get('hidden_dim', 512)
        )
    
    def get_embeddings(self, batch_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Get enhanced embeddings using KG-GNN"""
        embeddings = {}
        
        # Get player embeddings with context
        if 'players' in batch_data and 'context' in batch_data:
            player_embeddings = []
            
            for i, player in enumerate(batch_data['players']):
                # Extract context for this sample
                context = {
                    'phase': batch_data['context'].get('phase', ['middle'])[i],
                    'bowling_type': batch_data['context'].get('bowling_type', ['pace'])[i],
                    'pressure': batch_data['context'].get('pressure', [False])[i],
                    'required_run_rate': batch_data['context'].get('required_run_rate', [6.0])[i],
                    'wickets_lost': batch_data['context'].get('wickets_lost', [3])[i],
                    'balls_remaining': batch_data['context'].get('balls_remaining', [60])[i]
                }
                
                player_emb = self.kg_embedding_service.get_player_embedding(player, context)
                player_embeddings.append(player_emb)
            
            embeddings['players'] = torch.tensor(np.stack(player_embeddings), dtype=torch.float32)
        
        # Get venue embeddings
        if 'venues' in batch_data:
            venue_embeddings = []
            for venue in batch_data['venues']:
                venue_emb = self.kg_embedding_service.get_venue_embedding(venue)
                venue_embeddings.append(venue_emb)
            
            embeddings['venues'] = torch.tensor(np.stack(venue_embeddings), dtype=torch.float32)
        
        return embeddings
    
    def forward(self, batch_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced KG-GNN embeddings"""
        # Get enhanced embeddings
        embeddings = self.get_embeddings(batch_data)
        
        # Combine embeddings
        combined_input = []
        if 'players' in embeddings:
            combined_input.append(embeddings['players'])
        if 'venues' in embeddings:
            combined_input.append(embeddings['venues'])
        
        # Add other contextual features
        if 'context_features' in batch_data:
            combined_input.append(batch_data['context_features'])
        
        if combined_input:
            input_tensor = torch.cat(combined_input, dim=-1)
        else:
            # Fallback to parent implementation
            return super().forward(batch_data)
        
        # Project to hidden dimension
        hidden = self.input_projection(input_tensor)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        
        # Generate predictions
        predictions = self.output_head(hidden)
        
        return {
            'predictions': predictions,
            'embeddings': embeddings,
            'hidden_states': hidden
        }

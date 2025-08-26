# Purpose: Enhanced GNN architecture for comprehensive cricket knowledge graph
# Author: WicketWise Team, Last Modified: 2025-01-18

"""
Enhanced GNN that leverages the comprehensive situational statistics from the
unified knowledge graph, including powerplay/death overs, pace vs spin,
pressure situations, and venue-specific performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnhancedFeatureConfig:
    """Configuration for enhanced feature extraction from KG"""
    use_situational_stats: bool = True
    use_venue_performance: bool = True
    use_pressure_stats: bool = True
    use_bowling_matchups: bool = True
    feature_dim: int = 128
    normalize_features: bool = True


class SituationalFeatureExtractor:
    """
    Extracts rich features from the comprehensive knowledge graph built
    by UnifiedKGBuilder, leveraging all the advanced analytics.
    """
    
    def __init__(self, config: EnhancedFeatureConfig = None):
        self.config = config or EnhancedFeatureConfig()
        
    def extract_player_features(self, player_name: str, node_attrs: Dict[str, Any]) -> np.ndarray:
        """
        Extract comprehensive features for a player node from the KG.
        
        Features include:
        1. Basic batting/bowling stats
        2. Powerplay vs death overs performance 
        3. Pace vs spin performance
        4. Pressure situation performance
        5. Venue-specific aggregated performance
        """
        features = []
        
        # === BASIC BATTING STATS ===
        batting_stats = node_attrs.get('batting_stats', {})
        features.extend([
            batting_stats.get('runs', 0) / 10000,  # Normalized
            batting_stats.get('balls', 0) / 10000,
            batting_stats.get('average', 0) / 100,
            batting_stats.get('fours', 0) / 1000,
            batting_stats.get('sixes', 0) / 500
        ])
        
        # === BASIC BOWLING STATS ===
        bowling_stats = node_attrs.get('bowling_stats', {})
        features.extend([
            bowling_stats.get('balls', 0) / 10000,
            bowling_stats.get('runs', 0) / 10000, 
            bowling_stats.get('wickets', 0) / 1000,
            bowling_stats.get('economy', 0) / 15,  # Typical T20 economy
            bowling_stats.get('strike_rate', 0) / 30  # Typical bowling SR
        ])
        
        # === SITUATIONAL PERFORMANCE ===
        if self.config.use_situational_stats:
            # Powerplay performance
            powerplay = node_attrs.get('in_powerplay', {})
            features.extend([
                powerplay.get('runs', 0) / 1000,
                powerplay.get('balls', 0) / 1000,
                powerplay.get('strike_rate', 0) / 200,
                powerplay.get('fours', 0) / 100,
                powerplay.get('sixes', 0) / 50
            ])
            
            # Death overs performance
            death_overs = node_attrs.get('in_death_overs', {})
            features.extend([
                death_overs.get('runs', 0) / 1000,
                death_overs.get('balls', 0) / 1000,
                death_overs.get('strike_rate', 0) / 200,
                death_overs.get('fours', 0) / 100,
                death_overs.get('sixes', 0) / 50
            ])
        else:
            features.extend([0.0] * 10)  # Placeholder
            
        # === BOWLING MATCHUP PERFORMANCE ===
        if self.config.use_bowling_matchups:
            # vs Pace bowling
            vs_pace = node_attrs.get('vs_pace', {})
            features.extend([
                vs_pace.get('runs', 0) / 5000,
                vs_pace.get('balls', 0) / 5000,
                vs_pace.get('average', 0) / 50,
                vs_pace.get('strike_rate', 0) / 200
            ])
            
            # vs Spin bowling
            vs_spin = node_attrs.get('vs_spin', {})
            features.extend([
                vs_spin.get('runs', 0) / 5000,
                vs_spin.get('balls', 0) / 5000,
                vs_spin.get('average', 0) / 50,
                vs_spin.get('strike_rate', 0) / 200
            ])
        else:
            features.extend([0.0] * 8)  # Placeholder
            
        # === PRESSURE SITUATION PERFORMANCE ===
        if self.config.use_pressure_stats:
            under_pressure = node_attrs.get('under_pressure', {})
            features.extend([
                under_pressure.get('runs', 0) / 2000,
                under_pressure.get('balls', 0) / 2000,
                under_pressure.get('average', 0) / 50,
                under_pressure.get('strike_rate', 0) / 200,
                under_pressure.get('fours', 0) / 50,
                under_pressure.get('sixes', 0) / 25
            ])
        else:
            features.extend([0.0] * 6)  # Placeholder
            
        # === VENUE PERFORMANCE AGGREGATION ===
        if self.config.use_venue_performance:
            by_venue = node_attrs.get('by_venue', {})
            if by_venue:
                # Aggregate venue performance
                total_venue_runs = sum(v.get('runs', 0) for v in by_venue.values())
                total_venue_balls = sum(v.get('balls', 0) for v in by_venue.values())
                avg_venue_strike_rate = np.mean([v.get('strike_rate', 0) for v in by_venue.values()])
                num_venues = len(by_venue)
                
                features.extend([
                    total_venue_runs / 10000,
                    total_venue_balls / 10000,
                    avg_venue_strike_rate / 200,
                    num_venues / 100  # Number of venues played
                ])
            else:
                features.extend([0.0] * 4)
        else:
            features.extend([0.0] * 4)
            
        # === ROLE EMBEDDINGS (if available) ===
        if 'role_embedding' in node_attrs:
            role_emb = node_attrs['role_embedding']
            if isinstance(role_emb, np.ndarray):
                features.extend(role_emb.tolist())
            else:
                features.extend([0.0] * 16)  # Default role embedding size
        else:
            features.extend([0.0] * 16)
            
        # === STYLE EMBEDDINGS (if available) ===
        if 'style_embedding' in node_attrs:
            style_emb = node_attrs['style_embedding'] 
            if isinstance(style_emb, np.ndarray):
                features.extend(style_emb.tolist())
            else:
                features.extend([0.0] * 32)  # Default style embedding size
        else:
            features.extend([0.0] * 32)
            
        # Ensure consistent feature dimension
        target_dim = self.config.feature_dim
        if len(features) > target_dim:
            features = features[:target_dim]
        elif len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
            
        feature_array = np.array(features, dtype=np.float32)
        
        # Normalize if requested
        if self.config.normalize_features:
            # Apply min-max normalization to prevent any single feature from dominating
            feature_array = np.clip(feature_array, -5.0, 5.0)  # Clip outliers
            
        return feature_array
    
    def extract_venue_features(self, venue_name: str, node_attrs: Dict[str, Any]) -> np.ndarray:
        """Extract features for venue nodes"""
        features = []
        
        # Basic venue stats
        features.extend([
            node_attrs.get('total_matches', 0) / 1000,
            node_attrs.get('avg_score', 0) / 200,
            node_attrs.get('boundary_percentage', 0) / 100,
            node_attrs.get('six_percentage', 0) / 100
        ])
        
        # Pad to consistent size
        while len(features) < 32:
            features.append(0.0)
            
        return np.array(features[:32], dtype=np.float32)
    
    def extract_match_features(self, match_id: str, node_attrs: Dict[str, Any]) -> np.ndarray:
        """Extract features for match nodes"""
        features = []
        
        # Basic match info
        features.extend([
            node_attrs.get('total_runs', 0) / 400,
            node_attrs.get('total_wickets', 0) / 20,
            node_attrs.get('total_boundaries', 0) / 50,
            node_attrs.get('match_duration', 0) / 240  # Minutes
        ])
        
        # Pad to consistent size
        while len(features) < 16:
            features.append(0.0)
            
        return np.array(features[:16], dtype=np.float32)


class EnhancedCricketGNN(nn.Module):
    """
    Enhanced GNN architecture that leverages comprehensive situational statistics
    from the unified knowledge graph for superior cricket analytics.
    
    Key improvements:
    1. Rich feature extraction from situational stats
    2. Multi-head attention for complex relationships
    3. Hierarchical message passing (local -> global)
    4. Situational context awareness
    """
    
    def __init__(self, 
                 node_feature_dims: Dict[str, int],
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_attention: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Input projection layers for different node types
        self.input_projections = nn.ModuleDict()
        for node_type, input_dim in node_feature_dims.items():
            self.input_projections[node_type] = nn.Linear(input_dim, hidden_dim)
        
        # Multi-layer GNN architecture
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if use_attention:
                # Use GAT for attention-based message passing
                conv_dict = {}
                for node_type in node_feature_dims.keys():
                    for target_type in node_feature_dims.keys():
                        edge_type = (node_type, f"interacts_with", target_type)
                        conv_dict[edge_type] = GATv2Conv(
                            hidden_dim, hidden_dim // num_heads, 
                            heads=num_heads, dropout=dropout, concat=True
                        )
                
                self.gnn_layers.append(HeteroConv(conv_dict, aggr='mean'))
            else:
                # Use SAGE for scalable message passing
                conv_dict = {}
                for node_type in node_feature_dims.keys():
                    for target_type in node_feature_dims.keys():
                        edge_type = (node_type, f"interacts_with", target_type)
                        conv_dict[edge_type] = SAGEConv(hidden_dim, hidden_dim)
                
                self.gnn_layers.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Output projection layers
        self.output_projections = nn.ModuleDict()
        for node_type in node_feature_dims.keys():
            self.output_projections[node_type] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim)
            )
        
        # Situational context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)
        )
        
        # Prediction heads for different tasks
        self.performance_predictor = nn.Sequential(
            nn.Linear(output_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Performance score
        )
        
        self.matchup_predictor = nn.Sequential(
            nn.Linear(output_dim * 2 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 3)  # Favorable, Neutral, Unfavorable
        )
        
    def forward(self, data: HeteroData, 
                context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through enhanced cricket GNN.
        
        Args:
            data: HeteroData containing node features and edge indices
            context: Optional situational context (powerplay, death overs, etc.)
            
        Returns:
            Dictionary containing embeddings and predictions for each node type
        """
        # Project input features to hidden dimension
        x_dict = {}
        for node_type, features in data.x_dict.items():
            x_dict[node_type] = self.input_projections[node_type](features)
        
        # Multi-layer message passing
        for gnn_layer in self.gnn_layers:
            x_dict = gnn_layer(x_dict, data.edge_index_dict)
            
            # Apply activation and normalization
            for node_type in x_dict:
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.layer_norm(x_dict[node_type], [self.hidden_dim])
        
        # Generate final embeddings
        embeddings = {}
        for node_type, features in x_dict.items():
            embeddings[node_type] = self.output_projections[node_type](features)
        
        # Encode situational context
        if context is not None:
            context_emb = self.context_encoder(context)
        else:
            # Default context
            context_emb = torch.zeros(embeddings['player'].size(0), 64, 
                                    device=embeddings['player'].device)
        
        # Generate predictions
        results = {
            'embeddings': embeddings,
            'context': context_emb
        }
        
        # Performance prediction for players
        if 'player' in embeddings:
            player_context = context_emb[:embeddings['player'].size(0)]
            performance_input = torch.cat([embeddings['player'], player_context], dim=1)
            results['performance_scores'] = self.performance_predictor(performance_input)
        
        return results
    
    def predict_matchup(self, player1_emb: torch.Tensor, player2_emb: torch.Tensor,
                       context: torch.Tensor) -> torch.Tensor:
        """
        Predict matchup favorability between two players.
        
        Args:
            player1_emb: Embedding of first player (batter)
            player2_emb: Embedding of second player (bowler)
            context: Situational context
            
        Returns:
            Matchup prediction probabilities
        """
        matchup_input = torch.cat([player1_emb, player2_emb, context], dim=1)
        return F.softmax(self.matchup_predictor(matchup_input), dim=1)


class EnhancedKGGNNTrainer:
    """
    Trainer for the enhanced cricket GNN that leverages the comprehensive
    knowledge graph with situational statistics.
    """
    
    def __init__(self, 
                 unified_kg: nx.Graph,
                 feature_config: EnhancedFeatureConfig = None,
                 model_config: Dict[str, Any] = None):
        self.kg = unified_kg
        self.feature_config = feature_config or EnhancedFeatureConfig()
        self.model_config = model_config or {}
        
        self.feature_extractor = SituationalFeatureExtractor(self.feature_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Enhanced KG-GNN Trainer initialized with {self.kg.number_of_nodes()} nodes")
        logger.info(f"Feature config: {self.feature_config}")
        
    def prepare_hetero_data(self) -> HeteroData:
        """
        Convert the comprehensive knowledge graph to PyTorch Geometric HeteroData
        with rich feature extraction.
        """
        data = HeteroData()
        
        # Separate nodes by type
        node_types = {}
        for node, attrs in self.kg.nodes(data=True):
            node_type = attrs.get('type', 'player')  # Default to player
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append((node, attrs))
        
        logger.info(f"Found node types: {list(node_types.keys())}")
        
        # Extract features for each node type
        node_feature_dims = {}
        for node_type, nodes in node_types.items():
            features = []
            node_ids = []
            
            for node_id, attrs in nodes:
                if node_type == 'player':
                    feature_vec = self.feature_extractor.extract_player_features(node_id, attrs)
                elif node_type == 'venue':
                    feature_vec = self.feature_extractor.extract_venue_features(node_id, attrs)
                elif node_type == 'match':
                    feature_vec = self.feature_extractor.extract_match_features(node_id, attrs)
                else:
                    # Default features
                    feature_vec = np.zeros(64, dtype=np.float32)
                
                features.append(feature_vec)
                node_ids.append(node_id)
            
            if features:
                feature_tensor = torch.tensor(np.stack(features), dtype=torch.float32)
                data[node_type].x = feature_tensor
                data[node_type].num_nodes = len(features)
                node_feature_dims[node_type] = feature_tensor.size(1)
                
                logger.info(f"{node_type}: {len(features)} nodes, {feature_tensor.size(1)} features")
        
        # Add edges (handle sparse graphs gracefully)
        edge_counts = {}
        node_id_to_idx = {}
        
        # Create node ID to index mapping for efficiency
        for node_type, nodes in node_types.items():
            node_id_to_idx[node_type] = {node_id: idx for idx, (node_id, _) in enumerate(nodes)}
        
        # Process existing edges
        for source, target, attrs in self.kg.edges(data=True):
            source_type = self.kg.nodes[source].get('type', 'player')
            target_type = self.kg.nodes[target].get('type', 'player')
            
            edge_type = (source_type, 'interacts_with', target_type)
            
            if edge_type not in data.edge_index_dict:
                data.edge_index_dict[edge_type] = [[], []]
                edge_counts[edge_type] = 0
            
            # Map node IDs to indices safely
            if source in node_id_to_idx[source_type] and target in node_id_to_idx[target_type]:
                source_idx = node_id_to_idx[source_type][source]
                target_idx = node_id_to_idx[target_type][target]
                
                data.edge_index_dict[edge_type][0].append(source_idx)
                data.edge_index_dict[edge_type][1].append(target_idx)
                edge_counts[edge_type] += 1
        
        # Ensure all node types have at least self-loops for GNN processing
        for node_type in node_types.keys():
            self_loop_edge_type = (node_type, 'interacts_with', node_type)
            
            if self_loop_edge_type not in data.edge_index_dict or not data.edge_index_dict[self_loop_edge_type][0]:
                # Add self-loops for isolated nodes
                num_nodes = len(node_types[node_type])
                if num_nodes > 0:
                    self_loops = torch.arange(num_nodes)
                    data.edge_index_dict[self_loop_edge_type] = torch.stack([self_loops, self_loops])
                    edge_counts[self_loop_edge_type] = num_nodes
        
        # Convert edge lists to tensors
        for edge_type in list(data.edge_index_dict.keys()):
            if isinstance(data.edge_index_dict[edge_type], list):
                edge_list = data.edge_index_dict[edge_type]
                # Check if we have a nested list structure [source_list, target_list]
                if len(edge_list) == 2 and isinstance(edge_list[0], list) and isinstance(edge_list[1], list):
                    if edge_list[0] and edge_list[1]:  # Both source and target lists have edges
                        # Convert [source_list, target_list] to tensor
                        data.edge_index_dict[edge_type] = torch.tensor(edge_list, dtype=torch.long)
                    else:
                        # Remove empty edge types
                        del data.edge_index_dict[edge_type]
                        if edge_type in edge_counts:
                            del edge_counts[edge_type]
                elif edge_list:  # Direct edge list format
                    data.edge_index_dict[edge_type] = torch.tensor(edge_list, dtype=torch.long)
                else:
                    # Remove empty edge types
                    del data.edge_index_dict[edge_type]
                    if edge_type in edge_counts:
                        del edge_counts[edge_type]
        
        logger.info(f"Edge counts: {edge_counts}")
        
        # Ensure we have at least some edges for GNN processing
        if not data.edge_index_dict:
            logger.warning("No edges found in HeteroData, adding minimal self-loops for all node types")
            for node_type in node_types.keys():
                num_nodes = len(node_types[node_type])
                if num_nodes > 0:
                    self_loops = torch.arange(num_nodes)
                    edge_type = (node_type, 'self_loop', node_type)
                    data.edge_index_dict[edge_type] = torch.stack([self_loops, self_loops])
                    edge_counts[edge_type] = num_nodes
                    logger.info(f"Added {num_nodes} self-loops for {node_type}")
        
        logger.info(f"Final edge types: {list(data.edge_index_dict.keys())}")
        
        return data, node_feature_dims
    
    def create_model(self, node_feature_dims: Dict[str, int]) -> EnhancedCricketGNN:
        """Create the enhanced GNN model"""
        model = EnhancedCricketGNN(
            node_feature_dims=node_feature_dims,
            hidden_dim=self.model_config.get('hidden_dim', 256),
            output_dim=self.model_config.get('output_dim', 128),
            num_layers=self.model_config.get('num_layers', 3),
            num_heads=self.model_config.get('num_heads', 8),
            dropout=self.model_config.get('dropout', 0.1),
            use_attention=self.model_config.get('use_attention', True)
        )
        
        return model.to(self.device)
    
    def train(self, num_epochs: int = 100, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the enhanced GNN model.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size (None for full-batch training)
            
        Returns:
            Training results and metrics
        """
        # Prepare data
        data, node_feature_dims = self.prepare_hetero_data()
        data = data.to(self.device)
        
        # Create model
        model = self.create_model(node_feature_dims)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Training loop
        model.train()
        training_losses = []
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            results = model(data)
            
            # Compute loss (self-supervised reconstruction)
            loss = 0.0
            for node_type, embeddings in results['embeddings'].items():
                # Simple reconstruction loss
                reconstructed = torch.matmul(embeddings, embeddings.T)
                target = torch.eye(embeddings.size(0), device=self.device)
                loss += F.mse_loss(torch.sigmoid(reconstructed), target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            training_losses.append(loss.item())
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")
        
        logger.info("Training completed!")
        
        return {
            'model': model,
            'training_losses': training_losses,
            'final_loss': training_losses[-1],
            'data': data,
            'node_feature_dims': node_feature_dims
        }

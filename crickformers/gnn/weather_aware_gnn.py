#!/usr/bin/env python3
"""
Weather-Aware Cricket GNN
Extends the existing GNN to incorporate weather conditions, venue coordinates, 
and team squad information for enhanced predictions.

Author: WicketWise Team, Last Modified: 2025-01-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATv2Conv, Linear
import logging

logger = logging.getLogger(__name__)

class WeatherFeatureExtractor:
    """
    Extracts weather-based features for cricket prediction
    """
    
    @staticmethod
    def extract_weather_features(weather_data: Dict[str, Any]) -> torch.Tensor:
        """
        Extract numerical features from weather data
        
        Returns:
            torch.Tensor: Weather features (16 dimensions)
        """
        features = []
        
        # Temperature features (4 dims)
        temp_c = weather_data.get('temperature_c', 25.0)
        feels_like = weather_data.get('feels_like_c', 25.0)
        features.extend([
            temp_c / 50.0,  # Normalized temperature
            feels_like / 50.0,  # Normalized feels-like
            max(0, (temp_c - 35) / 10),  # Extreme heat indicator
            max(0, (15 - temp_c) / 10)   # Cold weather indicator
        ])
        
        # Humidity and precipitation (4 dims)
        humidity = weather_data.get('humidity_pct', 50) / 100.0
        precip_mm = weather_data.get('precip_mm', 0.0)
        precip_prob = weather_data.get('precip_prob_pct', 0) / 100.0
        features.extend([
            humidity,
            min(precip_mm / 10.0, 1.0),  # Capped precipitation
            precip_prob,
            1.0 if precip_mm > 0.5 else 0.0  # Rain indicator
        ])
        
        # Wind conditions (4 dims)
        wind_speed = weather_data.get('wind_speed_kph', 0.0)
        wind_gust = weather_data.get('wind_gust_kph', 0.0)
        wind_dir = weather_data.get('wind_dir_deg', 0)
        features.extend([
            min(wind_speed / 50.0, 1.0),  # Normalized wind speed
            min(wind_gust / 80.0, 1.0),   # Normalized gusts
            np.sin(np.radians(wind_dir)),  # Wind direction (sin)
            np.cos(np.radians(wind_dir))   # Wind direction (cos)
        ])
        
        # Atmospheric conditions (4 dims)
        cloud_cover = weather_data.get('cloud_cover_pct', 50) / 100.0
        pressure = weather_data.get('pressure_hpa', 1013)
        uv_index = weather_data.get('uv_index', 5.0)
        features.extend([
            cloud_cover,
            (pressure - 1000) / 50.0,  # Normalized pressure deviation
            min(uv_index / 12.0, 1.0),  # Normalized UV index
            1.0 if cloud_cover > 0.8 else 0.0  # Overcast indicator
        ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    @staticmethod
    def extract_venue_features(venue_data: Dict[str, Any]) -> torch.Tensor:
        """
        Extract venue-based features including coordinates
        
        Returns:
            torch.Tensor: Venue features (8 dimensions)
        """
        features = []
        
        # Geographic features (4 dims)
        lat = venue_data.get('latitude', 0.0)
        lon = venue_data.get('longitude', 0.0)
        features.extend([
            lat / 90.0,  # Normalized latitude
            lon / 180.0,  # Normalized longitude
            1.0 if abs(lat) < 23.5 else 0.0,  # Tropical zone
            1.0 if abs(lat) > 40.0 else 0.0   # High latitude
        ])
        
        # Altitude and climate proxies (4 dims)
        # These could be enhanced with actual altitude data
        altitude_proxy = abs(lat) * 100  # Rough altitude estimation
        features.extend([
            min(altitude_proxy / 3000.0, 1.0),  # Normalized altitude proxy
            1.0 if venue_data.get('country', '').lower() in ['india', 'australia', 'england'] else 0.0,  # Major cricket nations
            1.0 if 'stadium' in venue_data.get('city', '').lower() else 0.0,  # Stadium indicator
            1.0 if venue_data.get('coordinates_available', False) else 0.0  # Data quality
        ])
        
        return torch.tensor(features, dtype=torch.float32)

class TeamSquadEncoder(nn.Module):
    """
    Encodes team squad information into embeddings
    """
    
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Role embeddings
        self.role_embedding = nn.Embedding(5, embed_dim // 4)  # batter, bowler, allrounder, wk, unknown
        
        # Style embeddings
        self.batting_style_embedding = nn.Embedding(3, embed_dim // 4)  # RHB, LHB, unknown
        self.bowling_style_embedding = nn.Embedding(9, embed_dim // 4)  # Various bowling styles
        
        # Experience features
        self.experience_encoder = nn.Linear(4, embed_dim // 4)  # captain, wk, playing_xi, other stats
        
        # Squad aggregation
        self.squad_aggregator = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, squad_data: Dict[str, Any]) -> torch.Tensor:
        """
        Encode team squad into a fixed-size embedding
        
        Args:
            squad_data: Dictionary with player information
            
        Returns:
            torch.Tensor: Squad embedding (embed_dim dimensions)
        """
        player_embeddings = []
        
        for player_name, player_info in squad_data.items():
            # Role embedding
            role_map = {'batter': 0, 'bowler': 1, 'allrounder': 2, 'wk': 3, 'unknown': 4}
            role_idx = role_map.get(player_info.get('role', 'unknown'), 4)
            role_emb = self.role_embedding(torch.tensor(role_idx))
            
            # Batting style embedding
            batting_map = {'RHB': 0, 'LHB': 1, 'unknown': 2}
            batting_idx = batting_map.get(player_info.get('batting_style', 'unknown'), 2)
            batting_emb = self.batting_style_embedding(torch.tensor(batting_idx))
            
            # Bowling style embedding (simplified)
            bowling_map = {'RF': 0, 'RM': 1, 'LF': 2, 'LM': 3, 'OB': 4, 'LB': 5, 'SLA': 6, 'SLC': 7, 'unknown': 8}
            bowling_idx = bowling_map.get(player_info.get('bowling_style', 'unknown'), 8)
            bowling_emb = self.bowling_style_embedding(torch.tensor(bowling_idx))
            
            # Experience features
            exp_features = torch.tensor([
                1.0 if player_info.get('captain', False) else 0.0,
                1.0 if player_info.get('wicket_keeper', False) else 0.0,
                1.0 if player_info.get('playing_xi', True) else 0.0,
                player_info.get('captain_experience', 0) / 100.0  # Normalized
            ], dtype=torch.float32)
            exp_emb = self.experience_encoder(exp_features)
            
            # Combine player features
            player_emb = torch.cat([role_emb, batting_emb, bowling_emb, exp_emb], dim=0)
            player_embeddings.append(player_emb)
        
        if not player_embeddings:
            # Return zero embedding if no squad data
            return torch.zeros(self.embed_dim)
        
        # Aggregate squad embeddings (mean pooling)
        squad_tensor = torch.stack(player_embeddings)
        squad_mean = torch.mean(squad_tensor, dim=0)
        
        return self.squad_aggregator(squad_mean)

class WeatherAwareCricketGNN(nn.Module):
    """
    Enhanced Cricket GNN that incorporates weather conditions,
    venue coordinates, and team squad information
    """
    
    def __init__(self, 
                 node_feature_dims: Dict[str, int],
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 weather_dim: int = 16,
                 venue_dim: int = 8,
                 squad_dim: int = 32):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Weather and venue feature extractors
        self.weather_encoder = nn.Linear(weather_dim, hidden_dim)
        self.venue_encoder = nn.Linear(venue_dim, hidden_dim)
        
        # Team squad encoder
        self.squad_encoder = TeamSquadEncoder(squad_dim)
        self.squad_projector = nn.Linear(squad_dim, hidden_dim)
        
        # Node feature encoders
        self.node_encoders = nn.ModuleDict()
        for node_type, input_dim in node_feature_dims.items():
            self.node_encoders[node_type] = nn.Linear(input_dim, hidden_dim)
        
        # Heterogeneous graph convolutions
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            # Define edge types for heterogeneous convolution
            edge_types = [
                ('player', 'faced', 'player'),
                ('player', 'bowled_to', 'player'), 
                ('player', 'played_at', 'venue'),
                ('player', 'plays_for', 'team'),
                ('match', 'played_at', 'venue'),
                ('match', 'had_weather', 'weather'),
                ('team', 'played_in', 'match')
            ]
            
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(hidden_dim, hidden_dim)
            
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Attention mechanism for weather impact
        self.weather_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Final prediction layers
        self.match_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # player + weather + venue + team
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # Single output for win probability
        )
        
        # Weather impact predictor
        self.weather_impact_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # Favorable, neutral, unfavorable
        )
    
    def forward(self, 
                hetero_data: HeteroData,
                weather_features: torch.Tensor,
                venue_features: torch.Tensor,
                team_squad_data: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the weather-aware GNN
        
        Args:
            hetero_data: Heterogeneous graph data
            weather_features: Weather condition features
            venue_features: Venue coordinate and location features
            team_squad_data: Team squad information
            
        Returns:
            Dictionary with predictions and embeddings
        """
        
        # Encode node features
        x_dict = {}
        for node_type, features in hetero_data.x_dict.items():
            if node_type in self.node_encoders:
                x_dict[node_type] = self.node_encoders[node_type](features)
            else:
                x_dict[node_type] = torch.zeros(features.size(0), self.hidden_dim)
        
        # Process through graph convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, hetero_data.edge_index_dict)
            # Apply activation
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Encode weather features
        weather_emb = self.weather_encoder(weather_features)
        
        # Encode venue features
        venue_emb = self.venue_encoder(venue_features)
        
        # Encode team squad features
        team_embeddings = []
        for team_name, squad_data in team_squad_data.items():
            squad_emb = self.squad_encoder(squad_data)
            team_emb = self.squad_projector(squad_emb)
            team_embeddings.append(team_emb)
        
        # Aggregate team embeddings
        if team_embeddings:
            team_emb = torch.mean(torch.stack(team_embeddings), dim=0)
        else:
            team_emb = torch.zeros(self.hidden_dim)
        
        # Get player embeddings (aggregate all players)
        if 'player' in x_dict:
            player_emb = torch.mean(x_dict['player'], dim=0)
        else:
            player_emb = torch.zeros(self.hidden_dim)
        
        # Apply weather attention
        weather_context = weather_emb.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
        player_context = player_emb.unsqueeze(0).unsqueeze(0)
        
        attended_weather, _ = self.weather_attention(player_context, weather_context, weather_context)
        attended_weather = attended_weather.squeeze(0).squeeze(0)
        
        # Combine all features for final prediction
        combined_features = torch.cat([player_emb, attended_weather, venue_emb, team_emb], dim=0)
        
        # Make predictions
        match_prediction = self.match_predictor(combined_features)
        weather_impact = self.weather_impact_predictor(attended_weather)
        
        return {
            'match_prediction': match_prediction,
            'weather_impact': weather_impact,
            'player_embedding': player_emb,
            'weather_embedding': attended_weather,
            'venue_embedding': venue_emb,
            'team_embedding': team_emb
        }

class WeatherAwareTrainer:
    """
    Trainer for the weather-aware cricket GNN
    """
    
    def __init__(self, model: WeatherAwareCricketGNN, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.weather_loss_fn = nn.CrossEntropyLoss()
    
    def train_step(self, 
                   hetero_data: HeteroData,
                   weather_features: torch.Tensor,
                   venue_features: torch.Tensor,
                   team_squad_data: Dict[str, Dict[str, Any]],
                   target_win_prob: torch.Tensor,
                   target_weather_impact: torch.Tensor) -> Dict[str, float]:
        """
        Single training step
        """
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(hetero_data, weather_features, venue_features, team_squad_data)
        
        # Calculate losses
        match_loss = self.loss_fn(outputs['match_prediction'], target_win_prob)
        weather_loss = self.weather_loss_fn(outputs['weather_impact'], target_weather_impact)
        
        total_loss = match_loss + 0.3 * weather_loss  # Weight weather loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'match_loss': match_loss.item(),
            'weather_loss': weather_loss.item()
        }
    
    def evaluate(self,
                 hetero_data: HeteroData,
                 weather_features: torch.Tensor,
                 venue_features: torch.Tensor,
                 team_squad_data: Dict[str, Dict[str, Any]],
                 target_win_prob: torch.Tensor,
                 target_weather_impact: torch.Tensor) -> Dict[str, float]:
        """
        Evaluation step
        """
        
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(hetero_data, weather_features, venue_features, team_squad_data)
            
            match_loss = self.loss_fn(outputs['match_prediction'], target_win_prob)
            weather_loss = self.weather_loss_fn(outputs['weather_impact'], target_weather_impact)
            
            # Calculate accuracy for weather impact
            weather_pred = torch.argmax(outputs['weather_impact'], dim=-1)
            weather_acc = (weather_pred == target_weather_impact).float().mean()
        
        return {
            'match_loss': match_loss.item(),
            'weather_loss': weather_loss.item(),
            'weather_accuracy': weather_acc.item(),
            'match_prediction': outputs['match_prediction'].item(),
            'weather_impact_probs': F.softmax(outputs['weather_impact'], dim=-1).tolist()
        }

def create_weather_aware_model(enriched_kg: nx.Graph) -> WeatherAwareCricketGNN:
    """
    Create a weather-aware GNN model from an enriched knowledge graph
    """
    
    # Analyze node types and their feature dimensions
    node_feature_dims = {}
    for node_id, node_data in enriched_kg.nodes(data=True):
        node_type = node_data.get('type', 'unknown')
        
        if node_type not in node_feature_dims:
            # Estimate feature dimensions based on node type
            if node_type == 'player':
                node_feature_dims[node_type] = 32  # Player stats
            elif node_type == 'venue':
                node_feature_dims[node_type] = 16  # Venue stats + coordinates
            elif node_type == 'team':
                node_feature_dims[node_type] = 24  # Team stats
            elif node_type == 'weather':
                node_feature_dims[node_type] = 16  # Weather features
            elif node_type == 'match':
                node_feature_dims[node_type] = 20  # Match context
            else:
                node_feature_dims[node_type] = 8   # Default
    
    # Create model
    model = WeatherAwareCricketGNN(
        node_feature_dims=node_feature_dims,
        hidden_dim=64,
        num_layers=3,
        weather_dim=16,
        venue_dim=8,
        squad_dim=32
    )
    
    logger.info(f"Created weather-aware GNN with node types: {list(node_feature_dims.keys())}")
    return model

if __name__ == "__main__":
    # Example usage
    print("🌤️ Weather-Aware Cricket GNN")
    print("This module provides enhanced GNN capabilities for cricket prediction")
    print("incorporating weather conditions, venue coordinates, and team squad data.")

#!/usr/bin/env python3
"""
Weather-Aware Graph Neural Network for Cricket Analysis
Extends the existing GNN to process weather, venue coordinates, and team squad data

Author: WicketWise Team, Last Modified: 2025-01-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, HGTConv, global_mean_pool
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class WeatherEncoder(nn.Module):
    """Encodes weather conditions into dense representations"""
    
    def __init__(self, weather_dim: int = 64):
        super().__init__()
        self.weather_dim = weather_dim
        
        # Weather feature processing
        self.temperature_encoder = nn.Linear(1, 16)
        self.humidity_encoder = nn.Linear(1, 16)
        self.wind_encoder = nn.Linear(2, 16)  # speed + direction
        self.precipitation_encoder = nn.Linear(2, 16)  # amount + probability
        
        # Weather fusion
        self.weather_fusion = nn.Sequential(
            nn.Linear(64, weather_dim),  # 4 * 16 = 64
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(weather_dim, weather_dim)
        )
    
    def forward(self, weather_features: torch.Tensor) -> torch.Tensor:
        """
        Encode weather features
        
        Args:
            weather_features: [batch_size, 6] tensor with:
                [temp_c, humidity_pct, wind_speed_kph, wind_dir_deg, precip_mm, precip_prob_pct]
        
        Returns:
            Weather embeddings [batch_size, weather_dim]
        """
        batch_size = weather_features.shape[0]
        
        # Extract individual weather components
        temp = weather_features[:, 0:1]  # Temperature
        humidity = weather_features[:, 1:2]  # Humidity
        wind = weather_features[:, 2:4]  # Wind speed + direction
        precip = weather_features[:, 4:6]  # Precipitation + probability
        
        # Encode each component
        temp_emb = self.temperature_encoder(temp)
        humidity_emb = self.humidity_encoder(humidity)
        wind_emb = self.wind_encoder(wind)
        precip_emb = self.precipitation_encoder(precip)
        
        # Concatenate and fuse
        weather_concat = torch.cat([temp_emb, humidity_emb, wind_emb, precip_emb], dim=1)
        weather_embedding = self.weather_fusion(weather_concat)
        
        return weather_embedding

class VenueCoordinateEncoder(nn.Module):
    """Encodes venue coordinates and timezone into dense representations"""
    
    def __init__(self, coord_dim: int = 32):
        super().__init__()
        self.coord_dim = coord_dim
        
        # Coordinate processing with geographical awareness
        self.lat_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        self.lon_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Coordinate fusion
        self.coord_fusion = nn.Sequential(
            nn.Linear(32, coord_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(coord_dim, coord_dim)
        )
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Encode venue coordinates
        
        Args:
            coordinates: [batch_size, 2] tensor with [latitude, longitude]
        
        Returns:
            Coordinate embeddings [batch_size, coord_dim]
        """
        lat = coordinates[:, 0:1]
        lon = coordinates[:, 1:2]
        
        # Normalize coordinates to [-1, 1] range
        lat_norm = lat / 90.0  # Latitude range: -90 to 90
        lon_norm = lon / 180.0  # Longitude range: -180 to 180
        
        # Encode
        lat_emb = self.lat_encoder(lat_norm)
        lon_emb = self.lon_encoder(lon_norm)
        
        # Fuse
        coord_concat = torch.cat([lat_emb, lon_emb], dim=1)
        coord_embedding = self.coord_fusion(coord_concat)
        
        return coord_embedding

class TeamSquadEncoder(nn.Module):
    """Encodes team squad composition and roles"""
    
    def __init__(self, squad_dim: int = 48):
        super().__init__()
        self.squad_dim = squad_dim
        
        # Role encodings
        self.role_embedding = nn.Embedding(5, 12)  # batter, bowler, allrounder, wk, unknown
        self.batting_style_embedding = nn.Embedding(3, 8)  # RHB, LHB, unknown
        self.bowling_style_embedding = nn.Embedding(9, 8)  # RF, RM, LF, LM, OB, LB, SLA, SLC, unknown
        
        # Squad composition encoder
        self.squad_fusion = nn.Sequential(
            nn.Linear(28, squad_dim),  # 12 + 8 + 8 = 28
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(squad_dim, squad_dim)
        )
    
    def forward(self, squad_features: torch.Tensor) -> torch.Tensor:
        """
        Encode team squad features
        
        Args:
            squad_features: [batch_size, 3] tensor with role indices:
                [primary_role_id, batting_style_id, bowling_style_id]
        
        Returns:
            Squad embeddings [batch_size, squad_dim]
        """
        role_ids = squad_features[:, 0].long()
        batting_ids = squad_features[:, 1].long()
        bowling_ids = squad_features[:, 2].long()
        
        # Embed each component
        role_emb = self.role_embedding(role_ids)
        batting_emb = self.batting_style_embedding(batting_ids)
        bowling_emb = self.bowling_style_embedding(bowling_ids)
        
        # Concatenate and fuse
        squad_concat = torch.cat([role_emb, batting_emb, bowling_emb], dim=1)
        squad_embedding = self.squad_fusion(squad_concat)
        
        return squad_embedding

class WeatherAwareGNN(nn.Module):
    """
    Enhanced GNN that incorporates weather, venue coordinates, and team squad data
    for improved cricket match prediction
    """
    
    def __init__(
        self,
        player_feature_dim: int = 128,
        venue_feature_dim: int = 64,
        match_feature_dim: int = 96,
        weather_dim: int = 64,
        coord_dim: int = 32,
        squad_dim: int = 48,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.player_feature_dim = player_feature_dim
        self.venue_feature_dim = venue_feature_dim
        self.match_feature_dim = match_feature_dim
        self.weather_dim = weather_dim
        self.coord_dim = coord_dim
        self.squad_dim = squad_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Enhanced feature encoders
        self.weather_encoder = WeatherEncoder(weather_dim)
        self.coord_encoder = VenueCoordinateEncoder(coord_dim)
        self.squad_encoder = TeamSquadEncoder(squad_dim)
        
        # Feature projection layers
        self.player_proj = nn.Linear(player_feature_dim, hidden_dim)
        self.venue_proj = nn.Linear(venue_feature_dim + coord_dim, hidden_dim)  # venue + coordinates
        self.match_proj = nn.Linear(match_feature_dim + weather_dim, hidden_dim)  # match + weather
        
        # Heterogeneous Graph Transformer layers
        self.hgt_layers = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=(['player', 'venue', 'match'], [
                    ('player', 'played_at', 'venue'),
                    ('player', 'played_in', 'match'),
                    ('venue', 'hosted', 'match')
                ]),
                heads=4
            ) for _ in range(num_layers)
        ])
        
        # Graph attention for final aggregation
        self.attention_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
            for _ in range(2)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Weather impact prediction head
        self.weather_impact_head = nn.Sequential(
            nn.Linear(weather_dim + coord_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hetero_data: HeteroData,
        weather_features: Optional[torch.Tensor] = None,
        venue_coordinates: Optional[torch.Tensor] = None,
        squad_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through weather-aware GNN
        
        Args:
            hetero_data: Heterogeneous graph data with player, venue, match nodes
            weather_features: Weather conditions [num_matches, 6]
            venue_coordinates: Venue coordinates [num_venues, 2]
            squad_features: Squad composition [num_players, 3]
        
        Returns:
            Dictionary with node embeddings and weather impact predictions
        """
        # Extract node features
        player_x = hetero_data['player'].x
        venue_x = hetero_data['venue'].x
        match_x = hetero_data['match'].x
        
        # Encode enhanced features
        enhanced_features = {}
        
        # Weather encoding
        if weather_features is not None:
            weather_emb = self.weather_encoder(weather_features)
            enhanced_features['weather'] = weather_emb
            
            # Enhance match features with weather
            if match_x.shape[0] == weather_emb.shape[0]:
                match_x = torch.cat([match_x, weather_emb], dim=1)
        
        # Venue coordinate encoding
        if venue_coordinates is not None:
            coord_emb = self.coord_encoder(venue_coordinates)
            enhanced_features['coordinates'] = coord_emb
            
            # Enhance venue features with coordinates
            if venue_x.shape[0] == coord_emb.shape[0]:
                venue_x = torch.cat([venue_x, coord_emb], dim=1)
        
        # Squad encoding
        if squad_features is not None and squad_features.shape[0] > 0:
            # Ensure squad_features has valid indices
            squad_features = torch.clamp(squad_features, 0, 4)  # Clamp to valid range
            squad_emb = self.squad_encoder(squad_features)
            enhanced_features['squad'] = squad_emb
        
        # Project features to hidden dimension
        player_h = self.player_proj(player_x)
        venue_h = self.venue_proj(venue_x)
        match_h = self.match_proj(match_x)
        
        # Create heterogeneous node features
        hetero_h = {
            'player': player_h,
            'venue': venue_h,
            'match': match_h
        }
        
        # Apply HGT layers
        for hgt_layer in self.hgt_layers:
            hetero_h = hgt_layer(hetero_h, hetero_data.edge_index_dict)
            
            # Apply dropout and residual connections
            for node_type in hetero_h:
                hetero_h[node_type] = self.dropout(hetero_h[node_type])
        
        # Final embeddings
        final_embeddings = {}
        for node_type, node_h in hetero_h.items():
            final_embeddings[node_type] = self.output_proj(node_h)
        
        # Weather impact prediction
        weather_impact = None
        if weather_features is not None and venue_coordinates is not None:
            weather_venue_concat = torch.cat([
                enhanced_features['weather'], 
                enhanced_features['coordinates']
            ], dim=1)
            weather_impact = self.weather_impact_head(weather_venue_concat)
        
        return {
            'node_embeddings': final_embeddings,
            'enhanced_features': enhanced_features,
            'weather_impact': weather_impact
        }
    
    def get_player_embeddings(self, player_indices: torch.Tensor, node_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract embeddings for specific players"""
        return node_embeddings['player'][player_indices]
    
    def get_venue_embeddings(self, venue_indices: torch.Tensor, node_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract embeddings for specific venues"""
        return node_embeddings['venue'][venue_indices]
    
    def predict_weather_advantage(self, weather_features: torch.Tensor, venue_coordinates: torch.Tensor) -> torch.Tensor:
        """Predict weather advantage for batting/bowling"""
        weather_emb = self.weather_encoder(weather_features)
        coord_emb = self.coord_encoder(venue_coordinates)
        
        combined = torch.cat([weather_emb, coord_emb], dim=1)
        advantage = self.weather_impact_head(combined)
        
        return advantage

def create_weather_aware_gnn(config: Dict[str, Any]) -> WeatherAwareGNN:
    """Factory function to create weather-aware GNN from config"""
    return WeatherAwareGNN(
        player_feature_dim=config.get('player_feature_dim', 128),
        venue_feature_dim=config.get('venue_feature_dim', 64),
        match_feature_dim=config.get('match_feature_dim', 96),
        weather_dim=config.get('weather_dim', 64),
        coord_dim=config.get('coord_dim', 32),
        squad_dim=config.get('squad_dim', 48),
        hidden_dim=config.get('hidden_dim', 256),
        output_dim=config.get('output_dim', 128),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.1)
    )

if __name__ == "__main__":
    # Test the weather-aware GNN
    print("🌦️ Testing Weather-Aware GNN...")
    
    # Create sample data
    batch_size = 32
    num_players = 100
    num_venues = 20
    num_matches = 50
    
    # Sample weather features: [temp, humidity, wind_speed, wind_dir, precip, precip_prob]
    weather_features = torch.randn(batch_size, 6)
    
    # Sample venue coordinates: [lat, lon]
    venue_coordinates = torch.randn(batch_size, 2) * 90  # Random coordinates
    
    # Sample squad features: [role_id, batting_style_id, bowling_style_id]
    squad_features = torch.randint(0, 5, (batch_size, 3))
    
    # Create weather-aware GNN
    config = {
        'player_feature_dim': 128,
        'venue_feature_dim': 64,
        'match_feature_dim': 96,
        'weather_dim': 64,
        'coord_dim': 32,
        'squad_dim': 48,
        'hidden_dim': 256,
        'output_dim': 128,
        'num_layers': 3,
        'dropout': 0.1
    }
    
    gnn = create_weather_aware_gnn(config)
    
    # Test weather and coordinate encoding
    weather_emb = gnn.weather_encoder(weather_features)
    coord_emb = gnn.coord_encoder(venue_coordinates)
    squad_emb = gnn.squad_encoder(squad_features)
    
    print(f"✅ Weather embeddings shape: {weather_emb.shape}")
    print(f"✅ Coordinate embeddings shape: {coord_emb.shape}")
    print(f"✅ Squad embeddings shape: {squad_emb.shape}")
    
    # Test weather advantage prediction
    weather_advantage = gnn.predict_weather_advantage(weather_features, venue_coordinates)
    print(f"✅ Weather advantage shape: {weather_advantage.shape}")
    
    print("🎉 Weather-Aware GNN test completed successfully!")
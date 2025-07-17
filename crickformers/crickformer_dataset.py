# Purpose: PyTorch Dataset for Crickformer model training and inference
# Author: Assistant, Last Modified: 2024

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
import logging
import json

from .csv_data_schema import (
    CurrentBallFeatures, RecentBallHistoryEntry, VideoSignals,
    GNNEmbeddings, MarketOdds
)
from .csv_input_preprocessor import preprocess_ball_input
from .csv_data_adapter import CSVDataAdapter, CSVDataConfig

logger = logging.getLogger(__name__)


class CrickformerDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing cricket match data for Crickformer model.
    
    Supports two data formats:
    1. Directory structure with separate files (original format)
    2. CSV files using CSVDataAdapter (real data format)
    
    The dataset loads:
    - Current ball features (batting, bowling, match state)
    - Recent ball history (last N balls, zero-padded)
    - Video signals (ball tracking, player detection)
    - GNN embeddings (player/venue representations)
    - Market odds (betting data)
    
    All data is preprocessed into tensors ready for model input.
    """
    
    def __init__(
        self,
        data_root: str,
        manifest_file: Optional[str] = None,
        history_length: int = 5,
        video_dim: int = 96,
        gnn_dims: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None,
        load_video: bool = True,
        load_embeddings: bool = True,
        load_market_odds: bool = True,
        use_csv_adapter: bool = False,
        csv_config: Optional[CSVDataConfig] = None,
        match_id_list_path: Optional[str] = None
    ):
        """
        Initialize CrickformerDataset.
        
        Args:
            data_root: Root directory containing match data or CSV files
            manifest_file: Optional manifest file listing available samples
            history_length: Number of recent balls to include in history
            video_dim: Dimension of video feature vectors
            gnn_dims: Dictionary specifying GNN embedding dimensions
            transform: Optional data augmentation transform
            load_video: Whether to load video signals
            load_embeddings: Whether to load GNN embeddings
            load_market_odds: Whether to load market odds
            use_csv_adapter: Whether to use CSV data format
            csv_config: Configuration for CSV adapter
            match_id_list_path: Optional path to CSV file containing match IDs to filter by
        """
        self.data_root = Path(data_root)
        self.manifest_file = manifest_file
        self.history_length = history_length
        self.video_dim = video_dim
        self.transform = transform
        self.load_video = load_video
        self.load_embeddings = load_embeddings
        self.load_market_odds = load_market_odds
        self.use_csv_adapter = use_csv_adapter
        self.match_id_list_path = match_id_list_path
        
        # Load match filter list if provided
        self.match_filter_list = None
        if self.match_id_list_path:
            self.match_filter_list = self._load_match_filter_list(self.match_id_list_path)
        
        # Set default GNN dimensions
        self.gnn_dims = gnn_dims or {
            'batter': 128,
            'bowler': 128, 
            'venue': 64,
            'edge': 64
        }
        
        # Initialize data adapter
        if self.use_csv_adapter:
            self.csv_adapter = CSVDataAdapter(data_root, csv_config)
            self._load_csv_samples()
        else:
            self.csv_adapter = None
            self._load_directory_samples()
    
    def _load_match_filter_list(self, match_id_list_path: str) -> List[str]:
        """
        Load match IDs from CSV file for filtering.
        
        Args:
            match_id_list_path: Path to CSV file containing match IDs
            
        Returns:
            List of match IDs to filter by
            
        Raises:
            FileNotFoundError: If the filter file doesn't exist
            ValueError: If the CSV file doesn't have the expected format
        """
        import pandas as pd
        
        filter_path = Path(match_id_list_path)
        if not filter_path.exists():
            raise FileNotFoundError(f"Match filter file not found: {filter_path}")
        
        try:
            # Load CSV file
            df = pd.read_csv(filter_path)
            
            # Check if 'match_id' column exists
            if 'match_id' not in df.columns:
                raise ValueError(f"CSV file must contain 'match_id' column. Found columns: {list(df.columns)}")
            
            # Extract match IDs and remove duplicates
            match_ids = df['match_id'].dropna().unique().tolist()
            
            logger.info(f"Loaded {len(match_ids)} match IDs from filter file: {filter_path}")
            return match_ids
            
        except Exception as e:
            raise ValueError(f"Error loading match filter file {filter_path}: {e}")

    def _load_csv_samples(self):
        """Load samples using CSV adapter"""
        self.samples = []
        
        if self.manifest_file:
            # Load from manifest if provided
            manifest_path = self.data_root / self.manifest_file
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest_data = json.load(f)
                    self.samples = manifest_data.get('samples', [])
            else:
                logger.warning(f"Manifest file not found: {manifest_path}")
        
        if not self.samples:
            # Generate samples from all available balls
            for i in range(len(self.csv_adapter)):
                sample_info = self.csv_adapter.get_sample_info(i)
                self.samples.append({
                    'ball_id': sample_info['ball_id'],
                    'match_id': sample_info['match_id'],
                    'index': i
                })
        
        # Apply match filtering if filter list is provided
        if self.match_filter_list:
            original_count = len(self.samples)
            self.samples = [
                sample for sample in self.samples 
                if sample['match_id'] in self.match_filter_list
            ]
            filtered_count = len(self.samples)
            unique_matches = len(set(sample['match_id'] for sample in self.samples))
            
            logger.info(f"Applied match filtering: {original_count} → {filtered_count} samples")
            logger.info(f"Filtered to {unique_matches} matches from {len(self.match_filter_list)} requested matches")
        else:
            unique_matches = len(set(sample['match_id'] for sample in self.samples))
            logger.info(f"Loaded {len(self.samples)} samples from {unique_matches} matches (no filtering applied)")

    def _load_directory_samples(self):
        """Load samples from directory structure (original implementation)"""
        self.samples = []
        
        if self.manifest_file:
            # Load from manifest file
            manifest_path = self.data_root / self.manifest_file
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest_data = json.load(f)
                    self.samples = manifest_data.get('samples', [])
            else:
                logger.warning(f"Manifest file not found: {manifest_path}")
        
        if not self.samples:
            # Discover samples from directory structure
            for match_dir in self.data_root.glob('*'):
                if not match_dir.is_dir():
                    continue
                
                match_id = match_dir.name
                current_features_dir = match_dir / 'current_ball_features'
                
                if current_features_dir.exists():
                    for ball_file in current_features_dir.glob('*.json'):
                        ball_id = ball_file.stem
                        self.samples.append({
                            'match_id': match_id,
                            'ball_id': ball_id,
                            'current_features_path': str(ball_file)
                        })
        
        # Apply match filtering if filter list is provided
        if self.match_filter_list:
            original_count = len(self.samples)
            self.samples = [
                sample for sample in self.samples 
                if sample['match_id'] in self.match_filter_list
            ]
            filtered_count = len(self.samples)
            unique_matches = len(set(sample['match_id'] for sample in self.samples))
            
            logger.info(f"Applied match filtering: {original_count} → {filtered_count} samples")
            logger.info(f"Filtered to {unique_matches} matches from {len(self.match_filter_list)} requested matches")
        else:
            unique_matches = len(set(sample['match_id'] for sample in self.samples))
            logger.info(f"Loaded {len(self.samples)} samples from {unique_matches} matches (no filtering applied)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing preprocessed tensors for model input
        """
        sample = self.samples[idx]
        
        if self.use_csv_adapter:
            return self._get_csv_sample(sample)
        else:
            return self._get_directory_sample(sample)
    
    def _get_csv_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Get sample using CSV adapter"""
        ball_id = sample['ball_id']
        
        # Load current ball features
        current_features = self.csv_adapter.get_current_ball_features(ball_id)
        
        # Load ball history
        history = self.csv_adapter.get_ball_history(ball_id, self.history_length)
        
        # Load video signals
        video_signals = None
        if self.load_video:
            video_signals = self.csv_adapter.get_video_signals(ball_id)
        
        # Load GNN embeddings
        gnn_embeddings = None
        if self.load_embeddings:
            gnn_embeddings = self.csv_adapter.get_gnn_embeddings(ball_id)
        
        # Load market odds
        market_odds = None
        if self.load_market_odds:
            market_odds = self.csv_adapter.get_market_odds(ball_id)
        
        # Preprocess into tensor format
        return self._preprocess_sample(
            current_features, history, video_signals, gnn_embeddings, market_odds
        )
    
    def _get_directory_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Get sample from directory structure (original implementation)"""
        match_id = sample['match_id']
        ball_id = sample['ball_id']
        
        # Load current ball features
        current_features = self._load_current_ball_features(match_id, ball_id)
        
        # Load ball history
        history = self._load_ball_history(match_id, ball_id)
        
        # Load video signals
        video_signals = None
        if self.load_video:
            video_signals = self._load_video_signals(match_id, ball_id)
        
        # Load GNN embeddings
        gnn_embeddings = None
        if self.load_embeddings:
            gnn_embeddings = self._load_gnn_embeddings(match_id, ball_id)
        
        # Load market odds
        market_odds = None
        if self.load_market_odds:
            market_odds = self._load_market_odds(match_id, ball_id)
        
        # Preprocess into tensor format
        return self._preprocess_sample(
            current_features, history, video_signals, gnn_embeddings, market_odds
        )
    
    def _preprocess_sample(
        self,
        current_features: CurrentBallFeatures,
        history: List[RecentBallHistoryEntry],
        video_signals: Optional[VideoSignals],
        gnn_embeddings: Optional[GNNEmbeddings],
        market_odds: Optional[MarketOdds]
    ) -> Dict[str, torch.Tensor]:
        """Preprocess sample data into tensor format"""
        
        # Use the input preprocessor to convert to tensors
        preprocessed = preprocess_ball_input(
            current_features=current_features,
            recent_history=history,
            video_signals=video_signals,
            gnn_embeddings=gnn_embeddings,
            market_odds=market_odds
        )
        
        # Convert numpy arrays to tensors
        tensor_dict = {}
        for key, value in preprocessed.items():
            if isinstance(value, np.ndarray):
                tensor_dict[key] = torch.from_numpy(value).float()
            elif isinstance(value, torch.Tensor):
                tensor_dict[key] = value.float()
            else:
                # Handle scalar values
                tensor_dict[key] = torch.tensor(value, dtype=torch.float32)
        
        # Apply augmentation if specified
        if self.transform:
            tensor_dict = self._apply_augmentation(tensor_dict)
        
        return tensor_dict
    
    def _apply_augmentation(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply data augmentation transforms"""
        if self.transform:
            # Apply transform to numeric features only
            if 'numeric_ball_features' in data:
                data['numeric_ball_features'] = self.transform(data['numeric_ball_features'])
        return data

    def _load_current_ball_features(self, match_id: str, ball_id: str) -> CurrentBallFeatures:
        """Load current ball features from file"""
        features_file = self.data_root / match_id / 'current_ball_features' / f'{ball_id}.json'
        
        if not features_file.exists():
            raise FileNotFoundError(f"Current ball features not found: {features_file}")
        
        with open(features_file, 'r') as f:
            data = json.load(f)
        
        return CurrentBallFeatures(**data)
    
    def _load_ball_history(self, match_id: str, ball_id: str) -> List[RecentBallHistoryEntry]:
        """Load recent ball history from file"""
        history_file = self.data_root / match_id / 'ball_history' / f'{ball_id}.json'
        
        if not history_file.exists():
            # Return zero-padded history if file doesn't exist
            return self._get_zero_history()
        
        with open(history_file, 'r') as f:
            data = json.load(f)
        
        history = [RecentBallHistoryEntry(**entry) for entry in data.get('history', [])]
        
        # Pad or truncate to desired length
        while len(history) < self.history_length:
            history.append(RecentBallHistoryEntry(
                runs_scored=0, extras=0, is_wicket=False,
                batter_name="PADDING", bowler_name="PADDING",
                batter_hand="PADDING", bowler_type="PADDING"
            ))
        
        return history[:self.history_length]
    
    def _load_video_signals(self, match_id: str, ball_id: str) -> VideoSignals:
        """Load video signals from file"""
        video_file = self.data_root / match_id / 'video_signals' / f'{ball_id}.json'
        
        if not video_file.exists():
            # Return zero signals if file doesn't exist
            return self._get_zero_video_signals()
        
        with open(video_file, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        data['motion_vectors'] = np.array(data['motion_vectors'], dtype=np.float32)
        data['optical_flow'] = np.array(data['optical_flow'], dtype=np.float32)
        
        return VideoSignals(**data)
    
    def _load_gnn_embeddings(self, match_id: str, ball_id: str) -> GNNEmbeddings:
        """Load GNN embeddings from file"""
        embeddings_file = self.data_root / match_id / 'gnn_embeddings' / f'{ball_id}.json'
        
        if not embeddings_file.exists():
            # Return zero embeddings if file doesn't exist
            return self._get_zero_embeddings()
        
        with open(embeddings_file, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        data['batter_embedding'] = np.array(data['batter_embedding'], dtype=np.float32)
        data['bowler_embedding'] = np.array(data['bowler_embedding'], dtype=np.float32)
        data['venue_embedding'] = np.array(data['venue_embedding'], dtype=np.float32)
        data['edge_embeddings'] = np.array(data['edge_embeddings'], dtype=np.float32)
        
        return GNNEmbeddings(**data)
    
    def _load_market_odds(self, match_id: str, ball_id: str) -> Optional[MarketOdds]:
        """Load market odds from file"""
        odds_file = self.data_root / match_id / 'market_odds' / f'{ball_id}.json'
        
        if not odds_file.exists():
            return None
        
        with open(odds_file, 'r') as f:
            data = json.load(f)
        
        return MarketOdds(**data)
    
    def _get_zero_history(self) -> List[RecentBallHistoryEntry]:
        """Generate zero-padded history entries"""
        return [
            RecentBallHistoryEntry(
                runs_scored=0, extras=0, is_wicket=False,
                batter_name="PADDING", bowler_name="PADDING",
                batter_hand="PADDING", bowler_type="PADDING"
            ) for _ in range(self.history_length)
        ]
    
    def _get_zero_video_signals(self) -> VideoSignals:
        """Generate zero video signals"""
        return VideoSignals(
            ball_tracking_confidence=0.0,
            player_detection_confidence=0.0,
            scene_classification="unknown",
            motion_vectors=np.zeros(32, dtype=np.float32),
            optical_flow=np.zeros(64, dtype=np.float32)
        )
    
    def _get_zero_embeddings(self) -> GNNEmbeddings:
        """Generate zero GNN embeddings"""
        return GNNEmbeddings(
            batter_embedding=np.zeros(self.gnn_dims['batter'], dtype=np.float32),
            bowler_embedding=np.zeros(self.gnn_dims['bowler'], dtype=np.float32),
            venue_embedding=np.zeros(self.gnn_dims['venue'], dtype=np.float32),
            edge_embeddings=np.zeros(self.gnn_dims['edge'], dtype=np.float32)
        )
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a specific sample"""
        sample = self.samples[idx]
        
        if self.use_csv_adapter:
            return self.csv_adapter.get_sample_info(sample['index'])
        else:
            return {
                'match_id': sample['match_id'],
                'ball_id': sample['ball_id'],
                'index': idx
            }
    
    def get_match_ids(self) -> List[str]:
        """Get list of unique match IDs in the dataset"""
        # Always use the filtered samples to get match IDs
        return list(set(sample['match_id'] for sample in self.samples))
    
    def filter_by_match(self, match_ids: List[str]) -> 'CrickformerDataset':
        """Create a new dataset filtered to specific matches"""
        filtered_samples = [
            sample for sample in self.samples 
            if sample['match_id'] in match_ids
        ]
        
        # Create new dataset with filtered samples
        new_dataset = CrickformerDataset.__new__(CrickformerDataset)
        new_dataset.__dict__.update(self.__dict__)
        new_dataset.samples = filtered_samples
        
        return new_dataset 
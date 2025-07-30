# Purpose: Match Regime Detection system using Hidden Markov Models and clustering
# Author: Shamus Rae, Last Modified: 2024-01-15

"""
This module implements a Match Regime Detection system that classifies the current
cricket match situation into distinct regimes based on recent ball-by-ball data.
Uses Hidden Markov Models and unsupervised clustering to identify patterns like
"Post-Wicket Consolidation", "Spin Squeeze", and "All-Out Attack".
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import deque
import logging
import pickle
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    """Configuration for regime detection system."""
    
    # Window parameters
    window_size: int = 12              # Number of recent balls to analyze
    min_window_size: int = 6           # Minimum balls needed for detection
    
    # Feature parameters
    run_rate_window: int = 6           # Balls for run rate calculation
    boundary_window: int = 12          # Balls for boundary percentage
    wicket_window: int = 18            # Balls for wicket density
    biomech_window: int = 8            # Balls for biomechanical volatility
    
    # Model parameters
    n_regimes: int = 5                 # Number of distinct regimes
    hmm_n_components: int = 5          # HMM hidden states
    clustering_method: str = 'gmm'     # 'kmeans' or 'gmm'
    
    # Regime thresholds
    high_run_rate_threshold: float = 10.0    # Runs per over
    low_run_rate_threshold: float = 4.0      # Runs per over
    high_boundary_threshold: float = 0.3     # 30% boundaries
    high_wicket_density: float = 0.15        # Wickets per ball
    high_biomech_volatility: float = 0.3     # Volatility threshold
    
    # Model persistence
    model_save_path: str = "regime_detector_model.pkl"
    fallback_mode: bool = True         # Use rule-based fallback if model fails


class RegimeLabels:
    """Standardized regime labels."""
    
    POST_WICKET_CONSOLIDATION = "Post-Wicket Consolidation"
    SPIN_SQUEEZE = "Spin Squeeze"
    ALL_OUT_ATTACK = "All-Out Attack"
    STEADY_ACCUMULATION = "Steady Accumulation"
    PRESSURE_BUILDING = "Pressure Building"
    UNKNOWN = "Unknown"
    
    @classmethod
    def get_all_labels(cls) -> List[str]:
        """Get all regime labels."""
        return [
            cls.POST_WICKET_CONSOLIDATION,
            cls.SPIN_SQUEEZE,
            cls.ALL_OUT_ATTACK,
            cls.STEADY_ACCUMULATION,
            cls.PRESSURE_BUILDING
        ]
    
    @classmethod
    def get_label_encoding(cls) -> Dict[str, int]:
        """Get label to integer encoding."""
        labels = cls.get_all_labels()
        return {label: i for i, label in enumerate(labels)}
    
    @classmethod
    def get_reverse_encoding(cls) -> Dict[int, str]:
        """Get integer to label encoding."""
        encoding = cls.get_label_encoding()
        return {v: k for k, v in encoding.items()}


@dataclass
class RegimeFeatures:
    """Container for regime detection features."""
    
    run_rate: float                    # Current run rate (runs per over)
    boundary_percentage: float         # Percentage of boundaries in window
    wicket_count: int                  # Recent wickets in window
    wicket_density: float              # Wickets per ball in window
    biomech_volatility: float          # Biomechanical signal volatility
    balls_since_wicket: int            # Balls since last wicket
    balls_since_boundary: int          # Balls since last boundary
    phase: str                         # Match phase (powerplay, middle, death)
    over_number: float                 # Current over number
    required_run_rate: Optional[float] # Required run rate (if chasing)
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for model input."""
        # Encode phase as numeric
        phase_encoding = {'powerplay': 0, 'middle': 1, 'death': 2, 'unknown': 3}
        phase_num = phase_encoding.get(self.phase.lower(), 3)
        
        return np.array([
            self.run_rate,
            self.boundary_percentage,
            self.wicket_count,
            self.wicket_density,
            self.biomech_volatility,
            self.balls_since_wicket,
            self.balls_since_boundary,
            phase_num,
            self.over_number,
            self.required_run_rate or 0.0
        ])
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get feature names for interpretability."""
        return [
            'run_rate',
            'boundary_percentage', 
            'wicket_count',
            'wicket_density',
            'biomech_volatility',
            'balls_since_wicket',
            'balls_since_boundary',
            'phase',
            'over_number',
            'required_run_rate'
        ]


class RegimeFeatureExtractor:
    """Extracts features from ball-by-ball data for regime detection."""
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
    
    def extract_features(
        self,
        ball_data: pd.DataFrame,
        biomech_data: Optional[Dict[str, Dict[str, float]]] = None,
        target_info: Optional[Dict[str, Any]] = None
    ) -> RegimeFeatures:
        """
        Extract regime features from recent ball-by-ball data.
        
        Args:
            ball_data: DataFrame with recent balls (most recent last)
            biomech_data: Optional biomechanical signals per delivery
            target_info: Optional target score information for chasing
        
        Returns:
            RegimeFeatures object with extracted features
        """
        if len(ball_data) == 0:
            return self._get_default_features()
        
        # Ensure data is sorted by delivery order
        ball_data = ball_data.sort_values(['over', 'ball']).reset_index(drop=True)
        
        # Extract basic features
        run_rate = self._calculate_run_rate(ball_data)
        boundary_percentage = self._calculate_boundary_percentage(ball_data)
        wicket_count, wicket_density = self._calculate_wicket_metrics(ball_data)
        biomech_volatility = self._calculate_biomech_volatility(ball_data, biomech_data)
        balls_since_wicket = self._calculate_balls_since_wicket(ball_data)
        balls_since_boundary = self._calculate_balls_since_boundary(ball_data)
        phase = self._determine_phase(ball_data)
        over_number = ball_data['over'].iloc[-1] if 'over' in ball_data.columns else 0.0
        required_run_rate = self._calculate_required_run_rate(ball_data, target_info)
        
        return RegimeFeatures(
            run_rate=run_rate,
            boundary_percentage=boundary_percentage,
            wicket_count=wicket_count,
            wicket_density=wicket_density,
            biomech_volatility=biomech_volatility,
            balls_since_wicket=balls_since_wicket,
            balls_since_boundary=balls_since_boundary,
            phase=phase,
            over_number=over_number,
            required_run_rate=required_run_rate
        )
    
    def _calculate_run_rate(self, ball_data: pd.DataFrame) -> float:
        """Calculate current run rate over recent balls."""
        window = min(self.config.run_rate_window, len(ball_data))
        recent_balls = ball_data.tail(window)
        
        total_runs = recent_balls.get('runs_scored', pd.Series([0])).sum()
        total_balls = len(recent_balls)
        
        if total_balls == 0:
            return 0.0
        
        # Convert to runs per over
        return (total_runs / total_balls) * 6.0
    
    def _calculate_boundary_percentage(self, ball_data: pd.DataFrame) -> float:
        """Calculate percentage of boundaries in recent balls."""
        window = min(self.config.boundary_window, len(ball_data))
        recent_balls = ball_data.tail(window)
        
        if len(recent_balls) == 0:
            return 0.0
        
        boundaries = recent_balls.get('runs_scored', pd.Series([0])) >= 4
        return boundaries.sum() / len(recent_balls)
    
    def _calculate_wicket_metrics(self, ball_data: pd.DataFrame) -> Tuple[int, float]:
        """Calculate wicket count and density."""
        window = min(self.config.wicket_window, len(ball_data))
        recent_balls = ball_data.tail(window)
        
        if len(recent_balls) == 0:
            return 0, 0.0
        
        # Count wickets (non-null wicket_type)
        wickets = recent_balls.get('wicket_type', pd.Series()).notna()
        wicket_count = wickets.sum()
        wicket_density = wicket_count / len(recent_balls)
        
        return int(wicket_count), float(wicket_density)
    
    def _calculate_biomech_volatility(
        self,
        ball_data: pd.DataFrame,
        biomech_data: Optional[Dict[str, Dict[str, float]]]
    ) -> float:
        """Calculate biomechanical signal volatility."""
        if not biomech_data:
            return 0.0
        
        window = min(self.config.biomech_window, len(ball_data))
        recent_balls = ball_data.tail(window)
        
        # Extract biomechanical signals for recent deliveries
        signal_values = []
        
        for _, ball_row in recent_balls.iterrows():
            delivery_id = f"{ball_row.get('match_id', 'unknown')}_{ball_row.get('innings', 1)}_{ball_row.get('over', 1)}_{ball_row.get('ball', 1)}"
            
            if delivery_id in biomech_data:
                signals = biomech_data[delivery_id]
                # Use head_stability and shot_commitment as key volatility indicators
                key_signals = [
                    signals.get('head_stability', 0.5),
                    signals.get('shot_commitment', 0.5)
                ]
                signal_values.extend(key_signals)
        
        if len(signal_values) < 2:
            return 0.0
        
        # Calculate coefficient of variation as volatility measure
        signal_array = np.array(signal_values)
        mean_val = np.mean(signal_array)
        std_val = np.std(signal_array)
        
        if mean_val == 0:
            return 0.0
        
        return std_val / mean_val
    
    def _calculate_balls_since_wicket(self, ball_data: pd.DataFrame) -> int:
        """Calculate balls since last wicket."""
        wicket_col = ball_data.get('wicket_type', pd.Series())
        wickets = wicket_col.notna() & (wicket_col != '') & (wicket_col != 'not_out')
        
        if not wickets.any():
            return len(ball_data)  # No wickets in window
        
        # Find last wicket index (using iloc position)
        wicket_positions = wickets[wickets].index.tolist()
        if not wicket_positions:
            return len(ball_data)
        
        last_wicket_position = wicket_positions[-1]
        last_wicket_iloc = ball_data.index.get_loc(last_wicket_position)
        balls_since = len(ball_data) - 1 - last_wicket_iloc
        
        return max(0, balls_since)
    
    def _calculate_balls_since_boundary(self, ball_data: pd.DataFrame) -> int:
        """Calculate balls since last boundary."""
        runs_col = ball_data.get('runs_scored', pd.Series([0]))
        boundaries = runs_col >= 4
        
        if not boundaries.any():
            return len(ball_data)  # No boundaries in window
        
        # Find last boundary index (using iloc position)
        boundary_positions = boundaries[boundaries].index.tolist()
        if not boundary_positions:
            return len(ball_data)
        
        last_boundary_position = boundary_positions[-1]
        last_boundary_iloc = ball_data.index.get_loc(last_boundary_position)
        balls_since = len(ball_data) - 1 - last_boundary_iloc
        
        return max(0, balls_since)
    
    def _determine_phase(self, ball_data: pd.DataFrame) -> str:
        """Determine current match phase."""
        if len(ball_data) == 0:
            return 'unknown'
        
        current_over = ball_data['over'].iloc[-1] if 'over' in ball_data.columns else 0
        
        # T20 phase classification
        if current_over <= 6:
            return 'powerplay'
        elif current_over <= 15:
            return 'middle'
        else:
            return 'death'
    
    def _calculate_required_run_rate(
        self,
        ball_data: pd.DataFrame,
        target_info: Optional[Dict[str, Any]]
    ) -> Optional[float]:
        """Calculate required run rate if chasing."""
        if not target_info:
            return None
        
        target_runs = target_info.get('target_runs')
        current_score = target_info.get('current_score', 0)
        balls_remaining = target_info.get('balls_remaining', 120)
        
        if target_runs is None or balls_remaining <= 0:
            return None
        
        runs_needed = target_runs - current_score
        overs_remaining = balls_remaining / 6.0
        
        if overs_remaining <= 0:
            return 999.0  # Impossible rate
        
        return runs_needed / overs_remaining
    
    def _get_default_features(self) -> RegimeFeatures:
        """Get default features when no data is available."""
        return RegimeFeatures(
            run_rate=6.0,
            boundary_percentage=0.1,
            wicket_count=0,
            wicket_density=0.0,
            biomech_volatility=0.2,
            balls_since_wicket=6,
            balls_since_boundary=3,
            phase='middle',
            over_number=10.0,
            required_run_rate=None
        )


class RegimeDetector:
    """Main regime detection system using HMM and clustering."""
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        self.feature_extractor = RegimeFeatureExtractor(self.config)
        self.scaler = StandardScaler()
        self.hmm_model: Optional[hmm.GaussianHMM] = None
        self.clustering_model: Optional[Union[KMeans, GaussianMixture]] = None
        self.is_trained = False
        self.regime_mapping: Dict[int, str] = {}
        
        # Initialize regime mapping
        self._initialize_regime_mapping()
    
    def _initialize_regime_mapping(self):
        """Initialize mapping from cluster/state IDs to regime labels."""
        labels = RegimeLabels.get_all_labels()
        for i in range(self.config.n_regimes):
            if i < len(labels):
                self.regime_mapping[i] = labels[i]
            else:
                self.regime_mapping[i] = RegimeLabels.UNKNOWN
    
    def train(
        self,
        training_data: List[pd.DataFrame],
        biomech_data_list: Optional[List[Dict[str, Dict[str, float]]]] = None
    ) -> None:
        """
        Train the regime detection models on historical data.
        
        Args:
            training_data: List of DataFrames, each containing ball-by-ball data for a match
            biomech_data_list: Optional list of biomechanical data for each match
        """
        logger.info(f"Training regime detector on {len(training_data)} matches")
        
        # Extract features from all matches
        all_features = []
        all_sequences = []
        
        for i, match_data in enumerate(training_data):
            biomech_data = biomech_data_list[i] if biomech_data_list else None
            match_features = self._extract_match_features(match_data, biomech_data)
            
            if len(match_features) > 0:
                all_features.extend(match_features)
                all_sequences.append(match_features)
        
        if len(all_features) == 0:
            logger.warning("No features extracted from training data")
            return
        
        # Convert to numpy arrays
        feature_matrix = np.array([f.to_array() for f in all_features])
        
        # Fit scaler
        self.scaler.fit(feature_matrix)
        scaled_features = self.scaler.transform(feature_matrix)
        
        # Train clustering model
        self._train_clustering_model(scaled_features)
        
        # Train HMM model
        self._train_hmm_model(all_sequences)
        
        # Map clusters to regime labels
        self._map_clusters_to_regimes(scaled_features)
        
        self.is_trained = True
        logger.info("Regime detector training completed")
    
    def _extract_match_features(
        self,
        match_data: pd.DataFrame,
        biomech_data: Optional[Dict[str, Dict[str, float]]] = None
    ) -> List[RegimeFeatures]:
        """Extract features from a single match using sliding window."""
        features = []
        
        for i in range(self.config.min_window_size, len(match_data) + 1):
            # Get window of recent balls
            start_idx = max(0, i - self.config.window_size)
            window_data = match_data.iloc[start_idx:i]
            
            # Extract features for this window
            match_features = self.feature_extractor.extract_features(
                window_data, biomech_data
            )
            features.append(match_features)
        
        return features
    
    def _train_clustering_model(self, scaled_features: np.ndarray) -> None:
        """Train clustering model on scaled features."""
        if self.config.clustering_method == 'gmm':
            self.clustering_model = GaussianMixture(
                n_components=self.config.n_regimes,
                random_state=42,
                max_iter=200
            )
        else:
            self.clustering_model = KMeans(
                n_clusters=self.config.n_regimes,
                random_state=42,
                max_iter=300
            )
        
        self.clustering_model.fit(scaled_features)
        logger.info(f"Trained {self.config.clustering_method} clustering model")
    
    def _train_hmm_model(self, all_sequences: List[List[RegimeFeatures]]) -> None:
        """Train HMM model on feature sequences."""
        if len(all_sequences) == 0:
            return
        
        # Convert sequences to scaled feature arrays
        hmm_sequences = []
        sequence_lengths = []
        
        for sequence in all_sequences:
            if len(sequence) > 1:  # Need at least 2 observations for HMM
                seq_features = np.array([f.to_array() for f in sequence])
                seq_scaled = self.scaler.transform(seq_features)
                hmm_sequences.append(seq_scaled)
                sequence_lengths.append(len(seq_scaled))
        
        if len(hmm_sequences) == 0:
            logger.warning("No valid sequences for HMM training")
            return
        
        # Concatenate all sequences
        X = np.vstack(hmm_sequences)
        
        # Train HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.config.hmm_n_components,
            covariance_type="full",
            random_state=42,
            n_iter=100
        )
        
        try:
            self.hmm_model.fit(X, sequence_lengths)
            logger.info("Trained HMM model")
        except Exception as e:
            logger.warning(f"HMM training failed: {e}")
            self.hmm_model = None
    
    def _map_clusters_to_regimes(self, scaled_features: np.ndarray) -> None:
        """Map cluster IDs to meaningful regime labels based on feature characteristics."""
        if not self.clustering_model:
            return
        
        cluster_labels = self.clustering_model.predict(scaled_features)
        
        # Analyze cluster characteristics
        cluster_stats = {}
        for cluster_id in range(self.config.n_regimes):
            mask = cluster_labels == cluster_id
            if np.sum(mask) == 0:
                continue
            
            cluster_features = scaled_features[mask]
            
            # Calculate mean characteristics (using original feature indices)
            mean_features = np.mean(cluster_features, axis=0)
            
            cluster_stats[cluster_id] = {
                'run_rate': mean_features[0],
                'boundary_pct': mean_features[1],
                'wicket_count': mean_features[2],
                'wicket_density': mean_features[3],
                'biomech_volatility': mean_features[4],
                'balls_since_wicket': mean_features[5],
                'balls_since_boundary': mean_features[6]
            }
        
        # Map clusters to regimes based on characteristics
        self._assign_regime_labels(cluster_stats)
    
    def _assign_regime_labels(self, cluster_stats: Dict[int, Dict[str, float]]) -> None:
        """Assign regime labels to clusters based on their characteristics."""
        # Sort clusters by different characteristics to assign labels
        
        # Find cluster with highest wicket density and low run rate -> Post-Wicket Consolidation
        post_wicket_candidates = sorted(
            cluster_stats.keys(),
            key=lambda x: (cluster_stats[x]['wicket_density'], -cluster_stats[x]['run_rate']),
            reverse=True
        )
        
        # Find cluster with low run rate and low boundary percentage -> Spin Squeeze
        spin_squeeze_candidates = sorted(
            cluster_stats.keys(),
            key=lambda x: (-cluster_stats[x]['run_rate'], -cluster_stats[x]['boundary_pct'])
        )
        
        # Find cluster with high run rate and high boundary percentage -> All-Out Attack
        attack_candidates = sorted(
            cluster_stats.keys(),
            key=lambda x: (cluster_stats[x]['run_rate'], cluster_stats[x]['boundary_pct']),
            reverse=True
        )
        
        # Assign labels (avoiding duplicates)
        assigned = set()
        
        if post_wicket_candidates and post_wicket_candidates[0] not in assigned:
            self.regime_mapping[post_wicket_candidates[0]] = RegimeLabels.POST_WICKET_CONSOLIDATION
            assigned.add(post_wicket_candidates[0])
        
        if spin_squeeze_candidates and spin_squeeze_candidates[0] not in assigned:
            self.regime_mapping[spin_squeeze_candidates[0]] = RegimeLabels.SPIN_SQUEEZE
            assigned.add(spin_squeeze_candidates[0])
        
        if attack_candidates and attack_candidates[0] not in assigned:
            self.regime_mapping[attack_candidates[0]] = RegimeLabels.ALL_OUT_ATTACK
            assigned.add(attack_candidates[0])
        
        # Assign remaining clusters
        remaining_labels = [RegimeLabels.STEADY_ACCUMULATION, RegimeLabels.PRESSURE_BUILDING]
        remaining_clusters = [c for c in cluster_stats.keys() if c not in assigned]
        
        for i, cluster_id in enumerate(remaining_clusters):
            if i < len(remaining_labels):
                self.regime_mapping[cluster_id] = remaining_labels[i]
            else:
                self.regime_mapping[cluster_id] = RegimeLabels.UNKNOWN
    
    def detect_regime(
        self,
        recent_balls: pd.DataFrame,
        biomech_data: Optional[Dict[str, Dict[str, float]]] = None,
        target_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float]:
        """
        Detect current match regime based on recent balls.
        
        Args:
            recent_balls: DataFrame with recent ball-by-ball data
            biomech_data: Optional biomechanical signals
            target_info: Optional target information for chasing scenarios
        
        Returns:
            Tuple of (regime_label, confidence_score)
        """
        if len(recent_balls) < self.config.min_window_size:
            return RegimeLabels.UNKNOWN, 0.0
        
        # Extract features
        features = self.feature_extractor.extract_features(
            recent_balls, biomech_data, target_info
        )
        
        if not self.is_trained:
            # Use rule-based fallback
            return self._rule_based_detection(features)
        
        # Use trained models
        try:
            return self._model_based_detection(features)
        except Exception as e:
            logger.warning(f"Model-based detection failed: {e}")
            if self.config.fallback_mode:
                return self._rule_based_detection(features)
            else:
                return RegimeLabels.UNKNOWN, 0.0
    
    def _model_based_detection(self, features: RegimeFeatures) -> Tuple[str, float]:
        """Use trained models for regime detection."""
        feature_array = features.to_array().reshape(1, -1)
        scaled_features = self.scaler.transform(feature_array)
        
        # Get cluster prediction
        if hasattr(self.clustering_model, 'predict_proba'):
            # GMM provides probabilities
            probabilities = self.clustering_model.predict_proba(scaled_features)[0]
            cluster_id = np.argmax(probabilities)
            confidence = probabilities[cluster_id]
        else:
            # KMeans doesn't provide probabilities
            cluster_id = self.clustering_model.predict(scaled_features)[0]
            # Calculate confidence based on distance to cluster center
            distances = self.clustering_model.transform(scaled_features)[0]
            min_distance = np.min(distances)
            confidence = 1.0 / (1.0 + min_distance)  # Convert distance to confidence
        
        regime_label = self.regime_mapping.get(cluster_id, RegimeLabels.UNKNOWN)
        
        return regime_label, float(confidence)
    
    def _rule_based_detection(self, features: RegimeFeatures) -> Tuple[str, float]:
        """Fallback rule-based regime detection."""
        confidence = 0.7  # Default confidence for rule-based detection
        
        # Post-Wicket Consolidation: Recent wicket and low run rate
        if features.balls_since_wicket <= 3 and features.run_rate < self.config.low_run_rate_threshold:
            return RegimeLabels.POST_WICKET_CONSOLIDATION, confidence
        
        # All-Out Attack: High run rate and boundaries
        if (features.run_rate > self.config.high_run_rate_threshold and 
            features.boundary_percentage > self.config.high_boundary_threshold):
            return RegimeLabels.ALL_OUT_ATTACK, confidence
        
        # Spin Squeeze: Low run rate, no recent boundaries, middle overs
        if (features.run_rate < self.config.low_run_rate_threshold and
            features.balls_since_boundary > 6 and
            features.phase == 'middle'):
            return RegimeLabels.SPIN_SQUEEZE, confidence
        
        # Pressure Building: High required run rate or death overs with moderate scoring
        if ((features.required_run_rate and features.required_run_rate > 12) or
            (features.phase == 'death' and features.run_rate < 8)):
            return RegimeLabels.PRESSURE_BUILDING, confidence
        
        # Default to Steady Accumulation
        return RegimeLabels.STEADY_ACCUMULATION, confidence
    
    def get_regime_probabilities(
        self,
        recent_balls: pd.DataFrame,
        biomech_data: Optional[Dict[str, Dict[str, float]]] = None,
        target_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Get probabilities for all regimes.
        
        Returns:
            Dictionary mapping regime labels to probabilities
        """
        if not self.is_trained or len(recent_balls) < self.config.min_window_size:
            # Return uniform probabilities
            labels = RegimeLabels.get_all_labels()
            return {label: 1.0 / len(labels) for label in labels}
        
        features = self.feature_extractor.extract_features(
            recent_balls, biomech_data, target_info
        )
        
        try:
            feature_array = features.to_array().reshape(1, -1)
            scaled_features = self.scaler.transform(feature_array)
            
            if hasattr(self.clustering_model, 'predict_proba'):
                probabilities = self.clustering_model.predict_proba(scaled_features)[0]
            else:
                # Convert distances to probabilities for KMeans
                distances = self.clustering_model.transform(scaled_features)[0]
                probabilities = 1.0 / (1.0 + distances)
                probabilities = probabilities / np.sum(probabilities)
            
            # Map to regime labels
            regime_probs = {}
            for i, prob in enumerate(probabilities):
                regime_label = self.regime_mapping.get(i, RegimeLabels.UNKNOWN)
                regime_probs[regime_label] = float(prob)
            
            return regime_probs
            
        except Exception as e:
            logger.warning(f"Probability calculation failed: {e}")
            labels = RegimeLabels.get_all_labels()
            return {label: 1.0 / len(labels) for label in labels}
    
    def save_model(self, filepath: str = None) -> None:
        """Save trained model to disk."""
        filepath = filepath or self.config.model_save_path
        
        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'clustering_model': self.clustering_model,
            'hmm_model': self.hmm_model,
            'regime_mapping': self.regime_mapping,
            'is_trained': self.is_trained
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str = None) -> bool:
        """Load trained model from disk."""
        filepath = filepath or self.config.model_save_path
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.config = model_data['config']
            self.scaler = model_data['scaler']
            self.clustering_model = model_data['clustering_model']
            self.hmm_model = model_data['hmm_model']
            self.regime_mapping = model_data['regime_mapping']
            self.is_trained = model_data['is_trained']
            
            # Recreate feature extractor with loaded config
            self.feature_extractor = RegimeFeatureExtractor(self.config)
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_regime_encoding(self, regime_label: str) -> int:
        """Get integer encoding for regime label (for CrickFormer input)."""
        encoding = RegimeLabels.get_label_encoding()
        return encoding.get(regime_label, len(encoding))  # Unknown gets highest value
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores if available."""
        if not self.is_trained or not hasattr(self.clustering_model, 'cluster_centers_'):
            return {}
        
        # Calculate feature variance across cluster centers
        centers = self.clustering_model.cluster_centers_
        feature_variance = np.var(centers, axis=0)
        feature_names = RegimeFeatures.get_feature_names()
        
        # Normalize to sum to 1
        total_variance = np.sum(feature_variance)
        if total_variance > 0:
            importance = feature_variance / total_variance
        else:
            importance = np.ones(len(feature_variance)) / len(feature_variance)
        
        return dict(zip(feature_names, importance))
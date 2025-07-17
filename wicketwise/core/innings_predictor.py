# Purpose: Predicts match outcomes and win probabilities from ball-by-ball data
# Author: WicketWise Team, Last Modified: 2024-12-07

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import warnings

from .feature_generator import FeatureGenerator, FeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a prediction."""
    win_probability: float
    next_ball_outcome: str
    next_ball_probabilities: Dict[str, float]
    match_outcome: str
    confidence: float
    tactical_insights: Dict[str, Any]


@dataclass
class PredictorConfig:
    """Configuration for the innings predictor."""
    feature_config: FeatureConfig
    model_type: str = "random_forest"  # random_forest, logistic_regression
    use_ensemble: bool = True
    confidence_threshold: float = 0.7
    save_models: bool = True
    model_path: Optional[str] = None


class InningsPredictor:
    """
    Predicts match outcomes and win probabilities from cricket ball-by-ball data.
    
    Uses machine learning models trained on historical data to provide:
    - Win probability predictions
    - Next ball outcome predictions
    - Match outcome predictions
    - Tactical insights and recommendations
    """
    
    def __init__(self, config: Optional[PredictorConfig] = None):
        """
        Initialize the innings predictor.
        
        Args:
            config: Configuration for the predictor
        """
        self.config = config or PredictorConfig(feature_config=FeatureConfig())
        self.feature_generator = FeatureGenerator(self.config.feature_config)
        
        # Models
        self.win_prob_model = None
        self.outcome_model = None
        self.match_outcome_model = None
        
        # Model metadata
        self.is_fitted = False
        self.feature_names = []
        self.outcome_classes = ['0', '1', '2', '3', '4', '6', 'wicket']
        self.match_outcome_classes = ['win', 'loss']
        
        # Performance metrics
        self.training_metrics = {}
        
    def fit(self, data: pd.DataFrame, target_column: str = 'match_result') -> 'InningsPredictor':
        """
        Fit the predictor on training data.
        
        Args:
            data: Training data DataFrame
            target_column: Column containing match results
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting innings predictor on training data...")
        
        # Generate features
        logger.info("Generating features...")
        features_df = self.feature_generator.fit_transform(data)
        self.feature_names = self.feature_generator.get_feature_names()
        
        # Prepare targets
        targets = self._prepare_targets(data, target_column)
        
        # Train models
        self._train_models(features_df, targets)
        
        # Evaluate models
        self._evaluate_models(features_df, targets)
        
        # Save models if requested
        if self.config.save_models and self.config.model_path:
            self._save_models()
        
        self.is_fitted = True
        logger.info("Innings predictor fitted successfully")
        
        return self
    
    def predict(self, data: pd.DataFrame) -> List[PredictionResult]:
        """
        Make predictions on new data.
        
        Args:
            data: Input data DataFrame
            
        Returns:
            List of prediction results
        """
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first.")
        
        logger.info(f"Making predictions on {len(data):,} rows...")
        
        # Generate features
        features_df = self.feature_generator.transform(data)
        
        # Make predictions
        predictions = []
        for idx, (_, row) in enumerate(data.iterrows()):
            feature_row = features_df.iloc[idx:idx+1]
            prediction = self._predict_single(feature_row, row)
            predictions.append(prediction)
        
        return predictions
    
    def predict_live(self, current_data: pd.DataFrame) -> PredictionResult:
        """
        Make a prediction for live match data.
        
        Args:
            current_data: Current match state data
            
        Returns:
            Prediction result
        """
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first.")
        
        predictions = self.predict(current_data)
        return predictions[0] if predictions else None
    
    def _prepare_targets(self, data: pd.DataFrame, target_column: str) -> Dict[str, np.ndarray]:
        """Prepare target variables for training."""
        targets = {}
        
        # Win probability target (0-1)
        if 'win_probability' in data.columns:
            targets['win_prob'] = data['win_probability'].values
        else:
            # Create synthetic win probability based on match state
            targets['win_prob'] = self._create_synthetic_win_prob(data)
        
        # Next ball outcome target
        if 'next_ball_outcome' in data.columns:
            targets['outcome'] = data['next_ball_outcome'].values
        else:
            # Use current ball outcome as proxy
            targets['outcome'] = data['runs_scored'].apply(
                lambda x: str(min(x, 6)) if x < 7 else 'wicket'
            ).values
        
        # Match outcome target
        if target_column in data.columns:
            targets['match_outcome'] = data[target_column].values
        else:
            # Create synthetic match outcome
            targets['match_outcome'] = self._create_synthetic_match_outcome(data)
        
        return targets
    
    def _create_synthetic_win_prob(self, data: pd.DataFrame) -> np.ndarray:
        """Create synthetic win probability based on match state."""
        win_probs = []
        
        for _, row in data.iterrows():
            # Simple heuristic based on run rate and wickets
            over = row.get('over', 0)
            runs_scored = row.get('runs_scored', 0)
            is_wicket = row.get('is_wicket', False)
            
            # Base probability
            base_prob = 0.5
            
            # Adjust for phase of play
            if over < 6:  # Powerplay
                base_prob += 0.1
            elif over > 15:  # Death overs
                base_prob -= 0.1
            
            # Adjust for current ball outcome
            if runs_scored >= 4:
                base_prob += 0.05
            elif is_wicket:
                base_prob -= 0.1
            
            # Ensure valid probability
            win_prob = max(0.1, min(0.9, base_prob))
            win_probs.append(win_prob)
        
        return np.array(win_probs)
    
    def _create_synthetic_match_outcome(self, data: pd.DataFrame) -> np.ndarray:
        """Create synthetic match outcome based on final scores."""
        outcomes = []
        
        # Group by match
        for match_id, match_data in data.groupby('match_id'):
            # Get final scores for both innings
            innings_scores = match_data.groupby('innings')['runs_scored'].sum()
            
            if len(innings_scores) >= 2:
                # Compare scores
                if innings_scores.iloc[1] > innings_scores.iloc[0]:
                    outcome = 'win'
                else:
                    outcome = 'loss'
            else:
                # Default to win for single innings
                outcome = 'win'
            
            # Assign outcome to all balls in match
            for _ in range(len(match_data)):
                outcomes.append(outcome)
        
        return np.array(outcomes)
    
    def _train_models(self, features_df: pd.DataFrame, targets: Dict[str, np.ndarray]):
        """Train all prediction models."""
        logger.info("Training prediction models...")
        
        # Win probability model (regression)
        logger.info("Training win probability model...")
        if self.config.model_type == "random_forest":
            self.win_prob_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            # Use logistic regression for probability
            self.win_prob_model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        
        self.win_prob_model.fit(features_df, targets['win_prob'])
        
        # Next ball outcome model (classification)
        logger.info("Training next ball outcome model...")
        if self.config.model_type == "random_forest":
            self.outcome_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.outcome_model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='multinomial'
            )
        
        self.outcome_model.fit(features_df, targets['outcome'])
        
        # Match outcome model (classification)
        logger.info("Training match outcome model...")
        if self.config.model_type == "random_forest":
            self.match_outcome_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.match_outcome_model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        
        self.match_outcome_model.fit(features_df, targets['match_outcome'])
        
        logger.info("All models trained successfully")
    
    def _evaluate_models(self, features_df: pd.DataFrame, targets: Dict[str, np.ndarray]):
        """Evaluate model performance."""
        logger.info("Evaluating model performance...")
        
        # Win probability model
        win_prob_pred = self.win_prob_model.predict(features_df)
        win_prob_mse = mean_squared_error(targets['win_prob'], win_prob_pred)
        
        # Next ball outcome model
        outcome_pred = self.outcome_model.predict(features_df)
        outcome_acc = accuracy_score(targets['outcome'], outcome_pred)
        
        # Match outcome model
        match_outcome_pred = self.match_outcome_model.predict(features_df)
        match_outcome_acc = accuracy_score(targets['match_outcome'], match_outcome_pred)
        
        # Store metrics
        self.training_metrics = {
            'win_prob_mse': win_prob_mse,
            'win_prob_rmse': np.sqrt(win_prob_mse),
            'outcome_accuracy': outcome_acc,
            'match_outcome_accuracy': match_outcome_acc
        }
        
        logger.info(f"Model Performance:")
        logger.info(f"  Win Probability RMSE: {self.training_metrics['win_prob_rmse']:.4f}")
        logger.info(f"  Next Ball Outcome Accuracy: {self.training_metrics['outcome_accuracy']:.4f}")
        logger.info(f"  Match Outcome Accuracy: {self.training_metrics['match_outcome_accuracy']:.4f}")
    
    def _predict_single(self, feature_row: pd.DataFrame, data_row: pd.Series) -> PredictionResult:
        """Make prediction for a single ball."""
        # Win probability
        win_prob = float(self.win_prob_model.predict(feature_row)[0])
        win_prob = max(0.0, min(1.0, win_prob))  # Ensure valid probability
        
        # Next ball outcome
        outcome_probs = self.outcome_model.predict_proba(feature_row)[0]
        outcome_classes = self.outcome_model.classes_
        
        # Get most likely outcome
        best_outcome_idx = np.argmax(outcome_probs)
        next_ball_outcome = outcome_classes[best_outcome_idx]
        
        # Create probability dictionary
        next_ball_probabilities = {
            str(cls): float(prob) for cls, prob in zip(outcome_classes, outcome_probs)
        }
        
        # Match outcome
        match_outcome_probs = self.match_outcome_model.predict_proba(feature_row)[0]
        match_outcome_classes = self.match_outcome_model.classes_
        best_match_outcome_idx = np.argmax(match_outcome_probs)
        match_outcome = match_outcome_classes[best_match_outcome_idx]
        
        # Confidence (average of max probabilities)
        confidence = float(np.mean([
            win_prob if win_prob > 0.5 else 1 - win_prob,
            outcome_probs[best_outcome_idx],
            match_outcome_probs[best_match_outcome_idx]
        ]))
        
        # Tactical insights
        tactical_insights = self._generate_tactical_insights(
            feature_row, data_row, win_prob, next_ball_probabilities
        )
        
        return PredictionResult(
            win_probability=win_prob,
            next_ball_outcome=next_ball_outcome,
            next_ball_probabilities=next_ball_probabilities,
            match_outcome=match_outcome,
            confidence=confidence,
            tactical_insights=tactical_insights
        )
    
    def _generate_tactical_insights(self, 
                                   feature_row: pd.DataFrame, 
                                   data_row: pd.Series,
                                   win_prob: float,
                                   outcome_probs: Dict[str, float]) -> Dict[str, Any]:
        """Generate tactical insights based on predictions."""
        insights = {}
        
        # Match situation
        over = data_row.get('over', 0)
        innings = data_row.get('innings', 1)
        
        # Phase-based insights
        if over < 6:
            insights['phase'] = 'powerplay'
            insights['recommendation'] = 'Aggressive batting recommended' if win_prob < 0.6 else 'Consolidate wickets'
        elif over < 16:
            insights['phase'] = 'middle_overs'
            insights['recommendation'] = 'Build partnerships' if win_prob > 0.4 else 'Accelerate scoring'
        else:
            insights['phase'] = 'death_overs'
            insights['recommendation'] = 'Maximum aggression' if win_prob < 0.7 else 'Finish safely'
        
        # Risk assessment
        wicket_prob = outcome_probs.get('wicket', 0)
        boundary_prob = outcome_probs.get('4', 0) + outcome_probs.get('6', 0)
        
        if wicket_prob > 0.15:
            insights['risk_level'] = 'high'
            insights['batting_advice'] = 'Play cautiously, avoid risky shots'
        elif boundary_prob > 0.3:
            insights['risk_level'] = 'low'
            insights['batting_advice'] = 'Good conditions for aggressive batting'
        else:
            insights['risk_level'] = 'medium'
            insights['batting_advice'] = 'Balanced approach recommended'
        
        # Bowling insights
        if innings == 1:
            insights['bowling_advice'] = 'Restrict scoring' if win_prob > 0.5 else 'Attack for wickets'
        else:
            insights['bowling_advice'] = 'Defend total' if win_prob > 0.5 else 'Pressure bowling needed'
        
        # Key metrics
        insights['key_metrics'] = {
            'win_probability': win_prob,
            'wicket_risk': wicket_prob,
            'boundary_chance': boundary_prob,
            'dot_ball_probability': outcome_probs.get('0', 0)
        }
        
        return insights
    
    def _save_models(self):
        """Save trained models to disk."""
        if not self.config.model_path:
            return
        
        model_path = Path(self.config.model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        joblib.dump(self.win_prob_model, model_path / 'win_prob_model.pkl')
        joblib.dump(self.outcome_model, model_path / 'outcome_model.pkl')
        joblib.dump(self.match_outcome_model, model_path / 'match_outcome_model.pkl')
        
        # Save feature generator
        joblib.dump(self.feature_generator, model_path / 'feature_generator.pkl')
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'outcome_classes': self.outcome_classes,
            'match_outcome_classes': self.match_outcome_classes,
            'training_metrics': self.training_metrics,
            'config': self.config
        }
        joblib.dump(metadata, model_path / 'metadata.pkl')
        
        logger.info(f"Models saved to {model_path}")
    
    def load_models(self, model_path: str):
        """Load trained models from disk."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Load models
        self.win_prob_model = joblib.load(model_path / 'win_prob_model.pkl')
        self.outcome_model = joblib.load(model_path / 'outcome_model.pkl')
        self.match_outcome_model = joblib.load(model_path / 'match_outcome_model.pkl')
        
        # Load feature generator
        self.feature_generator = joblib.load(model_path / 'feature_generator.pkl')
        
        # Load metadata
        metadata = joblib.load(model_path / 'metadata.pkl')
        self.feature_names = metadata['feature_names']
        self.outcome_classes = metadata['outcome_classes']
        self.match_outcome_classes = metadata['match_outcome_classes']
        self.training_metrics = metadata['training_metrics']
        
        self.is_fitted = True
        logger.info(f"Models loaded from {model_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models."""
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first.")
        
        importance_dict = {}
        
        # Get importance from random forest models
        if hasattr(self.win_prob_model, 'feature_importances_'):
            win_prob_importance = self.win_prob_model.feature_importances_
            for i, importance in enumerate(win_prob_importance):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
                importance_dict[f'win_prob_{feature_name}'] = float(importance)
        
        if hasattr(self.outcome_model, 'feature_importances_'):
            outcome_importance = self.outcome_model.feature_importances_
            for i, importance in enumerate(outcome_importance):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
                importance_dict[f'outcome_{feature_name}'] = float(importance)
        
        return importance_dict
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Get training performance metrics."""
        return self.training_metrics.copy()
    
    def analyze_match(self, match_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze a complete match and provide insights."""
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first.")
        
        predictions = self.predict(match_data)
        
        # Calculate match statistics
        win_probs = [p.win_probability for p in predictions]
        
        analysis = {
            'match_id': match_data['match_id'].iloc[0] if 'match_id' in match_data.columns else 'unknown',
            'total_balls': len(predictions),
            'avg_win_probability': np.mean(win_probs),
            'win_prob_volatility': np.std(win_probs),
            'max_win_prob': np.max(win_probs),
            'min_win_prob': np.min(win_probs),
            'final_prediction': predictions[-1] if predictions else None,
            'key_moments': self._identify_key_moments(predictions),
            'tactical_summary': self._summarize_tactics(predictions)
        }
        
        return analysis
    
    def _identify_key_moments(self, predictions: List[PredictionResult]) -> List[Dict[str, Any]]:
        """Identify key moments in the match based on win probability swings."""
        key_moments = []
        
        win_probs = [p.win_probability for p in predictions]
        
        # Find significant changes in win probability
        for i in range(1, len(win_probs)):
            prob_change = abs(win_probs[i] - win_probs[i-1])
            
            if prob_change > 0.1:  # Significant change
                key_moments.append({
                    'ball_number': i,
                    'win_prob_change': prob_change,
                    'new_win_prob': win_probs[i],
                    'prediction': predictions[i]
                })
        
        # Sort by magnitude of change
        key_moments.sort(key=lambda x: x['win_prob_change'], reverse=True)
        
        return key_moments[:5]  # Top 5 key moments
    
    def _summarize_tactics(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Summarize tactical insights from all predictions."""
        all_insights = [p.tactical_insights for p in predictions]
        
        # Count recommendations
        recommendations = {}
        for insights in all_insights:
            rec = insights.get('recommendation', 'unknown')
            recommendations[rec] = recommendations.get(rec, 0) + 1
        
        # Average risk levels
        risk_levels = [insights.get('risk_level', 'medium') for insights in all_insights]
        risk_distribution = {
            'high': risk_levels.count('high'),
            'medium': risk_levels.count('medium'),
            'low': risk_levels.count('low')
        }
        
        return {
            'dominant_recommendation': max(recommendations.items(), key=lambda x: x[1])[0],
            'recommendation_distribution': recommendations,
            'risk_distribution': risk_distribution,
            'avg_wicket_risk': np.mean([
                insights.get('key_metrics', {}).get('wicket_risk', 0) 
                for insights in all_insights
            ]),
            'avg_boundary_chance': np.mean([
                insights.get('key_metrics', {}).get('boundary_chance', 0) 
                for insights in all_insights
            ])
        } 
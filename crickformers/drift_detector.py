# Purpose: Detects model drift by monitoring prediction errors over time
# Author: Shamus Rae, Last Modified: 2024-12-19

"""
This module provides functionality to detect model drift by analyzing prediction errors.
It monitors error trends, standard deviation changes, and statistical significance
to identify when a model's performance is degrading and may need retraining.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from scipy import stats
from dataclasses import dataclass
import warnings
from collections import deque


@dataclass
class DriftThresholds:
    """Configuration for drift detection thresholds."""
    error_threshold: float = 1.5  # Multiplier for error increase detection
    std_threshold: float = 1.3    # Multiplier for std dev increase detection
    min_samples: int = 20         # Minimum samples needed for drift detection
    recent_window: int = 10       # Size of recent window for comparison
    long_term_window: int = 50    # Size of long-term window for baseline
    significance_level: float = 0.05  # P-value threshold for statistical tests


class DriftDetector:
    """
    Real-time drift detection system for monitoring model performance.
    
    This class monitors feature distributions and model predictions to detect
    when the data distribution has shifted significantly from the training distribution.
    """
    
    def __init__(self, feature_dim: int, threshold: float = 0.1, window_size: int = 1000):
        """
        Initialize the drift detector.
        
        Args:
            feature_dim: Dimensionality of the feature vectors
            threshold: Drift detection threshold (higher = less sensitive)
            window_size: Size of the reference window for drift detection
        """
        self.feature_dim = feature_dim
        self.threshold = threshold
        self.window_size = window_size
        
        # Store reference distribution statistics
        self.reference_features = deque(maxlen=window_size)
        self.reference_mean = None
        self.reference_std = None
        
        # Track drift scores
        self.drift_scores = []
        self.last_drift_score = 0.0
        
        # Initialize statistics
        self.is_initialized = False
        
    def _initialize_reference(self, features: torch.Tensor):
        """Initialize reference distribution statistics."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        # Flatten if needed
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Store features
        for i in range(features.shape[0]):
            self.reference_features.append(features[i])
        
        # Calculate reference statistics
        if len(self.reference_features) >= 20:  # Minimum samples
            ref_array = np.array(self.reference_features)
            self.reference_mean = np.mean(ref_array, axis=0)
            self.reference_std = np.std(ref_array, axis=0) + 1e-8  # Add small epsilon
            self.is_initialized = True
    
    def _calculate_drift_score(self, features: torch.Tensor) -> float:
        """Calculate drift score using KL divergence approximation."""
        if not self.is_initialized:
            return 0.0
            
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        # Flatten if needed
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Calculate current statistics
        current_mean = np.mean(features, axis=0)
        current_std = np.std(features, axis=0) + 1e-8
        
        # Calculate approximate KL divergence
        # KL(P||Q) ≈ log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²)/(2σ_q²) - 1/2
        kl_div = np.mean(
            np.log(current_std / self.reference_std) + 
            (self.reference_std**2 + (self.reference_mean - current_mean)**2) / 
            (2 * current_std**2) - 0.5
        )
        
        # Take absolute value and normalize
        drift_score = abs(kl_div)
        
        return drift_score
    
    def detect_drift(self, features: torch.Tensor) -> bool:
        """
        Detect drift in the current batch of features.
        
        Args:
            features: Input features tensor
            
        Returns:
            True if drift is detected, False otherwise
        """
        # Initialize reference if not done
        if not self.is_initialized:
            self._initialize_reference(features)
            self.last_drift_score = 0.0
            self.drift_scores.append(0.0)
            return False
        
        # Calculate drift score
        drift_score = self._calculate_drift_score(features)
        self.last_drift_score = drift_score
        self.drift_scores.append(drift_score)
        
        # Update reference window (sliding window)
        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features
            
        if features_np.ndim > 2:
            features_np = features_np.reshape(features_np.shape[0], -1)
        
        # Add new features to reference (with sliding window)
        for i in range(min(features_np.shape[0], 10)):  # Limit to prevent memory issues
            self.reference_features.append(features_np[i])
        
        # Update reference statistics
        if len(self.reference_features) >= 20:
            ref_array = np.array(self.reference_features)
            self.reference_mean = np.mean(ref_array, axis=0)
            self.reference_std = np.std(ref_array, axis=0) + 1e-8
        
        # Return drift detection result
        return drift_score > self.threshold
    
    def get_last_drift_score(self) -> float:
        """Get the last calculated drift score."""
        return self.last_drift_score
    
    def get_drift_history(self) -> List[float]:
        """Get the history of drift scores."""
        return self.drift_scores.copy()
    
    def reset(self):
        """Reset the drift detector."""
        self.reference_features.clear()
        self.reference_mean = None
        self.reference_std = None
        self.drift_scores.clear()
        self.last_drift_score = 0.0
        self.is_initialized = False
    
    def get_state(self) -> Dict:
        """Get the current state of the drift detector."""
        return {
            'threshold': self.threshold,
            'window_size': self.window_size,
            'feature_dim': self.feature_dim,
            'is_initialized': self.is_initialized,
            'drift_scores': self.drift_scores,
            'last_drift_score': self.last_drift_score
        }


def detect_model_drift(
    predictions: List[float], 
    actuals: List[float],
    thresholds: Optional[DriftThresholds] = None
) -> Dict[str, any]:
    """
    Detects model drift by analyzing prediction errors over time.
    
    Args:
        predictions: List of model predictions
        actuals: List of actual values
        thresholds: Configuration for drift detection thresholds
        
    Returns:
        Dictionary containing drift detection results:
        - drift_detected: Boolean indicating if drift was detected
        - error_trend: Trend analysis of prediction errors
        - std_dev_trend: Trend analysis of error standard deviation
        - recent_window_stats: Statistics for recent predictions
        - long_term_stats: Statistics for long-term baseline
        - drift_reasons: List of reasons why drift was detected
        - statistical_tests: Results of statistical significance tests
    """
    if thresholds is None:
        thresholds = DriftThresholds()
    
    # Validate inputs
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have the same length")
    
    if len(predictions) < thresholds.min_samples:
        return {
            'drift_detected': False,
            'error_trend': None,
            'std_dev_trend': None,
            'recent_window_stats': None,
            'long_term_stats': None,
            'drift_reasons': ['Insufficient data for drift detection'],
            'statistical_tests': None
        }
    
    # Calculate errors
    errors = np.array(predictions) - np.array(actuals)
    absolute_errors = np.abs(errors)
    
    # Calculate rolling statistics
    error_trend = _calculate_error_trend(absolute_errors)
    std_dev_trend = _calculate_std_dev_trend(absolute_errors)
    
    # Get recent and long-term windows
    recent_window_size = min(thresholds.recent_window, len(errors))
    long_term_window_size = min(thresholds.long_term_window, len(errors))
    
    recent_errors = absolute_errors[-recent_window_size:]
    long_term_errors = absolute_errors[-long_term_window_size:]
    
    # Calculate window statistics
    recent_window_stats = _calculate_window_stats(recent_errors, "recent")
    long_term_stats = _calculate_window_stats(long_term_errors, "long_term")
    
    # Perform statistical tests
    statistical_tests = _perform_statistical_tests(
        recent_errors, long_term_errors, thresholds.significance_level
    )
    
    # Detect drift
    drift_detected, drift_reasons = _detect_drift_conditions(
        recent_window_stats, long_term_stats, thresholds, statistical_tests
    )
    
    return {
        'drift_detected': drift_detected,
        'error_trend': error_trend,
        'std_dev_trend': std_dev_trend,
        'recent_window_stats': recent_window_stats,
        'long_term_stats': long_term_stats,
        'drift_reasons': drift_reasons,
        'statistical_tests': statistical_tests
    }


def _calculate_error_trend(errors: np.ndarray, window_size: int = 10) -> Dict[str, any]:
    """Calculate rolling error trend analysis."""
    if len(errors) < window_size:
        return {
            'rolling_mean': errors.mean(),
            'trend_slope': 0.0,
            'trend_correlation': 0.0,
            'is_increasing': False
        }
    
    # Calculate rolling mean
    rolling_means = []
    for i in range(len(errors) - window_size + 1):
        window_mean = errors[i:i + window_size].mean()
        rolling_means.append(window_mean)
    
    rolling_means = np.array(rolling_means)
    
    # Calculate trend
    x = np.arange(len(rolling_means))
    if len(x) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, rolling_means)
        trend_slope = slope
        trend_correlation = r_value
        is_increasing = slope > 0 and p_value < 0.05
    else:
        trend_slope = 0.0
        trend_correlation = 0.0
        is_increasing = False
    
    return {
        'rolling_mean': rolling_means[-1] if len(rolling_means) > 0 else errors.mean(),
        'trend_slope': trend_slope,
        'trend_correlation': trend_correlation,
        'is_increasing': is_increasing,
        'rolling_means': rolling_means.tolist()
    }


def _calculate_std_dev_trend(errors: np.ndarray, window_size: int = 10) -> Dict[str, any]:
    """Calculate rolling standard deviation trend analysis."""
    if len(errors) < window_size:
        return {
            'rolling_std': errors.std(),
            'trend_slope': 0.0,
            'trend_correlation': 0.0,
            'is_increasing': False
        }
    
    # Calculate rolling std
    rolling_stds = []
    for i in range(len(errors) - window_size + 1):
        window_std = errors[i:i + window_size].std()
        rolling_stds.append(window_std)
    
    rolling_stds = np.array(rolling_stds)
    
    # Calculate trend
    x = np.arange(len(rolling_stds))
    if len(x) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, rolling_stds)
        trend_slope = slope
        trend_correlation = r_value
        is_increasing = slope > 0 and p_value < 0.05
    else:
        trend_slope = 0.0
        trend_correlation = 0.0
        is_increasing = False
    
    return {
        'rolling_std': rolling_stds[-1] if len(rolling_stds) > 0 else errors.std(),
        'trend_slope': trend_slope,
        'trend_correlation': trend_correlation,
        'is_increasing': is_increasing,
        'rolling_stds': rolling_stds.tolist()
    }


def _calculate_window_stats(errors: np.ndarray, window_name: str) -> Dict[str, any]:
    """Calculate statistics for a window of errors."""
    return {
        'window_name': window_name,
        'mean_error': errors.mean(),
        'std_error': errors.std(),
        'median_error': np.median(errors),
        'min_error': errors.min(),
        'max_error': errors.max(),
        'q25_error': np.percentile(errors, 25),
        'q75_error': np.percentile(errors, 75),
        'sample_size': len(errors)
    }


def _perform_statistical_tests(
    recent_errors: np.ndarray, 
    long_term_errors: np.ndarray,
    significance_level: float
) -> Dict[str, any]:
    """Perform statistical tests to detect significant changes."""
    tests = {}
    
    # T-test for mean difference
    try:
        t_stat, t_pvalue = stats.ttest_ind(recent_errors, long_term_errors)
        tests['t_test'] = {
            'statistic': t_stat,
            'p_value': t_pvalue,
            'significant': t_pvalue < significance_level,
            'interpretation': 'Recent errors significantly different from long-term'
        }
    except Exception as e:
        tests['t_test'] = {'error': str(e)}
    
    # F-test for variance difference (Levene's test)
    try:
        f_stat, f_pvalue = stats.levene(recent_errors, long_term_errors)
        tests['levene_test'] = {
            'statistic': f_stat,
            'p_value': f_pvalue,
            'significant': f_pvalue < significance_level,
            'interpretation': 'Variance significantly different between windows'
        }
    except Exception as e:
        tests['levene_test'] = {'error': str(e)}
    
    # Mann-Whitney U test (non-parametric alternative)
    try:
        u_stat, u_pvalue = stats.mannwhitneyu(
            recent_errors, long_term_errors, alternative='two-sided'
        )
        tests['mann_whitney'] = {
            'statistic': u_stat,
            'p_value': u_pvalue,
            'significant': u_pvalue < significance_level,
            'interpretation': 'Distribution significantly different between windows'
        }
    except Exception as e:
        tests['mann_whitney'] = {'error': str(e)}
    
    # Kolmogorov-Smirnov test for distribution difference
    try:
        ks_stat, ks_pvalue = stats.ks_2samp(recent_errors, long_term_errors)
        tests['ks_test'] = {
            'statistic': ks_stat,
            'p_value': ks_pvalue,
            'significant': ks_pvalue < significance_level,
            'interpretation': 'Cumulative distributions significantly different'
        }
    except Exception as e:
        tests['ks_test'] = {'error': str(e)}
    
    return tests


def _detect_drift_conditions(
    recent_stats: Dict[str, any],
    long_term_stats: Dict[str, any],
    thresholds: DriftThresholds,
    statistical_tests: Dict[str, any]
) -> Tuple[bool, List[str]]:
    """Detect drift based on various conditions."""
    drift_detected = False
    drift_reasons = []
    
    # Check error increase
    error_ratio = recent_stats['mean_error'] / long_term_stats['mean_error']
    if error_ratio > thresholds.error_threshold:
        drift_detected = True
        drift_reasons.append(
            f"Recent error ({recent_stats['mean_error']:.4f}) exceeds "
            f"long-term average ({long_term_stats['mean_error']:.4f}) by "
            f"{error_ratio:.2f}x (threshold: {thresholds.error_threshold}x)"
        )
    
    # Check std dev increase
    if long_term_stats['std_error'] > 0:
        std_ratio = recent_stats['std_error'] / long_term_stats['std_error']
        if std_ratio > thresholds.std_threshold:
            drift_detected = True
            drift_reasons.append(
                f"Recent error std dev ({recent_stats['std_error']:.4f}) exceeds "
                f"long-term std dev ({long_term_stats['std_error']:.4f}) by "
                f"{std_ratio:.2f}x (threshold: {thresholds.std_threshold}x)"
            )
    
    # Check statistical significance
    significant_tests = []
    for test_name, test_result in statistical_tests.items():
        if isinstance(test_result, dict) and test_result.get('significant', False):
            significant_tests.append(test_name)
    
    if significant_tests:
        drift_detected = True
        drift_reasons.append(
            f"Statistical tests indicate significant change: {', '.join(significant_tests)}"
        )
    
    # Check for extreme values
    if recent_stats['max_error'] > long_term_stats['max_error'] * 2:
        drift_detected = True
        drift_reasons.append(
            f"Recent maximum error ({recent_stats['max_error']:.4f}) is "
            f"significantly higher than long-term maximum ({long_term_stats['max_error']:.4f})"
        )
    
    if not drift_detected:
        drift_reasons.append("No drift detected - model performance is stable")
    
    return drift_detected, drift_reasons


def calculate_drift_score(drift_result: Dict[str, any]) -> float:
    """
    Calculate a single drift score from 0 to 1 based on drift detection results.
    
    Args:
        drift_result: Result from detect_model_drift function
        
    Returns:
        Float score from 0 (no drift) to 1 (high drift)
    """
    if not drift_result['drift_detected']:
        return 0.0
    
    score = 0.0
    
    # Error ratio component
    recent_stats = drift_result['recent_window_stats']
    long_term_stats = drift_result['long_term_stats']
    
    if long_term_stats['mean_error'] > 0:
        error_ratio = recent_stats['mean_error'] / long_term_stats['mean_error']
        score += min(0.4, (error_ratio - 1) * 0.2)  # Up to 0.4 points
    
    # Std dev ratio component
    if long_term_stats['std_error'] > 0:
        std_ratio = recent_stats['std_error'] / long_term_stats['std_error']
        score += min(0.3, (std_ratio - 1) * 0.15)  # Up to 0.3 points
    
    # Statistical significance component
    significant_tests = 0
    for test_name, test_result in drift_result['statistical_tests'].items():
        if isinstance(test_result, dict) and test_result.get('significant', False):
            significant_tests += 1
    
    score += min(0.3, significant_tests * 0.075)  # Up to 0.3 points
    
    return min(1.0, score)


def get_drift_recommendations(drift_result: Dict[str, any]) -> List[str]:
    """
    Get actionable recommendations based on drift detection results.
    
    Args:
        drift_result: Result from detect_model_drift function
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    if not drift_result['drift_detected']:
        recommendations.append("✅ Model performance is stable - continue monitoring")
        return recommendations
    
    drift_score = calculate_drift_score(drift_result)
    
    if drift_score < 0.3:
        recommendations.append("⚠️ Mild drift detected - increase monitoring frequency")
    elif drift_score < 0.6:
        recommendations.append("🔶 Moderate drift detected - consider model retraining")
    else:
        recommendations.append("🔴 Severe drift detected - immediate model retraining recommended")
    
    # Specific recommendations based on drift reasons
    for reason in drift_result['drift_reasons']:
        if "error" in reason.lower() and "exceeds" in reason.lower():
            recommendations.append("📊 Investigate recent data quality and feature distributions")
        if "std dev" in reason.lower():
            recommendations.append("📈 Check for increased prediction uncertainty - may need model ensemble")
        if "statistical" in reason.lower():
            recommendations.append("🔬 Statistically significant change detected - validate with domain experts")
    
    # Error trend recommendations
    if drift_result['error_trend'] and drift_result['error_trend']['is_increasing']:
        recommendations.append("📈 Error trend is increasing - monitor closely and prepare for retraining")
    
    # Std dev trend recommendations
    if drift_result['std_dev_trend'] and drift_result['std_dev_trend']['is_increasing']:
        recommendations.append("📊 Prediction variance is increasing - consider model calibration")
    
    return recommendations 
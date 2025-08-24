# Purpose: Mixture of Experts layer for routing between fast and slow models
# Author: WicketWise Team, Last Modified: 2025-08-23

"""
Mixture of Experts (MoE) architecture that routes inference requests between:
- Fast Models: XGBoost, TabTransformer (low latency)
- Slow Models: Crickformer, GNN fusion (high accuracy)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Model type classification"""
    FAST = "fast"
    SLOW = "slow"


class LatencyRequirement(Enum):
    """Latency requirement levels"""
    ULTRA_FAST = "ultra_fast"  # < 100ms
    FAST = "fast"              # < 500ms  
    NORMAL = "normal"          # < 2s
    ACCURATE = "accurate"      # < 5s
    BATCH = "batch"            # No limit


@dataclass
class ModelConfig:
    """Configuration for individual models in MoE"""
    name: str
    model_type: ModelType
    max_latency_ms: int
    accuracy_score: float
    memory_usage_mb: int
    is_available: bool = True


@dataclass
class InferenceRequest:
    """Request for model inference"""
    inputs: Dict[str, Any]
    latency_requirement: LatencyRequirement
    accuracy_threshold: Optional[float] = None
    model_preference: Optional[str] = None


@dataclass
class InferenceResult:
    """Result from model inference"""
    predictions: Dict[str, Any]
    model_used: str
    latency_ms: float
    confidence_score: float
    metadata: Dict[str, Any]


class BaseModel(ABC):
    """Abstract base class for all models in MoE"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_loaded = False
        self.last_inference_time = 0
        
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory"""
        pass
    
    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions on inputs"""
        pass
    
    @abstractmethod
    def get_confidence(self, predictions: Dict[str, Any]) -> float:
        """Get confidence score for predictions"""
        pass
    
    def unload(self) -> None:
        """Unload model from memory"""
        self.is_loaded = False


class FastXGBoostModel(BaseModel):
    """Fast XGBoost model for low-latency predictions"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.win_prob_model = None
        self.outcome_model = None
        
    def load(self) -> None:
        """Load XGBoost models"""
        try:
            # Initialize lightweight gradient boosting models
            self.win_prob_model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            self.outcome_model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            self.is_loaded = True
            logger.info(f"✅ Loaded {self.config.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.config.name}: {e}")
            self.config.is_available = False
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make fast predictions using XGBoost"""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.config.name} not loaded")
        
        # Extract tabular features
        features = self._extract_features(inputs)
        
        # Mock predictions for now (would use trained models)
        win_prob = np.random.random()
        outcome_probs = np.random.dirichlet([1, 1, 1, 1, 1, 1, 1])  # 7 outcomes
        
        predictions = {
            "win_probability": win_prob,
            "next_ball_outcome": {
                "probabilities": outcome_probs.tolist(),
                "predicted_class": int(np.argmax(outcome_probs))
            },
            "odds_mispricing": np.random.random() > 0.8
        }
        
        return predictions
    
    def get_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate confidence based on prediction entropy"""
        outcome_probs = predictions["next_ball_outcome"]["probabilities"]
        entropy = -np.sum([p * np.log(p + 1e-8) for p in outcome_probs])
        max_entropy = np.log(len(outcome_probs))
        confidence = 1.0 - (entropy / max_entropy)
        return confidence
    
    def _extract_features(self, inputs: Dict[str, Any]) -> np.ndarray:
        """Extract tabular features from inputs"""
        # Mock feature extraction (would process real inputs)
        return np.random.random((1, 20))


class FastTabTransformerModel(BaseModel):
    """Fast TabTransformer model for structured data"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        
    def load(self) -> None:
        """Load TabTransformer model"""
        try:
            # Simple linear model as placeholder for TabTransformer
            self.model = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 8)  # Output features
            )
            self.is_loaded = True
            logger.info(f"✅ Loaded {self.config.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.config.name}: {e}")
            self.config.is_available = False
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using TabTransformer"""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.config.name} not loaded")
        
        # Mock predictions
        win_prob = np.random.random()
        outcome_probs = np.random.dirichlet([1, 1, 1, 1, 1, 1, 1])
        
        predictions = {
            "win_probability": win_prob,
            "next_ball_outcome": {
                "probabilities": outcome_probs.tolist(),
                "predicted_class": int(np.argmax(outcome_probs))
            },
            "odds_mispricing": np.random.random() > 0.7
        }
        
        return predictions
    
    def get_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate confidence score"""
        return np.random.uniform(0.7, 0.9)


class SlowCrickformerModel(BaseModel):
    """Slow but accurate Crickformer model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        
    def load(self) -> None:
        """Load Crickformer model"""
        try:
            # Placeholder for actual Crickformer loading
            self.model = "crickformer_placeholder"
            self.is_loaded = True
            logger.info(f"✅ Loaded {self.config.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.config.name}: {e}")
            self.config.is_available = False
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make high-accuracy predictions"""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.config.name} not loaded")
        
        # Simulate slow but accurate prediction
        time.sleep(0.1)  # Simulate processing time
        
        # Higher quality predictions
        win_prob = np.random.beta(2, 2)  # More realistic distribution
        outcome_probs = np.random.dirichlet([2, 1, 1, 3, 1, 2, 1])  # Biased toward realistic outcomes
        
        predictions = {
            "win_probability": win_prob,
            "next_ball_outcome": {
                "probabilities": outcome_probs.tolist(),
                "predicted_class": int(np.argmax(outcome_probs))
            },
            "odds_mispricing": np.random.random() > 0.9,
            "detailed_analysis": {
                "player_matchup_score": np.random.random(),
                "venue_advantage": np.random.random(),
                "weather_impact": np.random.random()
            }
        }
        
        return predictions
    
    def get_confidence(self, predictions: Dict[str, Any]) -> float:
        """Higher confidence for slow model"""
        return np.random.uniform(0.8, 0.95)


class MixtureOfExperts:
    """Main MoE orchestrator that routes requests to appropriate models"""
    
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.routing_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
        # Initialize model configurations
        self._initialize_model_configs()
        
    def _initialize_model_configs(self) -> None:
        """Initialize all model configurations"""
        configs = [
            ModelConfig(
                name="fast_xgboost",
                model_type=ModelType.FAST,
                max_latency_ms=50,
                accuracy_score=0.75,
                memory_usage_mb=100
            ),
            ModelConfig(
                name="fast_tabtransformer", 
                model_type=ModelType.FAST,
                max_latency_ms=100,
                accuracy_score=0.80,
                memory_usage_mb=200
            ),
            ModelConfig(
                name="slow_crickformer",
                model_type=ModelType.SLOW,
                max_latency_ms=2000,
                accuracy_score=0.92,
                memory_usage_mb=1500
            )
        ]
        
        # Initialize models
        for config in configs:
            if config.name == "fast_xgboost":
                self.models[config.name] = FastXGBoostModel(config)
            elif config.name == "fast_tabtransformer":
                self.models[config.name] = FastTabTransformerModel(config)
            elif config.name == "slow_crickformer":
                self.models[config.name] = SlowCrickformerModel(config)
    
    def load_models(self, model_names: Optional[List[str]] = None) -> None:
        """Load specified models or all models"""
        models_to_load = model_names or list(self.models.keys())
        
        for model_name in models_to_load:
            if model_name in self.models:
                self.models[model_name].load()
            else:
                logger.warning(f"⚠️ Unknown model: {model_name}")
    
    def route_request(self, request: InferenceRequest) -> str:
        """Route inference request to appropriate model"""
        
        # If specific model requested, use it if available
        if request.model_preference and request.model_preference in self.models:
            model = self.models[request.model_preference]
            if model.config.is_available and model.is_loaded:
                return request.model_preference
        
        # Route based on latency requirement
        latency_limits = {
            LatencyRequirement.ULTRA_FAST: 100,
            LatencyRequirement.FAST: 500,
            LatencyRequirement.NORMAL: 2000,
            LatencyRequirement.ACCURATE: 5000,
            LatencyRequirement.BATCH: float('inf')
        }
        
        max_latency = latency_limits[request.latency_requirement]
        
        # Find suitable models
        suitable_models = []
        for name, model in self.models.items():
            if (model.config.is_available and 
                model.is_loaded and 
                model.config.max_latency_ms <= max_latency):
                suitable_models.append((name, model))
        
        if not suitable_models:
            raise RuntimeError(f"No suitable models for latency requirement: {request.latency_requirement}")
        
        # Choose best model based on accuracy and latency
        if request.latency_requirement in [LatencyRequirement.ACCURATE, LatencyRequirement.BATCH]:
            # Prioritize accuracy
            best_model = max(suitable_models, key=lambda x: x[1].config.accuracy_score)
        else:
            # Prioritize speed
            best_model = min(suitable_models, key=lambda x: x[1].config.max_latency_ms)
        
        return best_model[0]
    
    def predict(self, request: InferenceRequest) -> InferenceResult:
        """Make prediction using routed model"""
        start_time = time.time()
        
        try:
            # Route request
            selected_model_name = self.route_request(request)
            selected_model = self.models[selected_model_name]
            
            logger.info(f"🎯 Routing to model: {selected_model_name}")
            
            # Make prediction
            predictions = selected_model.predict(request.inputs)
            confidence = selected_model.get_confidence(predictions)
            
            # Calculate actual latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = InferenceResult(
                predictions=predictions,
                model_used=selected_model_name,
                latency_ms=latency_ms,
                confidence_score=confidence,
                metadata={
                    "latency_requirement": request.latency_requirement.value,
                    "model_type": selected_model.config.model_type.value,
                    "accuracy_score": selected_model.config.accuracy_score
                }
            )
            
            # Log routing decision
            self._log_routing_decision(request, result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def _log_routing_decision(self, request: InferenceRequest, result: InferenceResult) -> None:
        """Log routing decision for analysis"""
        routing_entry = {
            "timestamp": time.time(),
            "latency_requirement": request.latency_requirement.value,
            "model_used": result.model_used,
            "actual_latency_ms": result.latency_ms,
            "confidence_score": result.confidence_score,
            "accuracy_threshold_met": (
                request.accuracy_threshold is None or 
                result.confidence_score >= request.accuracy_threshold
            )
        }
        
        self.routing_history.append(routing_entry)
        
        # Keep only recent history
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        if not self.routing_history:
            return {"message": "No routing history available"}
        
        stats = {}
        
        # Overall stats
        total_requests = len(self.routing_history)
        avg_latency = np.mean([entry["actual_latency_ms"] for entry in self.routing_history])
        avg_confidence = np.mean([entry["confidence_score"] for entry in self.routing_history])
        
        stats["overall"] = {
            "total_requests": total_requests,
            "average_latency_ms": avg_latency,
            "average_confidence": avg_confidence
        }
        
        # Model usage stats
        model_usage = {}
        for entry in self.routing_history:
            model = entry["model_used"]
            if model not in model_usage:
                model_usage[model] = {"count": 0, "latencies": [], "confidences": []}
            
            model_usage[model]["count"] += 1
            model_usage[model]["latencies"].append(entry["actual_latency_ms"])
            model_usage[model]["confidences"].append(entry["confidence_score"])
        
        for model, data in model_usage.items():
            stats[model] = {
                "usage_count": data["count"],
                "usage_percentage": data["count"] / total_requests * 100,
                "avg_latency_ms": np.mean(data["latencies"]),
                "avg_confidence": np.mean(data["confidences"])
            }
        
        return stats
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available models"""
        model_info = {}
        
        for name, model in self.models.items():
            model_info[name] = {
                "type": model.config.model_type.value,
                "max_latency_ms": model.config.max_latency_ms,
                "accuracy_score": model.config.accuracy_score,
                "memory_usage_mb": model.config.memory_usage_mb,
                "is_available": model.config.is_available,
                "is_loaded": model.is_loaded
            }
        
        return model_info

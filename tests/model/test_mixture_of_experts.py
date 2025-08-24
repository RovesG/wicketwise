# Purpose: Unit tests for Mixture of Experts layer
# Author: WicketWise Team, Last Modified: 2025-08-23

import pytest
import time
from unittest.mock import Mock, patch
import numpy as np

from crickformers.model.mixture_of_experts import (
    MixtureOfExperts,
    InferenceRequest,
    LatencyRequirement,
    ModelType,
    ModelConfig,
    FastXGBoostModel,
    FastTabTransformerModel,
    SlowCrickformerModel
)


class TestModelConfig:
    """Test ModelConfig dataclass"""
    
    def test_model_config_creation(self):
        """Test creating a model configuration"""
        config = ModelConfig(
            name="test_model",
            model_type=ModelType.FAST,
            max_latency_ms=100,
            accuracy_score=0.8,
            memory_usage_mb=200
        )
        
        assert config.name == "test_model"
        assert config.model_type == ModelType.FAST
        assert config.max_latency_ms == 100
        assert config.accuracy_score == 0.8
        assert config.memory_usage_mb == 200
        assert config.is_available is True  # Default value


class TestFastXGBoostModel:
    """Test FastXGBoostModel implementation"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        config = ModelConfig(
            name="xgboost_test",
            model_type=ModelType.FAST,
            max_latency_ms=50,
            accuracy_score=0.75,
            memory_usage_mb=100
        )
        
        model = FastXGBoostModel(config)
        assert model.config.name == "xgboost_test"
        assert not model.is_loaded
        assert model.win_prob_model is None
        assert model.outcome_model is None
    
    def test_model_loading(self):
        """Test model loading"""
        config = ModelConfig(
            name="xgboost_test",
            model_type=ModelType.FAST,
            max_latency_ms=50,
            accuracy_score=0.75,
            memory_usage_mb=100
        )
        
        model = FastXGBoostModel(config)
        model.load()
        
        assert model.is_loaded
        assert model.win_prob_model is not None
        assert model.outcome_model is not None
    
    def test_model_prediction(self):
        """Test model prediction"""
        config = ModelConfig(
            name="xgboost_test",
            model_type=ModelType.FAST,
            max_latency_ms=50,
            accuracy_score=0.75,
            memory_usage_mb=100
        )
        
        model = FastXGBoostModel(config)
        model.load()
        
        inputs = {"numeric_features": [1, 2, 3], "categorical_features": {"team": "RCB"}}
        predictions = model.predict(inputs)
        
        assert "win_probability" in predictions
        assert "next_ball_outcome" in predictions
        assert "odds_mispricing" in predictions
        assert 0 <= predictions["win_probability"] <= 1
        assert len(predictions["next_ball_outcome"]["probabilities"]) == 7
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        config = ModelConfig(
            name="xgboost_test",
            model_type=ModelType.FAST,
            max_latency_ms=50,
            accuracy_score=0.75,
            memory_usage_mb=100
        )
        
        model = FastXGBoostModel(config)
        predictions = {
            "next_ball_outcome": {
                "probabilities": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]
            }
        }
        
        confidence = model.get_confidence(predictions)
        assert 0 <= confidence <= 1
    
    def test_prediction_without_loading(self):
        """Test prediction fails without loading model"""
        config = ModelConfig(
            name="xgboost_test",
            model_type=ModelType.FAST,
            max_latency_ms=50,
            accuracy_score=0.75,
            memory_usage_mb=100
        )
        
        model = FastXGBoostModel(config)
        inputs = {"test": "data"}
        
        with pytest.raises(RuntimeError, match="Model .* not loaded"):
            model.predict(inputs)


class TestFastTabTransformerModel:
    """Test FastTabTransformerModel implementation"""
    
    def test_model_initialization(self):
        """Test TabTransformer initialization"""
        config = ModelConfig(
            name="tabtransformer_test",
            model_type=ModelType.FAST,
            max_latency_ms=100,
            accuracy_score=0.80,
            memory_usage_mb=200
        )
        
        model = FastTabTransformerModel(config)
        assert model.config.name == "tabtransformer_test"
        assert not model.is_loaded
        assert model.model is None
    
    def test_model_loading(self):
        """Test TabTransformer loading"""
        config = ModelConfig(
            name="tabtransformer_test",
            model_type=ModelType.FAST,
            max_latency_ms=100,
            accuracy_score=0.80,
            memory_usage_mb=200
        )
        
        model = FastTabTransformerModel(config)
        model.load()
        
        assert model.is_loaded
        assert model.model is not None
    
    def test_model_prediction(self):
        """Test TabTransformer prediction"""
        config = ModelConfig(
            name="tabtransformer_test",
            model_type=ModelType.FAST,
            max_latency_ms=100,
            accuracy_score=0.80,
            memory_usage_mb=200
        )
        
        model = FastTabTransformerModel(config)
        model.load()
        
        inputs = {"features": [1, 2, 3]}
        predictions = model.predict(inputs)
        
        assert "win_probability" in predictions
        assert "next_ball_outcome" in predictions
        assert "odds_mispricing" in predictions


class TestSlowCrickformerModel:
    """Test SlowCrickformerModel implementation"""
    
    def test_model_initialization(self):
        """Test Crickformer initialization"""
        config = ModelConfig(
            name="crickformer_test",
            model_type=ModelType.SLOW,
            max_latency_ms=2000,
            accuracy_score=0.92,
            memory_usage_mb=1500
        )
        
        model = SlowCrickformerModel(config)
        assert model.config.name == "crickformer_test"
        assert not model.is_loaded
        assert model.model is None
    
    def test_model_prediction_has_detailed_analysis(self):
        """Test that slow model provides detailed analysis"""
        config = ModelConfig(
            name="crickformer_test",
            model_type=ModelType.SLOW,
            max_latency_ms=2000,
            accuracy_score=0.92,
            memory_usage_mb=1500
        )
        
        model = SlowCrickformerModel(config)
        model.load()
        
        inputs = {"complex_features": "test"}
        predictions = model.predict(inputs)
        
        assert "detailed_analysis" in predictions
        assert "player_matchup_score" in predictions["detailed_analysis"]
        assert "venue_advantage" in predictions["detailed_analysis"]
        assert "weather_impact" in predictions["detailed_analysis"]
    
    def test_slow_model_latency(self):
        """Test that slow model actually takes time"""
        config = ModelConfig(
            name="crickformer_test",
            model_type=ModelType.SLOW,
            max_latency_ms=2000,
            accuracy_score=0.92,
            memory_usage_mb=1500
        )
        
        model = SlowCrickformerModel(config)
        model.load()
        
        start_time = time.time()
        inputs = {"test": "data"}
        model.predict(inputs)
        elapsed = (time.time() - start_time) * 1000
        
        # Should take at least 100ms due to sleep
        assert elapsed >= 90  # Allow for some timing variance


class TestMixtureOfExperts:
    """Test MixtureOfExperts orchestrator"""
    
    def test_moe_initialization(self):
        """Test MoE initialization"""
        moe = MixtureOfExperts()
        
        assert len(moe.models) == 3
        assert "fast_xgboost" in moe.models
        assert "fast_tabtransformer" in moe.models
        assert "slow_crickformer" in moe.models
        assert len(moe.routing_history) == 0
    
    def test_model_loading(self):
        """Test loading all models"""
        moe = MixtureOfExperts()
        moe.load_models()
        
        for model in moe.models.values():
            assert model.is_loaded
    
    def test_selective_model_loading(self):
        """Test loading specific models"""
        moe = MixtureOfExperts()
        moe.load_models(["fast_xgboost"])
        
        assert moe.models["fast_xgboost"].is_loaded
        assert not moe.models["fast_tabtransformer"].is_loaded
        assert not moe.models["slow_crickformer"].is_loaded
    
    def test_routing_ultra_fast_requirement(self):
        """Test routing for ultra-fast latency requirement"""
        moe = MixtureOfExperts()
        moe.load_models()
        
        request = InferenceRequest(
            inputs={"test": "data"},
            latency_requirement=LatencyRequirement.ULTRA_FAST
        )
        
        selected_model = moe.route_request(request)
        
        # Should select fastest model (XGBoost with 50ms limit)
        assert selected_model == "fast_xgboost"
    
    def test_routing_accurate_requirement(self):
        """Test routing for accuracy requirement"""
        moe = MixtureOfExperts()
        moe.load_models()
        
        request = InferenceRequest(
            inputs={"test": "data"},
            latency_requirement=LatencyRequirement.ACCURATE
        )
        
        selected_model = moe.route_request(request)
        
        # Should select most accurate model (Crickformer with 0.92 accuracy)
        assert selected_model == "slow_crickformer"
    
    def test_routing_with_model_preference(self):
        """Test routing with specific model preference"""
        moe = MixtureOfExperts()
        moe.load_models()
        
        request = InferenceRequest(
            inputs={"test": "data"},
            latency_requirement=LatencyRequirement.FAST,
            model_preference="fast_tabtransformer"
        )
        
        selected_model = moe.route_request(request)
        assert selected_model == "fast_tabtransformer"
    
    def test_routing_no_suitable_models(self):
        """Test routing when no models meet requirements"""
        moe = MixtureOfExperts()
        # Don't load any models
        
        request = InferenceRequest(
            inputs={"test": "data"},
            latency_requirement=LatencyRequirement.FAST
        )
        
        with pytest.raises(RuntimeError, match="No suitable models"):
            moe.route_request(request)
    
    def test_end_to_end_prediction(self):
        """Test complete prediction flow"""
        moe = MixtureOfExperts()
        moe.load_models()
        
        request = InferenceRequest(
            inputs={"numeric_features": [1, 2, 3]},
            latency_requirement=LatencyRequirement.FAST
        )
        
        result = moe.predict(request)
        
        assert result.predictions is not None
        assert result.model_used in moe.models
        assert result.latency_ms > 0
        assert 0 <= result.confidence_score <= 1
        assert "latency_requirement" in result.metadata
        assert "model_type" in result.metadata
        assert "accuracy_score" in result.metadata
    
    def test_routing_history_logging(self):
        """Test that routing decisions are logged"""
        moe = MixtureOfExperts()
        moe.load_models()
        
        request = InferenceRequest(
            inputs={"test": "data"},
            latency_requirement=LatencyRequirement.FAST
        )
        
        initial_history_length = len(moe.routing_history)
        moe.predict(request)
        
        assert len(moe.routing_history) == initial_history_length + 1
        
        latest_entry = moe.routing_history[-1]
        assert "timestamp" in latest_entry
        assert "latency_requirement" in latest_entry
        assert "model_used" in latest_entry
        assert "actual_latency_ms" in latest_entry
        assert "confidence_score" in latest_entry
    
    def test_routing_stats_generation(self):
        """Test routing statistics generation"""
        moe = MixtureOfExperts()
        moe.load_models()
        
        # Make several predictions
        for i in range(5):
            request = InferenceRequest(
                inputs={"test": f"data_{i}"},
                latency_requirement=LatencyRequirement.FAST
            )
            moe.predict(request)
        
        stats = moe.get_routing_stats()
        
        assert "overall" in stats
        assert stats["overall"]["total_requests"] == 5
        assert "average_latency_ms" in stats["overall"]
        assert "average_confidence" in stats["overall"]
        
        # Should have stats for models that were used
        used_models = set(entry["model_used"] for entry in moe.routing_history)
        for model in used_models:
            assert model in stats
            assert "usage_count" in stats[model]
            assert "usage_percentage" in stats[model]
            assert "avg_latency_ms" in stats[model]
            assert "avg_confidence" in stats[model]
    
    def test_get_available_models(self):
        """Test getting available models information"""
        moe = MixtureOfExperts()
        moe.load_models(["fast_xgboost"])
        
        model_info = moe.get_available_models()
        
        assert len(model_info) == 3
        assert "fast_xgboost" in model_info
        assert "fast_tabtransformer" in model_info
        assert "slow_crickformer" in model_info
        
        # Check XGBoost is loaded
        xgboost_info = model_info["fast_xgboost"]
        assert xgboost_info["type"] == "fast"
        assert xgboost_info["max_latency_ms"] == 50
        assert xgboost_info["is_loaded"] is True
        
        # Check TabTransformer is not loaded
        tabtransformer_info = model_info["fast_tabtransformer"]
        assert tabtransformer_info["is_loaded"] is False
    
    def test_routing_history_size_limit(self):
        """Test that routing history doesn't grow indefinitely"""
        moe = MixtureOfExperts()
        moe.load_models()
        
        # Simulate many predictions
        for i in range(1100):  # More than the 1000 limit
            routing_entry = {
                "timestamp": time.time(),
                "latency_requirement": "fast",
                "model_used": "fast_xgboost",
                "actual_latency_ms": 50,
                "confidence_score": 0.8,
                "accuracy_threshold_met": True
            }
            moe.routing_history.append(routing_entry)
        
        # Trigger the size limit check
        request = InferenceRequest(
            inputs={"test": "data"},
            latency_requirement=LatencyRequirement.FAST
        )
        moe.predict(request)
        
        # Should be limited to 1000 entries
        assert len(moe.routing_history) <= 1000


class TestInferenceRequest:
    """Test InferenceRequest dataclass"""
    
    def test_inference_request_creation(self):
        """Test creating an inference request"""
        request = InferenceRequest(
            inputs={"test": "data"},
            latency_requirement=LatencyRequirement.FAST,
            accuracy_threshold=0.8,
            model_preference="fast_xgboost"
        )
        
        assert request.inputs == {"test": "data"}
        assert request.latency_requirement == LatencyRequirement.FAST
        assert request.accuracy_threshold == 0.8
        assert request.model_preference == "fast_xgboost"
    
    def test_inference_request_minimal(self):
        """Test creating minimal inference request"""
        request = InferenceRequest(
            inputs={"test": "data"},
            latency_requirement=LatencyRequirement.NORMAL
        )
        
        assert request.accuracy_threshold is None
        assert request.model_preference is None


class TestLatencyRequirements:
    """Test latency requirement routing logic"""
    
    def test_all_latency_requirements(self):
        """Test that all latency requirements can be handled"""
        moe = MixtureOfExperts()
        moe.load_models()
        
        latency_requirements = [
            LatencyRequirement.ULTRA_FAST,
            LatencyRequirement.FAST,
            LatencyRequirement.NORMAL,
            LatencyRequirement.ACCURATE,
            LatencyRequirement.BATCH
        ]
        
        for req in latency_requirements:
            request = InferenceRequest(
                inputs={"test": "data"},
                latency_requirement=req
            )
            
            # Should not raise an exception
            selected_model = moe.route_request(request)
            assert selected_model in moe.models
            
            # Verify the selected model meets latency requirement
            model = moe.models[selected_model]
            if req == LatencyRequirement.ULTRA_FAST:
                assert model.config.max_latency_ms <= 100
            elif req == LatencyRequirement.FAST:
                assert model.config.max_latency_ms <= 500
            elif req == LatencyRequirement.NORMAL:
                assert model.config.max_latency_ms <= 2000

# Purpose: Unit tests for MoE Orchestrator
# Author: WicketWise Team, Last Modified: 2025-08-23

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from crickformers.orchestration.moe_orchestrator import (
    MoEOrchestrator,
    create_moe_orchestrator
)
from crickformers.model.mixture_of_experts import LatencyRequirement


class TestMoEOrchestrator:
    """Test MoE Orchestrator functionality"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing"""
        return MoEOrchestrator()
    
    @pytest.fixture
    def mock_moe(self):
        """Create mock MoE instance"""
        mock = Mock()
        mock.get_available_models.return_value = {
            "fast_xgboost": {"is_loaded": True},
            "slow_crickformer": {"is_loaded": True}
        }
        mock.get_routing_stats.return_value = {"overall": {"total_requests": 0}}
        return mock
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert not orchestrator.is_initialized
        assert orchestrator.performance_metrics["total_requests"] == 0
        assert orchestrator.performance_metrics["error_count"] == 0
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, orchestrator, mock_moe):
        """Test successful initialization"""
        with patch.object(orchestrator, 'moe', mock_moe):
            await orchestrator.initialize()
            
            assert orchestrator.is_initialized
            mock_moe.load_models.assert_called_once_with(None)
            mock_moe.get_available_models.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_no_models_loaded(self, orchestrator):
        """Test initialization fails when no models loaded"""
        mock_moe = Mock()
        mock_moe.get_available_models.return_value = {
            "model1": {"is_loaded": False},
            "model2": {"is_loaded": False}
        }
        
        with patch.object(orchestrator, 'moe', mock_moe):
            with pytest.raises(RuntimeError, match="No models successfully loaded"):
                await orchestrator.initialize()
    
    @pytest.mark.asyncio
    async def test_initialize_with_config(self):
        """Test initialization with specific configuration"""
        config = {"models_to_load": ["fast_xgboost"]}
        orchestrator = MoEOrchestrator(config)
        
        mock_moe = Mock()
        mock_moe.get_available_models.return_value = {
            "fast_xgboost": {"is_loaded": True}
        }
        
        with patch.object(orchestrator, 'moe', mock_moe):
            await orchestrator.initialize()
            
            mock_moe.load_models.assert_called_once_with(["fast_xgboost"])
            assert orchestrator.is_initialized
    
    @pytest.mark.asyncio
    async def test_predict_not_initialized(self, orchestrator):
        """Test prediction fails when not initialized"""
        inputs = {"test": "data"}
        
        result = await orchestrator.predict(inputs)
        
        assert result["status"] == "error"
        assert "not initialized" in result["metadata"]["error"]
    
    @pytest.mark.asyncio
    async def test_predict_success(self, orchestrator, mock_moe):
        """Test successful prediction"""
        # Setup mock
        mock_result = Mock()
        mock_result.predictions = {"win_probability": 0.7}
        mock_result.model_used = "fast_xgboost"
        mock_result.latency_ms = 45.0
        mock_result.confidence_score = 0.85
        mock_result.metadata = {"model_type": "fast"}
        
        mock_moe.predict.return_value = mock_result
        
        with patch.object(orchestrator, 'moe', mock_moe):
            orchestrator.is_initialized = True
            
            inputs = {"numeric_features": [1, 2, 3]}
            result = await orchestrator.predict(inputs, latency_requirement="fast")
            
            assert result["status"] == "success"
            assert result["predictions"] == {"win_probability": 0.7}
            assert result["metadata"]["model_used"] == "fast_xgboost"
            assert result["metadata"]["latency_ms"] == 45.0
            assert result["metadata"]["confidence_score"] == 0.85
            assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_predict_with_all_parameters(self, orchestrator, mock_moe):
        """Test prediction with all optional parameters"""
        mock_result = Mock()
        mock_result.predictions = {"outcome": "boundary"}
        mock_result.model_used = "slow_crickformer"
        mock_result.latency_ms = 150.0
        mock_result.confidence_score = 0.92
        mock_result.metadata = {"model_type": "slow"}
        
        mock_moe.predict.return_value = mock_result
        
        with patch.object(orchestrator, 'moe', mock_moe):
            orchestrator.is_initialized = True
            
            result = await orchestrator.predict(
                inputs={"complex": "data"},
                latency_requirement="accurate",
                accuracy_threshold=0.9,
                model_preference="slow_crickformer"
            )
            
            assert result["status"] == "success"
            assert result["metadata"]["model_used"] == "slow_crickformer"
            
            # Verify the request was created correctly
            call_args = mock_moe.predict.call_args[0][0]
            assert call_args.latency_requirement == LatencyRequirement.ACCURATE
            assert call_args.accuracy_threshold == 0.9
            assert call_args.model_preference == "slow_crickformer"
    
    @pytest.mark.asyncio
    async def test_predict_invalid_latency_requirement(self, orchestrator):
        """Test prediction with invalid latency requirement"""
        orchestrator.is_initialized = True
        
        result = await orchestrator.predict(
            inputs={"test": "data"},
            latency_requirement="invalid_requirement"
        )
        
        assert result["status"] == "error"
        assert "Invalid latency requirement" in result["metadata"]["error"]
    
    @pytest.mark.asyncio
    async def test_predict_model_error(self, orchestrator, mock_moe):
        """Test prediction when model raises exception"""
        mock_moe.predict.side_effect = Exception("Model prediction failed")
        
        with patch.object(orchestrator, 'moe', mock_moe):
            orchestrator.is_initialized = True
            
            result = await orchestrator.predict({"test": "data"})
            
            assert result["status"] == "error"
            assert "Model prediction failed" in result["metadata"]["error"]
            assert orchestrator.performance_metrics["error_count"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_predict_success(self, orchestrator, mock_moe):
        """Test successful batch prediction"""
        # Setup mock
        mock_result = Mock()
        mock_result.predictions = {"win_probability": 0.7}
        mock_result.model_used = "fast_xgboost"
        mock_result.latency_ms = 45.0
        mock_result.confidence_score = 0.85
        mock_result.metadata = {"model_type": "fast"}
        
        mock_moe.predict.return_value = mock_result
        
        with patch.object(orchestrator, 'moe', mock_moe):
            orchestrator.is_initialized = True
            
            batch_inputs = [
                {"data": "batch1"},
                {"data": "batch2"},
                {"data": "batch3"}
            ]
            
            results = await orchestrator.batch_predict(batch_inputs)
            
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result["status"] == "success"
                assert result["metadata"]["batch_index"] == i
                assert result["predictions"] == {"win_probability": 0.7}
    
    @pytest.mark.asyncio
    async def test_batch_predict_with_errors(self, orchestrator):
        """Test batch prediction with some failures"""
        # Mock predict method to fail on second call
        async def mock_predict(inputs, latency_requirement="batch"):
            if "fail" in inputs.get("data", ""):
                raise Exception("Prediction failed")
            return {
                "predictions": {"result": "success"},
                "metadata": {"model_used": "test"},
                "status": "success",
                "timestamp": 123456
            }
        
        orchestrator.is_initialized = True
        orchestrator.predict = mock_predict
        
        batch_inputs = [
            {"data": "success1"},
            {"data": "fail_this"},
            {"data": "success2"}
        ]
        
        results = await orchestrator.batch_predict(batch_inputs)
        
        assert len(results) == 3
        assert results[0]["status"] == "success"
        assert results[1]["status"] == "error"
        assert results[2]["status"] == "success"
        assert results[1]["metadata"]["batch_index"] == 1
    
    def test_get_performance_metrics(self, orchestrator, mock_moe):
        """Test performance metrics retrieval"""
        # Setup some metrics
        orchestrator.performance_metrics["total_requests"] = 10
        orchestrator.performance_metrics["total_latency_ms"] = 500
        orchestrator.performance_metrics["error_count"] = 1
        
        mock_routing_stats = {"overall": {"total_requests": 10}}
        mock_models = {"model1": {"is_loaded": True}}
        
        mock_moe.get_routing_stats.return_value = mock_routing_stats
        mock_moe.get_available_models.return_value = mock_models
        
        with patch.object(orchestrator, 'moe', mock_moe):
            metrics = orchestrator.get_performance_metrics()
            
            assert "orchestrator" in metrics
            assert "routing" in metrics
            assert "models" in metrics
            
            orch_metrics = metrics["orchestrator"]
            assert orch_metrics["total_requests"] == 10
            assert orch_metrics["average_latency_ms"] == 50.0
            assert orch_metrics["error_rate"] == 0.1
    
    def test_get_performance_metrics_no_requests(self, orchestrator, mock_moe):
        """Test performance metrics with no requests"""
        mock_moe.get_routing_stats.return_value = {"overall": {"total_requests": 0}}
        mock_moe.get_available_models.return_value = {}
        
        with patch.object(orchestrator, 'moe', mock_moe):
            metrics = orchestrator.get_performance_metrics()
            
            orch_metrics = metrics["orchestrator"]
            assert orch_metrics["average_latency_ms"] == 0
            assert orch_metrics["error_rate"] == 0
    
    def test_get_health_status_not_initialized(self, orchestrator, mock_moe):
        """Test health status when not initialized"""
        with patch.object(orchestrator, 'moe', mock_moe):
            status = orchestrator.get_health_status()
            
            assert status["status"] == "unhealthy"
            assert "not initialized" in status["message"]
            assert not status["initialized"]
    
    def test_get_health_status_no_models(self, orchestrator, mock_moe):
        """Test health status with no models loaded"""
        mock_moe.get_available_models.return_value = {
            "model1": {"is_loaded": False},
            "model2": {"is_loaded": False}
        }
        
        with patch.object(orchestrator, 'moe', mock_moe):
            orchestrator.is_initialized = True
            status = orchestrator.get_health_status()
            
            assert status["status"] == "unhealthy"
            assert "No models loaded" in status["message"]
            assert status["loaded_models"] == []
            assert status["total_models"] == 2
    
    def test_get_health_status_degraded(self, orchestrator, mock_moe):
        """Test health status with only one model loaded"""
        mock_moe.get_available_models.return_value = {
            "model1": {"is_loaded": True},
            "model2": {"is_loaded": False}
        }
        
        with patch.object(orchestrator, 'moe', mock_moe):
            orchestrator.is_initialized = True
            status = orchestrator.get_health_status()
            
            assert status["status"] == "degraded"
            assert "Only 1 model(s) loaded" in status["message"]
            assert status["loaded_models"] == ["model1"]
    
    def test_get_health_status_healthy(self, orchestrator, mock_moe):
        """Test healthy status"""
        mock_moe.get_available_models.return_value = {
            "model1": {"is_loaded": True},
            "model2": {"is_loaded": True}
        }
        
        with patch.object(orchestrator, 'moe', mock_moe):
            orchestrator.is_initialized = True
            status = orchestrator.get_health_status()
            
            assert status["status"] == "healthy"
            assert "All systems operational" in status["message"]
            assert len(status["loaded_models"]) == 2
    
    def test_parse_latency_requirement_valid(self, orchestrator):
        """Test parsing valid latency requirements"""
        valid_requirements = [
            ("ultra_fast", LatencyRequirement.ULTRA_FAST),
            ("fast", LatencyRequirement.FAST),
            ("normal", LatencyRequirement.NORMAL),
            ("accurate", LatencyRequirement.ACCURATE),
            ("batch", LatencyRequirement.BATCH)
        ]
        
        for req_str, expected_enum in valid_requirements:
            result = orchestrator._parse_latency_requirement(req_str)
            assert result == expected_enum
    
    def test_parse_latency_requirement_invalid(self, orchestrator):
        """Test parsing invalid latency requirement"""
        with pytest.raises(ValueError, match="Invalid latency requirement"):
            orchestrator._parse_latency_requirement("invalid")
    
    def test_update_metrics(self, orchestrator):
        """Test metrics updating"""
        mock_result = Mock()
        mock_result.latency_ms = 100.0
        mock_result.model_used = "test_model"
        
        initial_requests = orchestrator.performance_metrics["total_requests"]
        initial_latency = orchestrator.performance_metrics["total_latency_ms"]
        
        orchestrator._update_metrics(mock_result)
        
        assert orchestrator.performance_metrics["total_requests"] == initial_requests + 1
        assert orchestrator.performance_metrics["total_latency_ms"] == initial_latency + 100.0
        assert orchestrator.performance_metrics["model_usage"]["test_model"] == 1
        
        # Test updating same model again
        orchestrator._update_metrics(mock_result)
        assert orchestrator.performance_metrics["model_usage"]["test_model"] == 2


class TestFactoryFunction:
    """Test factory function for creating orchestrator"""
    
    def test_create_moe_orchestrator_default(self):
        """Test creating orchestrator with default config"""
        orchestrator = create_moe_orchestrator()
        
        assert isinstance(orchestrator, MoEOrchestrator)
        assert orchestrator.config == {}
        assert not orchestrator.is_initialized
    
    def test_create_moe_orchestrator_with_config(self):
        """Test creating orchestrator with custom config"""
        config = {"models_to_load": ["fast_xgboost"]}
        orchestrator = create_moe_orchestrator(config)
        
        assert isinstance(orchestrator, MoEOrchestrator)
        assert orchestrator.config == config


class TestIntegration:
    """Integration tests for orchestrator"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow from initialization to prediction"""
        orchestrator = MoEOrchestrator()
        
        # Mock the MoE system
        mock_moe = Mock()
        mock_moe.get_available_models.return_value = {
            "fast_xgboost": {"is_loaded": True}
        }
        mock_moe.get_routing_stats.return_value = {"overall": {"total_requests": 0}}
        
        mock_result = Mock()
        mock_result.predictions = {"win_probability": 0.75}
        mock_result.model_used = "fast_xgboost"
        mock_result.latency_ms = 42.0
        mock_result.confidence_score = 0.88
        mock_result.metadata = {"model_type": "fast"}
        mock_moe.predict.return_value = mock_result
        
        with patch.object(orchestrator, 'moe', mock_moe):
            # Initialize
            await orchestrator.initialize()
            assert orchestrator.is_initialized
            
            # Make prediction
            result = await orchestrator.predict(
                inputs={"runs": 150, "wickets": 3},
                latency_requirement="fast"
            )
            
            assert result["status"] == "success"
            assert result["predictions"]["win_probability"] == 0.75
            assert result["metadata"]["model_used"] == "fast_xgboost"
            
            # Check health
            health = orchestrator.get_health_status()
            assert health["status"] == "degraded"  # Only 1 model loaded
            
            # Check metrics
            metrics = orchestrator.get_performance_metrics()
            assert metrics["orchestrator"]["total_requests"] == 1

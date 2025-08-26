# Purpose: Tests for Model Selection Service
# Author: WicketWise Team, Last Modified: 2025-08-25

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import json

from crickformers.models.model_selection_service import (
    ModelSelectionService,
    TaskContext,
    TaskPriority,
    ResponseTimeRequirement,
    AccuracyRequirement,
    CostSensitivity,
    ModelConfig,
    create_chat_context,
    create_betting_context,
    create_simulation_context
)


class TestModelSelectionService:
    """Test suite for Model Selection Service"""
    
    @pytest.fixture
    def service(self):
        """Create model selection service for testing"""
        return ModelSelectionService()
    
    def test_initialization(self, service):
        """Test service initialization"""
        assert service is not None
        assert len(service.model_configs) > 0
        assert len(service.task_model_mapping) > 0
        assert "gpt-5" in service.model_configs
        assert "gpt-5-mini" in service.model_configs
        assert "gpt-4-nano" in service.model_configs
    
    def test_model_configs_structure(self, service):
        """Test model configuration structure"""
        for model_name, config in service.model_configs.items():
            assert isinstance(config, ModelConfig)
            assert config.name == model_name
            assert config.max_tokens > 0
            assert config.cost_per_1k_input_tokens >= 0
            assert config.cost_per_1k_output_tokens >= 0
            assert isinstance(config.capabilities, list)
    
    def test_select_model_critical_priority(self, service):
        """Test model selection for critical priority tasks"""
        context = TaskContext(
            task_type="betting_decision",
            priority=TaskPriority.CRITICAL,
            expected_input_tokens=500,
            expected_output_tokens=200,
            response_time_requirement=ResponseTimeRequirement.NORMAL,
            accuracy_requirement=AccuracyRequirement.MAXIMUM,
            cost_sensitivity=CostSensitivity.LOW
        )
        
        selected = service.select_model(context)
        assert selected == "gpt-5"  # Critical tasks should use best model
    
    def test_select_model_realtime_requirement(self, service):
        """Test model selection for real-time requirements"""
        context = TaskContext(
            task_type="simulation_decision",
            priority=TaskPriority.NORMAL,
            expected_input_tokens=200,
            expected_output_tokens=50,
            response_time_requirement=ResponseTimeRequirement.REALTIME,
            accuracy_requirement=AccuracyRequirement.NORMAL,
            cost_sensitivity=CostSensitivity.MEDIUM
        )
        
        selected = service.select_model(context)
        assert selected == "gpt-4-nano"  # Real-time tasks should use fastest model
    
    def test_select_model_cost_sensitive(self, service):
        """Test model selection for cost-sensitive tasks"""
        context = TaskContext(
            task_type="data_enrichment",
            priority=TaskPriority.NORMAL,
            expected_input_tokens=1000,
            expected_output_tokens=500,
            response_time_requirement=ResponseTimeRequirement.NORMAL,
            accuracy_requirement=AccuracyRequirement.NORMAL,
            cost_sensitivity=CostSensitivity.HIGH
        )
        
        selected = service.select_model(context)
        # Should select most cost-effective model
        assert selected in ["gpt-5-mini", "gpt-4-nano"]
    
    def test_select_model_task_mapping(self, service):
        """Test task-based model mapping"""
        # KG chat should use GPT-5 Mini
        context = TaskContext(
            task_type="kg_chat",
            priority=TaskPriority.NORMAL,
            expected_input_tokens=500,
            expected_output_tokens=300,
            response_time_requirement=ResponseTimeRequirement.FAST,
            accuracy_requirement=AccuracyRequirement.HIGH,
            cost_sensitivity=CostSensitivity.MEDIUM
        )
        
        selected = service.select_model(context)
        assert selected == "gpt-5-mini"
    
    def test_get_model_config(self, service):
        """Test getting model configuration"""
        config = service.get_model_config("gpt-5")
        
        assert isinstance(config, ModelConfig)
        assert config.name == "gpt-5"
        assert config.max_tokens > 0
        assert "reasoning" in config.capabilities
    
    def test_get_model_config_fallback(self, service):
        """Test fallback for unknown model"""
        config = service.get_model_config("unknown-model")
        
        # Should fallback to default
        assert isinstance(config, ModelConfig)
        assert config.name == "gpt-4o"
    
    def test_estimate_cost(self, service):
        """Test cost estimation"""
        context = TaskContext(
            task_type="kg_chat",
            priority=TaskPriority.NORMAL,
            expected_input_tokens=1000,
            expected_output_tokens=500,
            response_time_requirement=ResponseTimeRequirement.FAST,
            accuracy_requirement=AccuracyRequirement.HIGH,
            cost_sensitivity=CostSensitivity.MEDIUM
        )
        
        cost = service.estimate_cost(context, "gpt-5-mini")
        
        assert cost > 0
        assert isinstance(cost, float)
        
        # GPT-5 should be more expensive than GPT-5 Mini
        gpt5_cost = service.estimate_cost(context, "gpt-5")
        gpt5_mini_cost = service.estimate_cost(context, "gpt-5-mini")
        assert gpt5_cost > gpt5_mini_cost
    
    def test_validate_model_for_task(self, service):
        """Test model validation for tasks"""
        # Function calling requirement
        context = TaskContext(
            task_type="kg_chat",
            priority=TaskPriority.NORMAL,
            expected_input_tokens=500,
            expected_output_tokens=200,
            response_time_requirement=ResponseTimeRequirement.NORMAL,
            accuracy_requirement=AccuracyRequirement.NORMAL,
            cost_sensitivity=CostSensitivity.MEDIUM,
            requires_function_calling=True
        )
        
        # GPT-5 should support function calling
        assert service.validate_model_for_task("gpt-5", context) is True
        
        # Real-time requirement
        realtime_context = TaskContext(
            task_type="simulation",
            priority=TaskPriority.NORMAL,
            expected_input_tokens=200,
            expected_output_tokens=50,
            response_time_requirement=ResponseTimeRequirement.REALTIME,
            accuracy_requirement=AccuracyRequirement.NORMAL,
            cost_sensitivity=CostSensitivity.HIGH
        )
        
        # GPT-4 Nano should be suitable for real-time
        assert service.validate_model_for_task("gpt-4-nano", realtime_context) is True
        # GPT-5 might be too slow for real-time
        gpt5_config = service.get_model_config("gpt-5")
        if gpt5_config.timeout_seconds > 5:
            assert service.validate_model_for_task("gpt-5", realtime_context) is False
    
    def test_get_available_models(self, service):
        """Test getting available models"""
        models = service.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-5" in models
        assert "gpt-5-mini" in models
        assert "gpt-4-nano" in models
    
    def test_get_model_capabilities(self, service):
        """Test getting model capabilities"""
        capabilities = service.get_model_capabilities("gpt-5")
        
        assert isinstance(capabilities, list)
        assert "reasoning" in capabilities
        assert "analysis" in capabilities
    
    def test_usage_statistics_tracking(self, service):
        """Test usage statistics tracking"""
        context = TaskContext(
            task_type="kg_chat",
            priority=TaskPriority.NORMAL,
            expected_input_tokens=500,
            expected_output_tokens=200,
            response_time_requirement=ResponseTimeRequirement.FAST,
            accuracy_requirement=AccuracyRequirement.HIGH,
            cost_sensitivity=CostSensitivity.MEDIUM
        )
        
        # Make several selections
        for _ in range(3):
            service.select_model(context)
        
        stats = service.get_usage_statistics()
        
        assert "model_usage" in stats
        assert "total_requests" in stats
        assert stats["total_requests"] >= 3
    
    def test_fallback_model_chain(self, service):
        """Test fallback model selection"""
        # Test with non-existent primary model
        fallback = service._get_fallback_model("non-existent-model")
        
        assert fallback in service.model_configs
        assert fallback in ["gpt-5-mini", "gpt-4o"]
    
    def test_cost_effective_model_selection(self, service):
        """Test most cost-effective model selection"""
        context = TaskContext(
            task_type="data_processing",
            priority=TaskPriority.LOW,
            expected_input_tokens=1000,
            expected_output_tokens=200,
            response_time_requirement=ResponseTimeRequirement.NORMAL,
            accuracy_requirement=AccuracyRequirement.NORMAL,
            cost_sensitivity=CostSensitivity.HIGH
        )
        
        cost_effective = service._get_most_cost_effective_model(context)
        
        # Should be one of the cheaper models
        assert cost_effective in ["gpt-4-nano", "gpt-5-mini"]


class TestTaskContextHelpers:
    """Test task context helper functions"""
    
    def test_create_chat_context(self):
        """Test chat context creation"""
        context = create_chat_context(expected_tokens=800)
        
        assert context.task_type == "kg_chat"
        assert context.priority == TaskPriority.NORMAL
        assert context.expected_input_tokens == 800
        assert context.response_time_requirement == ResponseTimeRequirement.FAST
        assert context.requires_function_calling is True
        assert context.cricket_domain_specific is True
    
    def test_create_betting_context(self):
        """Test betting context creation"""
        context = create_betting_context(expected_tokens=1200)
        
        assert context.task_type == "betting_decision"
        assert context.priority == TaskPriority.CRITICAL
        assert context.expected_input_tokens == 1200
        assert context.accuracy_requirement == AccuracyRequirement.MAXIMUM
        assert context.cost_sensitivity == CostSensitivity.LOW
        assert context.requires_function_calling is True
    
    def test_create_simulation_context(self):
        """Test simulation context creation"""
        context = create_simulation_context(expected_tokens=300)
        
        assert context.task_type == "simulation_decision"
        assert context.priority == TaskPriority.NORMAL
        assert context.expected_input_tokens == 300
        assert context.response_time_requirement == ResponseTimeRequirement.REALTIME
        assert context.cost_sensitivity == CostSensitivity.HIGH
        assert context.requires_function_calling is False


class TestModelConfigValidation:
    """Test model configuration validation"""
    
    def test_model_config_dataclass(self):
        """Test ModelConfig dataclass"""
        config = ModelConfig(
            name="test-model",
            display_name="Test Model",
            max_tokens=4000,
            temperature=0.7,
            timeout_seconds=10,
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            capabilities=["reasoning"],
            fallback_model="gpt-4o",
            rate_limits={"rpm": 100},
            supported_features=["chat"]
        )
        
        assert config.name == "test-model"
        assert config.max_tokens == 4000
        assert config.temperature == 0.7
        assert "reasoning" in config.capabilities
    
    def test_task_context_dataclass(self):
        """Test TaskContext dataclass"""
        context = TaskContext(
            task_type="test_task",
            priority=TaskPriority.HIGH,
            expected_input_tokens=500,
            expected_output_tokens=200,
            response_time_requirement=ResponseTimeRequirement.FAST,
            accuracy_requirement=AccuracyRequirement.HIGH,
            cost_sensitivity=CostSensitivity.MEDIUM
        )
        
        assert context.task_type == "test_task"
        assert context.priority == TaskPriority.HIGH
        assert context.expected_input_tokens == 500
        assert context.response_time_requirement == ResponseTimeRequirement.FAST


class TestModelSelectionEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_model_configs(self):
        """Test behavior with empty model configs"""
        service = ModelSelectionService()
        service.model_configs = {}
        
        context = TaskContext(
            task_type="test",
            priority=TaskPriority.NORMAL,
            expected_input_tokens=100,
            expected_output_tokens=50,
            response_time_requirement=ResponseTimeRequirement.NORMAL,
            accuracy_requirement=AccuracyRequirement.NORMAL,
            cost_sensitivity=CostSensitivity.MEDIUM
        )
        
        # Should handle gracefully
        selected = service.select_model(context)
        assert selected is not None
    
    def test_invalid_task_type(self):
        """Test with invalid task type"""
        service = ModelSelectionService()
        
        context = TaskContext(
            task_type="unknown_task_type",
            priority=TaskPriority.NORMAL,
            expected_input_tokens=100,
            expected_output_tokens=50,
            response_time_requirement=ResponseTimeRequirement.NORMAL,
            accuracy_requirement=AccuracyRequirement.NORMAL,
            cost_sensitivity=CostSensitivity.MEDIUM
        )
        
        selected = service.select_model(context)
        # Should fall back to default mapping
        assert selected == "gpt-5-mini"
    
    def test_extreme_token_requirements(self):
        """Test with extreme token requirements"""
        service = ModelSelectionService()
        
        # Very large token requirement
        context = TaskContext(
            task_type="large_task",
            priority=TaskPriority.NORMAL,
            expected_input_tokens=10000,
            expected_output_tokens=5000,
            response_time_requirement=ResponseTimeRequirement.NORMAL,
            accuracy_requirement=AccuracyRequirement.NORMAL,
            cost_sensitivity=CostSensitivity.MEDIUM
        )
        
        # Should still select a model
        selected = service.select_model(context)
        assert selected is not None
        
        # Validation should catch oversized requests
        valid = service.validate_model_for_task(selected, context)
        # May or may not be valid depending on model limits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

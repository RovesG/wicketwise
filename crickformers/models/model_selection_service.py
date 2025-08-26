# Purpose: Intelligent OpenAI Model Selection Service for WicketWise
# Author: WicketWise Team, Last Modified: 2025-08-25

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high" 
    NORMAL = "normal"
    LOW = "low"


class ResponseTimeRequirement(Enum):
    """Response time requirements"""
    REALTIME = "realtime"  # <100ms
    FAST = "fast"          # <500ms
    NORMAL = "normal"      # <2s
    RELAXED = "relaxed"    # >2s


class AccuracyRequirement(Enum):
    """Accuracy requirements"""
    MAXIMUM = "maximum"    # Best possible accuracy
    HIGH = "high"          # High accuracy needed
    NORMAL = "normal"      # Standard accuracy
    BASIC = "basic"        # Basic accuracy sufficient


class CostSensitivity(Enum):
    """Cost sensitivity levels"""
    LOW = "low"           # Cost not a concern
    MEDIUM = "medium"     # Balanced cost/quality
    HIGH = "high"         # Cost-optimized


@dataclass
class ModelConfig:
    """Configuration for an OpenAI model"""
    name: str
    display_name: str
    max_tokens: int
    temperature: float
    timeout_seconds: int
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    capabilities: List[str]
    fallback_model: Optional[str]
    rate_limits: Dict[str, int]
    supported_features: List[str]


@dataclass
class TaskContext:
    """Context information for task-based model selection"""
    task_type: str
    priority: TaskPriority
    expected_input_tokens: int
    expected_output_tokens: int
    response_time_requirement: ResponseTimeRequirement
    accuracy_requirement: AccuracyRequirement
    cost_sensitivity: CostSensitivity
    requires_function_calling: bool = False
    requires_structured_output: bool = False
    cricket_domain_specific: bool = True


class ModelSelectionService:
    """
    Intelligent model selection service for WicketWise
    
    Selects optimal OpenAI models based on task characteristics,
    performance requirements, and cost considerations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize model selection service
        
        Args:
            config_path: Path to model configuration file
        """
        self.config_path = config_path or "config/model_configs.json"
        self.model_configs: Dict[str, ModelConfig] = {}
        self.task_model_mapping: Dict[str, str] = {}
        self.usage_stats: Dict[str, Dict] = {}
        
        # Load configurations
        self._load_model_configs()
        self._load_task_mappings()
        
        logger.info(f"🚀 Model Selection Service initialized with {len(self.model_configs)} models")
    
    def _load_model_configs(self) -> None:
        """Load model configurations"""
        try:
            # Default model configurations (using latest GPT-5 models available on platform)
            default_configs = {
                "gpt-5": ModelConfig(
                    name="gpt-5",
                    display_name="GPT-5 (Premium)",
                    max_tokens=8192,  # Higher capacity for complex analysis
                    temperature=0.7,
                    timeout_seconds=30,
                    cost_per_1k_input_tokens=0.015,  # Estimated premium pricing
                    cost_per_1k_output_tokens=0.060,
                    capabilities=["advanced_reasoning", "complex_analysis", "function_calling", "multimodal"],
                    fallback_model="gpt-4o",
                    rate_limits={"requests_per_minute": 5000, "tokens_per_minute": 800000},
                    supported_features=["critical_decisions", "advanced_reasoning", "complex_cricket_analysis"]
                ),
                "gpt-5-mini": ModelConfig(
                    name="gpt-5-mini",
                    display_name="GPT-5 Mini",
                    max_tokens=4096,
                    temperature=1.0,  # GPT-5-mini only supports temperature=1.0
                    timeout_seconds=10,
                    cost_per_1k_input_tokens=0.0015,  # Estimated mini pricing
                    cost_per_1k_output_tokens=0.006,
                    capabilities=["reasoning", "analysis", "function_calling", "structured_output"],
                    fallback_model="gpt-4o-mini",
                    rate_limits={"requests_per_minute": 5000, "tokens_per_minute": 4000000},
                    supported_features=["fast_queries", "interactive_chat", "data_processing"]
                ),
                "gpt-5-nano": ModelConfig(
                    name="gpt-5-nano",
                    display_name="GPT-5 Nano",
                    max_tokens=2048,
                    temperature=0.3,
                    timeout_seconds=5,
                    cost_per_1k_input_tokens=0.0001,  # Estimated nano pricing
                    cost_per_1k_output_tokens=0.0004,
                    capabilities=["basic_reasoning", "pattern_recognition", "simple_function_calling"],
                    fallback_model="gpt-5-mini",
                    rate_limits={"requests_per_minute": 5000, "tokens_per_minute": 4000000},
                    supported_features=["realtime_decisions", "simple_tasks", "high_volume"]
                ),
                "gpt-4o": ModelConfig(
                    name="gpt-4o",
                    display_name="GPT-4o (Legacy Fallback)",
                    max_tokens=4096,
                    temperature=0.7,
                    timeout_seconds=20,
                    cost_per_1k_input_tokens=0.005,
                    cost_per_1k_output_tokens=0.015,
                    capabilities=["reasoning", "analysis", "function_calling", "multimodal"],
                    fallback_model="gpt-4o-mini",
                    rate_limits={"requests_per_minute": 500, "tokens_per_minute": 100000},
                    supported_features=["general_purpose", "multimodal", "legacy_support"]
                ),
                "gpt-4o-mini": ModelConfig(
                    name="gpt-4o-mini",
                    display_name="GPT-4o Mini (Legacy Fallback)",
                    max_tokens=4096,
                    temperature=0.7,
                    timeout_seconds=10,
                    cost_per_1k_input_tokens=0.00015,
                    cost_per_1k_output_tokens=0.0006,
                    capabilities=["reasoning", "analysis", "function_calling"],
                    fallback_model="gpt-4o",
                    rate_limits={"requests_per_minute": 1000, "tokens_per_minute": 200000},
                    supported_features=["fast_queries", "legacy_support"]
                )
            }
            
            # Try to load from file, fallback to defaults
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    file_configs = json.load(f)
                    # Merge with defaults
                    for name, config_data in file_configs.items():
                        if name in default_configs:
                            # Update existing config
                            default_configs[name].__dict__.update(config_data)
            
            self.model_configs = default_configs
            logger.info(f"✅ Loaded {len(self.model_configs)} model configurations")
            
        except Exception as e:
            logger.error(f"❌ Error loading model configs: {e}")
            # Use minimal fallback
            self.model_configs = {
                "gpt-4o": ModelConfig(
                    name="gpt-4o", display_name="GPT-4o", max_tokens=4096,
                    temperature=0.7, timeout_seconds=20, cost_per_1k_input_tokens=0.005,
                    cost_per_1k_output_tokens=0.015, capabilities=["reasoning"],
                    fallback_model=None, rate_limits={}, supported_features=[]
                )
            }
    
    def _load_task_mappings(self) -> None:
        """Load task to model mappings"""
        self.task_model_mapping = {
            # Critical decision tasks - Use GPT-5 (maximum accuracy)
            "betting_decision": "gpt-5",
            "risk_assessment": "gpt-5", 
            "financial_analysis": "gpt-5",
            "strategy_development": "gpt-5",
            "complex_cricket_analysis": "gpt-5",
            "model_explanation": "gpt-5",
            
            # Fast query tasks - Use GPT-5 Mini (optimal balance)
            "kg_chat": "gpt-5-mini",
            "player_analysis": "gpt-5-mini",
            "match_insights": "gpt-5-mini",
            "data_enrichment": "gpt-5-mini",
            "entity_harmonization": "gpt-5-mini",
            "data_validation": "gpt-5-mini",
            
            # Real-time tasks - Use GPT-5 Nano (ultra-fast)
            "simulation_decision": "gpt-5-nano",
            "live_update": "gpt-5-nano",
            "quick_prediction": "gpt-5-nano",
            "system_notification": "gpt-5-nano",
            
            # Fallback for unknown tasks
            "default": "gpt-5-mini"
        }
        
        logger.info(f"📋 Loaded {len(self.task_model_mapping)} task mappings")
    
    def select_model(self, task_context: TaskContext) -> str:
        """
        Select optimal model based on task context
        
        Args:
            task_context: Task requirements and constraints
            
        Returns:
            Selected model name
        """
        try:
            # Start with task-based mapping
            base_model = self.task_model_mapping.get(task_context.task_type, "gpt-5-mini")
            
            # Apply priority-based overrides
            if task_context.priority == TaskPriority.CRITICAL:
                selected_model = "gpt-5"
            elif task_context.response_time_requirement == ResponseTimeRequirement.REALTIME:
                selected_model = "gpt-5-nano"
            elif task_context.cost_sensitivity == CostSensitivity.HIGH:
                selected_model = self._get_most_cost_effective_model(task_context)
            elif task_context.accuracy_requirement == AccuracyRequirement.MAXIMUM:
                selected_model = "gpt-5"
            else:
                selected_model = base_model
            
            # Validate model availability
            if selected_model not in self.model_configs:
                logger.warning(f"⚠️ Model {selected_model} not available, using fallback")
                selected_model = self._get_fallback_model(selected_model)
            
            # Log selection reasoning
            logger.info(f"🎯 Selected {selected_model} for {task_context.task_type} "
                       f"(priority: {task_context.priority.value}, "
                       f"response_time: {task_context.response_time_requirement.value})")
            
            # Update usage stats
            self._update_usage_stats(selected_model, task_context)
            
            return selected_model
            
        except Exception as e:
            logger.error(f"❌ Error selecting model: {e}")
            return "gpt-4o"  # Safe fallback
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Get configuration for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration
        """
        if model_name not in self.model_configs:
            logger.warning(f"⚠️ Model {model_name} not found, using default")
            model_name = "gpt-4o"
        
        return self.model_configs[model_name]
    
    def estimate_cost(self, task_context: TaskContext, model_name: Optional[str] = None) -> float:
        """
        Estimate cost for a task
        
        Args:
            task_context: Task context with token estimates
            model_name: Optional specific model, otherwise auto-select
            
        Returns:
            Estimated cost in USD
        """
        if not model_name:
            model_name = self.select_model(task_context)
        
        config = self.get_model_config(model_name)
        
        input_cost = (task_context.expected_input_tokens / 1000) * config.cost_per_1k_input_tokens
        output_cost = (task_context.expected_output_tokens / 1000) * config.cost_per_1k_output_tokens
        
        total_cost = input_cost + output_cost
        
        logger.debug(f"💰 Estimated cost for {model_name}: ${total_cost:.4f}")
        
        return total_cost
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.model_configs.keys())
    
    def get_model_capabilities(self, model_name: str) -> List[str]:
        """Get capabilities for a specific model"""
        config = self.get_model_config(model_name)
        return config.capabilities
    
    def validate_model_for_task(self, model_name: str, task_context: TaskContext) -> bool:
        """
        Validate if a model is suitable for a task
        
        Args:
            model_name: Model to validate
            task_context: Task requirements
            
        Returns:
            True if model is suitable
        """
        config = self.get_model_config(model_name)
        
        # Check function calling requirement
        if task_context.requires_function_calling and "function_calling" not in config.capabilities:
            return False
        
        # Check response time requirement
        if (task_context.response_time_requirement == ResponseTimeRequirement.REALTIME and 
            config.timeout_seconds > 5):
            return False
        
        # Check token limits
        total_tokens = task_context.expected_input_tokens + task_context.expected_output_tokens
        if total_tokens > config.max_tokens:
            return False
        
        return True
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for all models"""
        return {
            "model_usage": self.usage_stats,
            "total_requests": sum(stats.get("request_count", 0) for stats in self.usage_stats.values()),
            "available_models": self.get_available_models()
        }
    
    def _get_most_cost_effective_model(self, task_context: TaskContext) -> str:
        """Get the most cost-effective model for a task"""
        costs = {}
        
        for model_name in self.model_configs:
            if self.validate_model_for_task(model_name, task_context):
                cost = self.estimate_cost(task_context, model_name)
                costs[model_name] = cost
        
        if costs:
            return min(costs, key=costs.get)
        else:
            return "gpt-5-mini"  # Default fallback
    
    def _get_fallback_model(self, primary_model: str) -> str:
        """Get fallback model for a primary model"""
        if primary_model in self.model_configs:
            fallback = self.model_configs[primary_model].fallback_model
            if fallback and fallback in self.model_configs:
                return fallback
        
        # Ultimate fallback chain (GPT-5 models first, then legacy)
        fallback_chain = ["gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-4o-mini", "gpt-4o"]
        for model in fallback_chain:
            if model in self.model_configs:
                return model
        
        # If nothing else works
        return list(self.model_configs.keys())[0]
    
    def _update_usage_stats(self, model_name: str, task_context: TaskContext) -> None:
        """Update usage statistics"""
        if model_name not in self.usage_stats:
            self.usage_stats[model_name] = {
                "request_count": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "task_types": {}
            }
        
        stats = self.usage_stats[model_name]
        stats["request_count"] += 1
        stats["total_input_tokens"] += task_context.expected_input_tokens
        stats["total_output_tokens"] += task_context.expected_output_tokens
        stats["total_cost"] += self.estimate_cost(task_context, model_name)
        
        # Track task type usage
        task_type = task_context.task_type
        if task_type not in stats["task_types"]:
            stats["task_types"][task_type] = 0
        stats["task_types"][task_type] += 1


# Convenience functions for common use cases
def create_chat_context(expected_tokens: int = 500) -> TaskContext:
    """Create task context for chat queries"""
    return TaskContext(
        task_type="kg_chat",
        priority=TaskPriority.NORMAL,
        expected_input_tokens=expected_tokens,
        expected_output_tokens=expected_tokens,
        response_time_requirement=ResponseTimeRequirement.FAST,
        accuracy_requirement=AccuracyRequirement.HIGH,
        cost_sensitivity=CostSensitivity.MEDIUM,
        requires_function_calling=True,
        cricket_domain_specific=True
    )


def create_betting_context(expected_tokens: int = 1000) -> TaskContext:
    """Create task context for betting decisions"""
    return TaskContext(
        task_type="betting_decision",
        priority=TaskPriority.CRITICAL,
        expected_input_tokens=expected_tokens,
        expected_output_tokens=expected_tokens // 2,
        response_time_requirement=ResponseTimeRequirement.NORMAL,
        accuracy_requirement=AccuracyRequirement.MAXIMUM,
        cost_sensitivity=CostSensitivity.LOW,
        requires_function_calling=True,
        cricket_domain_specific=True
    )


def create_simulation_context(expected_tokens: int = 200) -> TaskContext:
    """Create task context for simulation decisions"""
    return TaskContext(
        task_type="simulation_decision",
        priority=TaskPriority.NORMAL,
        expected_input_tokens=expected_tokens,
        expected_output_tokens=expected_tokens // 4,
        response_time_requirement=ResponseTimeRequirement.REALTIME,
        accuracy_requirement=AccuracyRequirement.NORMAL,
        cost_sensitivity=CostSensitivity.HIGH,
        requires_function_calling=False,
        cricket_domain_specific=True
    )

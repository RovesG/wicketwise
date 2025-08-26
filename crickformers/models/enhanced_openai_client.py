# Purpose: Enhanced OpenAI Client with Intelligent Model Selection
# Author: WicketWise Team, Last Modified: 2025-08-25

import logging
import time
from typing import Dict, List, Optional, Any, Union
from openai import OpenAI
import os
from dataclasses import asdict

from .model_selection_service import (
    ModelSelectionService, 
    TaskContext, 
    TaskPriority,
    ResponseTimeRequirement,
    AccuracyRequirement,
    CostSensitivity
)

logger = logging.getLogger(__name__)


class EnhancedOpenAIClient:
    """
    Enhanced OpenAI client with intelligent model selection
    
    Automatically selects the optimal model based on task characteristics,
    provides cost tracking, fallback handling, and performance monitoring.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize enhanced OpenAI client
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model_service = ModelSelectionService()
        
        # Performance tracking
        self.request_history: List[Dict] = []
        self.total_cost = 0.0
        self.total_requests = 0
        
        logger.info("🚀 Enhanced OpenAI Client initialized with intelligent model selection")
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]],
                       task_context: Optional[TaskContext] = None,
                       model: Optional[str] = None,
                       **kwargs) -> Any:
        """
        Create chat completion with intelligent model selection
        
        Args:
            messages: Chat messages
            task_context: Task context for model selection
            model: Override model selection
            **kwargs: Additional OpenAI parameters
            
        Returns:
            OpenAI chat completion response
        """
        start_time = time.time()
        
        try:
            # Select optimal model if not specified
            if not model:
                if not task_context:
                    # Create default context for chat
                    estimated_tokens = self._estimate_tokens(messages)
                    task_context = TaskContext(
                        task_type="kg_chat",
                        priority=TaskPriority.NORMAL,
                        expected_input_tokens=estimated_tokens,
                        expected_output_tokens=estimated_tokens // 2,
                        response_time_requirement=ResponseTimeRequirement.FAST,
                        accuracy_requirement=AccuracyRequirement.HIGH,
                        cost_sensitivity=CostSensitivity.MEDIUM,
                        requires_function_calling=kwargs.get('tools') is not None
                    )
                
                model = self.model_service.select_model(task_context)
            
            # Get model configuration
            model_config = self.model_service.get_model_config(model)
            
            # Apply model-specific parameters
            enhanced_kwargs = self._apply_model_config(kwargs, model_config)
            enhanced_kwargs['model'] = model
            
            # Log request
            logger.info(f"🎯 Making request with {model} for task: {task_context.task_type if task_context else 'unknown'}")
            
            # Make API call with retries
            response = self._make_request_with_fallback(messages, model, enhanced_kwargs, task_context)
            
            # Track performance and cost
            end_time = time.time()
            self._track_request(model, task_context, start_time, end_time, response)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in chat completion: {e}")
            raise
    
    def _make_request_with_fallback(self, 
                                   messages: List[Dict[str, str]], 
                                   model: str, 
                                   kwargs: Dict,
                                   task_context: Optional[TaskContext]) -> Any:
        """Make request with automatic fallback on failure"""
        try:
            # Primary request
            return self.client.chat.completions.create(messages=messages, **kwargs)
            
        except Exception as e:
            logger.warning(f"⚠️ Primary model {model} failed: {e}")
            
            # Try fallback model
            fallback_model = self.model_service.get_model_config(model).fallback_model
            if fallback_model and fallback_model != model:
                logger.info(f"🔄 Trying fallback model: {fallback_model}")
                
                kwargs['model'] = fallback_model
                fallback_config = self.model_service.get_model_config(fallback_model)
                kwargs = self._apply_model_config(kwargs, fallback_config)
                
                try:
                    return self.client.chat.completions.create(messages=messages, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback model {fallback_model} also failed: {fallback_error}")
            
            # Ultimate fallback to GPT-4o
            logger.info("🆘 Using ultimate fallback: gpt-4o")
            kwargs['model'] = 'gpt-4o'
            kwargs['temperature'] = 0.7
            kwargs['max_completion_tokens'] = 4000
            
            return self.client.chat.completions.create(messages=messages, **kwargs)
    
    def _apply_model_config(self, kwargs: Dict, model_config) -> Dict:
        """Apply model-specific configuration"""
        enhanced_kwargs = kwargs.copy()
        
        # Apply defaults from model config if not specified
        if 'temperature' not in enhanced_kwargs:
            enhanced_kwargs['temperature'] = model_config.temperature
        
        if 'max_completion_tokens' not in enhanced_kwargs and 'max_tokens' not in enhanced_kwargs:
            enhanced_kwargs['max_completion_tokens'] = min(model_config.max_tokens, 4000)
        
        # Set timeout based on model config
        enhanced_kwargs['timeout'] = model_config.timeout_seconds
        
        return enhanced_kwargs
    
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token count for messages"""
        # Simple estimation: ~4 characters per token
        total_chars = sum(len(msg.get('content', '') or '') for msg in messages)
        return max(total_chars // 4, 100)  # Minimum 100 tokens
    
    def _track_request(self, 
                      model: str, 
                      task_context: Optional[TaskContext],
                      start_time: float, 
                      end_time: float, 
                      response: Any) -> None:
        """Track request performance and cost"""
        try:
            duration = end_time - start_time
            
            # Extract token usage
            usage = getattr(response, 'usage', None)
            input_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
            output_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
            
            # Calculate cost
            model_config = self.model_service.get_model_config(model)
            cost = ((input_tokens / 1000) * model_config.cost_per_1k_input_tokens + 
                   (output_tokens / 1000) * model_config.cost_per_1k_output_tokens)
            
            # Store request info
            request_info = {
                'timestamp': time.time(),
                'model': model,
                'task_type': task_context.task_type if task_context else 'unknown',
                'duration_seconds': duration,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'cost_usd': cost,
                'success': True
            }
            
            self.request_history.append(request_info)
            self.total_cost += cost
            self.total_requests += 1
            
            # Log performance
            logger.info(f"📊 Request completed: {model} | {duration:.2f}s | "
                       f"{input_tokens + output_tokens} tokens | ${cost:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error tracking request: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.request_history:
            return {"message": "No requests tracked yet"}
        
        recent_requests = self.request_history[-100:]  # Last 100 requests
        
        # Calculate averages
        avg_duration = sum(r['duration_seconds'] for r in recent_requests) / len(recent_requests)
        avg_tokens = sum(r['total_tokens'] for r in recent_requests) / len(recent_requests)
        avg_cost = sum(r['cost_usd'] for r in recent_requests) / len(recent_requests)
        
        # Model usage breakdown
        model_usage = {}
        for request in recent_requests:
            model = request['model']
            if model not in model_usage:
                model_usage[model] = {'count': 0, 'total_cost': 0.0, 'total_duration': 0.0}
            model_usage[model]['count'] += 1
            model_usage[model]['total_cost'] += request['cost_usd']
            model_usage[model]['total_duration'] += request['duration_seconds']
        
        # Task type breakdown
        task_usage = {}
        for request in recent_requests:
            task = request['task_type']
            if task not in task_usage:
                task_usage[task] = {'count': 0, 'total_cost': 0.0}
            task_usage[task]['count'] += 1
            task_usage[task]['total_cost'] += request['cost_usd']
        
        return {
            'total_requests': self.total_requests,
            'total_cost_usd': self.total_cost,
            'recent_performance': {
                'avg_duration_seconds': avg_duration,
                'avg_tokens_per_request': avg_tokens,
                'avg_cost_per_request': avg_cost
            },
            'model_usage': model_usage,
            'task_usage': task_usage,
            'available_models': self.model_service.get_available_models()
        }
    
    def estimate_cost(self, 
                     messages: List[Dict[str, str]], 
                     task_context: Optional[TaskContext] = None) -> Dict[str, float]:
        """Estimate cost for different models"""
        if not task_context:
            estimated_tokens = self._estimate_tokens(messages)
            task_context = TaskContext(
                task_type="estimate",
                priority=TaskPriority.NORMAL,
                expected_input_tokens=estimated_tokens,
                expected_output_tokens=estimated_tokens // 2,
                response_time_requirement=ResponseTimeRequirement.NORMAL,
                accuracy_requirement=AccuracyRequirement.NORMAL,
                cost_sensitivity=CostSensitivity.MEDIUM
            )
        
        costs = {}
        for model_name in self.model_service.get_available_models():
            cost = self.model_service.estimate_cost(task_context, model_name)
            costs[model_name] = cost
        
        return costs
    
    def get_recommended_model(self, task_context: TaskContext) -> Dict[str, Any]:
        """Get recommended model with explanation"""
        selected_model = self.model_service.select_model(task_context)
        model_config = self.model_service.get_model_config(selected_model)
        estimated_cost = self.model_service.estimate_cost(task_context, selected_model)
        
        return {
            'recommended_model': selected_model,
            'model_config': asdict(model_config),
            'estimated_cost_usd': estimated_cost,
            'selection_reasoning': {
                'task_type': task_context.task_type,
                'priority': task_context.priority.value,
                'response_time_req': task_context.response_time_requirement.value,
                'accuracy_req': task_context.accuracy_requirement.value,
                'cost_sensitivity': task_context.cost_sensitivity.value
            }
        }


# Convenience functions for common WicketWise use cases
class WicketWiseOpenAI:
    """Convenience wrapper for WicketWise-specific OpenAI usage"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = EnhancedOpenAIClient(api_key)
    
    def kg_chat(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Optimized for KG chat queries"""
        from .model_selection_service import create_chat_context
        
        estimated_tokens = self.client._estimate_tokens(messages)
        context = create_chat_context(estimated_tokens)
        
        return self.client.chat_completion(messages, task_context=context, **kwargs)
    
    def betting_decision(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Optimized for betting decisions"""
        from .model_selection_service import create_betting_context
        
        estimated_tokens = self.client._estimate_tokens(messages)
        context = create_betting_context(estimated_tokens)
        
        return self.client.chat_completion(messages, task_context=context, **kwargs)
    
    def simulation_decision(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Optimized for simulation decisions"""
        from .model_selection_service import create_simulation_context
        
        estimated_tokens = self.client._estimate_tokens(messages)
        context = create_simulation_context(estimated_tokens)
        
        return self.client.chat_completion(messages, task_context=context, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self.client.get_performance_stats()

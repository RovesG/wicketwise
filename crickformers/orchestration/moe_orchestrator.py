# Purpose: MoE orchestrator for production inference
# Author: WicketWise Team, Last Modified: 2025-08-23

"""
Production orchestrator that integrates the Mixture of Experts layer
with the existing WicketWise prediction pipeline.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from crickformers.model.mixture_of_experts import (
    MixtureOfExperts,
    InferenceRequest,
    LatencyRequirement,
    InferenceResult
)

logger = logging.getLogger(__name__)


class MoEOrchestrator:
    """Production orchestrator for MoE-based predictions"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.moe = MixtureOfExperts()
        self.is_initialized = False
        self.performance_metrics = {
            "total_requests": 0,
            "total_latency_ms": 0,
            "error_count": 0,
            "model_usage": {}
        }
        
    async def initialize(self) -> None:
        """Initialize the orchestrator and load models"""
        try:
            logger.info("🚀 Initializing MoE Orchestrator...")
            
            # Load models based on configuration
            models_to_load = self.config.get("models_to_load", None)
            self.moe.load_models(models_to_load)
            
            # Validate at least one model is loaded
            available_models = self.moe.get_available_models()
            loaded_models = [
                name for name, info in available_models.items() 
                if info["is_loaded"]
            ]
            
            if not loaded_models:
                raise RuntimeError("No models successfully loaded")
            
            self.is_initialized = True
            logger.info(f"✅ MoE Orchestrator initialized with models: {loaded_models}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MoE Orchestrator: {e}")
            raise
    
    async def predict(
        self,
        inputs: Dict[str, Any],
        latency_requirement: str = "normal",
        accuracy_threshold: Optional[float] = None,
        model_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make prediction using MoE routing
        
        Args:
            inputs: Input data for prediction
            latency_requirement: "ultra_fast", "fast", "normal", "accurate", "batch"
            accuracy_threshold: Minimum confidence threshold
            model_preference: Preferred model name
            
        Returns:
            Prediction result with metadata
        """
        if not self.is_initialized:
            return {
                "predictions": None,
                "metadata": {
                    "error": "Orchestrator not initialized. Call initialize() first.",
                    "latency_ms": 0
                },
                "status": "error",
                "timestamp": time.time()
            }
        
        start_time = time.time()
        
        try:
            # Convert string latency requirement to enum
            latency_enum = self._parse_latency_requirement(latency_requirement)
            
            # Create inference request
            request = InferenceRequest(
                inputs=inputs,
                latency_requirement=latency_enum,
                accuracy_threshold=accuracy_threshold,
                model_preference=model_preference
            )
            
            # Make prediction
            result = self.moe.predict(request)
            
            # Update performance metrics
            self._update_metrics(result)
            
            # Format response
            response = {
                "predictions": result.predictions,
                "metadata": {
                    "model_used": result.model_used,
                    "latency_ms": result.latency_ms,
                    "confidence_score": result.confidence_score,
                    "latency_requirement": latency_requirement,
                    **result.metadata
                },
                "status": "success",
                "timestamp": time.time()
            }
            
            logger.info(
                f"🎯 Prediction completed: {result.model_used} "
                f"({result.latency_ms:.1f}ms, confidence={result.confidence_score:.3f})"
            )
            
            return response
            
        except Exception as e:
            self.performance_metrics["error_count"] += 1
            logger.error(f"❌ Prediction failed: {e}")
            
            return {
                "predictions": None,
                "metadata": {
                    "error": str(e),
                    "latency_ms": (time.time() - start_time) * 1000
                },
                "status": "error",
                "timestamp": time.time()
            }
    
    async def batch_predict(
        self,
        batch_inputs: List[Dict[str, Any]],
        latency_requirement: str = "batch"
    ) -> List[Dict[str, Any]]:
        """
        Make batch predictions
        
        Args:
            batch_inputs: List of input dictionaries
            latency_requirement: Latency requirement for batch
            
        Returns:
            List of prediction results
        """
        logger.info(f"📦 Starting batch prediction for {len(batch_inputs)} items")
        
        # Create tasks for concurrent processing
        tasks = []
        for i, inputs in enumerate(batch_inputs):
            task = self.predict(
                inputs=inputs,
                latency_requirement=latency_requirement
            )
            tasks.append(task)
        
        # Execute batch predictions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "predictions": None,
                    "metadata": {"error": str(result), "batch_index": i},
                    "status": "error",
                    "timestamp": time.time()
                })
            else:
                result["metadata"]["batch_index"] = i
                processed_results.append(result)
        
        logger.info(f"✅ Batch prediction completed: {len(processed_results)} results")
        return processed_results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics"""
        routing_stats = self.moe.get_routing_stats()
        
        metrics = {
            "orchestrator": self.performance_metrics.copy(),
            "routing": routing_stats,
            "models": self.moe.get_available_models()
        }
        
        # Calculate derived metrics
        if self.performance_metrics["total_requests"] > 0:
            metrics["orchestrator"]["average_latency_ms"] = (
                self.performance_metrics["total_latency_ms"] / 
                self.performance_metrics["total_requests"]
            )
            metrics["orchestrator"]["error_rate"] = (
                self.performance_metrics["error_count"] / 
                self.performance_metrics["total_requests"]
            )
        else:
            metrics["orchestrator"]["average_latency_ms"] = 0
            metrics["orchestrator"]["error_rate"] = 0
        
        return metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        available_models = self.moe.get_available_models()
        loaded_models = [
            name for name, info in available_models.items() 
            if info["is_loaded"]
        ]
        
        # Determine health status
        if not self.is_initialized:
            status = "unhealthy"
            message = "Orchestrator not initialized"
        elif not loaded_models:
            status = "unhealthy"
            message = "No models loaded"
        elif len(loaded_models) < 2:
            status = "degraded"
            message = f"Only {len(loaded_models)} model(s) loaded"
        else:
            status = "healthy"
            message = f"All systems operational ({len(loaded_models)} models)"
        
        return {
            "status": status,
            "message": message,
            "initialized": self.is_initialized,
            "loaded_models": loaded_models,
            "total_models": len(available_models),
            "performance": self.get_performance_metrics()["orchestrator"]
        }
    
    def _parse_latency_requirement(self, requirement: str) -> LatencyRequirement:
        """Parse string latency requirement to enum"""
        requirement_map = {
            "ultra_fast": LatencyRequirement.ULTRA_FAST,
            "fast": LatencyRequirement.FAST,
            "normal": LatencyRequirement.NORMAL,
            "accurate": LatencyRequirement.ACCURATE,
            "batch": LatencyRequirement.BATCH
        }
        
        if requirement not in requirement_map:
            raise ValueError(
                f"Invalid latency requirement: {requirement}. "
                f"Must be one of: {list(requirement_map.keys())}"
            )
        
        return requirement_map[requirement]
    
    def _update_metrics(self, result: InferenceResult) -> None:
        """Update performance metrics"""
        self.performance_metrics["total_requests"] += 1
        self.performance_metrics["total_latency_ms"] += result.latency_ms
        
        # Track model usage
        model_used = result.model_used
        if model_used not in self.performance_metrics["model_usage"]:
            self.performance_metrics["model_usage"][model_used] = 0
        self.performance_metrics["model_usage"][model_used] += 1


# Factory function for easy instantiation
def create_moe_orchestrator(config: Optional[Dict[str, Any]] = None) -> MoEOrchestrator:
    """Create and return a MoE orchestrator instance"""
    return MoEOrchestrator(config)

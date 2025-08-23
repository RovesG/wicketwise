#!/usr/bin/env python3
"""
Service Container - Dependency Injection and Service Management
Microservices-ready architecture with clean service boundaries

Author: WicketWise Team, Last Modified: 2025-01-21
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Type, Callable, List, Protocol
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import threading
from contextlib import asynccontextmanager
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class ServiceLifecycle(Enum):
    """Service lifecycle states"""
    REGISTERED = "registered"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class ServiceHealth:
    """Service health status"""
    name: str
    status: ServiceLifecycle
    uptime: float
    last_check: datetime
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None

class ServiceInterface(Protocol):
    """Protocol that all services must implement"""
    
    @property
    def name(self) -> str:
        """Service name"""
        ...
    
    async def start(self) -> None:
        """Start the service"""
        ...
    
    async def stop(self) -> None:
        """Stop the service"""
        ...
    
    async def health_check(self) -> ServiceHealth:
        """Check service health"""
        ...

class BaseService(ABC):
    """Base class for all services"""
    
    def __init__(self, name: str):
        self._name = name
        self._status = ServiceLifecycle.REGISTERED
        self._start_time = None
        self._error_message = None
        self._lock = asyncio.Lock()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def status(self) -> ServiceLifecycle:
        return self._status
    
    async def start(self) -> None:
        """Start the service with lifecycle management"""
        async with self._lock:
            if self._status == ServiceLifecycle.RUNNING:
                return
            
            try:
                self._status = ServiceLifecycle.STARTING
                logger.info(f"🚀 Starting service: {self.name}")
                
                await self._start_implementation()
                
                self._status = ServiceLifecycle.RUNNING
                self._start_time = time.time()
                self._error_message = None
                
                logger.info(f"✅ Service started: {self.name}")
                
            except Exception as e:
                self._status = ServiceLifecycle.ERROR
                self._error_message = str(e)
                logger.error(f"❌ Failed to start service {self.name}: {e}")
                raise
    
    async def stop(self) -> None:
        """Stop the service with lifecycle management"""
        async with self._lock:
            if self._status in [ServiceLifecycle.STOPPED, ServiceLifecycle.STOPPING]:
                return
            
            try:
                self._status = ServiceLifecycle.STOPPING
                logger.info(f"🛑 Stopping service: {self.name}")
                
                await self._stop_implementation()
                
                self._status = ServiceLifecycle.STOPPED
                logger.info(f"✅ Service stopped: {self.name}")
                
            except Exception as e:
                self._status = ServiceLifecycle.ERROR
                self._error_message = str(e)
                logger.error(f"❌ Failed to stop service {self.name}: {e}")
                raise
    
    async def health_check(self) -> ServiceHealth:
        """Get service health status"""
        uptime = time.time() - self._start_time if self._start_time else 0
        
        # Get service-specific metrics
        metrics = await self._get_health_metrics()
        
        return ServiceHealth(
            name=self.name,
            status=self._status,
            uptime=uptime,
            last_check=datetime.utcnow(),
            error_message=self._error_message,
            metrics=metrics
        )
    
    @abstractmethod
    async def _start_implementation(self) -> None:
        """Service-specific start logic"""
        pass
    
    @abstractmethod
    async def _stop_implementation(self) -> None:
        """Service-specific stop logic"""
        pass
    
    async def _get_health_metrics(self) -> Dict[str, Any]:
        """Get service-specific health metrics"""
        return {}

# ==================== CORE SERVICES ====================

class EnrichmentService(BaseService):
    """Match enrichment service"""
    
    def __init__(self):
        super().__init__("enrichment_service")
        self.pipeline = None
        self.processed_matches = 0
        self.error_count = 0
    
    async def _start_implementation(self) -> None:
        """Start enrichment service"""
        from async_enrichment_pipeline import HighPerformanceEnrichmentPipeline, EnrichmentConfig
        from unified_configuration import get_config
        
        config = get_config()
        openai_key = config.get_api_key('openai')
        
        if not openai_key:
            raise RuntimeError("OpenAI API key not configured")
        
        enrichment_config = EnrichmentConfig(
            max_concurrent=config.apis.enrichment['max_concurrent'],
            batch_size=config.apis.enrichment['batch_size'],
            cache_ttl=config.apis.enrichment['cache_ttl']
        )
        
        self.pipeline = HighPerformanceEnrichmentPipeline(
            api_key=openai_key,
            config=enrichment_config
        )
    
    async def _stop_implementation(self) -> None:
        """Stop enrichment service"""
        if self.pipeline:
            # Close any open connections
            pass
    
    async def _get_health_metrics(self) -> Dict[str, Any]:
        """Get enrichment service metrics"""
        cache_stats = {}
        if self.pipeline:
            cache_stats = await self.pipeline.get_cache_stats()
        
        return {
            "processed_matches": self.processed_matches,
            "error_count": self.error_count,
            "cache_stats": cache_stats
        }
    
    async def enrich_matches(self, matches: List[Dict[str, Any]]) -> List[Any]:
        """Enrich matches using the pipeline"""
        if not self.pipeline:
            raise RuntimeError("Enrichment service not started")
        
        try:
            results = await self.pipeline.enrich_dataset_batch(matches)
            self.processed_matches += len(results)
            return results
        except Exception as e:
            self.error_count += 1
            raise

class KnowledgeGraphService(BaseService):
    """Knowledge graph management service"""
    
    def __init__(self):
        super().__init__("knowledge_graph_service")
        self.kg_builder = None
        self.graph = None
        self.build_count = 0
    
    async def _start_implementation(self) -> None:
        """Start KG service"""
        from optimized_kg_builder import OptimizedKGBuilder
        
        self.kg_builder = OptimizedKGBuilder()
    
    async def _stop_implementation(self) -> None:
        """Stop KG service"""
        # Save current graph if needed
        if self.graph:
            self.kg_builder.save_graph("models/latest_kg.pkl")
    
    async def _get_health_metrics(self) -> Dict[str, Any]:
        """Get KG service metrics"""
        return {
            "build_count": self.build_count,
            "graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
            "graph_edges": self.graph.number_of_edges() if self.graph else 0
        }
    
    async def build_graph(self, data_path: str) -> Any:
        """Build knowledge graph"""
        if not self.kg_builder:
            raise RuntimeError("KG service not started")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self.graph = await loop.run_in_executor(
            None, 
            self.kg_builder.build_from_data,
            data_path
        )
        
        self.build_count += 1
        return self.graph

class MatchAlignmentService(BaseService):
    """Match alignment service"""
    
    def __init__(self):
        super().__init__("match_alignment_service")
        self.aligner = None
        self.alignment_count = 0
    
    async def _start_implementation(self) -> None:
        """Start alignment service"""
        from unified_match_aligner import create_aligner
        
        self.aligner = create_aligner(
            strategy="hybrid",
            similarity_threshold=0.8
        )
    
    async def _stop_implementation(self) -> None:
        """Stop alignment service"""
        pass
    
    async def _get_health_metrics(self) -> Dict[str, Any]:
        """Get alignment service metrics"""
        return {
            "alignment_count": self.alignment_count
        }
    
    async def align_datasets(self, dataset1_path: str, dataset2_path: str) -> List[Any]:
        """Align two datasets"""
        if not self.aligner:
            raise RuntimeError("Alignment service not started")
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        matches = await loop.run_in_executor(
            None,
            self.aligner.align_datasets,
            dataset1_path,
            dataset2_path
        )
        
        self.alignment_count += 1
        return matches

class ModelTrainingService(BaseService):
    """ML model training service"""
    
    def __init__(self):
        super().__init__("model_training_service")
        self.trainer = None
        self.training_count = 0
        self.current_model = None
    
    async def _start_implementation(self) -> None:
        """Start training service"""
        # Would initialize trainer
        pass
    
    async def _stop_implementation(self) -> None:
        """Stop training service"""
        # Save current model if training
        pass
    
    async def _get_health_metrics(self) -> Dict[str, Any]:
        """Get training service metrics"""
        return {
            "training_count": self.training_count,
            "has_model": self.current_model is not None
        }
    
    async def train_model(self, config: Dict[str, Any]) -> Any:
        """Train ML model"""
        # Implementation would go here
        self.training_count += 1
        return {"status": "completed", "model_path": "models/latest.pt"}

# ==================== SERVICE CONTAINER ====================

class ServiceContainer:
    """Dependency injection container and service orchestrator"""
    
    def __init__(self):
        self._services: Dict[str, ServiceInterface] = {}
        self._singletons: Dict[str, Any] = {}
        self._service_dependencies: Dict[str, List[str]] = {}
        self._running = False
        self._lock = asyncio.Lock()
    
    def register_service(
        self,
        service_class: Type[ServiceInterface],
        dependencies: List[str] = None
    ) -> None:
        """Register a service with optional dependencies"""
        service_instance = service_class()
        service_name = service_instance.name
        
        self._services[service_name] = service_instance
        self._service_dependencies[service_name] = dependencies or []
        
        logger.info(f"📋 Registered service: {service_name}")
    
    def register_singleton(self, interface: str, instance: Any) -> None:
        """Register a singleton instance"""
        self._singletons[interface] = instance
        logger.info(f"📋 Registered singleton: {interface}")
    
    def resolve(self, interface: str) -> Any:
        """Resolve a service or singleton"""
        if interface in self._singletons:
            return self._singletons[interface]
        
        if interface in self._services:
            return self._services[interface]
        
        raise ValueError(f"Service not registered: {interface}")
    
    async def start_all_services(self) -> None:
        """Start all services in dependency order"""
        async with self._lock:
            if self._running:
                return
            
            logger.info("🚀 Starting all services...")
            
            # Determine startup order based on dependencies
            startup_order = self._calculate_startup_order()
            
            # Start services in order
            for service_name in startup_order:
                service = self._services[service_name]
                try:
                    await service.start()
                except Exception as e:
                    logger.error(f"❌ Failed to start {service_name}: {e}")
                    # Stop already started services
                    await self._stop_started_services(startup_order[:startup_order.index(service_name)])
                    raise
            
            self._running = True
            logger.info("✅ All services started successfully")
    
    async def stop_all_services(self) -> None:
        """Stop all services in reverse dependency order"""
        async with self._lock:
            if not self._running:
                return
            
            logger.info("🛑 Stopping all services...")
            
            # Stop in reverse order
            startup_order = self._calculate_startup_order()
            for service_name in reversed(startup_order):
                service = self._services[service_name]
                try:
                    await service.stop()
                except Exception as e:
                    logger.error(f"❌ Failed to stop {service_name}: {e}")
            
            self._running = False
            logger.info("✅ All services stopped")
    
    async def _stop_started_services(self, service_names: List[str]) -> None:
        """Stop a list of services (cleanup helper)"""
        for service_name in reversed(service_names):
            try:
                await self._services[service_name].stop()
            except Exception as e:
                logger.error(f"❌ Cleanup failed for {service_name}: {e}")
    
    def _calculate_startup_order(self) -> List[str]:
        """Calculate service startup order based on dependencies"""
        # Simple topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(service_name: str):
            if service_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {service_name}")
            
            if service_name not in visited:
                temp_visited.add(service_name)
                
                # Visit dependencies first
                for dep in self._service_dependencies.get(service_name, []):
                    if dep in self._services:
                        visit(dep)
                
                temp_visited.remove(service_name)
                visited.add(service_name)
                order.append(service_name)
        
        # Visit all services
        for service_name in self._services.keys():
            if service_name not in visited:
                visit(service_name)
        
        return order
    
    async def get_health_status(self) -> Dict[str, ServiceHealth]:
        """Get health status of all services"""
        health_status = {}
        
        for service_name, service in self._services.items():
            try:
                health = await service.health_check()
                health_status[service_name] = health
            except Exception as e:
                health_status[service_name] = ServiceHealth(
                    name=service_name,
                    status=ServiceLifecycle.ERROR,
                    uptime=0,
                    last_check=datetime.utcnow(),
                    error_message=str(e)
                )
        
        return health_status
    
    @asynccontextmanager
    async def lifecycle(self):
        """Context manager for service lifecycle"""
        try:
            await self.start_all_services()
            yield self
        finally:
            await self.stop_all_services()

# ==================== GLOBAL CONTAINER ====================

# Global service container instance
_container: Optional[ServiceContainer] = None

def get_container() -> ServiceContainer:
    """Get global service container"""
    global _container
    if _container is None:
        _container = ServiceContainer()
        _initialize_default_services()
    return _container

def _initialize_default_services():
    """Initialize default services"""
    container = _container
    
    # Register core services
    container.register_service(EnrichmentService)
    container.register_service(KnowledgeGraphService)
    container.register_service(MatchAlignmentService)
    container.register_service(ModelTrainingService)
    
    logger.info("🔧 Default services registered")

# Convenience functions
async def start_services():
    """Start all services"""
    container = get_container()
    await container.start_all_services()

async def stop_services():
    """Stop all services"""
    container = get_container()
    await container.stop_all_services()

def get_service(service_name: str) -> ServiceInterface:
    """Get a service by name"""
    container = get_container()
    return container.resolve(service_name)

# Example usage
if __name__ == "__main__":
    async def main():
        """Example usage of service container"""
        
        # Get container
        container = get_container()
        
        # Use lifecycle context manager
        async with container.lifecycle():
            # Get services
            enrichment_service = container.resolve("enrichment_service")
            kg_service = container.resolve("knowledge_graph_service")
            
            # Check health
            health_status = await container.get_health_status()
            
            print("🏥 Service Health Status:")
            for name, health in health_status.items():
                print(f"  {name}: {health.status.value} (uptime: {health.uptime:.1f}s)")
            
            # Use services
            print("\n🔧 Services ready for use!")
            
            # Example: Build knowledge graph
            # graph = await kg_service.build_graph("/path/to/data.csv")
            
            # Example: Enrich matches
            # results = await enrichment_service.enrich_matches([...])
    
    # Run example
    asyncio.run(main())

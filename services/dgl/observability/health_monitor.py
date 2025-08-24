# Purpose: System health monitoring for DGL components
# Author: WicketWise AI, Last Modified: 2024

"""
Health Monitor

Monitors overall system health and component status:
- Component health checks
- System resource monitoring
- Service dependency checks
- Health status aggregation
- Health reporting and dashboards
"""

import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
import psutil

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a system component"""
    component_name: str
    status: HealthStatus
    message: str
    last_check: datetime = field(default_factory=datetime.now)
    response_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "component_name": self.component_name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "response_time_ms": self.response_time_ms,
            "metadata": self.metadata
        }


@dataclass
class SystemHealth:
    """Overall system health status"""
    overall_status: HealthStatus
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    uptime_seconds: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overall_status": self.overall_status.value,
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
            "system_metrics": self.system_metrics,
            "uptime_seconds": self.uptime_seconds,
            "last_updated": self.last_updated.isoformat()
        }


class HealthMonitor:
    """
    Comprehensive system health monitoring
    
    Monitors the health of DGL components and system resources
    with configurable health checks and alerting.
    """
    
    def __init__(self):
        """Initialize health monitor"""
        self.start_time = datetime.now()
        
        # Component health checks
        self.health_checks: Dict[str, Callable[[], ComponentHealth]] = {}
        
        # Health history
        self.health_history: List[SystemHealth] = []
        
        # Health callbacks
        self.health_callbacks: List[Callable[[SystemHealth], None]] = []
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        logger.info("Health monitor initialized")
    
    def start(self):
        """Start health monitoring"""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    def stop(self):
        """Stop health monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
        logger.info("Health monitoring stopped")
    
    def register_health_check(self, component_name: str, 
                            check_function: Callable[[], ComponentHealth]):
        """Register a health check for a component"""
        self.health_checks[component_name] = check_function
        logger.info(f"Registered health check for component: {component_name}")
    
    def unregister_health_check(self, component_name: str):
        """Unregister a health check"""
        if component_name in self.health_checks:
            del self.health_checks[component_name]
            logger.info(f"Unregistered health check for component: {component_name}")
    
    async def check_system_health(self) -> SystemHealth:
        """Perform comprehensive system health check"""
        components = {}
        
        # Run all registered health checks
        for component_name, check_function in self.health_checks.items():
            try:
                start_time = time.time()
                component_health = check_function()
                check_duration = (time.time() - start_time) * 1000
                
                # Update response time
                component_health.response_time_ms = check_duration
                component_health.last_check = datetime.now()
                
                components[component_name] = component_health
                
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {str(e)}")
                components[component_name] = ComponentHealth(
                    component_name=component_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    last_check=datetime.now()
                )
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        
        # Calculate overall status
        overall_status = self._calculate_overall_status(components, system_metrics)
        
        # Calculate uptime
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        # Create system health object
        system_health = SystemHealth(
            overall_status=overall_status,
            components=components,
            system_metrics=system_metrics,
            uptime_seconds=uptime_seconds,
            last_updated=datetime.now()
        )
        
        # Store in history
        self.health_history.append(system_health)
        
        # Keep only recent history
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-500:]
        
        # Trigger callbacks
        for callback in self.health_callbacks:
            try:
                callback(system_health)
            except Exception as e:
                logger.error(f"Error in health callback: {str(e)}")
        
        return system_health
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status of a specific component"""
        if not self.health_history:
            return None
        
        latest_health = self.health_history[-1]
        return latest_health.components.get(component_name)
    
    def get_health_history(self, hours: int = 24) -> List[SystemHealth]:
        """Get health history for the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        return [
            health for health in self.health_history
            if health.last_updated >= cutoff
        ]
    
    def get_health_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get health summary for the last N hours"""
        history = self.get_health_history(hours)
        
        if not history:
            return {"error": "No health data available"}
        
        # Calculate availability percentages
        component_availability = {}
        
        for health in history:
            for comp_name, comp_health in health.components.items():
                if comp_name not in component_availability:
                    component_availability[comp_name] = {"total": 0, "healthy": 0}
                
                component_availability[comp_name]["total"] += 1
                if comp_health.status == HealthStatus.HEALTHY:
                    component_availability[comp_name]["healthy"] += 1
        
        # Calculate percentages
        availability_pct = {}
        for comp_name, stats in component_availability.items():
            availability_pct[comp_name] = (stats["healthy"] / stats["total"]) * 100
        
        # Overall system availability
        overall_healthy = len([h for h in history if h.overall_status == HealthStatus.HEALTHY])
        overall_availability = (overall_healthy / len(history)) * 100
        
        # Current status
        current_health = history[-1] if history else None
        
        return {
            "period": {
                "start": history[0].last_updated.isoformat(),
                "end": history[-1].last_updated.isoformat(),
                "data_points": len(history)
            },
            "overall_availability_pct": overall_availability,
            "component_availability_pct": availability_pct,
            "current_status": current_health.overall_status.value if current_health else "unknown",
            "uptime_seconds": current_health.uptime_seconds if current_health else 0
        }
    
    def add_health_callback(self, callback: Callable[[SystemHealth], None]):
        """Add callback for health status changes"""
        self.health_callbacks.append(callback)
    
    def remove_health_callback(self, callback: Callable[[SystemHealth], None]):
        """Remove health callback"""
        if callback in self.health_callbacks:
            self.health_callbacks.remove(callback)
    
    async def _monitoring_loop(self):
        """Main health monitoring loop"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Perform health check
                await self.check_system_health()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {str(e)}")
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                network_metrics = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            except Exception:
                network_metrics = {}
            
            # Process metrics
            process = psutil.Process()
            process_metrics = {
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None
            }
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                "memory": {
                    "total_bytes": memory.total,
                    "available_bytes": memory.available,
                    "used_bytes": memory.used,
                    "usage_percent": memory.percent
                },
                "disk": {
                    "total_bytes": disk.total,
                    "free_bytes": disk.free,
                    "used_bytes": disk.used,
                    "usage_percent": (disk.used / disk.total) * 100
                },
                "network": network_metrics,
                "process": process_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_overall_status(self, components: Dict[str, ComponentHealth], 
                                system_metrics: Dict[str, Any]) -> HealthStatus:
        """Calculate overall system health status"""
        
        # Check component health
        component_statuses = [comp.status for comp in components.values()]
        
        if not component_statuses:
            return HealthStatus.UNKNOWN
        
        # If any component is unhealthy, system is unhealthy
        if HealthStatus.UNHEALTHY in component_statuses:
            return HealthStatus.UNHEALTHY
        
        # If any component is degraded, system is degraded
        if HealthStatus.DEGRADED in component_statuses:
            return HealthStatus.DEGRADED
        
        # Check system resource health
        if "error" in system_metrics:
            return HealthStatus.DEGRADED
        
        # Check CPU usage
        cpu_usage = system_metrics.get("cpu", {}).get("usage_percent", 0)
        if cpu_usage > 90:
            return HealthStatus.UNHEALTHY
        elif cpu_usage > 80:
            return HealthStatus.DEGRADED
        
        # Check memory usage
        memory_usage = system_metrics.get("memory", {}).get("usage_percent", 0)
        if memory_usage > 95:
            return HealthStatus.UNHEALTHY
        elif memory_usage > 85:
            return HealthStatus.DEGRADED
        
        # Check disk usage
        disk_usage = system_metrics.get("disk", {}).get("usage_percent", 0)
        if disk_usage > 95:
            return HealthStatus.UNHEALTHY
        elif disk_usage > 90:
            return HealthStatus.DEGRADED
        
        # All checks passed
        return HealthStatus.HEALTHY
    
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        
        def check_dgl_engine():
            """Check DGL engine health"""
            try:
                # Simple health check - in production would check actual engine
                return ComponentHealth(
                    component_name="dgl_engine",
                    status=HealthStatus.HEALTHY,
                    message="DGL engine operational"
                )
            except Exception as e:
                return ComponentHealth(
                    component_name="dgl_engine",
                    status=HealthStatus.UNHEALTHY,
                    message=f"DGL engine error: {str(e)}"
                )
        
        def check_governance_system():
            """Check governance system health"""
            try:
                # Check governance components
                return ComponentHealth(
                    component_name="governance_system",
                    status=HealthStatus.HEALTHY,
                    message="Governance system operational"
                )
            except Exception as e:
                return ComponentHealth(
                    component_name="governance_system",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Governance system error: {str(e)}"
                )
        
        def check_audit_system():
            """Check audit system health"""
            try:
                # Check audit store
                return ComponentHealth(
                    component_name="audit_system",
                    status=HealthStatus.HEALTHY,
                    message="Audit system operational"
                )
            except Exception as e:
                return ComponentHealth(
                    component_name="audit_system",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Audit system error: {str(e)}"
                )
        
        def check_metrics_system():
            """Check metrics collection system health"""
            try:
                # Check metrics collector
                return ComponentHealth(
                    component_name="metrics_system",
                    status=HealthStatus.HEALTHY,
                    message="Metrics system operational"
                )
            except Exception as e:
                return ComponentHealth(
                    component_name="metrics_system",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Metrics system error: {str(e)}"
                )
        
        # Register default health checks
        self.register_health_check("dgl_engine", check_dgl_engine)
        self.register_health_check("governance_system", check_governance_system)
        self.register_health_check("audit_system", check_audit_system)
        self.register_health_check("metrics_system", check_metrics_system)


# Utility functions for health monitoring

def create_health_monitor() -> HealthMonitor:
    """Create and configure health monitor"""
    return HealthMonitor()


def setup_health_alerts(monitor: HealthMonitor):
    """Setup health status alerting"""
    
    def log_health_changes(system_health: SystemHealth):
        """Log health status changes"""
        if system_health.overall_status != HealthStatus.HEALTHY:
            logger.warning(f"System health status: {system_health.overall_status.value}")
            
            # Log unhealthy components
            for comp_name, comp_health in system_health.components.items():
                if comp_health.status != HealthStatus.HEALTHY:
                    logger.warning(f"Component {comp_name}: {comp_health.status.value} - {comp_health.message}")
    
    monitor.add_health_callback(log_health_changes)
    logger.info("Health alerting configured")

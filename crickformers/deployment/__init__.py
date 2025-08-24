# Purpose: Production deployment and infrastructure management module
# Author: WicketWise AI, Last Modified: 2024

"""
Production Deployment & Monitoring Module

This module provides comprehensive production deployment and infrastructure
management capabilities for the WicketWise cricket intelligence platform,
including containerization, CI/CD pipelines, and production monitoring.

Key Components:
- Docker containerization and orchestration
- Kubernetes deployment and scaling
- CI/CD pipeline management and automation
- Production health monitoring and alerting
- Infrastructure as Code (IaC) templates
- Load balancing and service discovery
- Backup and disaster recovery systems
- Environment configuration management
- Production logging and observability
- Performance monitoring and optimization
"""

__version__ = "1.0.0"
__author__ = "WicketWise AI"

# Core deployment components
from .container_manager import (
    ContainerManager,
    DockerConfig,
    ContainerStatus,
    ContainerMetrics,
    DeploymentError
)

# Kubernetes management (to be implemented in future phases)
# from .kubernetes_manager import (
#     KubernetesManager,
#     K8sConfig,
#     K8sDeployment,
#     K8sService,
#     K8sResource,
#     K8sStatus
# )

# CI/CD management (to be implemented in future phases)
# from .cicd_manager import (
#     CICDManager,
#     Pipeline,
#     PipelineStage,
#     PipelineStatus,
#     BuildConfig,
#     DeploymentConfig
# )

from .health_monitor import (
    HealthMonitor,
    HealthCheck,
    HealthStatus,
    ServiceHealth,
    SystemHealth,
    AlertManager
)

__all__ = [
    # Container management
    'ContainerManager',
    'DockerConfig',
    'ContainerStatus',
    'ContainerMetrics',
    'DeploymentError',
    
    # Kubernetes management (to be implemented in future phases)
    # 'KubernetesManager',
    # 'K8sConfig',
    # 'K8sDeployment',
    # 'K8sService',
    # 'K8sResource',
    # 'K8sStatus',
    
    # CI/CD management (to be implemented in future phases)
    # 'CICDManager',
    # 'Pipeline',
    # 'PipelineStage',
    # 'PipelineStatus',
    # 'BuildConfig',
    # 'DeploymentConfig',
    
    # Health monitoring
    'HealthMonitor',
    'HealthCheck',
    'HealthStatus',
    'ServiceHealth',
    'SystemHealth',
    'AlertManager'
]

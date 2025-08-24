# Purpose: Production deployment and infrastructure tests module initialization
# Author: WicketWise AI, Last Modified: 2024

"""
Production Deployment & Infrastructure Tests Module

This module contains comprehensive tests for the WicketWise production
deployment and infrastructure management system, including containerization,
health monitoring, and deployment automation.

Test Categories:
- Docker containerization and orchestration
- Health monitoring and alerting systems
- Production deployment configurations
- Infrastructure as Code validation
- CI/CD pipeline testing
- Load balancing and scaling tests
"""

__version__ = "1.0.0"
__author__ = "WicketWise AI"

# Test utilities and fixtures
# Container manager tests (require Docker to be available)
# from .test_container_manager import (
#     TestContainerManager,
#     TestDockerConfig,
#     TestContainerMetrics
# )

from .test_health_monitor import (
    TestHealthMonitor,
    TestHealthCheck,
    TestServiceHealth,
    TestAlertManager
)

__all__ = [
    # Container manager tests (require Docker to be available)
    # 'TestContainerManager',
    # 'TestDockerConfig',
    # 'TestContainerMetrics',
    'TestHealthMonitor',
    'TestHealthCheck',
    'TestServiceHealth',
    'TestAlertManager'
]

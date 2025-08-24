# Purpose: Performance optimization and caching module
# Author: WicketWise AI, Last Modified: 2024

"""
Performance Optimization Module

This module provides comprehensive performance optimization features including
intelligent caching, memory management, and resource optimization for the
WicketWise cricket intelligence platform.

Key Components:
- Multi-level caching system (Redis, in-memory, disk)
- Memory pool management and optimization
- Query result caching with intelligent invalidation
- Model prediction caching with TTL management
- Resource usage optimization and monitoring
- Cache warming and preloading strategies
"""

__version__ = "1.0.0"
__author__ = "WicketWise AI"

# Core optimization components
from .cache_manager import (
    CacheManager,
    CacheLevel,
    CachePolicy,
    CacheEntry,
    CacheStats
)

# Memory optimization (to be implemented in future phases)
# from .memory_optimizer import (
#     MemoryOptimizer,
#     MemoryPool,
#     MemoryUsageTracker,
#     OptimizationStrategy
# )

# Query caching (to be implemented in future phases)
# from .query_cache import (
#     QueryCache,
#     QueryCacheEntry,
#     CacheInvalidationRule,
#     QueryCacheManager
# )

__all__ = [
    # Cache management
    'CacheManager',
    'CacheLevel',
    'CachePolicy', 
    'CacheEntry',
    'CacheStats',
    
    # Memory optimization (to be implemented in future phases)
    # 'MemoryOptimizer',
    # 'MemoryPool',
    # 'MemoryUsageTracker',
    # 'OptimizationStrategy',
    
    # Query caching (to be implemented in future phases)
    # 'QueryCache',
    # 'QueryCacheEntry',
    # 'CacheInvalidationRule',
    # 'QueryCacheManager'
]

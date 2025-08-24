# Purpose: Performance optimization tests module initialization
# Author: WicketWise AI, Last Modified: 2024

"""
Performance Optimization Tests Module

This module contains comprehensive tests for the WicketWise performance optimization
system, including caching, memory management, and resource optimization.

Test Categories:
- Multi-level cache management and policies
- Memory optimization and pool management
- Query caching with intelligent invalidation
- Performance optimization strategies
- Integration testing with the main system
"""

__version__ = "1.0.0"
__author__ = "WicketWise AI"

# Test utilities and fixtures
from .test_cache_manager import (
    TestCacheManager,
    TestMemoryCache,
    TestCacheEntry
)

__all__ = [
    'TestCacheManager',
    'TestMemoryCache',
    'TestCacheEntry'
]

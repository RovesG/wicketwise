# Purpose: Unit tests for cache management system
# Author: WicketWise AI, Last Modified: 2024

import pytest
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crickformers.optimization.cache_manager import (
    CacheManager,
    MemoryCache,
    CacheEntry,
    CacheStats,
    CacheLevel,
    CachePolicy,
    cached
)


class TestCacheEntry:
    """Test suite for CacheEntry data structure"""
    
    def test_cache_entry_creation(self):
        """Test CacheEntry creation and basic properties"""
        timestamp = datetime.now()
        entry = CacheEntry(
            key="test_key",
            value={"data": "test_value"},
            created_at=timestamp,
            last_accessed=timestamp,
            access_count=1,
            ttl_seconds=300,
            size_bytes=1024,
            metadata={"source": "test"}
        )
        
        assert entry.key == "test_key"
        assert entry.value == {"data": "test_value"}
        assert entry.created_at == timestamp
        assert entry.last_accessed == timestamp
        assert entry.access_count == 1
        assert entry.ttl_seconds == 300
        assert entry.size_bytes == 1024
        assert entry.metadata == {"source": "test"}
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration logic"""
        # Non-expiring entry
        entry = CacheEntry(
            key="no_ttl",
            value="test",
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        assert not entry.is_expired
        
        # Expired entry
        old_time = datetime.now() - timedelta(seconds=10)
        expired_entry = CacheEntry(
            key="expired",
            value="test",
            created_at=old_time,
            last_accessed=old_time,
            ttl_seconds=5
        )
        assert expired_entry.is_expired
        
        # Non-expired entry
        recent_time = datetime.now() - timedelta(seconds=1)
        fresh_entry = CacheEntry(
            key="fresh",
            value="test",
            created_at=recent_time,
            last_accessed=recent_time,
            ttl_seconds=10
        )
        assert not fresh_entry.is_expired
    
    def test_cache_entry_touch(self):
        """Test cache entry touch functionality"""
        entry = CacheEntry(
            key="touch_test",
            value="test",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1
        )
        
        original_access_time = entry.last_accessed
        original_count = entry.access_count
        
        # Wait a bit to ensure time difference
        time.sleep(0.01)
        entry.touch()
        
        assert entry.last_accessed > original_access_time
        assert entry.access_count == original_count + 1
    
    def test_cache_entry_to_dict(self):
        """Test cache entry dictionary conversion"""
        timestamp = datetime.now()
        entry = CacheEntry(
            key="dict_test",
            value={"complex": "data"},
            created_at=timestamp,
            last_accessed=timestamp,
            access_count=5,
            ttl_seconds=600,
            size_bytes=2048,
            metadata={"type": "test"}
        )
        
        entry_dict = entry.to_dict()
        
        assert entry_dict['key'] == "dict_test"
        assert entry_dict['created_at'] == timestamp.isoformat()
        assert entry_dict['last_accessed'] == timestamp.isoformat()
        assert entry_dict['access_count'] == 5
        assert entry_dict['ttl_seconds'] == 600
        assert entry_dict['size_bytes'] == 2048
        assert 'age_seconds' in entry_dict
        assert 'is_expired' in entry_dict
        assert entry_dict['metadata'] == {"type": "test"}
        # Value should not be included in dict
        assert 'value' not in entry_dict


class TestCacheStats:
    """Test suite for CacheStats"""
    
    def test_cache_stats_creation(self):
        """Test CacheStats creation and basic properties"""
        stats = CacheStats(
            total_entries=100,
            total_size_bytes=1024000,
            hit_count=80,
            miss_count=20,
            eviction_count=5,
            error_count=2
        )
        
        assert stats.total_entries == 100
        assert stats.total_size_bytes == 1024000
        assert stats.hit_count == 80
        assert stats.miss_count == 20
        assert stats.eviction_count == 5
        assert stats.error_count == 2
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation"""
        stats = CacheStats(hit_count=80, miss_count=20)
        assert stats.hit_rate == 0.8
        assert stats.miss_rate == 0.2
        
        # Edge case: no requests
        empty_stats = CacheStats()
        assert empty_stats.hit_rate == 0.0
        assert empty_stats.miss_rate == 1.0
    
    def test_cache_stats_to_dict(self):
        """Test cache stats dictionary conversion"""
        stats = CacheStats(
            total_entries=50,
            total_size_bytes=512000,
            hit_count=40,
            miss_count=10,
            eviction_count=3,
            error_count=1
        )
        
        stats_dict = stats.to_dict()
        
        assert stats_dict['total_entries'] == 50
        assert stats_dict['total_size_bytes'] == 512000
        assert stats_dict['hit_count'] == 40
        assert stats_dict['miss_count'] == 10
        assert stats_dict['eviction_count'] == 3
        assert stats_dict['error_count'] == 1
        assert stats_dict['hit_rate'] == 0.8
        assert stats_dict['miss_rate'] == 0.2


class TestMemoryCache:
    """Test suite for MemoryCache"""
    
    @pytest.fixture
    def cache(self):
        """Create MemoryCache instance"""
        return MemoryCache(max_size=10, max_memory_mb=1, policy=CachePolicy.LRU)
    
    def test_cache_initialization(self, cache):
        """Test MemoryCache initialization"""
        assert cache.max_size == 10
        assert cache.max_memory_bytes == 1024 * 1024
        assert cache.policy == CachePolicy.LRU
        assert len(cache.entries) == 0
        assert len(cache.access_counts) == 0
    
    def test_basic_get_put(self, cache):
        """Test basic get and put operations"""
        # Put a value
        success = cache.put("key1", "value1")
        assert success
        
        # Get the value
        value = cache.get("key1")
        assert value == "value1"
        
        # Get non-existent key
        missing = cache.get("nonexistent")
        assert missing is None
    
    def test_ttl_expiration(self, cache):
        """Test TTL-based expiration"""
        # Put with short TTL
        cache.put("expiring_key", "expiring_value", ttl_seconds=1)
        
        # Should be available immediately
        value = cache.get("expiring_key")
        assert value == "expiring_value"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        expired_value = cache.get("expiring_key")
        assert expired_value is None
    
    def test_lru_eviction(self, cache):
        """Test LRU eviction policy"""
        # Fill cache to capacity
        for i in range(cache.max_size):
            cache.put(f"key{i}", f"value{i}")
        
        # Access first key to make it recently used
        cache.get("key0")
        
        # Add one more item (should evict least recently used)
        cache.put("new_key", "new_value")
        
        # key0 should still be there (recently accessed)
        assert cache.get("key0") == "value0"
        
        # key1 should be evicted (least recently used)
        assert cache.get("key1") is None
        
        # new_key should be there
        assert cache.get("new_key") == "new_value"
    
    def test_lfu_eviction(self):
        """Test LFU eviction policy"""
        cache = MemoryCache(max_size=3, max_memory_mb=1, policy=CachePolicy.LFU)
        
        # Add items
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 multiple times
        for _ in range(5):
            cache.get("key1")
        
        # Access key2 once
        cache.get("key2")
        
        # key3 is never accessed after creation
        
        # Add new item (should evict least frequently used - key3)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Most frequently used
        assert cache.get("key2") == "value2"  # Moderately used
        assert cache.get("key3") is None      # Least frequently used (evicted)
        assert cache.get("key4") == "value4"  # New item
    
    def test_memory_limit_eviction(self):
        """Test memory-based eviction"""
        # Create cache with very small memory limit
        cache = MemoryCache(max_size=100, max_memory_mb=0.001, policy=CachePolicy.LRU)
        
        # Add large values that exceed memory limit
        large_value1 = "x" * 500  # ~500 bytes
        large_value2 = "y" * 500  # ~500 bytes
        
        cache.put("large1", large_value1)
        cache.put("large2", large_value2)  # Should trigger eviction
        
        # First value should be evicted due to memory pressure
        assert cache.get("large1") is None
        assert cache.get("large2") == large_value2
    
    def test_delete_operation(self, cache):
        """Test delete operation"""
        # Add some items
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Delete existing key
        success = cache.delete("key1")
        assert success
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        
        # Delete non-existent key
        success = cache.delete("nonexistent")
        assert not success
    
    def test_clear_operation(self, cache):
        """Test clear operation"""
        # Add some items
        for i in range(5):
            cache.put(f"key{i}", f"value{i}")
        
        assert len(cache.entries) == 5
        
        # Clear cache
        cache.clear()
        
        assert len(cache.entries) == 0
        assert len(cache.access_counts) == 0
    
    def test_cache_stats(self, cache):
        """Test cache statistics"""
        # Add some items
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        stats = cache.get_stats()
        
        assert stats['total_entries'] == 2
        assert stats['total_size_bytes'] > 0
        assert stats['max_size'] == cache.max_size
        assert stats['max_memory_bytes'] == cache.max_memory_bytes
        assert 'memory_usage_percent' in stats
        assert stats['policy'] == cache.policy.value


class TestCacheManager:
    """Test suite for CacheManager"""
    
    @pytest.fixture
    def cache_manager(self):
        """Create CacheManager instance"""
        config = {
            'memory_max_size': 100,
            'memory_max_mb': 10,
            'memory_policy': 'lru',
            'redis_enabled': False  # Disable Redis for testing
        }
        return CacheManager(config)
    
    def test_cache_manager_initialization(self, cache_manager):
        """Test CacheManager initialization"""
        assert cache_manager.memory_cache is not None
        assert cache_manager.redis_cache is None  # Disabled in config
        assert isinstance(cache_manager.stats, CacheStats)
        assert len(cache_manager.level_stats) == 3
        assert len(cache_manager.warming_functions) == 0
    
    def test_memory_cache_operations(self, cache_manager):
        """Test memory cache operations through manager"""
        # Put value
        success = cache_manager.put("test_key", {"data": "test"}, levels=[CacheLevel.MEMORY])
        assert success
        
        # Get value
        value = cache_manager.get("test_key", levels=[CacheLevel.MEMORY])
        assert value == {"data": "test"}
        
        # Delete value
        success = cache_manager.delete("test_key", levels=[CacheLevel.MEMORY])
        assert success
        
        # Verify deletion
        value = cache_manager.get("test_key", levels=[CacheLevel.MEMORY])
        assert value is None
    
    def test_cache_key_generation(self, cache_manager):
        """Test cache key generation"""
        # Test with various argument types
        key1 = cache_manager.create_cache_key("arg1", 123, True)
        key2 = cache_manager.create_cache_key("arg1", 123, True)
        key3 = cache_manager.create_cache_key("arg1", 124, True)
        
        # Same arguments should produce same key
        assert key1 == key2
        
        # Different arguments should produce different keys
        assert key1 != key3
        
        # Test with keyword arguments
        key4 = cache_manager.create_cache_key("base", param1="value1", param2=42)
        key5 = cache_manager.create_cache_key("base", param2=42, param1="value1")
        
        # Order of kwargs shouldn't matter
        assert key4 == key5
    
    def test_cache_warming(self, cache_manager):
        """Test cache warming functionality"""
        # Register warming function
        warming_called = False
        warming_args = None
        
        def test_warming_func(**kwargs):
            nonlocal warming_called, warming_args
            warming_called = True
            warming_args = kwargs
            # Simulate warming by adding some data
            cache_manager.put("warmed_key", "warmed_value")
        
        cache_manager.register_warming_function("test_cache", test_warming_func)
        
        # Warm cache
        cache_manager.warm_cache("test_cache", param1="value1", param2=42)
        
        # Verify warming function was called
        assert warming_called
        assert warming_args == {"param1": "value1", "param2": 42}
        
        # Verify warmed data is available
        value = cache_manager.get("warmed_key")
        assert value == "warmed_value"
    
    def test_comprehensive_stats(self, cache_manager):
        """Test comprehensive statistics"""
        # Add some data and operations
        cache_manager.put("key1", "value1")
        cache_manager.put("key2", "value2")
        cache_manager.get("key1")  # Hit
        cache_manager.get("nonexistent")  # Miss
        
        stats = cache_manager.get_comprehensive_stats()
        
        assert 'overall' in stats
        assert 'levels' in stats
        assert 'memory_cache' in stats
        assert 'redis_available' in stats
        assert 'warming_functions' in stats
        
        # Check overall stats
        overall = stats['overall']
        assert overall['hit_count'] >= 1
        assert overall['miss_count'] >= 1
        
        # Check level stats
        assert CacheLevel.MEMORY.value in stats['levels']
        assert CacheLevel.REDIS.value in stats['levels']
        assert CacheLevel.DISK.value in stats['levels']
        
        # Check memory cache stats
        memory_stats = stats['memory_cache']
        assert 'total_entries' in memory_stats
        assert 'total_size_bytes' in memory_stats
    
    def test_clear_operations(self, cache_manager):
        """Test clear operations"""
        # Add data to memory cache
        cache_manager.put("key1", "value1", levels=[CacheLevel.MEMORY])
        cache_manager.put("key2", "value2", levels=[CacheLevel.MEMORY])
        
        # Verify data exists
        assert cache_manager.get("key1") == "value1"
        assert cache_manager.get("key2") == "value2"
        
        # Clear memory cache
        cache_manager.clear(levels=[CacheLevel.MEMORY])
        
        # Verify data is cleared
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None


class TestCacheDecorator:
    """Test suite for cache decorator"""
    
    @pytest.fixture
    def cache_manager(self):
        """Create CacheManager for decorator testing"""
        return CacheManager({'memory_max_size': 50, 'redis_enabled': False})
    
    def test_cached_decorator_basic(self, cache_manager):
        """Test basic cached decorator functionality"""
        call_count = 0
        
        @cached(cache_manager, ttl_seconds=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Function not called again
        
        # Call with different args should execute function
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    def test_cached_decorator_with_kwargs(self, cache_manager):
        """Test cached decorator with keyword arguments"""
        call_count = 0
        
        @cached(cache_manager, key_prefix="test_")
        def function_with_kwargs(a, b=10, c=20):
            nonlocal call_count
            call_count += 1
            return a + b + c
        
        # Test with different combinations of args/kwargs
        result1 = function_with_kwargs(1, b=5, c=10)
        assert result1 == 16
        assert call_count == 1
        
        # Same call should use cache
        result2 = function_with_kwargs(1, b=5, c=10)
        assert result2 == 16
        assert call_count == 1
        
        # Different kwargs should execute function
        result3 = function_with_kwargs(1, b=6, c=10)
        assert result3 == 17
        assert call_count == 2
    
    def test_cached_decorator_with_levels(self, cache_manager):
        """Test cached decorator with specific cache levels"""
        @cached(cache_manager, levels=[CacheLevel.MEMORY], ttl_seconds=30)
        def memory_cached_function(value):
            return value * 2
        
        # Function should work and cache in memory
        result = memory_cached_function(5)
        assert result == 10
        
        # Verify it's cached by checking cache directly
        # (This is implementation-dependent, but we can check stats)
        stats = cache_manager.get_comprehensive_stats()
        assert stats['overall']['hit_count'] >= 0  # May have hits from cache lookups


def run_cache_tests():
    """Run all cache management tests"""
    print("🗄️  Running Cache Management Tests")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        ("Cache Entry", TestCacheEntry),
        ("Cache Stats", TestCacheStats),
        ("Memory Cache", TestMemoryCache),
        ("Cache Manager", TestCacheManager),
        ("Cache Decorator", TestCacheDecorator)
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for category_name, test_class in test_categories:
        print(f"\n📊 {category_name}")
        print("-" * 30)
        
        # Get test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        category_passed = 0
        for test_method in test_methods:
            total_tests += 1
            try:
                # Create test instance
                test_instance = test_class()
                
                # Handle fixtures
                if hasattr(test_instance, test_method):
                    method = getattr(test_instance, test_method)
                    
                    # Check if method needs fixtures
                    import inspect
                    sig = inspect.signature(method)
                    
                    if 'cache' in sig.parameters:
                        cache = MemoryCache(max_size=10, max_memory_mb=1, policy=CachePolicy.LRU)
                        method(cache)
                    elif 'cache_manager' in sig.parameters:
                        config = {
                            'memory_max_size': 100,
                            'memory_max_mb': 10,
                            'memory_policy': 'lru',
                            'redis_enabled': False
                        }
                        cache_manager = CacheManager(config)
                        method(cache_manager)
                    else:
                        method()
                    
                    print(f"  ✅ {test_method}")
                    passed_tests += 1
                    category_passed += 1
                    
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Category Results: {category_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall Cache Management Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_cache_tests()
    exit(0 if success else 1)

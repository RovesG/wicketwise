# Purpose: Intelligent multi-level caching system
# Author: WicketWise AI, Last Modified: 2024

import time
import json
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
import logging
import pickle
import gzip


class CacheLevel(Enum):
    """Cache level types"""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheEntry:
    """Cache entry data structure"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def touch(self):
        """Update last accessed time and increment access count"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary (excluding value for serialization)"""
        return {
            'key': self.key,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'ttl_seconds': self.ttl_seconds,
            'size_bytes': self.size_bytes,
            'age_seconds': self.age_seconds,
            'is_expired': self.is_expired,
            'metadata': self.metadata
        }


@dataclass
class CacheStats:
    """Cache statistics"""
    total_entries: int = 0
    total_size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    error_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 1.0 - self.hit_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            'total_entries': self.total_entries,
            'total_size_bytes': self.total_size_bytes,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'eviction_count': self.eviction_count,
            'error_count': self.error_count,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate
        }


class MemoryCache:
    """In-memory cache implementation"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, 
                 policy: CachePolicy = CachePolicy.LRU):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.policy = policy
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.entries:
                return None
            
            entry = self.entries[key]
            
            # Check expiration
            if entry.is_expired:
                del self.entries[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                return None
            
            # Update access patterns
            entry.touch()
            self.access_counts[key] += 1
            
            # Move to end for LRU
            if self.policy == CachePolicy.LRU:
                self.entries.move_to_end(key)
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put value in cache"""
        with self.lock:
            try:
                # Calculate size
                size_bytes = self._calculate_size(value)
                
                # Check if we need to evict entries
                self._ensure_capacity(size_bytes)
                
                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    ttl_seconds=ttl_seconds,
                    size_bytes=size_bytes,
                    metadata=metadata or {}
                )
                
                # Remove existing entry if present
                if key in self.entries:
                    old_entry = self.entries[key]
                    del self.entries[key]
                
                # Add new entry
                self.entries[key] = entry
                self.access_counts[key] = 1
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error putting cache entry {key}: {str(e)}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self.lock:
            if key in self.entries:
                del self.entries[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                return True
            return False
    
    def clear(self):
        """Clear all entries"""
        with self.lock:
            self.entries.clear()
            self.access_counts.clear()
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            else:
                return 1024  # Default estimate
    
    def _ensure_capacity(self, new_size: int):
        """Ensure cache has capacity for new entry"""
        # Check memory limit
        current_memory = sum(entry.size_bytes for entry in self.entries.values())
        
        while (len(self.entries) >= self.max_size or 
               current_memory + new_size > self.max_memory_bytes):
            
            if not self.entries:
                break
            
            # Evict based on policy
            if self.policy == CachePolicy.LRU:
                # Remove least recently used (first in OrderedDict)
                key, entry = self.entries.popitem(last=False)
            elif self.policy == CachePolicy.LFU:
                # Remove least frequently used
                key = min(self.access_counts.keys(), 
                         key=lambda k: self.access_counts[k])
                entry = self.entries.pop(key)
            elif self.policy == CachePolicy.TTL:
                # Remove expired entries first, then oldest
                expired_keys = [k for k, e in self.entries.items() if e.is_expired]
                if expired_keys:
                    key = expired_keys[0]
                    entry = self.entries.pop(key)
                else:
                    key, entry = self.entries.popitem(last=False)
            else:  # FIFO
                key, entry = self.entries.popitem(last=False)
            
            # Clean up access counts
            if key in self.access_counts:
                del self.access_counts[key]
            
            current_memory -= entry.size_bytes
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_size = sum(entry.size_bytes for entry in self.entries.values())
            return {
                'total_entries': len(self.entries),
                'total_size_bytes': total_size,
                'max_size': self.max_size,
                'max_memory_bytes': self.max_memory_bytes,
                'memory_usage_percent': (total_size / self.max_memory_bytes) * 100,
                'policy': self.policy.value
            }
    
    def get_entries(self) -> List[CacheEntry]:
        """Get all cache entries (for debugging)"""
        with self.lock:
            return list(self.entries.values())


class CacheManager:
    """Multi-level cache manager"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache levels
        self.memory_cache = MemoryCache(
            max_size=self.config.get('memory_max_size', 1000),
            max_memory_mb=self.config.get('memory_max_mb', 100),
            policy=CachePolicy(self.config.get('memory_policy', 'lru'))
        )
        
        # Redis cache (optional)
        self.redis_cache = None
        if self.config.get('redis_enabled', False):
            try:
                import redis
                self.redis_cache = redis.Redis(
                    host=self.config.get('redis_host', 'localhost'),
                    port=self.config.get('redis_port', 6379),
                    db=self.config.get('redis_db', 0),
                    decode_responses=True
                )
                # Test connection
                self.redis_cache.ping()
                self.logger.info("Redis cache initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Redis cache: {str(e)}")
                self.redis_cache = None
        
        # Statistics
        self.stats = CacheStats()
        self.level_stats = {
            CacheLevel.MEMORY: CacheStats(),
            CacheLevel.REDIS: CacheStats(),
            CacheLevel.DISK: CacheStats()
        }
        
        # Cache warming functions
        self.warming_functions: Dict[str, Callable] = {}
    
    def get(self, key: str, levels: Optional[List[CacheLevel]] = None) -> Optional[Any]:
        """Get value from cache with multi-level fallback"""
        levels = levels or [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK]
        
        for level in levels:
            try:
                value = None
                
                if level == CacheLevel.MEMORY:
                    value = self.memory_cache.get(key)
                    if value is not None:
                        self.level_stats[level].hit_count += 1
                        self.stats.hit_count += 1
                        return value
                    else:
                        self.level_stats[level].miss_count += 1
                
                elif level == CacheLevel.REDIS and self.redis_cache:
                    cached_data = self.redis_cache.get(key)
                    if cached_data:
                        try:
                            # Try to deserialize
                            value = json.loads(cached_data)
                            self.level_stats[level].hit_count += 1
                            self.stats.hit_count += 1
                            
                            # Promote to memory cache
                            self.memory_cache.put(key, value)
                            
                            return value
                        except json.JSONDecodeError:
                            # Try pickle
                            try:
                                value = pickle.loads(cached_data.encode('latin1'))
                                self.level_stats[level].hit_count += 1
                                self.stats.hit_count += 1
                                
                                # Promote to memory cache
                                self.memory_cache.put(key, value)
                                
                                return value
                            except Exception:
                                pass
                    
                    self.level_stats[level].miss_count += 1
                
                # Disk cache implementation would go here
                elif level == CacheLevel.DISK:
                    # Placeholder for disk cache
                    self.level_stats[level].miss_count += 1
                
            except Exception as e:
                self.logger.error(f"Error accessing {level.value} cache for key {key}: {str(e)}")
                self.level_stats[level].error_count += 1
                self.stats.error_count += 1
        
        self.stats.miss_count += 1
        return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
            levels: Optional[List[CacheLevel]] = None,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put value in cache across multiple levels"""
        levels = levels or [CacheLevel.MEMORY]
        success = False
        
        for level in levels:
            try:
                if level == CacheLevel.MEMORY:
                    if self.memory_cache.put(key, value, ttl_seconds, metadata):
                        success = True
                
                elif level == CacheLevel.REDIS and self.redis_cache:
                    try:
                        # Try JSON serialization first
                        serialized = json.dumps(value)
                    except (TypeError, ValueError):
                        # Fallback to pickle
                        serialized = pickle.dumps(value).decode('latin1')
                    
                    if ttl_seconds:
                        self.redis_cache.setex(key, ttl_seconds, serialized)
                    else:
                        self.redis_cache.set(key, serialized)
                    success = True
                
                # Disk cache implementation would go here
                elif level == CacheLevel.DISK:
                    # Placeholder for disk cache
                    pass
                
            except Exception as e:
                self.logger.error(f"Error putting to {level.value} cache for key {key}: {str(e)}")
                self.level_stats[level].error_count += 1
                self.stats.error_count += 1
        
        return success
    
    def delete(self, key: str, levels: Optional[List[CacheLevel]] = None) -> bool:
        """Delete key from cache levels"""
        levels = levels or [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK]
        success = False
        
        for level in levels:
            try:
                if level == CacheLevel.MEMORY:
                    if self.memory_cache.delete(key):
                        success = True
                
                elif level == CacheLevel.REDIS and self.redis_cache:
                    if self.redis_cache.delete(key):
                        success = True
                
                # Disk cache implementation would go here
                elif level == CacheLevel.DISK:
                    pass
                
            except Exception as e:
                self.logger.error(f"Error deleting from {level.value} cache for key {key}: {str(e)}")
                self.level_stats[level].error_count += 1
        
        return success
    
    def clear(self, levels: Optional[List[CacheLevel]] = None):
        """Clear cache levels"""
        levels = levels or [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK]
        
        for level in levels:
            try:
                if level == CacheLevel.MEMORY:
                    self.memory_cache.clear()
                
                elif level == CacheLevel.REDIS and self.redis_cache:
                    self.redis_cache.flushdb()
                
                # Disk cache implementation would go here
                elif level == CacheLevel.DISK:
                    pass
                
            except Exception as e:
                self.logger.error(f"Error clearing {level.value} cache: {str(e)}")
    
    def warm_cache(self, cache_type: str, **kwargs):
        """Warm cache with precomputed data"""
        if cache_type in self.warming_functions:
            try:
                self.warming_functions[cache_type](**kwargs)
                self.logger.info(f"Cache warming completed for {cache_type}")
            except Exception as e:
                self.logger.error(f"Cache warming failed for {cache_type}: {str(e)}")
    
    def register_warming_function(self, cache_type: str, func: Callable):
        """Register a cache warming function"""
        self.warming_functions[cache_type] = func
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        
        return {
            'overall': self.stats.to_dict(),
            'levels': {
                level.value: stats.to_dict() 
                for level, stats in self.level_stats.items()
            },
            'memory_cache': memory_stats,
            'redis_available': self.redis_cache is not None,
            'warming_functions': list(self.warming_functions.keys())
        }
    
    def create_cache_key(self, *args, **kwargs) -> str:
        """Create a consistent cache key from arguments"""
        # Create a deterministic key from arguments
        key_parts = []
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(hash(str(arg))))
        
        # Add keyword arguments (sorted for consistency)
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}:{v}")
            else:
                key_parts.append(f"{k}:{hash(str(v))}")
        
        # Create hash of combined key
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()


# Decorator for automatic caching
def cached(cache_manager: CacheManager, ttl_seconds: Optional[int] = None,
          levels: Optional[List[CacheLevel]] = None, 
          key_prefix: str = ""):
    """Decorator for automatic function result caching"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key
            func_key = f"{key_prefix}{func.__name__}"
            cache_key = cache_manager.create_cache_key(func_key, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key, levels)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.put(cache_key, result, ttl_seconds, levels)
            
            return result
        
        return wrapper
    return decorator

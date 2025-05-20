import redis
import logging
from typing import Any, Optional, Dict
from functools import wraps
import json
import time
from config import get_redis_config, get_cache_ttl, ERROR_MESSAGES, FEATURES
from redis.exceptions import ConnectionError, TimeoutError, RedisError
from threading import Lock

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self._redis_client = None
        self._in_memory_cache = {}
        self._connection_lock = Lock()
        self._last_connection_attempt = 0
        self._connection_retry_interval = 60  # seconds
        self._max_retries = 3
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection if enabled."""
        if not FEATURES['use_redis_cache']:
            logger.info("Redis cache is disabled in configuration")
            return
        
        with self._connection_lock:
            current_time = time.time()
            if (current_time - self._last_connection_attempt) < self._connection_retry_interval:
                return
            
            self._last_connection_attempt = current_time
            
            try:
                config = get_redis_config()
                self._redis_client = redis.Redis(**config)
                self._redis_client.ping()  # Test connection
                logger.info("Successfully connected to Redis")
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"{ERROR_MESSAGES['redis_connection_error']} Error: {str(e)}")
                self._redis_client = None
                FEATURES['use_redis_cache'] = False
            except Exception as e:
                logger.error(f"Unexpected Redis error: {str(e)}")
                self._redis_client = None
                FEATURES['use_redis_cache'] = False
    
    def _ensure_redis_connection(self):
        """Ensure Redis connection is active or attempt reconnection."""
        if not self._redis_client and FEATURES['use_redis_cache']:
            self._initialize_redis()
    
    def _handle_redis_operation(self, operation, *args, **kwargs):
        """Handle Redis operations with retries and fallback."""
        for attempt in range(self._max_retries):
            try:
                self._ensure_redis_connection()
                if self._redis_client:
                    return operation(*args, **kwargs)
                return None
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Redis operation failed (attempt {attempt + 1}/{self._max_retries}): {str(e)}")
                if attempt == self._max_retries - 1:
                    self._redis_client = None
                    FEATURES['use_redis_cache'] = False
                    logger.error(ERROR_MESSAGES['redis_timeout'])
                else:
                    time.sleep(1)  # Wait before retry
            except Exception as e:
                logger.error(f"Unexpected Redis error: {str(e)}")
                self._redis_client = None
                FEATURES['use_redis_cache'] = False
                break
        return None
    
    def _serialize(self, value: Any) -> str:
        """Serialize value to JSON string."""
        try:
            return json.dumps(value)
        except (TypeError, ValueError) as e:
            logger.error(f"Error serializing value: {str(e)}")
            return json.dumps(str(value))
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize JSON string to value."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error deserializing value: {str(e)}")
            return value
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Try Redis first if available
        if self._redis_client:
            value = self._handle_redis_operation(self._redis_client.get, key)
            if value is not None:
                return self._deserialize(value)
        
        # Fallback to in-memory cache
        if key in self._in_memory_cache:
            expiry = self._in_memory_cache.get(f"{key}_expiry")
            if expiry and time.time() > expiry:
                self.delete(key)
                return None
            return self._in_memory_cache[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        # Try Redis first if available
        if self._redis_client:
            serialized = self._serialize(value)
            success = self._handle_redis_operation(
                self._redis_client.setex if ttl else self._redis_client.set,
                key,
                ttl if ttl else value,
                serialized if ttl else None
            )
            if success:
                return True
        
        # Fallback to in-memory cache
        try:
            self._in_memory_cache[key] = value
            if ttl:
                expiry = time.time() + ttl
                self._in_memory_cache[f"{key}_expiry"] = expiry
            return True
        except Exception as e:
            logger.error(f"Error setting value in in-memory cache: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        success = True
        
        # Try Redis first if available
        if self._redis_client:
            redis_success = self._handle_redis_operation(self._redis_client.delete, key)
            success = success and bool(redis_success)
        
        # Also clean in-memory cache
        try:
            self._in_memory_cache.pop(key, None)
            self._in_memory_cache.pop(f"{key}_expiry", None)
        except Exception as e:
            logger.error(f"Error deleting from in-memory cache: {str(e)}")
            success = False
        
        return success
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        # Try Redis first if available
        if self._redis_client:
            exists = self._handle_redis_operation(self._redis_client.exists, key)
            if exists:
                return True
        
        # Check in-memory cache
        if key in self._in_memory_cache:
            expiry = self._in_memory_cache.get(f"{key}_expiry")
            if expiry and time.time() > expiry:
                self.delete(key)
                return False
            return True
        
        return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        success = True
        
        # Clear Redis if available
        if self._redis_client:
            try:
                self._redis_client.flushdb()
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {str(e)}")
                success = False
        
        # Clear in-memory cache
        try:
            self._in_memory_cache.clear()
        except Exception as e:
            logger.error(f"Error clearing in-memory cache: {str(e)}")
            success = False
        
        return success
    
    def get_many(self, keys: list) -> Dict[str, Any]:
        """Get multiple values from cache."""
        result = {}
        
        # Try Redis first if available
        if self._redis_client:
            values = self._handle_redis_operation(self._redis_client.mget, keys)
            if values:
                for key, value in zip(keys, values):
                    if value:
                        result[key] = self._deserialize(value)
        
        # Get remaining keys from in-memory cache
        missing_keys = [k for k in keys if k not in result]
        for key in missing_keys:
            if self.exists(key):
                result[key] = self._in_memory_cache[key]
        
        return result
    
    def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        success = True
        
        # Try Redis first if available
        if self._redis_client:
            try:
                serialized = {k: self._serialize(v) for k, v in mapping.items()}
                pipeline = self._redis_client.pipeline()
                for key, value in serialized.items():
                    if ttl:
                        pipeline.setex(key, ttl, value)
                    else:
                        pipeline.set(key, value)
                pipeline.execute()
            except Exception as e:
                logger.error(f"Error setting multiple values in Redis: {str(e)}")
                success = False
        
        # Also set in in-memory cache
        try:
            for key, value in mapping.items():
                self.set(key, value, ttl)
        except Exception as e:
            logger.error(f"Error setting multiple values in in-memory cache: {str(e)}")
            success = False
        
        return success

# Cache decorator
def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = CacheService()
            
            # Create cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Calculate and cache result
            result = func(*args, **kwargs)
            if result is not None:
                cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Global cache instance
cache_service = CacheService() 
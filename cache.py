"""
cache.py — Simple LRU cache for query responses
"""

from collections import OrderedDict
from config import settings


class LRUCache:
    def __init__(self, capacity=settings.CACHE_MAX_SIZE):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        """Get value from cache"""
        if key not in self.cache:
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        """Insert into cache"""
        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = value

        # Remove least recently used
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


# Global cache instance
query_cache = LRUCache()

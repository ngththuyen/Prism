from functools import lru_cache
from ratelimit import limits, sleep_and_retry
import time
import requests
import json
from pathlib import Path

class APIManager:
    CALLS = 10
    RATE_LIMIT = 60

    def __init__(self, max_retries=3, backoff_factor=2):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)

    @sleep_and_retry
    @limits(calls=CALLS, period=RATE_LIMIT)
    def call_api_with_retry(self, api_func, *args, **kwargs):
        """Generic API call with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return api_func(*args, **kwargs)
            except Exception as e:
                wait_time = self.backoff_factor ** attempt
                if attempt == self.max_retries - 1:
                    raise Exception(f"Max retries exceeded: {str(e)}")
                time.sleep(wait_time)

    @lru_cache(maxsize=100)
    def get_cached_response(self, prompt, cache_key):
        """Cache API responses to reduce API calls"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        
        response = self.call_api_with_retry(prompt)
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False)
        
        return response
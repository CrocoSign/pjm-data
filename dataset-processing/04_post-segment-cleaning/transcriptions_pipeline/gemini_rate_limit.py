import time
from typing import Any
import requests
import logging
import json
from pathlib import Path


class APILimit:
    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name
        if model_name == "gemini-2.5-pro":
            self.per_day: int = 50
            self.per_minute: int = 2
        if model_name == "gemini-2.5-flash":
            self.per_day: int = 100
            self.per_minute: int = 5


class GeminiAPIManager:
    def __init__(self, api_keys: list[str], model_name: str = "gemini-2.5-pro", stats_file: str = ".gemini_rate_limits.json") -> None:
        self.api_keys = api_keys
        self.current_key_index = 0
        self.model_name = model_name
        self.base_url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent"
        self.rate_limit = APILimit(model_name)
        self.stats_file = Path(stats_file)
        
        self.key_id_map = {f"key_{i}": key for i, key in enumerate(api_keys)}
        self.key_to_id = {key: f"key_{i}" for i, key in enumerate(api_keys)}
        
        self.key_stats = self._load_stats()
        
        for i in range(len(api_keys)):
            key_id = f"key_{i}"
            if key_id not in self.key_stats:
                self.key_stats[key_id] = {}
            
            if self.model_name not in self.key_stats[key_id]:
                self.key_stats[key_id][self.model_name] = {
                    "daily_requests": 0,
                    "daily_reset_time": time.time(),
                    "last_request_time": 0,
                    "requests_per_minute": 0,
                    "minute_reset_time": time.time(),
                }

    def _load_stats(self) -> dict:
        if self.stats_file.exists():
            try:
                with open(self.stats_file, "r") as f:
                    stats = json.load(f)
                    logging.info(f"Loaded rate limit stats from {self.stats_file}")
                    return stats
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Could not load stats file: {e}. Starting fresh.")
        return {}
    
    def _save_stats(self) -> None:
        try:
            with open(self.stats_file, "w") as f:
                json.dump(self.key_stats, f, indent=2)
        except IOError as e:
            logging.error(f"Could not save stats file: {e}")

    @property
    def current_api_key(self) -> str:
        return self.api_keys[self.current_key_index]

    def _switch_to_next_key(self) -> bool:
        original_index = self.current_key_index

        for _ in range(len(self.api_keys)):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

            if self.current_key_index == original_index:
                continue

            if self._key_has_capacity(f"key_{self.current_key_index}"):
                print(f"Switched to API key #{self.current_key_index + 1}")
                return True

        self.current_key_index = original_index
        return False

    def _key_has_capacity(self, key_id: str) -> bool:
        stats = self.key_stats[key_id][self.model_name]
        current_time = time.time()
        needs_save = False

        if current_time - stats["daily_reset_time"] >= 86400:
            stats["daily_requests"] = 0
            stats["daily_reset_time"] = current_time
            needs_save = True

        if current_time - stats["minute_reset_time"] >= 60:
            stats["requests_per_minute"] = 0
            stats["minute_reset_time"] = current_time
            needs_save = True
        
        if needs_save:
            self._save_stats()

        daily_ok = stats["daily_requests"] < self.rate_limit.per_day
        minute_ok = stats["requests_per_minute"] < self.rate_limit.per_minute

        return daily_ok and minute_ok

    def _enforce_rate_limit(self):
        max_retries = len(self.api_keys)

        for _ in range(max_retries):
            key_id = f"key_{self.current_key_index}"
            stats = self.key_stats[key_id][self.model_name]
            current_time = time.time()

            if self._key_has_capacity(key_id):
                time_since_last = current_time - stats["last_request_time"]
                if time_since_last < 1:
                    time.sleep(1 - time_since_last)

                stats["last_request_time"] = time.time()
                stats["daily_requests"] += 1
                stats["requests_per_minute"] += 1
                
                logging.debug(
                    f"API Key #{self.current_key_index + 1}: "
                    f"daily={stats['daily_requests']}/{self.rate_limit.per_day}, "
                    f"per_min={stats['requests_per_minute']}/{self.rate_limit.per_minute}"
                )
                
                self._save_stats()
                return

            if not self._switch_to_next_key():
                break

        raise Exception(
            logging.exception(f"All {len(self.api_keys)} API keys have reached their limits. Please wait or add more keys.")
        )

    def generate_content(self, prompt: str) -> dict[str, Any]:
        max_retries = 3

        for retry in range(max_retries):
            try:
                self._enforce_rate_limit()

                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.current_api_key,
                }

                data = {"contents": [{"parts": [{"text": prompt}]}]}

                response = requests.post(
                    self.base_url, headers=headers, json=data, timeout=30
                )

                if response.status_code == 429:
                    logging.warning(
                        f"Rate limit hit for key #{self.current_key_index + 1}, switching..."
                    )
                    if self._switch_to_next_key():
                        continue
                
                if response.status_code == 503:
                    # Service temporarily unavailable, retry with exponential backoff
                    wait_time = 2 ** retry
                    logging.warning(
                        f"Service unavailable (503), retrying in {wait_time}s... (attempt {retry + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503:
                    # 503 errors should be retried
                    wait_time = 2 ** retry
                    logging.warning(
                        f"Service unavailable (503), retrying in {wait_time}s... (attempt {retry + 1}/{max_retries})"
                    )
                    if retry < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        return {"error": f"Service unavailable after {max_retries} retries: {str(e)}"}
                elif retry == max_retries - 1:
                    return {"error": str(e)}
                time.sleep(2**retry)
            except requests.exceptions.RequestException as e:
                if retry == max_retries - 1:
                    return {"error": str(e)}
                time.sleep(2**retry)

        return {"error": "Max retries exceeded"}

    def get_usage_stats(self) -> dict:
        stats = {}
        for i in range(len(self.api_keys)):
            key_id = f"key_{i}"
            model_stats = self.key_stats[key_id].get(self.model_name, {})
            stats[f"key_{i + 1}"] = {
                "model": self.model_name,
                "daily_requests": model_stats.get("daily_requests", 0),
                "requests_per_minute": model_stats.get("requests_per_minute", 0),
                "is_current": i == self.current_key_index,
            }
        return stats

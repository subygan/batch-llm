import asyncio
import json
import os
from typing import Dict, List, Any, Generator
from pathlib import Path
import aiofiles
from datetime import datetime, timedelta

import openai
import tiktoken
from dataclasses import dataclass
from .base import BaseProcessor
from ..config import ServiceConfig


@dataclass
class ProcessingStats:
    total_tokens: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    start_time: datetime = None


class TokenCounter:
    def __init__(self):
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))


class RateLimiter:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_timestamps: List[datetime] = []
        self.token_usage: List[tuple[datetime, int]] = []

    async def wait_if_needed(self, tokens: int):
        now = datetime.now()

        # Clean up old timestamps
        cutoff = now - timedelta(minutes=1)
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]
        self.token_usage = [(ts, tok) for ts, tok in self.token_usage if ts > cutoff]

        # Check rate limits
        while len(self.request_timestamps) >= self.requests_per_minute or \
                sum(tok for _, tok in self.token_usage) + tokens >= self.tokens_per_minute:
            await asyncio.sleep(0.1)
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)
            self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]
            self.token_usage = [(ts, tok) for ts, tok in self.token_usage if ts > cutoff]

        self.request_timestamps.append(now)
        self.token_usage.append((now, tokens))


class EnhancedOpenAIProcessor(BaseProcessor):
    def __init__(self, config: ServiceConfig):
        self.config = config
        openai.api_key = config.api_key
        self.token_counter = TokenCounter()
        self.rate_limiter = RateLimiter(
            requests_per_minute=5000,
            tokens_per_minute=2_000_000
        )
        self.stats = ProcessingStats()
        self.stats.start_time = datetime.now()

    @staticmethod
    def chunk_file(filepath: Path, max_size_mb: int = 190) -> Generator[List[Dict], None, None]:
        """Split large JSONL files into chunks under max_size_mb"""
        current_chunk: List[Dict] = []
        current_size = 0

        with open(filepath, 'r') as f:
            for line in f:
                item = json.loads(line)
                item_size = len(line.encode('utf-8')) / (1024 * 1024)  # Size in MB

                if item_size > max_size_mb:
                    raise ValueError(f"Single item exceeds maximum size: {item_size}MB")

                if current_size + item_size > max_size_mb:
                    yield current_chunk
                    current_chunk = []
                    current_size = 0

                current_chunk.append(item)
                current_size += item_size

            if current_chunk:
                yield current_chunk

    @staticmethod
    def create_batches(items: List[Dict], max_batch_size: int = 50) -> List[List[Dict]]:
        """Split items into batches of max_batch_size"""
        return [items[i:i + max_batch_size] for i in range(0, len(items), max_batch_size)]

    async def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item with rate limiting"""
        try:
            prompt_tokens = self.token_counter.count_tokens(item["prompt"])
            await self.rate_limiter.wait_if_needed(prompt_tokens)

            response = await openai.ChatCompletion.create(
                model=self.config.model,
                messages=[{"role": "user", "content": item["prompt"]}]
            )

            result = {
                "id": item.get("id"),
                "prompt": item["prompt"],
                "response": response.choices[0].message.content,
                "status": "success"
            }

            self.stats.total_tokens += prompt_tokens + self.token_counter.count_tokens(result["response"])
            self.stats.successful_requests += 1

            return result

        except Exception as e:
            self.stats.failed_requests += 1
            return {
                "id": item.get("id"),
                "prompt": item["prompt"],
                "error": str(e),
                "status": "error"
            }

    async def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of items concurrently"""
        tasks = [self.process_single_item(item) for item in batch]
        self.stats.total_requests += len(batch)
        return await asyncio.gather(*tasks)

    async def process_directory(self, directory_path: str) -> Dict[str, Any]:
        """Process all JSONL files in a directory"""
        directory = Path(directory_path)
        all_results = []

        for filepath in directory.glob("*.jsonl"):
            async with aiofiles.open(f"{filepath}.processed", "w") as outfile:
                # Process file in chunks to respect file size limits
                for chunk in self.chunk_file(filepath):
                    # Create batches for concurrent processing
                    batches = self.create_batches(chunk)

                    for batch in batches:
                        results = await self.process_batch(batch)
                        all_results.extend(results)

                        # Write results immediately to handle large datasets
                        for result in results:
                            await outfile.write(json.dumps(result) + "\n")

        # Calculate processing statistics
        processing_time = (datetime.now() - self.stats.start_time).total_seconds()
        stats = {
            "total_files_processed": len(list(directory.glob("*.jsonl"))),
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "total_tokens": self.stats.total_tokens,
            "processing_time_seconds": processing_time,
            "tokens_per_second": self.stats.total_tokens / processing_time if processing_time > 0 else 0,
            "requests_per_second": self.stats.total_requests / processing_time if processing_time > 0 else 0
        }

        return {
            "results": all_results,
            "stats": stats
        }
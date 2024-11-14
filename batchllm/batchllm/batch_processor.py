import json
import asyncio
from typing import Optional, List, Dict, Any
from .config import ServiceConfig
from .processors.base import BaseProcessor
from .processors.openai_processor import OpenAIProcessor


class BatchProcessor:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.processor = self._get_processor()

    def _get_processor(self) -> BaseProcessor:
        if self.config.service_type.lower() == "openai":
            return OpenAIProcessor(self.config)
        # Add more providers here
        raise ValueError(f"Unsupported service type: {self.config.service_type}")

    async def process_file(self, input_path: str, output_path: str, batch_size: int = 10):
        # Read input JSONL
        with open(input_path, 'r') as f:
            items = [json.loads(line) for line in f]

        # Process in batches
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await self.processor.process_batch(batch)
            results.extend(batch_results)

        # Write output JSONL
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
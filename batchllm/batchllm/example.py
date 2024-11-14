import asyncio

from batchllm.batchllm.batch_processor import BatchProcessor
from batchllm.batchllm.config import ServiceConfig

if __name__ == "__main__":
    async def main():
        config = ServiceConfig(
            api_key="your-api-key",
            service_type="openai",
            model="gpt-3.5-turbo"
        )

        processor = BatchProcessor(config)
        await processor.process_file(
            input_path="input.jsonl",
            output_path="output.jsonl",
            batch_size=10
        )


    asyncio.run(main())
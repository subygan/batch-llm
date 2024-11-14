from dataclasses import dataclass
from typing import Optional

@dataclass
class ServiceConfig:
    api_key: str
    service_type: str = "openai"  # default to OpenAI
    model: str = "gpt-3.5-turbo"  # default model
    file_size: int = 200 #in mb
    max_retries: int = 3
    timeout: int = 30
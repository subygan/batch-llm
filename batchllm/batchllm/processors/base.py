from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseProcessor(ABC):
    @abstractmethod
    async def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseSensor(ABC):
    """Abstract base class for all hardware and simulated sensors."""
    def __init__(self, name: str, mock: bool = False):
        self.name = name
        self.mock = mock

    @abstractmethod
    def read(self) -> Dict[str, Any]:
        """Polls sensor and returns a dictionary of readings."""
        pass

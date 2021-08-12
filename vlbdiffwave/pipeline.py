from typing import Any


class Pipeline:
    """Residual pipeline for efficient gradient computation of MC variance regularizer.
    """
    def __init__(self):
        """Initializer.
        """
        self.memory = None
    
    def put(self, value: Any):
        """Assign value to the memory.
        Args:
            value: any types of value.
        """
        self.memory = value
    
    def get(self) -> Any:
        """Return memory and reset it.
        Returns:
            stored value.
        """
        mem = self.memory
        self.memory = None
        return mem

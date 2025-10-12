from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseEmbedder(ABC):
    def __init__(self, embedding_dim: int, method: str):
        self.embedding_dim = embedding_dim
        self.method = method
        logger.info(f"initialized {method} embedder ({embedding_dim}D)")

    @abstractmethod
    def encode(self, expressions: Union[str, List[str]]) -> np.ndarray:
        """Encode expressions to embeddings
        Args:
            expressions: single expression or list of expressions
        Returns:
            embeddings: numpy array [n_expressions, embedding_dim]
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.method}, {self.embedding_dim}D)"

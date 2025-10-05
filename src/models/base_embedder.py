from abc import ABC, abstractmethod
from typing import List, Union
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseEmbedder(ABC):
    """Base class for embedders compatible with IntegralSimilaritySearch."""
    def __init__(self, embedding_dim: int, method: str):
        self.embedding_dim = embedding_dim
        self.method = method
        self.is_fitted = False
        logger.info(f"initialized {method} embedder ({embedding_dim}D)")

    @abstractmethod
    def fit(self, expressions: List[str]) -> 'BaseEmbedder':
        """Fit embedder on expressions. Returns self."""
        pass

    @abstractmethod
    def encode(self, expressions: Union[str, List[str]]) -> np.ndarray:
        """Encode expressions to embeddings (n_expressions, embedding_dim)."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save fitted model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> 'BaseEmbedder':
        """Load fitted model from disk."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.method}, {self.embedding_dim}D, fitted={self.is_fitted})"

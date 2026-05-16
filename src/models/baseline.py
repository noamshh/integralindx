import json
import pickle
import hmac
import hashlib
import logging
import os
import numpy as np
from pathlib import Path
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from src.models.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


def _get_hmac_key() -> bytes:
    key = os.environ.get('INTEGRALINDX_VECTORIZER_HMAC_KEY')
    if not key:
        raise RuntimeError(
            "INTEGRALINDX_VECTORIZER_HMAC_KEY env var not set. "
            "Set it to a random secret before saving or loading the TF-IDF vectorizer."
        )
    return key.encode('utf-8')


def _compute_pickle_hmac(data: bytes) -> str:
    return hmac.new(_get_hmac_key(), data, hashlib.sha256).hexdigest()


def _verify_pickle_hmac(pkl_path: Path) -> None:
    hmac_path = pkl_path.with_suffix('.pkl.hmac')
    if not hmac_path.exists():
        logger.warning(f"No HMAC sidecar for {pkl_path} — skipping integrity check")
        return
    pkl_bytes = pkl_path.read_bytes()
    expected = hmac_path.read_text(encoding='utf-8').strip()
    actual = _compute_pickle_hmac(pkl_bytes)
    if not hmac.compare_digest(expected, actual):
        raise ValueError(f"HMAC verification failed for {pkl_path}. File may have been tampered with.")


class BaselineEmbedder(BaseEmbedder):
    """Baseline embedder supporting tf-idf and sentence-bert. tf-idf requires fit() before encode()"""
    def __init__(self, method: str = 'tfidf', embedding_dim: int = 384):
        super().__init__(embedding_dim=embedding_dim, method=method)
        self.is_fitted = False
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 3), analyzer='char', max_features=embedding_dim,
                lowercase=False, token_pattern=None
            )
        elif method == 'sentence_bert':
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
            self.is_fitted = True
        else:
            raise ValueError(f"unknown method: {method}")

    def fit(self, expressions: List[str]) -> 'BaselineEmbedder':
        if self.method == 'tfidf':
            logger.info(f"fitting TF-IDF on {len(expressions)} expressions")
            self.vectorizer.fit(expressions)
            self.is_fitted = True
        return self

    def encode(self, expressions: Union[str, List[str]]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("embedder not fitted. call fit() first for TF-IDF")
        if isinstance(expressions, str):
            expressions = [expressions]
        if self.method == 'tfidf':
            return self.vectorizer.transform(expressions).toarray()
        else:  # sentence_bert
            return self.model.encode(expressions)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {'method': self.method, 'embedding_dim': self.embedding_dim, 'is_fitted': self.is_fitted}
        if self.method == 'tfidf':
            vectorizer_path = path.with_suffix('.vectorizer.pkl')
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            # write HMAC sidecar so load() can verify file integrity
            pkl_bytes = vectorizer_path.read_bytes()
            hmac_path = vectorizer_path.with_suffix('.pkl.hmac')
            hmac_path.write_text(_compute_pickle_hmac(pkl_bytes), encoding='utf-8')
            model_data['vectorizer_path'] = str(vectorizer_path)
        with open(path, 'w') as f:
            json.dump(model_data, f)
        logger.info(f"saved {self.method} model to {path}")

    @classmethod
    def load(cls, path: Path) -> 'BaselineEmbedder':
        with open(path) as f:
            data = json.load(f)
        embedder = cls(method=data['method'], embedding_dim=data['embedding_dim'])
        embedder.is_fitted = data['is_fitted']
        if data['method'] == 'tfidf' and 'vectorizer_path' in data:
            vectorizer_path = Path(data['vectorizer_path'])
            _verify_pickle_hmac(vectorizer_path)
            with open(vectorizer_path, 'rb') as f:
                embedder.vectorizer = pickle.load(f)
        logger.info(f"loaded {embedder.method} model")
        return embedder
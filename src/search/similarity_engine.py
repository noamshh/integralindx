import numpy as np
import faiss
import logging
from typing import List, Dict, Optional
import sympy as sp

from src.models.base_embedder import BaseEmbedder
from corpus.formula_models import IntegrandGroup
from src.utils.variable_normalization import normalize_parameters

logger = logging.getLogger(__name__)


class IntegrandGroupSearch:
    """FAISS-based similarity search for integrand groups.
    Each group represents a unique canonical integrand with multiple instances."""
    def __init__(self, embedding_dim, index_type: str = 'flat'):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.groups: List[IntegrandGroup] = []
        self.embedder: Optional[BaseEmbedder] = None
        self._id_to_idx = {}  # mapping from group ID to index position
        self._hash_to_idx = {}  # mapping from integrand_hash to index position
        if index_type == 'flat':
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            raise ValueError(f"unsupported index type: {index_type}")
        logger.info(f"initialized {index_type} FAISS index for groups with {embedding_dim}D embeddings")

    def build_index(self, embedder: BaseEmbedder, batch_size: int = 1000) -> None:
        if not self.groups:
            raise ValueError("no integrand groups loaded")
        self.embedder = embedder
        logger.info(f"building index for {len(self.groups)} integrand groups")
        # extract canonical integrands for embedding
        canonical_integrands = [group.integrand_canonical for group in self.groups]
        # encode in batches
        all_embeddings = []
        for i in range(0, len(canonical_integrands), batch_size):
            batch = canonical_integrands[i:i + batch_size]
            batch_embeddings = embedder.encode(batch)
            all_embeddings.append(batch_embeddings)
        # concatenate all embeddings
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        logger.info(f"generated embeddings shape: {embeddings.shape}")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        logger.info(f"built FAISS index with {self.index.ntotal} vectors")

    def _convert_to_latex(self, canonical: str) -> str:
        try:
            expr = sp.sympify(canonical)
            return sp.latex(expr)
        except Exception as e:
            logger.warning(f"failed to convert '{canonical}' to latex: {e}")
            return canonical

    def search(self, query: str, k: int = 5, embedder=None) -> List[Dict]:
        """
        Search for similar integrand groups.
        Args:
            query: sympy integrand string
            k: number of results to return
            embedder: optional embedder to use (defaults to self.embedder)
        Returns:
            list of dicts with group info and similarity scores
        """
        embedder = embedder or self.embedder
        if embedder is None:
            raise ValueError("no embedder provided and index not built")
        # normalize query parameters to match dataset canonicalization
        normalized_query = normalize_parameters(query, integration_var='x')
        logger.debug(f"normalized query: '{query}' → '{normalized_query}'")
        # encode normalized query
        query_embedding = embedder.encode([normalized_query])
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        # search
        scores, indices = self.index.search(query_embedding, k)
        # format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # no more results
                break
            group = self.groups[idx]
            total_instances = len(group.definite_instances) + len(group.indefinite_instances)
            results.append({
                'group_id': group.id,
                'integrand_canonical': group.integrand_canonical,
                'integrand_latex': self._convert_to_latex(group.integrand_canonical),
                'integrand_family': group.integrand_family,
                'integrand_hash': group.integrand_hash,
                'total_instances': total_instances,
                'definite_count': len(group.definite_instances),
                'indefinite_count': len(group.indefinite_instances),
                'unique_mse_questions': len(group.unique_mse_questions),
                'latex_variants': group.latex_variants[:3],  # show first 3
                'similarity_score': float(score)
            })
        return results

    def get_group_by_hash(self, integrand_hash: str) -> Optional[Dict]:
        if integrand_hash not in self._hash_to_idx:
            return None
        idx = self._hash_to_idx[integrand_hash]
        group = self.groups[idx]

        return {
            'group_id': group.id,
            'integrand_canonical': group.integrand_canonical,
            'integrand_latex': self._convert_to_latex(group.integrand_canonical),
            'integrand_family': group.integrand_family,
            'integrand_hash': group.integrand_hash,
            'total_instances': len(group.definite_instances) + len(group.indefinite_instances),
            'definite_count': len(group.definite_instances),
            'indefinite_count': len(group.indefinite_instances),
            'unique_mse_questions': len(group.unique_mse_questions),
            'latex_variants': group.latex_variants,
            'definite_instances': group.definite_instances,
            'indefinite_instances': group.indefinite_instances,
            'instances': []  # will be populated by caller if needed
        }

    def get_statistics(self) -> Dict:
        total_instances = sum(
            len(g.definite_instances) + len(g.indefinite_instances)
            for g in self.groups
        )

        return {
            'available': True,
            'total_groups': len(self.groups),
            'total_instances': total_instances,
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_dim,
            'embedder': self.embedder.__class__.__name__ if self.embedder else None
        }

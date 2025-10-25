import json
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import faiss

from src.models.baseline import BaselineEmbedder
from src.utils.paths import get_paths

logger = logging.getLogger(__name__)


def get_cache_dir(embedder_name: str) -> Path:
    paths = get_paths()
    cache_root = paths['data']['root'] / 'embeddings'
    cache_dir = cache_root / embedder_name
    return cache_dir


def cache_exists(embedder_name: str) -> bool:
    cache_dir = get_cache_dir(embedder_name)
    if not cache_dir.exists():
        return False
    index_path = cache_dir / 'index.faiss'
    hash_mapping_path = cache_dir / 'index_to_hash.pkl'
    metadata_path = cache_dir / 'metadata.json'
    return index_path.exists() and hash_mapping_path.exists() and metadata_path.exists()


def save_embedding_cache(search_engine: 'IntegrandGroupSearch', embedder_name: str,
                         metadata: Optional[Dict[str, Any]] = None) -> Path:
    """Args:
        search_engine: IntegrandGroupSearch with built index
        embedder_name: embedder name
        metadata: optional metadata dict
    Returns:
        path to cache directory"""
    if search_engine.index is None or search_engine.index.ntotal == 0:
        raise ValueError("search engine index is empty, build index first")
    cache_dir = get_cache_dir(embedder_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / 'index.faiss'
    hash_mapping_path = cache_dir / 'index_to_hash.pkl'
    metadata_path = cache_dir / 'metadata.json'
    embedder_path = cache_dir / 'embedder.json'
    logger.info(f"saving embedding cache to {cache_dir}")
    faiss.write_index(search_engine.index, str(index_path))
    logger.info(f"saved FAISS index: {index_path}")
    index_to_hash = {idx: group.integrand_hash for idx, group in enumerate(search_engine.groups)}
    with open(hash_mapping_path, 'wb') as f:
        pickle.dump(index_to_hash, f)
    logger.info(f"saved index->hash mapping: {hash_mapping_path} ({len(index_to_hash)} entries)")
    embedder_saved = False
    if search_engine.embedder and hasattr(search_engine.embedder, 'save'):
        try:
            search_engine.embedder.save(embedder_path)
            embedder_saved = True
            logger.info(f"saved embedder: {embedder_path}")
        except Exception as e:
            logger.warning(f"failed to save embedder (queries will require reloading): {e}")
    cache_metadata = {
        'embedder_name': embedder_name,
        'index_type': search_engine.index_type,
        'embedding_dim': search_engine.embedding_dim,
        'total_groups': len(search_engine.groups),
        'index_size': search_engine.index.ntotal,
        'embedder_saved': embedder_saved,
        'created_at': datetime.utcnow().isoformat(),
        'embedder_class': search_engine.embedder.__class__.__name__ if search_engine.embedder else None,
    }
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (str, Path)) and ('path' in key.lower() or 'dir' in key.lower()):
                cache_metadata[key] = Path(value).as_posix()
            else:
                cache_metadata[key] = value
    with open(metadata_path, 'w') as f:
        json.dump(cache_metadata, f, indent=2)
    logger.info(f"saved metadata: {metadata_path}")
    logger.info(f"embedding cache saved successfully: {cache_dir}")
    return cache_dir


def load_embedding_cache(embedder_name: str, embedder: Optional[Any] = None, database: Optional[Any] = None) -> 'IntegrandGroupSearch':
    from src.search.similarity_engine import IntegrandGroupSearch
    cache_dir = get_cache_dir(embedder_name)
    if not cache_dir.exists():
        raise FileNotFoundError(f"embedding cache not found: {cache_dir}")
    index_path = cache_dir / 'index.faiss'
    hash_mapping_path = cache_dir / 'index_to_hash.pkl'
    metadata_path = cache_dir / 'metadata.json'
    embedder_path = cache_dir / 'embedder.json'
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not hash_mapping_path.exists():
        raise FileNotFoundError(f"hash mapping not found: {hash_mapping_path}")
    if not metadata_path.exists():
        logger.warning(f"metadata file not found: {metadata_path}")
    logger.info(f"loading embedding cache from {cache_dir}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    logger.info(f"cache metadata: {metadata}")
    with open(hash_mapping_path, 'rb') as f:
        index_to_hash = pickle.load(f)
    logger.info(f"loaded index->hash mapping: {len(index_to_hash)} entries")
    embedding_dim = metadata.get('embedding_dim', 384)
    index_type = metadata.get('index_type', 'flat')
    search_engine = IntegrandGroupSearch(embedding_dim=embedding_dim, index_type=index_type, database=database)
    index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    search_engine.index = index
    logger.info(f"loaded FAISS index (mmap): {index.ntotal} vectors")
    search_engine.index_to_hash = index_to_hash
    if embedder_path.exists() and embedder is None:
        try:
            embedder = BaselineEmbedder.load(embedder_path)
            logger.info(f"loaded embedder from cache: {embedder.method}")
        except Exception as e:
            logger.warning(f"failed to load embedder from cache: {e}")
    search_engine.embedder = embedder
    logger.info(f"embedding cache loaded successfully from {cache_dir}")
    return search_engine


def get_cache_info(embedder_name: str) -> Optional[Dict[str, Any]]:
    if not cache_exists(embedder_name):
        return None
    cache_dir = get_cache_dir(embedder_name)
    metadata_path = cache_dir / 'metadata.json'
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata['cache_dir'] = str(cache_dir)
        return metadata
    except Exception as e:
        logger.warning(f"failed to read cache metadata: {e}")
        return None


def list_available_caches() -> list[Dict[str, Any]]:
    paths = get_paths()
    cache_root = paths['data']['root'] / 'embeddings'
    if not cache_root.exists():
        return []
    caches = []
    for cache_dir in cache_root.iterdir():
        if not cache_dir.is_dir():
            continue
        embedder_name = cache_dir.name
        cache_info = get_cache_info(embedder_name)
        if cache_info:
            caches.append(cache_info)
    return caches


def delete_cache(embedder_name: str) -> bool:
    cache_dir = get_cache_dir(embedder_name)
    if not cache_dir.exists():
        logger.warning(f"cache does not exist: {cache_dir}")
        return False

    try:
        import shutil
        shutil.rmtree(cache_dir)
        logger.info(f"deleted cache: {cache_dir}")
        return True
    except Exception as e:
        logger.error(f"failed to delete cache {cache_dir}: {e}")
        return False

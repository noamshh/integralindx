import argparse
import logging
from pathlib import Path
import sys

from src.database.integral_db import IntegralDatabase
from src.models.baseline import BaselineEmbedder
from src.models.egen.contrastive_embedder import CLEmbedder
from src.search.similarity_engine import IntegrandGroupSearch
from src.search import embedding_cache
from src.utils.paths import get_paths

paths = get_paths()
PROJECT_ROOT = paths['project_root']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def is_baseline_embedder(embedder_name: str) -> bool:
    return embedder_name in ['tfidf', 'sentence_bert']


def get_checkpoint_path(embedder_name: str, explicit_checkpoint: Path = None) -> Path:
    if explicit_checkpoint:
        return explicit_checkpoint
    paths = get_paths()
    checkpoint_path = paths['models']['checkpoints'] / embedder_name / 'best_model.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found at: {checkpoint_path}\n"
                                f"Please train the model first or specify --checkpoint")
    return checkpoint_path


def prompt_overwrite(embedder_name: str) -> bool:
    response = input(f"Cache '{embedder_name}' exists. Overwrite? [Y/n] ").strip().lower()
    return response in ['y', 'yes', '']


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings and build cache for fast webapp loading")
    parser.add_argument('--embedder', required=True, help="Embedder name (e.g., 'tfidf', 'sentence_bert', 'ii-cl-400k-v1.0')")
    parser.add_argument('--checkpoint', type=Path, default=None, help="Path to trained model checkpoint")
    parser.add_argument('--config',  type=Path, default=None, help="Model config path (e.g., config/model/ii-cl-400k.yaml). Defaults to ii-cl-19m.yaml")
    parser.add_argument('--force', action='store_true', help="Force rebuild without confirmation if cache exists")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of groups (for testing)")
    args = parser.parse_args()
    embedder_name = args.embedder
    db_path = paths['data']['root'] / 'integral.db'
    if not db_path.exists():
        sys.exit(1)
    if embedding_cache.cache_exists(embedder_name):
        if not args.force:
            if not prompt_overwrite(embedder_name):
                sys.exit(0)
        logger.info(f"overwriting existing cache: {embedder_name}")
        embedding_cache.delete_cache(embedder_name)
    logger.info("loading database...")
    database = IntegralDatabase(db_path, fallback_to_jsonl=False)
    logger.info(f"database: {database.count_groups(exclude_curated=False):,} groups, "
               f"{database.count_instances(exclude_curated=False):,} instances")
    logger.info("loading integrand groups...")
    all_groups = database.get_all_groups(limit=20000, exclude_curated=True)
    if args.limit:
        all_groups = all_groups[:args.limit]
        logger.info(f"limited to {len(all_groups)} groups for testing")
    logger.info(f"loaded {len(all_groups)} integrand groups")
    logger.info(f"initializing embedder: {embedder_name}")
    if is_baseline_embedder(embedder_name):
        embedding_dim = 384
        embedder = BaselineEmbedder(method=embedder_name, embedding_dim=embedding_dim)
        if embedder_name == 'tfidf':
            logger.info("fitting TF-IDF on canonical integrands...")
            embedder.fit([g.integrand_canonical for g in all_groups])
        metadata = {'embedder_type': 'baseline', 'method': embedder_name}

    else:
        # trained E-Gen model
        checkpoint_path = get_checkpoint_path(embedder_name, args.checkpoint)
        logger.info(f"loading trained model from: {checkpoint_path}")
        if args.config:
            config_path = args.config
        else:
            # default to ii-cl-19m.yaml
            config_path = PROJECT_ROOT / 'config' / 'model' / 'ii-cl-19m.yaml'
            logger.info(f"no --config specified, defaulting to: {config_path}")
        embedder = CLEmbedder.load(checkpoint_path, config_path=config_path)
        embedding_dim = embedder.embedding_dim
        metadata = {
            'embedder_type': 'egen',
            'checkpoint_path': str(checkpoint_path),
            'config_path': str(config_path)
        }

    logger.info(f"embedder initialized: {embedding_dim}D")
    logger.info("building FAISS index...")
    search_engine = IntegrandGroupSearch(embedding_dim=embedding_dim, index_type='flat')
    search_engine.groups = all_groups
    search_engine.build_index(embedder, batch_size=100)
    logger.info(f"FAISS index built: {search_engine.index.ntotal} vectors")
    logger.info("saving embedding cache...")
    search_engine.save_cache(embedder_name, metadata=metadata)
    logger.info("=" * 80)
    logger.info("EMBEDDING CACHE BUILT SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Embedder:       {embedder_name}")
    logger.info(f"Groups:         {len(all_groups):,}")
    logger.info(f"Embedding dim:  {embedding_dim}D")
    cache_dir = embedding_cache.get_cache_dir(embedder_name)
    logger.info(f"Cache location: {cache_dir}")
    logger.info("")
    logger.info("To use this cache in the webapp:")
    logger.info(f"  python -m src.web.main --embedder {embedder_name}")


if __name__ == '__main__':
    main()

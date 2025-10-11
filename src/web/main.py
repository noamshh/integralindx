import os
import logging
import argparse
import traceback
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.utils.paths import get_paths
from src.database.integral_db import IntegralDatabase
from src.web.routers import search, system, groups, theory
from src.models.baseline import BaselineEmbedder
from src.models.egen.contrastive_embedder import ContrastiveLearningEmbedder
from src.search.similarity_engine import IntegrandGroupSearch
from src.search import embedding_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent
search_engine = None
database = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_engine, database
    try:
        embedder_name = os.environ.get('INTEGRALINDX_EMBEDDER', 'tfidf')
        dev_mode = os.environ.get('INTEGRALINDX_DEV_MODE', 'false').lower() == 'true'
        paths = get_paths()
        db_path = Path(paths['data']['integral_db'])
        if not db_path.exists():
            logger.error(f"database not found: {db_path}")
            return
        database = IntegralDatabase(db_path, fallback_to_jsonl=False)
        logger.info(f"loaded {database.count_groups(exclude_curated=False):,} groups, "
                   f"{database.count_instances(exclude_curated=False):,} instances")
        is_baseline = embedder_name in ['tfidf', 'sentence_bert']
        if embedding_cache.cache_exists(embedder_name):
            logger.info(f"loading embedding cache: {embedder_name}")
            cache_info = embedding_cache.get_cache_info(embedder_name)
            logger.info(f"cache info: {cache_info}")
            embedder = None
            if not is_baseline:
                checkpoint_path = Path(cache_info.get('checkpoint_path'))
                if not checkpoint_path.exists():
                    logger.error(f"checkpoint not found: {checkpoint_path}")
                    logger.info("please rebuild cache with: python scripts/build_embedding_cache.py")
                    return
                logger.info(f"loading E-Gen model from: {checkpoint_path}")
                embedder = ContrastiveLearningEmbedder.load(checkpoint_path)
            search_engine = IntegrandGroupSearch.load_cache(embedder_name, embedder)
            logger.info(f"search engine loaded from cache: {len(search_engine.groups)} groups")
        else:
            logger.warning(f"embedding cache not found: {embedder_name}")
            logger.info("building index from scratch (this may take a while)...")
            logger.info(f"tip: pre-build cache with: python scripts/build_embedding_cache.py --embedder {embedder_name}")
            all_groups = database.get_all_groups(limit=20000, exclude_curated=True)
            if is_baseline:
                embedding_dim = 384
                embedder = BaselineEmbedder(method=embedder_name, embedding_dim=embedding_dim)
                if embedder_name == 'tfidf':
                    logger.info("fitting TF-IDF on canonical integrands...")
                    embedder.fit([g.integrand_canonical for g in all_groups])
                metadata = {'embedder_type': 'baseline', 'method': embedder_name}
            else:
                checkpoint_path = paths['data']['root'] / 'models' / 'egen' / embedder_name / 'best_model.pt'
                if not checkpoint_path.exists():
                    logger.error(f"checkpoint not found: {checkpoint_path}")
                    logger.info("please train model first or pre-build cache")
                    return
                logger.info(f"loading E-Gen model from: {checkpoint_path}")
                embedder = ContrastiveLearningEmbedder.load(checkpoint_path)
                embedding_dim = embedder.embedding_dim
                metadata = {'embedder_type': 'egen', 'checkpoint_path': str(checkpoint_path)}
            search_engine = IntegrandGroupSearch(embedding_dim=embedding_dim, index_type='flat')
            search_engine.groups = all_groups
            for idx, group in enumerate(all_groups):
                search_engine._id_to_idx[group.id] = idx
                search_engine._hash_to_idx[group.integrand_hash] = idx
            search_engine.build_index(embedder, batch_size=100)
            logger.info(f"search engine ready: {len(all_groups)} groups, {embedder_name} embedder")
            logger.info("saving embedding cache for future startups...")
            search_engine.save_cache(embedder_name, metadata=metadata)
            logger.info("cache saved successfully")
        templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
        search.set_search_engine(search_engine, {embedder_name: search_engine.embedder})
        system.set_search_engine(search_engine)
        for router_module in [system, groups, theory]:
            router_module.set_templates(templates)
        groups.set_database(database)
        groups.set_dev_mode(dev_mode)

    except Exception as e:
        logger.error(f"initialization failed: {e}")
        traceback.print_exc()
    yield
    logger.info("shutting down...")

app = FastAPI(
    title="IntegralIndx - Integral Similarity Search",
    description="Search through Math StackExchange integrals",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(system.router, tags=["system"])
app.include_router(search.router, tags=["search"])
app.include_router(groups.router, tags=["groups"])
app.include_router(theory.router, tags=["theory"])

if (WEB_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

def main():
    parser = argparse.ArgumentParser(description="IntegralIndx Web Application")
    parser.add_argument('--embedder', default='tfidf')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--no-reload', action='store_true')
    parser.add_argument('--dev-mode', action='store_true')
    args = parser.parse_args()
    os.environ['INTEGRALINDX_EMBEDDER'] = args.embedder
    os.environ['INTEGRALINDX_DEV_MODE'] = 'true' if args.dev_mode else 'false'
    logger.info(f"starting webapp: {args.embedder} embedder, port {args.port}")
    uvicorn.run("src.web.main:app", host=args.host, port=args.port,
                reload=not args.no_reload, log_level="info")

if __name__ == "__main__":
    main()
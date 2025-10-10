#!/usr/bin/env python3
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
from src.search.similarity_engine import IntegrandGroupSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

search_engine = None
database = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_engine, database
    try:
        embedder_method = os.environ.get('INTEGRALINDX_EMBEDDER', 'tfidf')
        dev_mode = os.environ.get('INTEGRALINDX_DEV_MODE', 'false').lower() == 'true'
        paths = get_paths()

        db_path = Path(paths['data']['integral_db'])

        if not db_path.exists():
            logger.error(f"database not found: {db_path}")
            logger.info("run: python -m scripts.migrate_to_database --rebuild")
            return

        database = IntegralDatabase(db_path, fallback_to_jsonl=False)
        logger.info(f"loaded {database.count_groups(exclude_curated=False):,} groups, "
                   f"{database.count_instances(exclude_curated=False):,} instances")

        all_groups = database.get_all_groups(limit=20000, exclude_curated=True)
        search_engine = IntegrandGroupSearch(embedding_dim=384, index_type='flat')
        search_engine.groups = all_groups
        for idx, group in enumerate(all_groups):
            search_engine._id_to_idx[group.id] = idx
            search_engine._hash_to_idx[group.integrand_hash] = idx

        embedder = BaselineEmbedder(method=embedder_method, embedding_dim=384)
        embedder.fit([g.integrand_canonical for g in all_groups])
        search_engine.build_index(embedder)
        logger.info(f"search engine ready: {len(all_groups)} groups, {embedder_method} embedder")

        web_dir = Path(__file__).parent
        templates = Jinja2Templates(directory=str(web_dir / "templates"))

        search.set_search_engine(search_engine, {embedder_method: embedder})
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

web_dir = Path(__file__).parent
if (web_dir / "static").exists():
    app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")

def main():
    parser = argparse.ArgumentParser(description="IntegralIndx Web Application")
    parser.add_argument('--embedder', default='tfidf', choices=['tfidf', 'sentence_bert'])
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
import os
import logging
import argparse
import traceback
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.exceptions import HTTPException

from src.utils.paths import get_paths
from src.database.integral_db import IntegralDatabase
from src.web.routers import search, system, groups, theory
from src.models.egen.contrastive_embedder import CLEmbedder
from src.search.similarity_engine import IntegrandGroupSearch
from src.search import embedding_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent
search_engine = None
database = None
templates = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_engine, database, templates
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
        if not embedding_cache.cache_exists(embedder_name):
            logger.error(f"embedding cache not found: {embedder_name}")
            logger.error(f"pre-build cache with: python scripts/build_embedding_cache.py --embedder {embedder_name}")
            return

        logger.info(f"loading embedding cache: {embedder_name}")
        cache_info = embedding_cache.get_cache_info(embedder_name)
        logger.info(f"cache info: {cache_info}")
        embedder = None
        if not is_baseline:
            checkpoint_path = Path(cache_info.get('checkpoint_path'))
            if not checkpoint_path.exists():
                logger.error(f"checkpoint not found: {checkpoint_path}")
                return
            config_path_str = cache_info.get('config_path')
            config_path = Path(config_path_str) if config_path_str else None
            logger.info(f"loading E-Gen model from: {checkpoint_path}")
            embedder = CLEmbedder.load(checkpoint_path, config_path=config_path)
        search_engine = IntegrandGroupSearch.load_cache(embedder_name, embedder)
        logger.info(f"search engine loaded from cache: {len(search_engine.groups)} groups")
        templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
        umami_site_id = os.getenv("UMAMI_SITE_ID", "")
        search.set_search_engine(search_engine, {embedder_name: search_engine.embedder})
        system.set_search_engine(search_engine)
        system.set_umami(umami_site_id)
        for router_module in [system, groups, theory]:
            router_module.set_templates(templates)
        groups.set_database(database)
        groups.set_dev_mode(dev_mode)
        system.set_dev_mode(dev_mode)
        search.set_dev_mode(dev_mode)

    except Exception as e:
        logger.error(f"initialization failed: {e}")
        traceback.print_exc()
    yield
    logger.info("shutting down...")

app = FastAPI(title="IntegralIndx - Integral Similarity Search", description="Search through Math StackExchange integrals",
              version="1.0.0", lifespan=lifespan)

app.include_router(system.router, tags=["system"])
app.include_router(search.router, tags=["search"])
app.include_router(groups.router, tags=["groups"])
app.include_router(theory.router, tags=["theory"])

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    if templates is None:
        return HTMLResponse(content="<h1>404 - Page Not Found</h1>", status_code=404)
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

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
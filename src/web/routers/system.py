from fastapi import APIRouter, Request, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()

search_engine = None
templates = None
dev_mode = False
umami_site_id = ""

HEALTH_TOKEN = os.getenv("HEALTH_TOKEN", "no-health-token")

def set_search_engine(engine):
    global search_engine
    search_engine = engine

def set_templates(template_instance):
    global templates
    templates = template_instance

def set_dev_mode(enabled: bool):
    global dev_mode
    dev_mode = enabled
    logger.info(f"system router: dev mode {'enabled' if enabled else 'disabled'}")

def set_umami(site_id: str):
    global umami_site_id
    umami_site_id = site_id

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "IntegralIndx - Integral Similarity Search",
        "dev_mode": dev_mode,
        "umami_site_id": umami_site_id
    })

@router.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok"})

@router.get("/health")
async def health_check(x_health_token: str = Header(None)):
    if x_health_token != HEALTH_TOKEN:
        raise HTTPException(status_code=404, detail="Not Found")
    return JSONResponse({
        "status": "healthy",
        "search_engine_available": search_engine is not None,
        "inference_available": search_engine.embedder is not None if search_engine else False
    })
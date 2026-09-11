from fastapi import APIRouter, Request, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
import hmac
import logging
import os
import secrets as _secrets

logger = logging.getLogger(__name__)

router = APIRouter()

search_engine = None
templates = None
dev_mode = False

_env_health_token = os.getenv("HEALTH_TOKEN")
HEALTH_TOKEN = _env_health_token if _env_health_token else _secrets.token_hex(32)
if not _env_health_token:
    logger.warning("HEALTH_TOKEN env var not set — using ephemeral random token for this process")

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

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "IntegralIndx - Integral Similarity Search",
        "dev_mode": dev_mode,
    })

# Cloud Run's front end answers /healthz itself, so the page polls /api/health.
@router.get("/api/health")
@router.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok"})

@router.get("/health")
async def health_check(x_health_token: str = Header(None)):
    if not hmac.compare_digest(x_health_token or "", HEALTH_TOKEN):
        raise HTTPException(status_code=404, detail="Not Found")
    return JSONResponse({"status": "healthy"})
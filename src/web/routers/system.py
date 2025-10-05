from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

search_engine = None
templates = None

def set_search_engine(engine):
    global search_engine
    search_engine = engine

def set_templates(template_instance):
    global templates
    templates = template_instance

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "IntegralIndx - Integral Similarity Search"
    })

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "search_engine_available": search_engine is not None,
        "inference_available": search_engine.embedder is not None if search_engine else False
    }

@router.get("/stats")
async def get_statistics():
    if search_engine is None:
        return {
            "available": False,
            "message": "search engine not initialized"
        }

    try:
        return search_engine.get_statistics()
    except Exception as e:
        logger.error(f"failed to get statistics: {e}")
        return {
            "available": False,
            "error": str(e)
        }
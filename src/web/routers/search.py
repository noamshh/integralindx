from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
from sympy import sympify

from src.web.utils.query_parser import parse_query_to_sympy_integrand

logger = logging.getLogger(__name__)

router = APIRouter()

# global search engine - will be set by main.py
search_engine = None
available_embedders = {}
dev_mode = False

class MathQuery(BaseModel):
    query: str = Field(..., max_length=200, description="mathematical expression (max 200 characters)")
    k: int = 5
    embedder: Optional[str] = None

class ValidationRequest(BaseModel):
    expression: str = Field(..., max_length=200, description="mathematical expression (max 200 characters)")

class SimilarityResult(BaseModel):
    query: str
    parsed_query: Optional[str] = None
    embedder_used: Optional[str] = None
    results: List[Dict]
    success: bool
    message: str = ""

def set_search_engine(engine, embedders: Dict):
    global search_engine, available_embedders
    search_engine = engine
    available_embedders = embedders

def set_dev_mode(enabled: bool):
    global dev_mode
    dev_mode = enabled
    logger.info(f"search router: dev mode {'enabled' if enabled else 'disabled'}")

@router.post("/api/validate")
async def validate_expression(request: ValidationRequest):
    """Validate sympy"""
    try:
        expr = sympify(request.expression)
        return {"valid": True, "sympy": str(expr)}
    except Exception as e:
        return {"valid": False, "error": f"Invalid SymPy expression: {str(e)}"}

@router.get("/embedders")
async def get_available_embedders():
    embedder_list = list(available_embedders.keys())
    return {
        "embedders": embedder_list,
        "default": embedder_list[0] if embedder_list else "tfidf"
    }

@router.post("/search", response_model=SimilarityResult)
async def search_similar_integrands(query: MathQuery):
    if search_engine is None:
        raise HTTPException(status_code=503, detail="search engine not available")
    try:
        default_embedder = list(available_embedders.keys())[0] if available_embedders else "tfidf"
        embedder_name = query.embedder or default_embedder
        if embedder_name not in available_embedders:
            return SimilarityResult(
                query=query.query,
                parsed_query=None,
                embedder_used=embedder_name,
                results=[],
                success=False,
                message=f"embedder '{embedder_name}' not available. Available: {list(available_embedders.keys())}"
            )
        embedder = available_embedders[embedder_name]
        sympy_integrand, parse_message = parse_query_to_sympy_integrand(query.query)
        if sympy_integrand is None:
            error_msg = parse_message if dev_mode else "Invalid expression"
            return SimilarityResult(
                query=query.query,
                parsed_query=None,
                embedder_used=embedder_name,
                results=[],
                success=False,
                message=error_msg
            )
        k = min(max(query.k, 1), 20)
        results = search_engine.search(sympy_integrand, k=k, embedder=embedder)
        return SimilarityResult(
            query=query.query,
            parsed_query=sympy_integrand,
            embedder_used=embedder_name,
            results=results,
            success=True,
            message=f"found {len(results)} similar integrand groups using {embedder_name} ({parse_message})"
        )
    except Exception as e:
        logger.error(f"search failed: {e}")
        # sanitize error message in non-dev mode
        error_msg = f"search failed: {str(e)}" if dev_mode else "Invalid expression"
        return SimilarityResult(
            query=query.query,
            parsed_query=None,
            embedder_used=query.embedder,
            results=[],
            success=False,
            message=error_msg
        )
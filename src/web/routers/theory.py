from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()
templates = None

def set_templates(template_instance):
    global templates
    templates = template_instance

@router.get("/theory", response_class=HTMLResponse)
async def theory_page(request: Request):
    return templates.TemplateResponse("theory.html", {
        "request": request,
        "title": "Theory - IntegralIndx",
    })

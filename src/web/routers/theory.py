from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()
templates = None
umami_site_id = ""

def set_templates(template_instance):
    global templates
    templates = template_instance

def set_umami(site_id: str):
    global umami_site_id
    umami_site_id = site_id

@router.get("/theory", response_class=HTMLResponse)
async def theory_page(request: Request):
    return templates.TemplateResponse("theory.html", {
        "request": request,
        "title": "Theory - IntegralIndx",
        "umami_site_id": umami_site_id
    })

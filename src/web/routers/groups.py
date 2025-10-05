from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import logging
from typing import List, Dict, Optional
from sympy import sympify
from sympy.printing import latex

logger = logging.getLogger(__name__)
router = APIRouter()

# global state
database = None  # IntegralDatabase instance (set by main.py)
templates: Optional[Jinja2Templates] = None
dev_mode: bool = False


def set_database(db):
    global database
    database = db
    if db:
        logger.info("Database instance set for groups router")

def set_templates(t: Jinja2Templates):
    global templates
    templates = t

def set_dev_mode(enabled: bool):
    global dev_mode
    dev_mode = enabled
    logger.info(f"Developer mode: {'enabled' if enabled else 'disabled'}")

def get_group_id_by_integral_id(integral_id: str) -> Optional[str]:
    if database is None:
        return None
    instance = database.get_integral_instance(integral_id, exclude_curated=False)
    if instance is None or instance.integrand_hash is None:
        return None
    group = database.get_group_by_hash(instance.integrand_hash, exclude_curated=False)
    return group.id if group else None

def generate_representative_integrand_latex(integrand_canonical: str) -> Optional[str]:
    if not integrand_canonical:
        return None
    try:
        expr = sympify(integrand_canonical)
        latex_str = latex(expr)
        return f"\\int {latex_str} \\, dx"
    except Exception as e:
        logger.warning(f"failed to generate LaTeX for '{integrand_canonical}': {e}")
        return None

def extract_closed_forms(instances: List[Dict]) -> List[Dict]:
    closed_forms = []
    seen = set()
    for instance in instances:
        source_integral = instance.get('normalized_latex', '')
        integral_type = instance.get('integral_type', 'unknown')
        equivalent_forms = instance.get('equivalent_forms', [])
        author_name = instance.get('author_name')
        author_link = instance.get('author_link')
        mse_question_id = instance.get('mse_question_id')
        for form in equivalent_forms:
            if (r'\int' not in form and
                r'\eqref' not in form and
                r'\ref' not in form and
                '\lim' not in form and
                form not in seen):
                closed_forms.append({
                    'closed_form': form,
                    'source_integral': source_integral,
                    'integral_type': integral_type,
                    'author_name': author_name,
                    'author_link': author_link,
                    'mse_question_id': mse_question_id
                })
                seen.add(form)
    return closed_forms

@router.get("/integrand/{integrand_hash}", response_class=HTMLResponse)
async def view_group(request: Request, integrand_hash: str):
    if templates is None:
        raise HTTPException(status_code=500, detail="Templates not initialized")
    if database is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    exclude_curated = not dev_mode
    group = database.get_group_by_hash(integrand_hash, exclude_curated=exclude_curated)
    if group is None:
        raise HTTPException(status_code=404, detail=f"Integrand not found: {integrand_hash}")
    removed_ids = set(database.get_removed_instance_ids())
    curation_log = {entry['target_id']: entry for entry in database.get_curation_log()}
    all_instance_ids = group.definite_instances + group.indefinite_instances
    instances = []
    for instance_id in all_instance_ids:
        integral = database.get_integral_instance(instance_id, exclude_curated=False)
        if integral:
            is_removed = instance_id in removed_ids
            removal_reason = curation_log.get(instance_id, {}).get('reason', 'No reason provided') if is_removed else None
            instances.append({
                'id': integral.id,
                'raw_latex': integral.raw_latex,
                'normalized_latex': integral.normalized_latex,
                'equivalent_forms': integral.equivalent_forms if integral.equivalent_forms else [],
                'integral_type': integral.integral_type,
                'sympy_integrand': integral.sympy_integrand,
                'sympy_variable': integral.sympy_variable,
                'sympy_lower_bound': integral.sympy_lower_bound,
                'sympy_upper_bound': integral.sympy_upper_bound,
                'mse_question_id': integral.mse_question_id,
                'mse_answer_id': integral.mse_answer_id,
                'source_url': integral.source_url,
                'author_name': integral.author_name if hasattr(integral, 'author_name') else None,
                'author_link': integral.author_link if hasattr(integral, 'author_link') else None,
                'removed': is_removed,
                'removal_reason': removal_reason
            })

    mse_questions = set()
    for instance in instances:
        if instance['mse_question_id']:
            mse_questions.add(instance['mse_question_id'])
    mse_links = [{'question_id': qid, 'url': f"https://math.stackexchange.com/questions/{qid}"}
                 for qid in sorted(mse_questions)]
    representative_integrand = generate_representative_integrand_latex(group.integrand_canonical)
    closed_forms = extract_closed_forms(instances)
    return templates.TemplateResponse(
        "group_detail.html",
        {
            "request": request,
            "group": {
                'id': group.id,
                'integrand_canonical': group.integrand_canonical,
                'integrand_hash': group.integrand_hash,
                'integrand_family': group.integrand_family,
                'total_instances': len(instances),
                'definite_count': len(group.definite_instances),
                'indefinite_count': len(group.indefinite_instances),
                'unique_questions': len(mse_questions),
                'latex_variants': group.latex_variants,
                'representative_integrand': representative_integrand
            },
            "instances": instances,
            "mse_links": mse_links,
            "closed_forms": closed_forms,
            "dev_mode": dev_mode
        }
    )

@router.post("/integrand/{integrand_hash}/remove-instance")
async def remove_instance_from_group(integrand_hash: str, instance_id: str = Form(...), reason: str = Form(...)):
    """
    remove a specific integral instance (dev mode only)
    marks instance as removed in curation_log table - does not delete from database
    curation log persists across database rebuilds
    """
    if not dev_mode:
        raise HTTPException(status_code=403, detail="Developer mode not enabled")
    if database is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    instance = database.get_integral_instance(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail=f"Instance not found: {instance_id}")
    group = database.get_group_by_hash(integrand_hash)
    if group is None:
        raise HTTPException(status_code=404, detail=f"Group not found: {integrand_hash}")
    database.mark_instance_removed(
        instance_id,
        reason=reason,
        context={'integrand_hash': integrand_hash, 'group_id': group.id}
    )
    logger.info(f"marked instance as removed: {instance_id} (reason: {reason})")
    removed_ids = set(database.get_removed_instance_ids())
    all_instance_ids = set(group.definite_instances + group.indefinite_instances)
    remaining_instances = all_instance_ids - removed_ids
    if not remaining_instances:
        logger.info(f"group {group.id} is now orphaned (all instances removed)")
        return RedirectResponse(url="/", status_code=303)
    else:
        logger.info(f"group {group.id} has {len(remaining_instances)} remaining instances")
        return RedirectResponse(url=f"/integrand/{integrand_hash}", status_code=303)

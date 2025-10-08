import json
from pathlib import Path
from typing import Optional, Tuple, Dict

# cache for raw formula lookups (source_id -> blob_path)
_blob_path_cache: Dict[str, Optional[str]] = {}

def load_raw_formula_blob_paths(raw_formulas_path: str = None) -> Dict[str, Dict[str, str]]:
    """
    Load mapping of source_id to blob paths from raw formulas JSONL
    Returns:
        dict mapping source_id to {'raw_blob': path, 'answer_blob': path (optional)}
    """
    global _blob_path_cache
    if _blob_path_cache:
        return _blob_path_cache
    if raw_formulas_path is None:
        from src.utils.paths import get_paths
        paths = get_paths()
        raw_formulas_path = paths['data']['raw']['formulas'] / 'mse.jsonl'

    blob_map = {}
    try:
        with open(raw_formulas_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    source_id = data.get('id')
                    blob_paths = data.get('blob_paths', {})
                    if source_id and blob_paths:
                        # store both raw_blob and answer_blob if available
                        blob_entry = {}
                        if blob_paths.get('raw_blob'):
                            blob_entry['raw_blob'] = blob_paths['raw_blob']
                        if blob_paths.get('answer_blob'):
                            blob_entry['answer_blob'] = blob_paths['answer_blob']

                        if blob_entry:
                            blob_map[source_id] = blob_entry
                except Exception:
                    continue
        _blob_path_cache = blob_map
    except Exception:
        pass
    return blob_map

def extract_author_from_blob(blob_path: str, item_id: int, origin: str) -> Tuple[Optional[str], Optional[str]]:
    """
    extract author name and profile link from MSE blob JSON
    args:
        blob_path: path to raw blob file (mse_*.txt)
        item_id: MSE question/answer ID from provenance
        origin: origin string (e.g., 'question_body', 'answer_body')

    Returns:
        tuple of (author_name, author_link) or (None, None) if not found
    """
    try:
        blob_file = Path(blob_path)
        if not blob_file.exists():
            return None, None
        with open(blob_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items = data.get('items', [])
        if not items:
            return None, None
        # determine if looking for question or answer author
        is_answer = 'answer' in origin
        for item in items:
            if is_answer:
                # for answers, match answer_id
                if item.get('answer_id') == item_id:
                    owner = item.get('owner', {})
                    return owner.get('display_name'), owner.get('link')
            else:
                # for questions, match question_id
                if item.get('question_id') == item_id:
                    owner = item.get('owner', {})
                    return owner.get('display_name'), owner.get('link')
        # if not found in answers, blob may not contain answer data
        # try to extract from question as fallback
        if is_answer and items:
            # answer blob may not be available, return None
            return None, None
        return None, None
    except Exception:
        # silently handle errors (file read issues, JSON parse errors)
        return None, None

def extract_author_from_formula(normalized_formula, blob_map: Dict[str, Dict[str, str]] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    extract author from normalized formula using provenance and blob lookup
    args:
        normalized_formula: NormalizedFormula object
        blob_map: optional preloaded blob path mapping (for performance)

    returns:
        tuple of (author_name, author_link) or (None, None)
    """
    if blob_map is None:
        blob_map = load_raw_formula_blob_paths()

    try:
        source_id = normalized_formula.source_id
        provenance = normalized_formula.provenance or {}
        item_id = provenance.get('item_id')
        origin = provenance.get('origin', '')

        if not source_id or not item_id or not origin:
            return None, None

        blob_paths = blob_map.get(source_id)
        if not blob_paths:
            return None, None
        # prioritize answer_blob for answer-origin formulas
        is_answer = 'answer' in origin
        blob_path = None
        if is_answer and blob_paths.get('answer_blob'):
            blob_path = blob_paths['answer_blob']
        elif blob_paths.get('raw_blob'):
            blob_path = blob_paths['raw_blob']
        if not blob_path:
            return None, None
        return extract_author_from_blob(blob_path, item_id, origin)
    except Exception:
        return None, None

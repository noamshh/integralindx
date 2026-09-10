"""
OcrIntegralRecord — a sibling to corpus.formula_models.IntegralFormula for records
produced by the OCR pipeline (corpus/ocr/). Unlike IntegralFormula, OCR records:
  * carry both `integrand_latex` AND `integrand_sympy` side-by-side
  * carry a solution (latex + sympy) when the source provides one
  * carry an OcrSourceMetadata block with extensible per-source-type `extra` fields

Source-type conventions for OcrSourceMetadata.extra:
  source_type="mse"        -> mse_question_id, mse_answer_id, mse_post_score,
                              mse_tags, mse_user_reputation
  source_type="arxiv"      -> arxiv_id, arxiv_version, doi, coauthors (list),
                              primary_category, submission_date
  source_type="book"       -> isbn, edition, publisher, year, chapter, section
  source_type="screenshot" -> captured_from_url, capture_timestamp

New source types: add a new `source_type` string and put fields in `extra`. No
schema migration needed.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from corpus.formula_models import BaseFormula
from src.utils.provenance import now_iso


@dataclass
class OcrSourceMetadata:
    source_type: str
    source_id: str
    source_title: Optional[str] = None
    source_url: Optional[str] = None
    page_number: Optional[int] = None
    page_image_path: Optional[str] = None
    page_image_sha256: Optional[str] = None
    bbox: Optional[List[float]] = None
    label: Optional[str] = None
    license: Optional[str] = None

    author_name: Optional[str] = None
    author_profile_url: Optional[str] = None
    author_id: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OcrSourceMetadata":
        return cls(**d)


@dataclass
class OcrIntegralRecord(BaseFormula):
    integrand_latex: str = ""
    integrand_sympy: str = ""
    variable: str = "x"
    lower_limit_latex: Optional[str] = None
    upper_limit_latex: Optional[str] = None
    lower_limit_sympy: Optional[str] = None
    upper_limit_sympy: Optional[str] = None
    integral_type: str = "indefinite"

    solution_latex: Optional[str] = None
    solution_sympy: Optional[str] = None
    has_solution: bool = False

    constants: List[Dict[str, str]] = field(default_factory=list)
    conditions: Optional[str] = None

    integrand_canonical: Optional[str] = None
    integrand_hash: Optional[str] = None

    source: Optional[OcrSourceMetadata] = None
    ocr_model: str = ""
    ocr_prompt_version: str = ""
    raw_model_output: Optional[str] = None
    created_at: str = field(default_factory=now_iso)

    @classmethod
    def make_id(cls, source_id: str, page: Optional[int], idx: int) -> str:
        page_part = f"p{page}" if page is not None else "p_"
        return f"ocr-{source_id}-{page_part}-{idx:03d}"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OcrIntegralRecord":
        d = dict(d)
        src = d.pop("source", None)
        if isinstance(src, dict):
            src = OcrSourceMetadata.from_dict(src)
        return cls(source=src, **d)

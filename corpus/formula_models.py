import json, hashlib
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Set
from pathlib import Path
import unicodedata

from src.utils.provenance import now_iso, sha256_hex

@dataclass
class BaseFormula:
    id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(**d)

    def compute_checksum(self, field_name: str = "raw_latex") -> Optional[str]:
        content = getattr(self, field_name, None)
        if not content:
            self.checksum = None
            return None
        checksum_value = "sha256:" + sha256_hex(_canonicalize_for_hash(content))
        self.checksum = checksum_value
        return checksum_value

@dataclass
class RawFormula(BaseFormula):
    raw_latex: str
    source: str
    source_url: str
    latex_origin: Optional[str] = None
    context_snippet: Optional[str] = None
    license: Optional[Dict[str, str]] = None
    scrape_timestamp: str = field(default_factory=now_iso)
    blob_paths: Optional[Dict[str, str]] = field(default_factory=dict)
    provenance: Optional[Dict[str, Any]] = field(default_factory=dict)
    checksum: Optional[str] = None
    created_at: str = field(default_factory=now_iso)

    @classmethod
    def make_id(cls, source: str, raw_latex: str) -> str:
        key = (source or "unknown") + ":" + (raw_latex or "")
        return (source or "unknown") + "-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]



@dataclass
class NormalizedFormula(BaseFormula):
    """Result of normalizing a raw LaTeX formula, keeping first expression as main and rest as equivalent forms"""
    raw_latex: str
    source_id: str
    accepted: bool
    leading_expression: Optional[str] = None           # main/first expression from chain
    equivalent_forms: List[str] = field(default_factory=list)  # other expressions from chain
    parametric_conditions: Optional[str] = None        # conditions like "for x > 0"
    rejection_reason: Optional[str] = None             # why rejected if not accepted
    provenance: Optional[Dict[str, Any]] = field(default_factory=dict)
    checksum: Optional[str] = None
    created_at: str = field(default_factory=now_iso)

    @classmethod
    def make_id(cls, source_id: str, raw_latex: str) -> str:
        """Create unique ID based on source and raw LaTeX"""
        key = (source_id or "unknown") + ":" + (raw_latex or "")
        return "normalized-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def rejected(cls, source_id: str, raw_latex: str, reason: str) -> "NormalizedFormula":
        """Create rejected NormalizedFormula"""
        return cls(
            id=cls.make_id(source_id, raw_latex),
            source_id=source_id,
            raw_latex=raw_latex,
            accepted=False,
            rejection_reason=reason
        )

    @classmethod
    def accepted(cls, source_id: str, raw_latex: str, leading_expression: str, 
                 equivalent_forms: List[str] = None, conditions: str = None) -> "NormalizedFormula":
        """Create accepted NormalizedFormula"""
        return cls(
            id=cls.make_id(source_id, raw_latex),
            source_id=source_id,
            raw_latex=raw_latex,
            accepted=True,
            leading_expression=leading_expression,
            equivalent_forms=equivalent_forms or [],
            parametric_conditions=conditions
        )


@dataclass
class IntegralFormula(BaseFormula):
    raw_latex: str
    source_id: str
    normalized_source_id: str
    chain_position: int
    mse_question_id: int
    source_url: str
    normalized_latex: str
    sympy_integrand: str
    sympy_variable: str
    integral_type: str
    parsing_success: bool
    mse_answer_id: Optional[int] = None
    sympy_lower_bound: Optional[str] = None
    sympy_upper_bound: Optional[str] = None
    parsing_error: Optional[str] = None
    equivalent_forms: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    provenance: Optional[Dict[str, Any]] = field(default_factory=dict)
    checksum: Optional[str] = None
    normalized_content_checksum: Optional[str] = None
    integrand_canonical: Optional[str] = None
    integrand_hash: Optional[str] = None
    integrand_family: Optional[str] = None
    author_name: Optional[str] = None
    author_link: Optional[str] = None
    created_at: str = field(default_factory=now_iso)

    @classmethod
    def make_id(cls, normalized_source_id: str, normalized_latex: str, chain_position: int) -> str:
        key = f"{normalized_source_id}:{normalized_latex}:{chain_position}"
        return "integral-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def from_sympy_parsing(cls, normalized_formula: "NormalizedFormula",
                          normalized_latex: str, chain_position: int,
                          sympy_result: Dict[str, str], equivalent_forms: List[str],
                          author_name: Optional[str] = None,
                          author_link: Optional[str] = None) -> "IntegralFormula":
        from src.utils.integrand_canonicalization import canonicalize_integrand
        mse_question_id = normalized_formula.provenance.get('item_id', 0) if normalized_formula.provenance else 0
        mse_answer_id = None
        origin = normalized_formula.provenance.get('origin', '') if normalized_formula.provenance else ''
        if 'answer' in origin:
            mse_answer_id = mse_question_id
        integrand_canonical, integrand_hash = canonicalize_integrand(sympy_result['integrand'])
        integrand_family = None

        return cls(
            id=cls.make_id(normalized_formula.id, normalized_latex, chain_position),
            source_id=normalized_formula.source_id,
            normalized_source_id=normalized_formula.id,
            chain_position=chain_position,
            mse_question_id=mse_question_id,
            mse_answer_id=mse_answer_id,
            source_url="",
            raw_latex=normalized_formula.raw_latex,
            normalized_latex=normalized_latex,
            sympy_integrand=sympy_result['integrand'],
            sympy_variable=sympy_result['variable'],
            sympy_lower_bound=sympy_result.get('lower_bound'),
            sympy_upper_bound=sympy_result.get('upper_bound'),
            integral_type=sympy_result['integral_type'],
            parsing_success=sympy_result['parsing_success'],
            parsing_error=sympy_result.get('parsing_error'),
            equivalent_forms=equivalent_forms,
            normalized_content_checksum=normalized_formula.checksum,
            integrand_canonical=integrand_canonical,
            integrand_hash=integrand_hash,
            integrand_family=integrand_family,
            author_name=author_name,
            author_link=author_link
        )


@dataclass
class IntegrandGroup(BaseFormula):
    """Deduplicated group of integrals with same canonical integrand"""
    integrand_canonical: str
    integrand_hash: str
    indefinite_instances: List[str] = field(default_factory=list)  # IntegralFormula IDs
    definite_instances: List[str] = field(default_factory=list)    # IntegralFormula IDs
    unique_mse_questions: Set[int] = field(default_factory=set)
    latex_variants: List[str] = field(default_factory=list)
    integrand_family: Optional[str] = None
    checksum: Optional[str] = None
    created_at: str = field(default_factory=now_iso)

    @classmethod
    def make_id(cls, integrand_hash: str) -> str:
        return f"integrand-group-{integrand_hash}"

    def to_dict(self) -> dict:
        """Convert to dictionary with JSON-serializable types"""
        return {
            'id': self.id,
            'integrand_canonical': self.integrand_canonical,
            'integrand_hash': self.integrand_hash,
            'indefinite_instances': self.indefinite_instances,
            'definite_instances': self.definite_instances,
            'unique_mse_questions': sorted(list(self.unique_mse_questions)),  # convert set to sorted list
            'latex_variants': self.latex_variants,
            'integrand_family': self.integrand_family,
            'checksum': self.checksum,
            'created_at': self.created_at
        }


def append_jsonl(path: str, item) -> None:
    """Append a single item (with .to_json() method) to JSONL file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(item.to_json() + "\n")


def write_jsonl(path: str, items: List) -> None:
    """Write a list of items (with .to_json() method) to JSONL file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.to_json() + "\n")

def _canonicalize_for_hash(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = " ".join(s.split())
    return s
